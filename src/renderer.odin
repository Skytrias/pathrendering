package src

import "core:mem"
import "core:fmt"
import "core:math"
import "core:slice"
import "core:runtime"
import "core:strings"
import "core:math/rand"
import glm "core:math/linalg/glsl"
import sa "core:container/small_array"
import gl "vendor:OpenGL"

MAX_CURVES :: 512
MAX_PATHS :: 512

SIZE_LANES :: mem.Megabyte
SIZE_FRAGMENTS :: mem.Megabyte
SIZE_COVERAGE :: mem.Megabyte

shader_vert := #load("shaders/vertex.glsl")
shader_frag := #load("shaders/fragment.glsl")
shader_header := #load("shaders/compute/header.comp")
shader_intersection0 := #load("shaders/compute/intersection0.comp")
shader_intersection1 := #load("shaders/compute/intersection1.comp")
shader_gen := #load("shaders/compute/gen.comp")
shader_mark := #load("shaders/compute/mark.comp")
shader_raster := #load("shaders/compute/raster.comp")

Xform :: [6]f32

Vertex :: struct {
	pos: [2]f32,
	uv: [2]f32,
}

KAPPA90 :: 0.5522847493

Renderer :: struct {
	// raw curves that were inserted by the user
	curves: []Curve,
	curve_index: int,
	curve_last: [2]f32, // temp last point in path creation
	curve_last_control: [2]f32, // temp last point in path creation

	// indices that will get advanced on the gpu
	indices: Indices,

	// paths per curve shape
	paths: []Path,
	path_index: int,

	fragments: []Fragment,
	lanes: []Lane,

	// font data
	font_pool: Pool(8, Font),

	// platform dependent implementation
	gpu: Renderer_GL,

	window_width: int,
	window_height: int,
}

Renderer_GL :: struct {
	fill: struct {
		vao: u32,
		vbo: u32,
		program: u32,
		loc_projection: i32,
	},

	raster_program: u32,
	raster_texture_id: u32,
	raster_texture_width: int,
	raster_texture_height: int,

	implicitize_program: u32,
	intersction0_program: u32,
	intersction1_program: u32,
	gen_program: u32,
	mark_program: u32,

	ssbo: struct {
		indices: u32,
		curves: u32,
		implicit_curves: u32,
		paths: u32,

		lanes: u32,
		fragments: u32,
		coverage: u32,
	},
}

// indices that will get advanced in compute shaders!
Indices :: struct #packed {
	paths: i32,
	fragments: i32,

	window_width: i32,
	window_height: i32,
}

// curve Linear, Quadratic, Cubic in flat structure
Curve :: struct #packed {
	B: [4][2]f32,
	count: i32, // 0-2 + 1
	path_index: i32,
}

Path :: struct #packed {
	color: [4]f32,
	box: [4]f32, // bounding box for all inserted vertices,
	clip: [4]f32,

	xform: Xform, // transformation used throughout vertice creation

	curve_index_start: i32,
	curve_index_current: i32,
}

Fragment :: struct #packed {
	x: i32,
	y: i32,
	
	index: i32,
	path_index: i32,
	
	winding_number: i32,
	
	fragment_flag: b32,
	span_flag: b32,

	fragment_next: i32,
}

Lane :: struct #packed {
	fragment_head: i32,
	fragment_tail: i32,
}

renderer_init :: proc(renderer: ^Renderer) {
	renderer.curves = make([]Curve, MAX_CURVES)
	renderer.paths = make([]Path, MAX_PATHS)
	renderer.fragments = make([]Fragment, 10_000)
	renderer.lanes = make([]Lane, 10_000)

	pool_clear(&renderer.font_pool)
	renderer_gpu_gl_init(&renderer.gpu)
}

renderer_make :: proc() -> (res: Renderer) {
	renderer_init(&res)
	return
}

renderer_destroy :: proc(renderer: ^Renderer) {
	delete(renderer.curves)
	delete(renderer.paths)
	delete(renderer.fragments)
	delete(renderer.lanes)
	renderer_gpu_gl_destroy(&renderer.gpu)
	// TODO destroy fonts
}

renderer_start :: proc(renderer: ^Renderer, width, height: int) {
	renderer.curve_index = 0
	renderer.indices = {}
	renderer.indices.window_width = i32(width)
	renderer.indices.window_height = i32(height)
	renderer.path_index = 0
	path_init(&renderer.paths[0], f32(width), f32(height))

	renderer.window_width = width
	renderer.window_height = height
	renderer_gpu_gl_start(renderer)
}

renderer_end :: proc(using renderer: ^Renderer) {
	renderer_gpu_gl_end(renderer)
}

renderer_text_push :: proc(
	using renderer: ^Renderer,
	text: string,
	x, y: f32,
	size: f32,
) {
	font := pool_at(&font_pool, 1)
	
	x_start := x
	x := x
	y := y
	for codepoint in text {
		x += renderer_font_glyph(renderer, font, codepoint, x, y, size)
		renderer_path_transition(renderer)
		// if x > 500 {
		// 	x = x_start
		// 	y += size
		// }
	}
}

renderer_font_push :: proc(renderer: ^Renderer, path: string) -> u32 {
	index := pool_alloc_index(&renderer.font_pool)
	font := pool_at(&renderer.font_pool, index)
	font_init(font, path)
	return index
}

// build unified compute shader with a shared header
renderer_gpu_shader_compute :: proc(
	builder: ^strings.Builder,
	header: []byte,
	data: []byte,
	x, y: int,
	loc := #caller_location,
) -> u32 {
	strings.builder_reset(builder)

	// write version + sizes
	fmt.sbprintf(builder, `#version 450 core
#extension GL_ARB_gpu_shader_int64 : enable
layout(local_size_x = %d, local_size_y = %d) in;
`, x, y)

	// write header
	strings.write_bytes(builder, header)
	strings.write_byte(builder, '\n')

	// write rest of the data
	strings.write_bytes(builder, data)

	// write result
	program, ok := gl.load_compute_source(strings.to_string(builder^))
	if !ok {
		panic("failed loading compute shader", loc)
	}

	return program
}

renderer_gpu_gl_init :: proc(using gpu: ^Renderer_GL) {
	{
		gl.GenVertexArrays(1, &fill.vao)
		gl.BindVertexArray(fill.vao)
		defer gl.BindVertexArray(0)

		gl.GenBuffers(1, &fill.vbo)
		gl.BindBuffer(gl.ARRAY_BUFFER, fill.vbo)

		size := i32(size_of(Vertex))
		gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, size, 0)
		gl.EnableVertexAttribArray(0)
		gl.VertexAttribPointer(1, 2, gl.FLOAT, gl.FALSE, size, offset_of(Vertex, uv))
		gl.EnableVertexAttribArray(1)

		ok: bool
		fill.program, ok = gl.load_shaders_source(string(shader_vert), string(shader_frag))
		if !ok {
			panic("failed loading frag/vert shader")
		}

		fill.loc_projection = gl.GetUniformLocation(fill.program, "projection")
	}

	builder := strings.builder_make(mem.Kilobyte * 4, context.temp_allocator)

	{
		raster_program = renderer_gpu_shader_compute(&builder, shader_header, shader_raster, 1, 1)

		gl.GenTextures(1, &raster_texture_id)
		gl.BindTexture(gl.TEXTURE_2D, raster_texture_id)
		gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
		gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
		gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
		gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
		gl.BindImageTexture(0, raster_texture_id, 0, gl.FALSE, 0, gl.WRITE_ONLY, gl.RGBA8)

		{
			// TODO dynamic width/height or finite big one
			raster_texture_width = 2560
			raster_texture_height = 1080
			gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, i32(raster_texture_width), i32(raster_texture_height), 0, gl.RGBA, gl.UNSIGNED_BYTE, nil)
		}

		gl.BindTexture(gl.TEXTURE_2D, 0)
	}

	intersction0_program = renderer_gpu_shader_compute(&builder, shader_header, shader_intersection0, 1, 1)
	gen_program = renderer_gpu_shader_compute(&builder, shader_header, shader_gen, 1, 1)
	mark_program = renderer_gpu_shader_compute(&builder, shader_header, shader_mark, 1, 1)

	// TODO revisit STREAM/STATIC?
	create :: proc(base: u32, size: int) -> (index: u32) {
		gl.CreateBuffers(1, &index)
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, index)
		gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, base, index)
		gl.NamedBufferData(index, size, nil, gl.STREAM_DRAW)
		fmt.eprintln("SSBO", base, size, size / mem.Kilobyte, size / mem.Megabyte)
		return index
	}

	ssbo.indices = create(0, 1 * size_of(Indices))
	ssbo.curves = create(1, MAX_CURVES * size_of(Curve))
	ssbo.paths = create(2, MAX_PATHS * size_of(Path))

	ssbo.lanes = create(3, SIZE_LANES)
	ssbo.fragments = create(4, SIZE_FRAGMENTS)
	ssbo.coverage = create(5, SIZE_COVERAGE)
}

renderer_gpu_gl_destroy :: proc(using gpu: ^Renderer_GL) {

}

renderer_gpu_gl_start :: proc(renderer: ^Renderer) {

}

renderer_gpu_sort_fragments :: proc(renderer: ^Renderer) {
	gpu := &renderer.gpu
	using gpu

	gl.GetNamedBufferSubData(ssbo.indices, 0, size_of(Indices), &renderer.indices)

	// fetch data
	size := int(renderer.indices.fragments * size_of(Fragment))
	gl.GetNamedBufferSubData(ssbo.fragments, 0, size, raw_data(renderer.fragments))

	call1 :: proc(a, b: Fragment) -> bool {
		return a.path_index < b.path_index
	}

	call2 :: proc(a, b: Fragment) -> bool {
		return a.x < b.x
	}

	// call3 :: proc(a, b: Fragment) -> bool {
	// 	return a.y < b.y
	// }

	slice.stable_sort_by(renderer.fragments[:renderer.indices.fragments], call1)
	slice.stable_sort_by(renderer.fragments[:renderer.indices.fragments], call2)
	// slice.stable_sort_by(renderer.fragments[:renderer.indices.fragments], call3)
	
	// setup lanes	
	for i in 0..<renderer.window_height {
		renderer.lanes[i] = { -1, -1 }
	}

	// insert fragments linearly to lanes
	for i in 0..<renderer.indices.fragments {
		frag := &renderer.fragments[i]
		assert(frag.y >= 0 && frag.y < i32(renderer.window_height))
		
		lane := &renderer.lanes[frag.y]

		// set starting fragment
		if lane.fragment_head == -1 {
			lane.fragment_head = i
			lane.fragment_tail = i
		} else {
			// chain last fragment to next
			last := &renderer.fragments[lane.fragment_tail]
			last.fragment_next = i
			
			// set tail
			lane.fragment_tail = i
		}
	}

	gl.NamedBufferSubData(ssbo.fragments, 0, size, raw_data(renderer.fragments))

	lanes_size := renderer.window_height * size_of(Lane)
	gl.NamedBufferSubData(ssbo.lanes, 0, lanes_size, raw_data(renderer.lanes))
}

// exclusive prefix scanning 
renderer_gpu_exclusive_scan :: proc(renderer: ^Renderer) {
	gpu := &renderer.gpu
	using gpu

	gl.GetNamedBufferSubData(ssbo.indices, 0, size_of(Indices), &renderer.indices)
	
	// fetch data
	size := int(renderer.indices.fragments * size_of(Fragment))
	gl.GetNamedBufferSubData(ssbo.fragments, 0, size, raw_data(renderer.fragments))

	// // perform exclusive search
	// sum: i32
	// temp: i32
	// old_y: i32
	// for i in 0..<renderer.indices.fragments {
	// 	frag := &renderer.fragments[i]
		
	// 	if old_y != frag.y {
	// 		sum = 0
	// 	}

	// 	temp = frag.winding_number
	// 	frag.winding_number = sum
	// 	sum += temp

	// 	old_y = frag.y
	// }	

	for i in 0..<renderer.window_height {
		lane := &renderer.lanes[i]
		fragment_index := lane.fragment_head
		sum: i32
		temp: i32

		for fragment_index != -1 {
			frag := &renderer.fragments[fragment_index]
			
			temp = frag.winding_number
			frag.winding_number = sum
			sum += temp

			fragment_index = frag.fragment_next
		}
	}

	// push data back
	gl.NamedBufferSubData(ssbo.fragments, 0, size, raw_data(renderer.fragments))
}

// inclusive prefix scanning 
renderer_gpu_inclusive_scan :: proc(renderer: ^Renderer) {
	gpu := &renderer.gpu
	using gpu


	gl.GetNamedBufferSubData(ssbo.indices, 0, size_of(Indices), &renderer.indices)
	
	// fetch data
	size := int(renderer.indices.fragments * size_of(Fragment))
	gl.GetNamedBufferSubData(ssbo.fragments, 0, size, raw_data(renderer.fragments))

	// // perform inclusive search
	// sum: i32
	// old_y: i32
	// for i in 0..<renderer.indices.fragments {
	// 	frag := &renderer.fragments[i]
		
	// 	// reset at new lines
	// 	if old_y != frag.y {
	// 		sum = 0
	// 	}
		
	// 	sum += frag.winding_number
	// 	frag.winding_number = sum
		
	// 	old_y = frag.y
	// 	// fmt.eprintln("i", frag.y, i, frag.winding_number)
	// }

	for i in 0..<renderer.window_height {
		lane := &renderer.lanes[i]
		fragment_index := lane.fragment_head
		sum: i32

		for fragment_index != -1 {
			frag := &renderer.fragments[fragment_index]
			
			sum += frag.winding_number
			frag.winding_number = sum

			fragment_index = frag.fragment_next
		}
	}

	// push data back
	gl.NamedBufferSubData(ssbo.fragments, 0, size, raw_data(renderer.fragments))	
}

renderer_gpu_gl_end :: proc(renderer: ^Renderer) {
	gpu := &renderer.gpu
	using gpu

	// check path count or unfinished path we might have pushed
	path_count := renderer.path_index + 1
	path := renderer.paths[renderer.path_index]
	if path.curve_index_start == path.curve_index_current {
		path_count -= 1
	}
	
	// write in the final path count
	// fmt.eprintln("PATH COUNT", path_count, size_of(Path))
	renderer.indices.paths = i32(path_count)

	// bind data
	{
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, ssbo.indices)
		gl.NamedBufferSubData(ssbo.indices, 0, size_of(Indices), &renderer.indices)

		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, ssbo.paths)
		gl.NamedBufferSubData(ssbo.paths, 0, path_count * size_of(Path), raw_data(renderer.paths))

		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, ssbo.curves)
		gl.NamedBufferSubData(ssbo.curves, 0, renderer.curve_index * size_of(Curve), raw_data(renderer.curves))
	}

	// // line intersections horizontally / vertically
	// {
	// 	gl.UseProgram(intersction0_program)
	// 	gl.DispatchCompute(u32(renderer.curve_index), 1, 1)
	// }

	// // line intersections horizontally / vertically
	// {
	// 	gl.UseProgram(intersction1_program)
	// 	gl.DispatchCompute(u32(renderer.curve_index), 1, 1)
	// }

	// generate fragments
	{
		gl.UseProgram(gen_program)
		gl.DispatchCompute(u32(renderer.curve_index), 1, 1)
	}

	renderer_gpu_sort_fragments(renderer)

	renderer_gpu_exclusive_scan(renderer)

	// mark fragments
	{
		gl.GetNamedBufferSubData(ssbo.indices, 0, size_of(Indices), &renderer.indices)
		gl.UseProgram(mark_program)
		gl.DispatchCompute(u32(renderer.indices.fragments - 1), 1, 1)
	}

	renderer_gpu_inclusive_scan(renderer)

	// raster stage
	{
		gl.UseProgram(raster_program)
		// max 1 necessary
		w := max(renderer.window_width, 1)
		// h := max(renderer.window_height, 1)
		gl.DispatchCompute(u32(w), u32(1), 1)
		gl.MemoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
	}

	// fill texture
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
	gl.Disable(gl.CULL_FACE)
	gl.Disable(gl.DEPTH_TEST)
	gl.BindVertexArray(fill.vao)
	gl.UseProgram(fill.program)

	projection := glm.mat4Ortho3d(0, f32(renderer.window_width), f32(renderer.window_height), 0, 0, 1)
	gl.UniformMatrix4fv(fill.loc_projection, 1, gl.FALSE, &projection[0][0])

	gl.BindTexture(gl.TEXTURE_2D, raster_texture_id)
	gl.BindBuffer(gl.ARRAY_BUFFER, fill.vbo)

	{
		w := f32(renderer.window_width)
		h := f32(renderer.window_height)
		u := w / f32(raster_texture_width)
		v := h / f32(raster_texture_height)

		data := [6]Vertex {
			{{ 0, 0 }, { 0, 0 }},
			{{ 0, h }, { 0, v }},
			{{ w, 0 }, { u, 0 }},
			{{ 0, h }, { 0, v }},
			{{ w, 0 }, { u, 0 }},
			{{ w, h }, { u, v }},
		}
		gl.BufferData(gl.ARRAY_BUFFER, size_of(data), &data[0], gl.STREAM_DRAW)
		gl.DrawArrays(gl.TRIANGLES, 0, 6)
	}
}

///////////////////////////////////////////////////////////
// PATHING
///////////////////////////////////////////////////////////

renderer_move_to :: proc(renderer: ^Renderer, x, y: f32) {
	path := renderer_path_get(renderer)
	
	// if path.curve_index_current > path.curve_index_start {
	// 	renderer_path_transition(renderer)
	// }

	renderer.curve_last = { x, y }
	renderer.curve_last_control = renderer.curve_last
}

renderer_move_to_rel :: proc(renderer: ^Renderer, x, y: f32) {
	path := renderer_path_get(renderer)
	// TODO add path transition
	renderer.curve_last = renderer.curve_last + { x, y }
}

renderer_line_to :: proc(using renderer: ^Renderer, x, y: f32) {
	path := renderer_path_get(renderer)
	curve_linear_init(
		&curves[curve_index],
		path_xform_v2(path, curve_last),
		path_xform_v2(path, { x, y }),
		renderer.path_index,
	)
	curve_index += 1
	path.curve_index_current = i32(curve_index)
	curve_last = { x, y }
}

renderer_line_to_rel :: proc(using renderer: ^Renderer, x, y: f32) {
	path := renderer_path_get(renderer)
	curve_linear_init(
		&curves[curve_index],
		path_xform_v2(path, curve_last),
		path_xform_v2(path, curve_last + { x, y }),
		renderer.path_index,
	)
	curve_index += 1
	path.curve_index_current = i32(curve_index)
	curve_last = curve_last + { x, y }
}

renderer_vertical_line_to :: proc(using renderer: ^Renderer, y: f32) {
	path := renderer_path_get(renderer)
	curve_linear_init(
		&curves[curve_index],
		path_xform_v2(path, curve_last),
		path_xform_v2(path, { curve_last.x, y }),
		renderer.path_index,
	)
	curve_index += 1
	path.curve_index_current = i32(curve_index)
	curve_last = { curve_last.x, y }
}

renderer_horizontal_line_to :: proc(using renderer: ^Renderer, x: f32) {
	path := renderer_path_get(renderer)
	curve_linear_init(
		&curves[curve_index],
		path_xform_v2(path, curve_last),
		path_xform_v2(path, { x, curve_last.y }),
		renderer.path_index,
	)
	curve_index += 1
	path.curve_index_current = i32(curve_index)
	curve_last = { x, curve_last.y }
}

renderer_quadratic_to :: proc(using renderer: ^Renderer, cx, cy, x, y: f32) {
	path := renderer_path_get(renderer)
	curve_quadratic_init(
		&curves[curve_index],
		path_xform_v2(path, curve_last),
		path_xform_v2(path, { cx, cy }),
		path_xform_v2(path, { x, y }),
		renderer.path_index,
	)
	curve_index += 1
	path.curve_index_current = i32(curve_index)
	curve_last = { x, y }
}

renderer_quadratic_to_rel :: proc(using renderer: ^Renderer, cx, cy, x, y: f32) {
	path := renderer_path_get(renderer)
	curve_quadratic_init(
		&curves[curve_index],
		path_xform_v2(path, curve_last),
		path_xform_v2(path, curve_last + { cx, cy }),
		path_xform_v2(path, curve_last + { x, y }),
		renderer.path_index,
	)
	curve_index += 1
	path.curve_index_current = i32(curve_index)
	curve_last = curve_last + { x, y }
}

renderer_cubic_to :: proc(using renderer: ^Renderer, c1x, c1y, c2x, c2y, x, y: f32) {
	path := renderer_path_get(renderer)
	curve_cubic_init(
		&curves[curve_index],
		path_xform_v2(path, curve_last),
		path_xform_v2(path, { c1x, c1y }),
		path_xform_v2(path, { c2x, c2y }),
		path_xform_v2(path, { x, y }),
		renderer.path_index,
	)
	curve_index += 1
	path.curve_index_current = i32(curve_index)
	curve_last = { x, y }
	curve_last_control = { c2x, c2y }
}

renderer_cubic_to_rel :: proc(using renderer: ^Renderer, c1x, c1y, c2x, c2y, x, y: f32) {
	path := renderer_path_get(renderer)
	curve_cubic_init(
		&curves[curve_index],
		path_xform_v2(path, curve_last),
		path_xform_v2(path, curve_last + { c1x, c1y }),
		path_xform_v2(path, curve_last + { c2x, c2y }),
		path_xform_v2(path, curve_last + { x, y }),
		renderer.path_index,
	)
	curve_index += 1
	path.curve_index_current = i32(curve_index)
	curve_last = curve_last + { x, y }
}

renderer_close :: proc(using renderer: ^Renderer) {
	if curve_index > 0 {
		path := renderer_path_get(renderer)
		curve_start := curves[0].B[0]
		curve_final := path_xform_v2(path, curve_last)

		if curve_start != curve_final {
			curve_linear_init(
				&curves[curve_index],
				curve_final,
				curve_start,
				renderer.path_index,
			)
		}

		curve_index += 1
		curve_last = { curve_start.x, curve_start.y }
		curve_last_control = curve_last
	}

	// renderer_path_transition(renderer)
	// renderer_move_to(renderer, curve_last.x, curve_last.y)
}

renderer_triangle :: proc(renderer: ^Renderer, x, y, r: f32) {
	renderer_move_to(renderer, x, y - r/2)
	renderer_line_to(renderer, x - r/2, y + r/2)
	renderer_line_to(renderer, x + r/2, y + r/2)
	renderer_close(renderer)
}

renderer_rect :: proc(renderer: ^Renderer, x, y, w, h: f32) {
	renderer_move_to(renderer, x, y)
	renderer_line_to(renderer, x, y + h)
	renderer_line_to(renderer, x + w, y + h)
	renderer_line_to(renderer, x + w, y)
	renderer_close(renderer)
}

renderer_ellipse :: proc(renderer: ^Renderer, cx, cy, rx, ry: f32) {
	renderer_move_to(renderer, cx-rx, cy)
	renderer_cubic_to(renderer, cx, cy+ry, cx-rx, cy+ry*KAPPA90, cx-rx*KAPPA90, cy+ry)
	renderer_cubic_to(renderer, cx+rx, cy, cx+rx*KAPPA90, cy+ry, cx+rx, cy+ry*KAPPA90)
	renderer_cubic_to(renderer, cx, cy-ry, cx+rx, cy-ry*KAPPA90, cx+rx*KAPPA90, cy-ry)
	renderer_cubic_to(renderer, cx-rx, cy, cx-rx*KAPPA90, cy-ry, cx-rx, cy-ry*KAPPA90)
	renderer_close(renderer)
}

renderer_circle :: proc(renderer: ^Renderer, cx, cy, r: f32) {
	renderer_ellipse(renderer, cx, cy, r, r)
}

// TODO still looks weird
// arc to for svg
renderer_arc_to :: proc(
	renderer: ^Renderer,
	rx, ry: f32,
	rotation: f32,
	large_arc: f32,
	sweep_direction: f32,
	x2, y2: f32,
) {
	PI :: 3.14159265358979323846264338327

	square :: #force_inline proc(a: f32) -> f32 {
		return a * a
	}

	vmag :: #force_inline proc(x, y: f32) -> f32 {
		return math.sqrt(x*x + y*y)
	}

	vecrat :: proc(ux, uy, vx, vy: f32) -> f32 {
		return (ux*vx + uy*vy) / (vmag(ux,uy) * vmag(vx,vy))
	}

	vecang :: proc(ux, uy, vx, vy: f32) -> f32 {
		r := vecrat(ux,uy, vx,vy)

		if r < -1.0 {
			r = -1
		}

		if r > 1.0 {
			r = 1
		}

		return ((ux*vy < uy*vx) ? -1 : 1) * math.acos(r)
	}

	// Ported from canvg (https://code.google.com/p/canvg/)
	rx := abs(rx)				// y radius
	ry := abs(ry)				// x radius
	rotx := rotation / 180.0 * PI		// x rotation angle
	fa := abs(large_arc) > 1e-6 ? 1 : 0	// Large arc
	fs := abs(sweep_direction) > 1e-6 ? 1 : 0	// Sweep direction
	// fs = 1
	// fa = 0
	x1 := renderer.curve_last.x // start point
	y1 := renderer.curve_last.y

	dx := x1 - x2
	dy := y1 - y2
	d := math.sqrt(dx*dx + dy*dy)
	if d < 1e-6 || rx < 1e-6 || ry < 1e-6 {
		// The arc degenerates to a line
		renderer_line_to(renderer, x2, y2)
		return
	}

	sinrx := math.sin(rotx)
	cosrx := math.cos(rotx)

	// Convert to center point parameterization.
	// http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
	// 1) Compute x1', y1'
	x1p := cosrx * dx / 2.0 + sinrx * dy / 2.0
	y1p := -sinrx * dx / 2.0 + cosrx * dy / 2.0
	d = square(x1p)/square(rx) + square(y1p)/square(ry)
	if d > 1 {
		d = math.sqrt(d)
		rx *= d
		ry *= d
	}
	// 2) Compute cx', cy'
	s := f32(0)
	sa := square(rx)*square(ry) - square(rx)*square(y1p) - square(ry)*square(x1p)
	sb := square(rx)*square(y1p) + square(ry)*square(x1p)
	if sa < 0.0 {
		sa = 0
	}
	if sb > 0.0 {
		s = math.sqrt(sa / sb)
	}
	if fa == fs {
		s = -s
	}
	cxp := s * rx * y1p / ry
	cyp := s * -ry * x1p / rx

	// 3) Compute cx,cy from cx',cy'
	cx := (x1 + x2)/2.0 + cosrx*cxp - sinrx*cyp
	cy := (y1 + y2)/2.0 + sinrx*cxp + cosrx*cyp

	// 4) Calculate theta1, and delta theta.
	ux := (x1p - cxp) / rx
	uy := (y1p - cyp) / ry
	vx := (-x1p - cxp) / rx
	vy := (-y1p - cyp) / ry
	a1 := vecang(1, 0, ux, uy)	// Initial angle
	da := vecang(ux, uy, vx, vy)		// Delta angle

	if fs == 0 && da > 0.0 {
		da -= 2 * PI
	} else if fs == 1.0 && da < 0.0 {
		da += 2 * PI
	}

	// cosrx += 0.1

	// Approximate the arc using cubic spline segments.
	t := Xform {
		cosrx, sinrx,
		-sinrx, cosrx,
		cx, cy,
	}

	// Split arc into max 90 degree segments.
	// The loop assumes an iteration per end point (including start and end), this +1.
	ndivs := int(abs(da) / (PI*0.5) + 1)
	hda := (da / f32(ndivs)) / 2.0
	// Fix for ticket #179: division by 0: avoid cotangens around 0 (infinite)
	if hda < 1e-3 && hda > -1e-3 {
		hda *= 0.5
	} else {
		hda = (1 - math.cos(hda)) / math.sin(hda)
	}

	kappa := abs(4.0 / 3.0 * hda)
	if da < 0.0 {
		kappa = -kappa
	}

	path := renderer_path_get(renderer)
	p, ptan: [2]f32
	for i := 0; i <= ndivs; i += 1 {
		a := a1 + da * (f32(i)/f32(ndivs))
		dx = math.cos(a)
		dy = math.sin(a)

		curr := xform_point_v2(t, { dx * rx, dy * ry }) // position
		tan := xform_v2(t, { -dy*rx * kappa, dx*ry * kappa }) // tangent

		if i > 0 {
			renderer_cubic_to(renderer, p.x+ptan.x, p.y+ptan.y, curr.x-tan.x, curr.y-tan.y, curr.x, curr.y)
		}

		p = curr
		ptan = tan
	}
}

///////////////////////////////////////////////////////////
// POOL
///////////////////////////////////////////////////////////

Pool_Invalid_Slot_Index :: 0

Pool :: struct($N: int, $T: typeid) {
	data: [N + 1]T,
	queue_top: int,
	free_queue: [N]u32,
}

pool_clear :: proc(pool: ^Pool($N, $T)) {
	pool.queue_top = 0

	for i := u32(N); i >= 1; i -= 1 {
		pool.free_queue[pool.queue_top] = i
		pool.queue_top += 1
	}
}

pool_alloc_index :: proc(p: ^Pool($N, $T)) -> u32 #no_bounds_check {
	if p.queue_top > 0 {
		p.queue_top -= 1
		slot_index := p.free_queue[p.queue_top]
		assert(slot_index > Pool_Invalid_Slot_Index && slot_index < u32(N))
		return slot_index
	} else {
		return Pool_Invalid_Slot_Index
	}
}

pool_free_index :: proc(p: ^Pool($N, $T), slot_index: u32, loc := #caller_location) #no_bounds_check {
	runtime.bounds_check_error_loc(loc, int(slot_index), N)
	assert(slot_index > Pool_Invalid_Slot_Index)
	assert(p.queue_top < N)

	// debug check?

	p.free_queue[p.queue_top] = u32(slot_index)
	p.queue_top += 1
	assert(p.queue_top <= N - 1)
}

pool_at :: proc(p: ^Pool($N, $T), slot_index: u32, loc := #caller_location) -> ^T #no_bounds_check {
	runtime.bounds_check_error_loc(loc, int(slot_index), N)
	assert(slot_index > Pool_Invalid_Slot_Index)
	return &p.data[slot_index]
}

///////////////////////////////////////////////////////////
// PATH
///////////////////////////////////////////////////////////

path_init :: proc(using path: ^Path, width, height: f32) {
	path.box = {
		max(f32),
		max(f32),
		-max(f32),
		-max(f32),
	}

	path.clip = {
		0,
		0,
		width,
		height,
	}

	xform_identity(&path.xform)
	path.color = { 0, 0, 1, 1 }
	path.curve_index_start = 0
	path.curve_index_current = 0
}

// add to bounding box
path_box_push :: proc(path: ^Path, output: [2]f32) {
	path.box.x = min(path.box.x, output.x)
	path.box.z = max(path.box.z, output.x)
	path.box.y = min(path.box.y, output.y)
	path.box.w = max(path.box.w, output.y)
}

path_xform_v2 :: proc(path: ^Path, input: [2]f32) -> (res: [2]f32) {
	res = {
		input.x * path.xform[0] + input.y * path.xform[2] + path.xform[4],
		input.x * path.xform[1] + input.y * path.xform[3] + path.xform[5],
	}
	path_box_push(path, res)
	return
}

curve_linear_init :: proc(curve: ^Curve, a, b: [2]f32, path_index: int) {
	curve.B[0] = a
	curve.B[1] = b
	curve.count = 0
	curve.path_index = i32(path_index)
}

curve_quadratic_init :: proc(curve: ^Curve, a, b, c: [2]f32, path_index: int) {
	curve.B[0] = a
	curve.B[1] = b
	curve.B[2] = c
	curve.count = 1
	curve.path_index = i32(path_index)
}

curve_cubic_init :: proc(curve: ^Curve, a, b, c, d: [2]f32, path_index: int) {
	curve.B[0] = a
	curve.B[1] = b
	curve.B[2] = c
	curve.B[3] = d
	curve.count = 2
	curve.path_index = i32(path_index)
}

curve_set_endpoint :: proc(curve: ^Curve, to: [2]f32) {
	curve.B[curve.count + 1] = to
}

renderer_path_get :: #force_inline proc(renderer: ^Renderer) -> ^Path {
	return &renderer.paths[renderer.path_index]
}

renderer_path_push :: proc(renderer: ^Renderer) -> (res: ^Path) {
	renderer.path_index += 1
	res = &renderer.paths[renderer.path_index]
	path_init(res, f32(renderer.window_width), f32(renderer.window_height))
	return res
}

renderer_path_transition :: proc(renderer: ^Renderer) {
	old := renderer_path_get(renderer)
	next := renderer_path_push(renderer)
	next.curve_index_start = i32(renderer.curve_index)
	next.curve_index_current = i32(renderer.curve_index)
	next.xform = old.xform
	next.color = old.color
	// next.color = { 0, 1, 0, 1 }
}

renderer_path_reset_transform :: proc(using renderer: ^Renderer) {
	path := renderer_path_get(renderer)
	xform_identity(&path.xform)
}

renderer_path_translate :: proc(using renderer: ^Renderer, x, y: f32) {
	path := renderer_path_get(renderer)
	temp := xform_translate(x, y)
	xform_premultiply(&path.xform, temp)
}

renderer_path_rotate :: proc(using renderer: ^Renderer, angle: f32) {
	path := renderer_path_get(renderer)
	temp := xform_rotate(angle)
	xform_premultiply(&path.xform, temp)
}

renderer_path_scale :: proc(using renderer: ^Renderer, x, y: f32) {
	path := renderer_path_get(renderer)
	temp := xform_scale(x, y)
	xform_premultiply(&path.xform, temp)
}

xform_point_v2 :: proc(xform: Xform, input: [2]f32) -> [2]f32 {
	return {
		input.x * xform[0] + input.y * xform[2] + xform[4],
		input.x * xform[1] + input.y * xform[3] + xform[5],
	}
}

xform_point_xy :: proc(xform: Xform, x, y: f32) -> (outx, outy: f32) {
	outx = x * xform[0] + y * xform[2] + xform[4]
	outy = x * xform[1] + y * xform[3] + xform[5]
	return
}

// without offset
xform_v2 :: proc(xform: Xform, input: [2]f32) -> [2]f32 {
	return {
		input.x * xform[0] + input.y * xform[2],
		input.x * xform[1] + input.y * xform[3],
	}
}

xform_identity :: proc(xform: ^Xform) {
	xform^ = {
		1, 0,
		0, 1,
		0, 0,
	}
}

xform_translate :: proc(tx, ty: f32) -> Xform {
	return {
		1, 0,
		0, 1,
		tx, ty,
	}
}

xform_scale :: proc(sx, sy: f32) -> Xform {
	return {
		sx, 0,
		0, sy,
		0, 0,
	}
}

xform_rotate :: proc(angle: f32) -> Xform {
	cs := math.cos(angle)
	sn := math.sin(angle)
	return {
		cs, sn,
		-sn, cs,
		0, 0,
	}
}

xform_multiply :: proc(t: ^Xform, s: Xform) {
	t0 := t[0] * s[0] + t[1] * s[2]
	t2 := t[2] * s[0] + t[3] * s[2]
	t4 := t[4] * s[0] + t[5] * s[2] + s[4]
	t[1] = t[0] * s[1] + t[1] * s[3]
	t[3] = t[2] * s[1] + t[3] * s[3]
	t[5] = t[4] * s[1] + t[5] * s[3] + s[5]
	t[0] = t0
	t[2] = t2
	t[4] = t4
}

xform_premultiply :: proc(a: ^Xform, b: Xform) {
	temp := b
	xform_multiply(&temp, a^)
	a^ = temp
}
