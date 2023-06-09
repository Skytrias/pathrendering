package src

import "core:time"
import "core:fmt"
import "core:math"
import "core:os"
import "core:bytes"
import "core:strings"
import "core:runtime"
import "core:strconv"
import "core:math/linalg"
import glm "core:math/linalg/glsl"
import glfw "vendor:glfw"
import gl "vendor:OpenGL"

length :: linalg.vector_length

App :: struct {
	mouse: Mouse,
	renderer: Renderer,
	ctrl: bool,
	shift: bool,

	window: glfw.WindowHandle,
	window_width: int,
	window_height: int,
}
app: App

Mouse :: struct {
	x: f32,
	y: f32,
	left: bool,
	right: bool,
}

TESTING :: #config(TESTING, true)

window_cursor_pos_callback :: proc "c" (handle: glfw.WindowHandle, x, y: f64) {
	app.mouse.x = f32(x)
	app.mouse.y = f32(y)
}

window_mouse_button_callback :: proc "c" (handle: glfw.WindowHandle, button, action, mods: i32) {
	if button == glfw.MOUSE_BUTTON_LEFT {
		app.mouse.left = action == glfw.PRESS
	}

	if button == glfw.MOUSE_BUTTON_RIGHT {
		app.mouse.right = action == glfw.PRESS
	}
}

window_key_callback :: proc "c" (handle: glfw.WindowHandle, key, scancode, action, mods: i32) {
	context = runtime.default_context()
	down := action == glfw.PRESS || action == glfw.REPEAT

	if mods == glfw.MOD_CONTROL {
		app.ctrl = down
	}

	if mods == glfw.MOD_SHIFT {
		app.shift = down
	}

	if mods != 0 {
		return
	}

	if key == glfw.KEY_LEFT_CONTROL {
		app.ctrl = down
	}

	if key == glfw.KEY_LEFT_SHIFT {
		app.shift = down
	}

	// if down {
	// 	switch key {
	// 	case glfw.KEY_1: app.renderer.color_mode = 0
	// 	case glfw.KEY_2: app.renderer.color_mode = 1
	// 	case glfw.KEY_3: app.renderer.color_mode = 2
	// 	case glfw.KEY_4: app.renderer.color_mode = 3

	// 	case glfw.KEY_Q: app.renderer.fill_rule = 0
	// 	case glfw.KEY_W: app.renderer.fill_rule = 1
	// 	case glfw.KEY_E: app.renderer.fill_rule = 2
	// 	case glfw.KEY_R: app.renderer.fill_rule = 3

	// 	case glfw.KEY_SPACE: app.renderer.ignore_temp = app.renderer.ignore_temp == 0 ? 1 : 0
	// 	}
	// }
}

window_size_callback :: proc "c" (handle: glfw.WindowHandle, width, height: i32) {
	app.window_width = int(width)
	app.window_height = int(height)
}

main :: proc() {
	glfw.Init()
	defer glfw.Terminate()

	error_callback :: proc "c" (code: i32, desc: cstring) {
		context = runtime.default_context()
		fmt.eprintln(desc, code)
	}
	glfw.SetErrorCallback(error_callback)

	glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 4)
	glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 5)

	app.window_width = 800
	app.window_height = 800
	app.window = glfw.CreateWindow(i32(app.window_width), i32(app.window_height), "mpvg", nil, nil)
	defer glfw.DestroyWindow(app.window)
	if app.window == nil {
		return
	}

	glfw.SetCursorPosCallback(app.window, window_cursor_pos_callback)
	glfw.SetMouseButtonCallback(app.window, window_mouse_button_callback)
	glfw.SetKeyCallback(app.window, window_key_callback)
	glfw.SetWindowSizeCallback(app.window, window_size_callback)

	glfw.MakeContextCurrent(app.window)
	gl.load_up_to(4, 5, glfw.gl_set_proc_address)

	renderer_init(&app.renderer)
	defer renderer_destroy(&app.renderer)

	renderer_font_push(&app.renderer, "Lato-Regular.ttf")

	svg_curves := svg_gen_temp(svg_AB)
	defer delete(svg_curves)

	count: f32
	duration: time.Duration
	for !glfw.WindowShouldClose(app.window) {
		time.SCOPED_TICK_DURATION(&duration)
		free_all(context.temp_allocator)

		window_text := fmt.ctprintf(
			"mpvg %fms, mousex: %d, mousey: %d", 
			time.duration_milliseconds(duration),
			int(app.mouse.x),
			int(app.mouse.y),
		)
		glfw.SetWindowTitle(app.window, window_text)

		gl.Viewport(0, 0, i32(app.window_width), i32(app.window_height))
		gl.ClearColor(1, 1, 1, 1)
		gl.Clear(gl.COLOR_BUFFER_BIT)

		{
			renderer_start(&app.renderer, app.window_width, app.window_height)
			defer renderer_end(&app.renderer)

			// renderer_move_to(&app.renderer, 300, 300)
			// renderer_line_to(&app.renderer, app.mouse.x, app.mouse.y)
			// renderer_cubic_to(&app.renderer, p1.x, p1.y, p3.x, p3.y, p2.x, p2.y)
			// renderer_arc_to(&app.renderer, 20, 20, 0, 1, 0, p2.x, p2.y)
			// renderer_quadratic_to(&app.renderer, p3.x, p3.y, p2.x, p2.y)

			// renderer_move_to(&app.renderer, p1.x, p1.y)
			// renderer_line_to(&app.renderer, p2.x, p2.y)
			// renderer_close(&app.renderer)

			renderer_move_to(&app.renderer, 100, 100)
			renderer_line_to(&app.renderer, 200, 140)
			// renderer_line_to(&app.renderer, 300, 80)
			renderer_line_to(&app.renderer, app.mouse.x, app.mouse.y)
			// renderer_close(&app.renderer)

			// // // NOTE: NEW
			// // renderer_path_translate(&app.renderer, 200, 200)
			// renderer_path_translate(&app.renderer, app.mouse.x, app.mouse.y)
			// renderer_path_rotate(&app.renderer, count * 0.01)
			// renderer_rect(&app.renderer, -100, -50, 200, 100)

			// renderer_path_translate(&app.renderer, app.mouse.x, app.mouse.y)
			// // renderer_path_translate(&app.renderer, 100, 100)
			// // renderer_path_scale(&app.renderer, 5, 5)
			// // renderer_path_scale(&app.renderer, 10, 10)
			// renderer_path_scale(&app.renderer, 20, 20)
			// // renderer_path_scale(&app.renderer, 50, 50)
			// renderer_svg(&app.renderer, svg_curves)

			// renderer_path_translate(&app.renderer, app.mouse.x, app.mouse.y)
			// renderer_path_scale(&app.renderer, 1, 1)
			// // renderer_text_push(&app.renderer, "xyzlp", 0, 0, math.sin(count * 0.05) * 20 + 200)
			// // renderer_text_push(&app.renderer, "text works", 0, 0, math.sin(count * 0.05) * 20 + 100)
			// renderer_text_push(&app.renderer, "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna ", 0, 0, math.sin(count * 0.05) * 10 + 40)
		}

		glfw.SwapBuffers(app.window)
		glfw.PollEvents()
		count += 1
	}
}