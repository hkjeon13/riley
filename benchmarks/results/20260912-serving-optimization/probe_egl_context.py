"""Create and destroy a one-pixel offscreen GL context; no window or user data."""
import ctypes as C
import json
from pathlib import Path


def main():
    egl, gl = C.CDLL('libEGL.so.1'), C.CDLL('libGL.so.1')
    signatures = {
        'eglGetDisplay': (C.c_void_p, [C.c_void_p]),
        'eglInitialize': (C.c_uint, [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int)]),
        'eglGetError': (C.c_int, []),
        'eglBindAPI': (C.c_uint, [C.c_uint]),
        'eglChooseConfig': (C.c_uint, [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_void_p), C.c_int, C.POINTER(C.c_int)]),
        'eglCreatePbufferSurface': (C.c_void_p, [C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]),
        'eglCreateContext': (C.c_void_p, [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]),
        'eglMakeCurrent': (C.c_uint, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]),
        'eglDestroySurface': (C.c_uint, [C.c_void_p, C.c_void_p]),
        'eglDestroyContext': (C.c_uint, [C.c_void_p, C.c_void_p]),
        'eglTerminate': (C.c_uint, [C.c_void_p]),
    }
    for name, (result, arguments) in signatures.items():
        fn = getattr(egl, name)
        fn.restype, fn.argtypes = result, arguments
    gl.glGetString.argtypes, gl.glGetString.restype = [C.c_uint], C.c_char_p
    def check(value, label):
        if not value:
            raise RuntimeError(f'{label} failed; EGL error 0x{egl.eglGetError():x}')
        return value
    display = check(egl.eglGetDisplay(None), 'eglGetDisplay')
    major, minor, count = C.c_int(), C.c_int(), C.c_int()
    surface = context = None
    initialized = False
    try:
        check(egl.eglInitialize(display, C.byref(major), C.byref(minor)), 'eglInitialize')
        initialized = True
        check(egl.eglBindAPI(0x30A2), 'eglBindAPI(OpenGL)')
        config = C.c_void_p()
        attributes = (C.c_int * 11)(0x3033, 1, 0x3040, 8, 0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3038)
        check(egl.eglChooseConfig(display, attributes, C.byref(config), 1, C.byref(count)), 'eglChooseConfig')
        assert count.value == 1 and config.value
        size = (C.c_int * 5)(0x3057, 1, 0x3056, 1, 0x3038)
        surface = check(egl.eglCreatePbufferSurface(display, config, size), 'eglCreatePbufferSurface')
        context = check(egl.eglCreateContext(display, config, None, (C.c_int * 1)(0x3038)), 'eglCreateContext')
        check(egl.eglMakeCurrent(display, surface, surface, context), 'eglMakeCurrent')
        details = {name: gl.glGetString(code).decode() for name, code in
                   (('vendor', 0x1F00), ('renderer', 0x1F01), ('version', 0x1F02))}
        assert details['vendor'] == 'NVIDIA Corporation' and 'RTX 4090' in details['renderer']
        assert '580.173.02' in details['version']
        maps = [line for line in Path('/proc/self/maps').read_text().splitlines()
                if 'libnvidia' in line or 'libGLX_nvidia' in line or 'libEGL_nvidia' in line]
        result = {'completed': True, 'window_created': False, 'egl_version': [major.value, minor.value],
                  'gl': details, 'driver_mappings': maps}
    finally:
        if initialized:
            check(egl.eglMakeCurrent(display, None, None, None), 'unbind context')
            if context:
                check(egl.eglDestroyContext(display, context), 'destroy context')
            if surface:
                check(egl.eglDestroySurface(display, surface), 'destroy surface')
            check(egl.eglTerminate(display), 'terminate display')
    result['cleanup_completed'] = True
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
