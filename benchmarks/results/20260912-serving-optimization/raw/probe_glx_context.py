"""Check a private 1x1 GLX pbuffer on DISPLAY; no window, Blender or file edits.

Run only in the controller's explicitly selected child environment. JSON is
written to stdout. A nonzero exit or incomplete cleanup is a failed probe.
GLX 1.3 signatures/constants follow Khronos OpenGL-Registry api/GL/glxext.h.
"""
import ctypes as C
import json
import os
from pathlib import Path
import re
import sys


class XErrorEvent(C.Structure):
    _fields_ = [("type", C.c_int), ("display", C.c_void_p),
                ("resourceid", C.c_ulong), ("serial", C.c_ulong),
                ("error_code", C.c_ubyte), ("request_code", C.c_ubyte),
                ("minor_code", C.c_ubyte)]


def require(value, label):
    if not value:
        raise RuntimeError(label)
    return value


def bind(library, signatures):
    for name, (result, arguments) in signatures.items():
        fn = getattr(library, name)
        fn.restype, fn.argtypes = result, arguments


def probe():
    result = {"schema_version": "riley.glx-runtime-probe.v1", "completed": False,
              "window_created": False, "pbuffer_size": [1, 1],
              "display": os.environ.get("DISPLAY"), "cleanup_completed": False,
              "driver_mappings": [], "x_errors": [], "cleanup_errors": [],
              "performance_claim": False}
    x11 = gl = None
    display = context = configs = None
    pbuffer = 0
    current = handler_installed = False
    old_handler = None
    operation = "initialization"
    error_count = 0

    @C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(XErrorEvent))
    def error_handler(_display, event):
        value = event.contents
        result["x_errors"].append({"operation": operation,
            "code": int(value.error_code), "request": int(value.request_code),
            "minor": int(value.minor_code), "resource": int(value.resourceid),
            "serial": int(value.serial)})
        return 0

    def sync(label):
        nonlocal operation, error_count
        operation = label
        x11.XSync(display, 0)
        require(len(result["x_errors"]) == error_count, label + " returned an X protocol error")

    def cleanup(label, function):
        nonlocal operation, error_count
        operation = label
        error_count = len(result["x_errors"])
        try:
            function()
            if display:
                sync(label)
        except Exception as error:
            result["cleanup_errors"].append({"operation": label, "error": str(error)})
        error_count = len(result["x_errors"])

    try:
        require(sys.platform == "linux", "Linux /proc and X11 are required")
        require(result["display"], "DISPLAY must be supplied by the controller")
        kernel = Path("/proc/driver/nvidia/version").read_text()
        require(re.search(r"\b580\.173\.02\b", kernel), "loaded NVIDIA kernel must be 580.173.02")
        result["kernel_version"] = kernel
        # SONAME lookup respects the controller's child-only LD_LIBRARY_PATH.
        x11, gl = C.CDLL("libX11.so.6"), C.CDLL("libGL.so.1")
        bind(x11, {
            "XOpenDisplay": (C.c_void_p, [C.c_char_p]),
            "XDefaultScreen": (C.c_int, [C.c_void_p]),
            "XFree": (C.c_int, [C.c_void_p]),
            "XSync": (C.c_int, [C.c_void_p, C.c_int]),
            "XCloseDisplay": (C.c_int, [C.c_void_p]),
            "XSetErrorHandler": (C.c_void_p, [C.c_void_p]),
        })
        bind(gl, {
            "glXQueryVersion": (C.c_int, [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int)]),
            "glXChooseFBConfig": (C.POINTER(C.c_void_p), [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int)]),
            "glXCreatePbuffer": (C.c_ulong, [C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]),
            "glXQueryDrawable": (None, [C.c_void_p, C.c_ulong, C.c_int, C.POINTER(C.c_uint)]),
            "glXCreateNewContext": (C.c_void_p, [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int]),
            "glXMakeContextCurrent": (C.c_int, [C.c_void_p, C.c_ulong, C.c_ulong, C.c_void_p]),
            "glXIsDirect": (C.c_int, [C.c_void_p, C.c_void_p]),
            "glXDestroyContext": (None, [C.c_void_p, C.c_void_p]),
            "glXDestroyPbuffer": (None, [C.c_void_p, C.c_ulong]),
            "glGetString": (C.c_char_p, [C.c_uint]),
        })
        old_handler = x11.XSetErrorHandler(C.cast(error_handler, C.c_void_p))
        handler_installed = True
        display = require(x11.XOpenDisplay(None), "XOpenDisplay failed")
        major, minor, count = C.c_int(), C.c_int(), C.c_int()
        require(gl.glXQueryVersion(display, C.byref(major), C.byref(minor)), "glXQueryVersion failed")
        sync("query GLX version")
        require((major.value, minor.value) >= (1, 3), "GLX 1.3 or later is required")
        result["glx_version"] = [major.value, minor.value]
        screen = x11.XDefaultScreen(display)
        result["screen"] = screen
        # X-renderable RGBA8, pbuffer drawable, single buffered. None terminates.
        attributes = (C.c_int * 17)(0x8012, 1, 0x8010, 4, 0x8011, 1,
                                   8, 8, 9, 8, 10, 8, 11, 8, 5, 0, 0)
        configs = gl.glXChooseFBConfig(display, screen, attributes, C.byref(count))
        sync("choose framebuffer configuration")
        require(bool(configs) and count.value > 0, "no matching GLX framebuffer configuration")
        config = configs[0]
        require(config, "null GLX framebuffer configuration")
        size = (C.c_int * 7)(0x8041, 1, 0x8040, 1, 0x801C, 0, 0)
        pbuffer = gl.glXCreatePbuffer(display, config, size)
        sync("create pbuffer")
        require(pbuffer, "glXCreatePbuffer failed")
        width, height = C.c_uint(), C.c_uint()
        gl.glXQueryDrawable(display, pbuffer, 0x801D, C.byref(width))
        gl.glXQueryDrawable(display, pbuffer, 0x801E, C.byref(height))
        sync("query pbuffer dimensions")
        require(width.value == height.value == 1, "pbuffer is not exactly 1x1")
        context = gl.glXCreateNewContext(display, config, 0x8014, None, 1)
        sync("create context")
        require(context, "glXCreateNewContext failed")
        current = bool(gl.glXMakeContextCurrent(display, pbuffer, pbuffer, context))
        sync("make context current")
        require(current, "glXMakeContextCurrent failed")
        result["direct"] = bool(gl.glXIsDirect(display, context))
        sync("query direct rendering")
        require(result["direct"], "direct rendering context is required")
        details = {}
        for name, code in (("vendor", 0x1F00), ("renderer", 0x1F01), ("version", 0x1F02)):
            value = require(gl.glGetString(code), "glGetString(" + name + ") failed")
            details[name] = value.decode("utf-8", errors="strict")
        result["gl"] = details
        result["driver_mappings"] = [line for line in Path("/proc/self/maps").read_text().splitlines()
            if "libnvidia" in line or "libGLX_nvidia" in line or "libEGL_nvidia" in line or "libcuda.so" in line]
        require(details["vendor"] == "NVIDIA Corporation", "GL vendor is not NVIDIA")
        require(re.search(r"\bRTX\s+4090\b", details["renderer"]), "GL renderer is not RTX 4090")
        require(re.search(r"\b580\.173\.02\b", details["version"]), "GL driver is not 580.173.02")
        require(result["driver_mappings"], "NVIDIA driver mappings are missing")
        sync("complete GL queries")
        result["completed"] = True
    except Exception as error:
        result["error"] = str(error)
        result["error_type"] = type(error).__name__
    finally:
        if display:
            if current:
                cleanup("unbind context", lambda: require(gl.glXMakeContextCurrent(display, 0, 0, None), "context unbind failed"))
            if context:
                cleanup("destroy context", lambda: gl.glXDestroyContext(display, context))
            if pbuffer:
                cleanup("destroy pbuffer", lambda: gl.glXDestroyPbuffer(display, pbuffer))
            if configs:
                cleanup("free framebuffer configuration array", lambda: x11.XFree(C.cast(configs, C.c_void_p)))
            # Close even after an earlier cleanup error. Do not XSync a closed display.
            operation = "close X display"
            error_count = len(result["x_errors"])
            try:
                x11.XCloseDisplay(display)
                require(len(result["x_errors"]) == error_count,
                        "close X display returned an X protocol error")
            except Exception as error:
                result["cleanup_errors"].append({"operation": operation, "error": str(error)})
            display = None
        if handler_installed:
            try:
                x11.XSetErrorHandler(old_handler)
            except Exception as error:
                result["cleanup_errors"].append({"operation": "restore X error handler", "error": str(error)})
        result["cleanup_completed"] = not result["cleanup_errors"]
        result["completed"] = result["completed"] and result["cleanup_completed"] and not result["x_errors"]
    return result


def main():
    result = probe()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
