from __future__ import annotations
import ctypes, struct
from tinygrad.device import Compiled, LRUAllocator, BufferSpec, Program, TinyELF, CompileError
from tinygrad.renderer.glsl import GLRenderer
from tinygrad.runtime.support.c import DLL
from tinygrad.helpers import round_up, suppress_finalizing, from_mv, DEBUG, Context

egl_dll = DLL('EGL', ['libEGL.so.1', 'libEGL.so', 'libEGL', 'EGL'])
gl_dll = DLL('GL', ['libOpenGL.so.0', 'libGL.so.1', 'libGL.so', 'libGL', 'OpenGL', 'opengl32'])

EGL_OPENGL_API = 0x30A2
EGL_CONTEXT_MAJOR_VERSION = 0x3098
EGL_CONTEXT_MINOR_VERSION = 0x30FB
EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD
EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001
EGL_SURFACE_TYPE = 0x3033
EGL_PBUFFER_BIT = 0x0001
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_BIT = 0x0008
EGL_NONE = 0x3038
EGL_PLATFORM_DEVICE_EXT = 0x313F

GL_COMPUTE_SHADER = 0x91B9
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_INFO_LOG_LENGTH = 0x8B84
GL_SHADER_STORAGE_BUFFER = 0x90D2
GL_UNIFORM_BUFFER = 0x8A11
GL_DYNAMIC_DRAW = 0x88E8
GL_SHADER_STORAGE_BARRIER_BIT = 0x2000
GL_BUFFER_UPDATE_BARRIER_BIT = 0x0200
GL_TIME_ELAPSED = 0x88BF
GL_QUERY_RESULT = 0x8866
GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02

egl_dll.eglGetProcAddress.restype = ctypes.c_void_p
egl_dll.eglGetProcAddress.argtypes = [ctypes.c_char_p]
egl_dll.eglGetDisplay.restype = ctypes.c_void_p
egl_dll.eglGetDisplay.argtypes = [ctypes.c_void_p]
egl_dll.eglInitialize.restype = ctypes.c_int
egl_dll.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
egl_dll.eglBindAPI.restype = ctypes.c_int
egl_dll.eglBindAPI.argtypes = [ctypes.c_uint]
egl_dll.eglChooseConfig.restype = ctypes.c_int
egl_dll.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
egl_dll.eglCreateContext.restype = ctypes.c_void_p
egl_dll.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
egl_dll.eglMakeCurrent.restype = ctypes.c_int
egl_dll.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
egl_dll.eglDestroyContext.restype = ctypes.c_int
egl_dll.eglDestroyContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
egl_dll.eglTerminate.restype = ctypes.c_int
egl_dll.eglTerminate.argtypes = [ctypes.c_void_p]
egl_dll.eglGetError.restype = ctypes.c_int
egl_dll.eglGetError.argtypes = []

def _get_proc(name: str, restype, *argtypes):
  addr = None
  if hasattr(egl_dll, 'eglGetProcAddress'):
    addr = egl_dll.eglGetProcAddress(name.encode())
  if not addr:
    if hasattr(gl_dll, name):
      fn = getattr(gl_dll, name)
      fn.restype, fn.argtypes = restype, list(argtypes)
      return fn
    raise RuntimeError(f"GL function {name} not found")
  return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

def _init_egl_context(device_id: int = 0) -> tuple[int, int]:
  dpy = None
  if device_id > 0 and hasattr(egl_dll, 'eglGetProcAddress'):
    qdev_addr = egl_dll.eglGetProcAddress(b'eglQueryDevicesEXT')
    gplat_addr = egl_dll.eglGetProcAddress(b'eglGetPlatformDisplayEXT')
    if qdev_addr and gplat_addr:
      qdev = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(qdev_addr)
      gplat = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(gplat_addr)
      num_devs = ctypes.c_int()
      if qdev(0, None, ctypes.byref(num_devs)) and num_devs.value > device_id:
        devs = (ctypes.c_void_p * num_devs.value)()
        qdev(num_devs.value, devs, ctypes.byref(num_devs))
        dpy = gplat(EGL_PLATFORM_DEVICE_EXT, devs[device_id], None)
  if not dpy:
    dpy = egl_dll.eglGetDisplay(None)
  if not dpy:
    raise RuntimeError("Failed to get EGL display")

  maj, min_ = ctypes.c_int(), ctypes.c_int()
  if not egl_dll.eglInitialize(dpy, ctypes.byref(maj), ctypes.byref(min_)):
    raise RuntimeError(f"Failed to initialize EGL: {egl_dll.eglGetError():#x}")

  egl_dll.eglBindAPI(EGL_OPENGL_API)

  config = ctypes.c_void_p()
  num_configs = ctypes.c_int()
  config_attribs = (ctypes.c_int * 9)(
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE
  )
  egl_dll.eglChooseConfig(dpy, config_attribs, ctypes.byref(config), 1, ctypes.byref(num_configs))

  ctx_attribs = (ctypes.c_int * 7)(
    EGL_CONTEXT_MAJOR_VERSION, 4,
    EGL_CONTEXT_MINOR_VERSION, 3,
    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
    EGL_NONE
  )
  ctx = egl_dll.eglCreateContext(dpy, config if num_configs.value > 0 else None, None, ctx_attribs)
  if not ctx:
    raise RuntimeError(f"Failed to create EGL context (GL 4.3): {egl_dll.eglGetError():#x}")

  if not egl_dll.eglMakeCurrent(dpy, None, None, ctx):
    raise RuntimeError(f"Failed to make EGL context current: {egl_dll.eglGetError():#x}")

  return dpy, ctx

glGetString = _get_proc('glGetString', ctypes.c_char_p, ctypes.c_uint)
glGetIntegerv = _get_proc('glGetIntegerv', None, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
glGetError = _get_proc('glGetError', ctypes.c_uint)
glCreateShader = _get_proc('glCreateShader', ctypes.c_uint, ctypes.c_uint)
glShaderSource = _get_proc('glShaderSource', None, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int))
glCompileShader = _get_proc('glCompileShader', None, ctypes.c_uint)
glGetShaderiv = _get_proc('glGetShaderiv', None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
glGetShaderInfoLog = _get_proc('glGetShaderInfoLog', None, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p)
glCreateProgram = _get_proc('glCreateProgram', ctypes.c_uint)
glAttachShader = _get_proc('glAttachShader', None, ctypes.c_uint, ctypes.c_uint)
glLinkProgram = _get_proc('glLinkProgram', None, ctypes.c_uint)
glGetProgramiv = _get_proc('glGetProgramiv', None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
glGetProgramInfoLog = _get_proc('glGetProgramInfoLog', None, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p)
glUseProgram = _get_proc('glUseProgram', None, ctypes.c_uint)
glDeleteShader = _get_proc('glDeleteShader', None, ctypes.c_uint)
glDeleteProgram = _get_proc('glDeleteProgram', None, ctypes.c_uint)
glGenBuffers = _get_proc('glGenBuffers', None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
glDeleteBuffers = _get_proc('glDeleteBuffers', None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
glBindBuffer = _get_proc('glBindBuffer', None, ctypes.c_uint, ctypes.c_uint)
glBufferData = _get_proc('glBufferData', None, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint)
glBufferSubData = _get_proc('glBufferSubData', None, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p)
glGetBufferSubData = _get_proc('glGetBufferSubData', None, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p)
glBindBufferBase = _get_proc('glBindBufferBase', None, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
glBindBufferRange = _get_proc('glBindBufferRange', None, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t)
glDispatchCompute = _get_proc('glDispatchCompute', None, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
glMemoryBarrier = _get_proc('glMemoryBarrier', None, ctypes.c_uint)
glFinish = _get_proc('glFinish', None)
glGenQueries = _get_proc('glGenQueries', None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
glDeleteQueries = _get_proc('glDeleteQueries', None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
glBeginQuery = _get_proc('glBeginQuery', None, ctypes.c_uint, ctypes.c_uint)
glEndQuery = _get_proc('glEndQuery', None, ctypes.c_uint)
glGetQueryObjectui64v = _get_proc('glGetQueryObjectui64v', None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64))

class GLBuffer:
  def __init__(self, buf: int, size: int): self.buf, self.size = buf, size

GLBuf = GLBuffer | tuple[GLBuffer, int]

class GLProgram(Program['GLDevice']):
  def __init__(self, dev:'GLDevice', obj:TinyELF):
    self.dev, self.name = dev, obj.name
    self.dev.make_current()

    sh = glCreateShader(GL_COMPUTE_SHADER)
    src_bytes = obj.lib
    c_src = ctypes.c_char_p(src_bytes)
    glShaderSource(sh, 1, ctypes.cast(ctypes.byref(c_src), ctypes.POINTER(ctypes.c_char_p)), None)
    glCompileShader(sh)

    status = ctypes.c_int()
    glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(status))
    if not status.value:
      log_len = ctypes.c_int()
      glGetShaderiv(sh, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
      log_buf = ctypes.create_string_buffer(log_len.value + 1)
      glGetShaderInfoLog(sh, log_len.value, None, log_buf)
      glDeleteShader(sh)
      raise CompileError(f"GL Shader Compile Error:\n{log_buf.value.decode()}")

    self.prg = glCreateProgram()
    glAttachShader(self.prg, sh)
    glLinkProgram(self.prg)
    glGetProgramiv(self.prg, GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
      log_len = ctypes.c_int()
      glGetProgramiv(self.prg, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
      log_buf = ctypes.create_string_buffer(log_len.value + 1)
      glGetProgramInfoLog(self.prg, log_len.value, None, log_buf)
      glDeleteShader(sh)
      glDeleteProgram(self.prg)
      raise CompileError(f"GL Program Link Error:\n{log_buf.value.decode()}")

    glDeleteShader(sh)
    self.global_slots = [slot for _, slot, _, _ in obj.signature]
    self.alu_dtypes = [dt for _, _, dt, _ in obj.signature]
    self.ubo: int | None = None
    self.ubo_size = 0

  @suppress_finalizing
  def __del__(self):
    if hasattr(self, 'prg') and self.prg:
      self.dev.make_current()
      glDeleteProgram(self.prg)
    if self.ubo is not None:
      self.dev.make_current()
      glDeleteBuffers(1, ctypes.byref(ctypes.c_uint(self.ubo)))

  def __call__(self, *bufs:GLBuf, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]|None=None,
               vals:tuple[int, ...]=(), wait=False, **kw) -> float|None:
    self.dev.make_current()
    glUseProgram(self.prg)
    for i, b in enumerate(bufs):
      buf, offset = (b[0], b[1]) if isinstance(b, tuple) else (b, 0)
      if offset == 0:
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, self.global_slots[i], buf.buf)
      else:
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, self.global_slots[i], buf.buf, offset, buf.size - offset)
    if len(vals) and vals[0] is not None and all(v is not None for v in vals):
      n_alu = len(vals)
      alu_dtypes = self.alu_dtypes[-n_alu:] if len(self.alu_dtypes) >= n_alu else self.alu_dtypes
      req_sz = round_up(4 * n_alu, 16)
      if self.ubo is None or self.ubo_size < req_sz:
        if self.ubo is not None:
          glDeleteBuffers(1, ctypes.byref(ctypes.c_uint(self.ubo)))
        u = ctypes.c_uint()
        glGenBuffers(1, ctypes.byref(u))
        self.ubo, self.ubo_size = u.value, req_sz
        glBindBuffer(GL_UNIFORM_BUFFER, self.ubo)
        glBufferData(GL_UNIFORM_BUFFER, self.ubo_size, None, GL_DYNAMIC_DRAW)
      fmt = "<" + "".join("i" if dt.name == "int" else "I" if dt.name == "unsigned int" else
                          "f" if dt.name == "float" else "i" for dt in alu_dtypes)
      packed = struct.pack(fmt, *vals)
      glBindBuffer(GL_UNIFORM_BUFFER, self.ubo)
      glBufferSubData(GL_UNIFORM_BUFFER, 0, len(packed), packed)
      glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.ubo)

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)
    if wait:
      q = ctypes.c_uint()
      glGenQueries(1, ctypes.byref(q))
      glBeginQuery(GL_TIME_ELAPSED, q)
      glDispatchCompute(*global_size)
      glEndQuery(GL_TIME_ELAPSED)
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)
      glFinish()
      elapsed = ctypes.c_uint64()
      glGetQueryObjectui64v(q, GL_QUERY_RESULT, ctypes.byref(elapsed))
      glDeleteQueries(1, ctypes.byref(q))
      return elapsed.value / 1e9
    glDispatchCompute(*global_size)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)
    return None

class GLAllocator(LRUAllocator['GLDevice']):
  def __init__(self, dev:'GLDevice'): super().__init__(dev, supports_transfer=False)

  def _alloc(self, size:int, options:BufferSpec) -> GLBuffer:
    self.dev.make_current()
    buf = ctypes.c_uint()
    glGenBuffers(1, ctypes.byref(buf))
    sz = round_up(size + 256, 4)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf.value)
    glBufferData(GL_SHADER_STORAGE_BUFFER, sz, None, GL_DYNAMIC_DRAW)
    return GLBuffer(buf.value, sz)

  @suppress_finalizing
  def _free(self, opaque:GLBuffer, options:BufferSpec):
    self.dev.make_current()
    buf = (ctypes.c_uint * 1)(opaque.buf)
    glDeleteBuffers(1, buf)

  def _offset(self, buf:GLBuf, size:int, offset:int) -> tuple[GLBuffer, int]:
    base, base_off = (buf[0], buf[1]) if isinstance(buf, tuple) else (buf, 0)
    return base, base_off + offset

  def _copyin(self, dest:GLBuf, src:memoryview):
    self.dev.make_current()
    buf, offset = (dest[0], dest[1]) if isinstance(dest, tuple) else (dest, 0)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf.buf)
    if src.nbytes % 4:
      padded = bytearray(round_up(src.nbytes, 4))
      padded[:src.nbytes] = src
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, len(padded), from_mv(memoryview(padded)))
    else:
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, src.nbytes, from_mv(src))

  def _copyout(self, dest:memoryview, src:GLBuf):
    self.dev.make_current()
    buf, offset = (src[0], src[1]) if isinstance(src, tuple) else (src, 0)
    glFinish()
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf.buf)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, dest.nbytes, from_mv(dest))

class GLDevice(Compiled):
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.dpy, self.ctx = _init_egl_context(self.device_id)
    self.make_current()
    ver_str = glGetString(GL_VERSION)
    if DEBUG >= 1:
      vendor = glGetString(GL_VENDOR)
      renderer = glGetString(GL_RENDERER)
      print(f"GLDevice: {vendor.decode() if vendor else ''} {renderer.decode() if renderer else ''} ({ver_str.decode() if ver_str else ''})")
    super().__init__(device, GLAllocator(self), [GLRenderer], GLProgram, arch="gl430")

  def make_current(self):
    egl_dll.eglMakeCurrent(self.dpy, None, None, self.ctx)

  def synchronize(self):
    self.make_current()
    glFinish()

# radeonsi (Mesa 26.1) miscompiles winograd conv kernels, so skip the winograd path on GL/MGL and
# always build the direct conv graph (correct, just not as fast). confined to this backend module.
from tinygrad.device import Device
from tinygrad.mixin.op import OpMixin
_conv2d_orig = OpMixin.conv2d
def _conv2d_gl(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, dtype=None):
  if Device.DEFAULT.startswith("GL") or Device.DEFAULT.startswith("MGL"):
    with Context(WINO=0): return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)
  return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)
setattr(OpMixin, "conv2d", _conv2d_gl)
