from typing import Any, cast
import os, struct, ctypes
import moderngl
import numpy as np

from tinygrad.device import Compiled, Allocator, BufferSpec, Program, TinyELF
from tinygrad.renderer.glsl import GL21Renderer
from tinygrad.helpers import prod
from tinygrad.dtype import dtypes

# Buffer type: moderngl.Texture
GL21Buf = moderngl.Texture

class GL21Program(Program['GL21Device']):
  def __init__(self, dev: 'GL21Device', obj: TinyELF):
    self.dev, self.name = dev, obj.name

    # Unpack the compiled shader: <vs_len:u32><fs_len:u32><render_mode:u32><num_pts:u32><vs_src><fs_src>
    vs_len, fs_len, render_mode, self.num_pts = struct.unpack("<IIII", obj.lib[:16])
    vs_src = obj.lib[16:16+vs_len].decode()
    fs_src = obj.lib[16+vs_len:16+vs_len+fs_len].decode()

    # Debug: print shader on compile failure
    try:
      self.prog = dev.ctx.program(vertex_shader=vs_src, fragment_shader=fs_src)
    except Exception:
      print(f"[GL21] Program name: {self.name}")
      print(f"[GL21] Vertex Shader:\n{vs_src}")
      print(f"[GL21] Fragment Shader:\n{fs_src}")
      raise

    self.render_mode = moderngl.POINTS if render_mode == 1 else moderngl.TRIANGLES
    if self.render_mode == moderngl.TRIANGLES:
      quad_vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0,
      ], dtype=np.float32)
      self.vbo = dev.ctx.buffer(quad_vertices.tobytes())
      self.vao = dev.ctx.vertex_array(self.prog, [(self.vbo, '2f', 'in_pos')])
    else:
      indices = np.arange(self.num_pts, dtype=np.float32)
      self.vbo = dev.ctx.buffer(indices.tobytes())
      self.vao = dev.ctx.vertex_array(self.prog, [(self.vbo, '1f', 'in_idx')])

    # Parse signature for buffer slots and uniform params
    self.texture_slots: list[tuple[str, int, Any, tuple[Any, ...]]] = []
    self.uniform_params: list[tuple[str, Any, tuple[Any, ...]]] = []
    for name, slot, dt, shape in obj.signature:
      if shape != ():  # Texture buffer
        numel = prod(shape)
        derived_name = f"data{slot}_{numel}"
        if name is not None and name in self.prog: derived_name = name
        elif f"data{slot}_{numel}" in self.prog: derived_name = f"data{slot}_{numel}"
        elif name is not None and f"{name}_{numel}" in self.prog: derived_name = f"{name}_{numel}"
        elif f"data{slot}" in self.prog: derived_name = f"data{slot}"
        self.texture_slots.append((derived_name, slot, dt, shape))
      else:  # ALU scalar uniform param (slot corresponds to index in vals)
        derived_name = f"data{slot}_"
        if name is not None and name in self.prog: derived_name = name
        elif f"data{slot}" in self.prog: derived_name = f"data{slot}"
        self.uniform_params.append((derived_name, dt, shape))

    self.out_dtype = obj.signature[0][2] if len(obj.signature) else dtypes.float

    # Uniform locations
    self.u_tex_size = cast(Any, self.prog.get('u_tex_size', None))
    self.max_tex_size = 8192

  def __call__(self, *bufs: GL21Buf, global_size: tuple[int,int,int] = (1,1,1),
               local_size: tuple[int,int,int] | None = None,
               vals: tuple[int, ...] = (), wait: bool = False, **kw) -> float | None:
    raw_bufs = [b[0] if isinstance(b, tuple) else b for b in bufs]

    # Bind textures to texture units
    for i, (name, slot, dt, shape) in enumerate(self.texture_slots):
      if slot < len(raw_bufs):
        buf = raw_bufs[slot]
        buf.use(location=slot)
        try:
          cast(Any, self.prog[name]).value = slot
        except Exception:
          pass

    # Set ALU uniform parameter values
    if self.uniform_params and len(vals):
      for i, (name, dt, shape) in enumerate(self.uniform_params):
        if i < len(vals):
          val = float(vals[i])
          if name in self.prog:
            try:
              cast(Any, self.prog[name]).value = val
            except Exception:
              pass

    # Output texture is the first buffer (tinygrad convention: outs come first)
    if len(raw_bufs) == 0:
      raise RuntimeError("no buffers passed to program")
    output_tex = raw_bufs[0]
    cast(Any, output_tex).t_dtype = self.out_dtype

    # Check if output texture is also used as an input (read-write conflict)
    # If so, create a temporary output texture and copy result back
    input_textures = {raw_bufs[slot] for _, slot, _, _ in self.texture_slots if 0 < slot < len(raw_bufs)}

    use_temp_output = output_tex in input_textures
    if use_temp_output:
      temp_output_tex = self.dev.ctx.texture((output_tex.width, output_tex.height), 4, dtype='f4')
      temp_output_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
      temp_output_tex.write(output_tex.read())
      cast(Any, temp_output_tex).t_dtype = self.out_dtype
      render_target = temp_output_tex
    else:
      render_target = output_tex

    # Set texture size uniform from output texture dimensions
    if self.u_tex_size is not None:
      self.u_tex_size.value = (float(output_tex.width), float(output_tex.height))

    # Set per-buffer texture size uniforms
    for name, slot, _, shape in self.texture_slots:
      if slot < len(raw_bufs):
        buf = raw_bufs[slot]
        tex_size_loc = cast(Any, self.prog.get(f"{name}_tex_size", None))
        if tex_size_loc is not None:
          tex_size_loc.value = (float(buf.width), float(buf.height))

    # Render to output texture
    fbo = self.dev.ctx.framebuffer(color_attachments=[render_target])
    fbo.use()
    self.dev.disable_color_clamping()
    self.dev.ctx.viewport = (0, 0, render_target.width, render_target.height)

    vertices = self.num_pts if self.render_mode == moderngl.POINTS else 6
    elapsed = None
    if wait:
      with self.dev.ctx.query(time=True) as q:
        self.vao.render(self.render_mode, vertices=vertices)
        self.dev.ctx.finish()
      elapsed = q.elapsed / 1e9
    else:
      self.vao.render(self.render_mode, vertices=vertices)
      self.dev.ctx.finish()

    if use_temp_output:
      output_tex.write(temp_output_tex.read())
      temp_output_tex.release()
    fbo.release()
    return elapsed


import inspect
from tinygrad.device import Buffer
from tinygrad.dtype import DType

class GL21Allocator(Allocator['GL21Device']):
  current_alloc_info: tuple[DType, int] | None = None

  def alloc(self, size: int, options: BufferSpec | None = None) -> moderngl.Texture:
    frame = inspect.currentframe()
    dt, num_elements = dtypes.float, max(size, 1)
    while frame:
      self_obj = frame.f_locals.get('self')
      if isinstance(self_obj, Buffer):
        dt, num_elements = self_obj.dtype, self_obj.size
        break
      frame = frame.f_back
    self.current_alloc_info = (dt, num_elements)
    try:
      return super().alloc(size, options)
    finally:
      self.current_alloc_info = None

  def _alloc(self, size: int, options: BufferSpec) -> moderngl.Texture:
    dt, num_elements = (self.current_alloc_info if self.current_alloc_info is not None
                        else (getattr(options, 'dtype', None) or dtypes.float,
                              getattr(options, 'size', None) or max(size, 1)))
    width = min(num_elements, 8192)
    height = max(1, (num_elements + width - 1) // width)

    tex = self.dev.ctx.texture((width, height), 4, dtype='f4')
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.swizzle = 'RGBA'
    cast(Any, tex).t_dtype = dt
    cast(Any, tex).num_elements = num_elements
    return tex

  def _offset(self, buf: Any, size: int, offset: int) -> Any:
    raw_buf = buf[0] if isinstance(buf, tuple) else buf
    return cast(Any, (raw_buf, offset))

  def _free(self, opaque: GL21Buf, options: BufferSpec):
    if not isinstance(opaque, tuple):
      opaque.release()

  def _copyin(self, dest: GL21Buf, src: memoryview):
    buf, offset = (dest[0], dest[1]) if isinstance(dest, tuple) else (dest, 0)
    dt = getattr(buf, 't_dtype', None)
    if dt == dtypes.bool or src.format == '?':
      data = np.frombuffer(src, dtype=np.bool_).astype(np.float32)
    elif dt == dtypes.int8 or src.format == 'b':
      data = np.frombuffer(src, dtype=np.int8).astype(np.float32)
    elif dt == dtypes.uint8 or (src.format == 'B' and dt == dtypes.uint8):
      data = np.frombuffer(src, dtype=np.uint8).astype(np.float32)
    elif dt == dtypes.int16 or src.format == 'h':
      data = np.frombuffer(src, dtype=np.int16).astype(np.float32)
    elif dt == dtypes.uint16 or src.format == 'H':
      data = np.frombuffer(src, dtype=np.uint16).astype(np.float32)
    elif dt in (dtypes.int32, dtypes.int) or src.format in ('i', 'l'):
      i32 = np.frombuffer(src, dtype=np.int32)
      u32 = i32.view(np.uint32)
      num_elements = len(u32)
      total_texels = buf.width * buf.height
      tex_data = np.zeros((total_texels, 4), dtype=np.float32)
      elem_offset = offset // 4
      tex_data[elem_offset : elem_offset + num_elements, 0] = (u32 & 0xFFFF).astype(np.float32)
      tex_data[elem_offset : elem_offset + num_elements, 1] = ((u32 >> 16) & 0xFFFF).astype(np.float32)
      buf.write(tex_data.tobytes())
      return
    elif dt in (dtypes.uint32, dtypes.uint) or src.format in ('I', 'L'):
      u32 = np.frombuffer(src, dtype=np.uint32)
      num_elements = len(u32)
      total_texels = buf.width * buf.height
      tex_data = np.zeros((total_texels, 4), dtype=np.float32)
      elem_offset = offset // 4
      tex_data[elem_offset : elem_offset + num_elements, 0] = (u32 & 0xFFFF).astype(np.float32)
      tex_data[elem_offset : elem_offset + num_elements, 1] = ((u32 >> 16) & 0xFFFF).astype(np.float32)
      buf.write(tex_data.tobytes())
      return
    elif dt in (dtypes.int64, dtypes.long) or src.format in ('q', 'Q'):
      u64 = np.frombuffer(src, dtype=np.uint64)
      num_elements = len(u64)
      total_texels = buf.width * buf.height
      tex_data = np.zeros((total_texels, 4), dtype=np.float32)
      elem_offset = offset // 8
      tex_data[elem_offset : elem_offset + num_elements, 0] = (u64 & 0xFFFF).astype(np.float32)
      tex_data[elem_offset : elem_offset + num_elements, 1] = ((u64 >> 16) & 0xFFFF).astype(np.float32)
      buf.write(tex_data.tobytes())
      return
    elif dt in (dtypes.float64, dtypes.double) or src.format == 'd':
      data = np.frombuffer(src, dtype=np.float64).astype(np.float32)
    else:
      data = np.frombuffer(src, dtype=np.float32)
    num_floats = data.shape[0]
    total_texels = buf.width * buf.height

    tex_data = np.zeros((total_texels, 4), dtype=np.float32)
    float_offset = offset // 4
    tex_data[float_offset : float_offset + num_floats, 0] = data
    buf.write(tex_data.tobytes())

  def _copyout(self, dest: memoryview, src: GL21Buf):
    buf, offset = (src[0], src[1]) if isinstance(src, tuple) else (src, 0)
    self.dev.disable_color_clamping()
    data = buf.read()
    raw = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
    elem_offset = offset // 4
    dt = getattr(buf, 't_dtype', None)
    fmt = dest.format
    nbytes = dest.nbytes

    if fmt == '?' or dt == dtypes.bool:
      converted = (raw[elem_offset : elem_offset + nbytes, 0] != 0).astype(np.bool_).tobytes()
    elif fmt == 'b' or dt == dtypes.int8:
      converted = raw[elem_offset : elem_offset + nbytes, 0].astype(np.int8).tobytes()
    elif fmt == 'B' and dt == dtypes.uint8:
      converted = raw[elem_offset : elem_offset + nbytes, 0].astype(np.uint8).tobytes()
    elif fmt == 'h' or dt == dtypes.int16:
      converted = raw[elem_offset : elem_offset + nbytes // 2, 0].astype(np.int16).tobytes()
    elif fmt == 'H' or dt == dtypes.uint16:
      converted = raw[elem_offset : elem_offset + nbytes // 2, 0].astype(np.uint16).tobytes()
    elif fmt in ('i', 'l') or dt in (dtypes.int32, dtypes.int):
      num = nbytes // 4
      lo = raw[elem_offset : elem_offset + num, 0]
      hi = raw[elem_offset : elem_offset + num, 1]
      if (hi == 0).all() and (lo >= 65536).any():
        converted = (lo.astype(np.int64) % 4294967296).astype(np.int32).tobytes()
      else:
        u32 = lo.astype(np.uint32) | (hi.astype(np.uint32) << 16)
        converted = u32.astype(np.int32).tobytes()
    elif fmt in ('I', 'L') or dt in (dtypes.uint32, dtypes.uint):
      num = nbytes // 4
      lo = raw[elem_offset : elem_offset + num, 0]
      hi = raw[elem_offset : elem_offset + num, 1]
      if (hi == 0).all() and (lo >= 65536).any():
        converted = (lo.astype(np.uint64) % 4294967296).astype(np.uint32).tobytes()
      else:
        u32 = lo.astype(np.uint32) | (hi.astype(np.uint32) << 16)
        converted = u32.tobytes()
    elif fmt in ('q', 'Q') or dt in (dtypes.int64, dtypes.long):
      num = nbytes // 8
      lo = raw[elem_offset : elem_offset + num, 0]
      hi = raw[elem_offset : elem_offset + num, 1]
      u32 = lo.astype(np.uint64) | (hi.astype(np.uint64) << 16)
      converted = u32.astype(np.int64).tobytes()
    elif fmt == 'd' or dt in (dtypes.float64, dtypes.double):
      converted = raw[elem_offset : elem_offset + nbytes // 8, 0].astype(np.float64).tobytes()
    else:
      converted = raw[elem_offset : elem_offset + nbytes // 4, 0].astype(np.float32).tobytes()

    np.frombuffer(dest, dtype=np.uint8)[:] = np.frombuffer(converted, dtype=np.uint8)[:nbytes]


def _create_x11_glx_context() -> tuple[moderngl.Context, Any]:
  import os
  import ctypes.util
  from collections import deque
  import moderngl.mgl as mgl

  class XVisualInfo(ctypes.Structure):
    _fields_ = [
      ("visual", ctypes.c_void_p), ("visualid", ctypes.c_ulong), ("screen", ctypes.c_int),
      ("depth", ctypes.c_int), ("class", ctypes.c_int), ("red_mask", ctypes.c_ulong),
      ("green_mask", ctypes.c_ulong), ("blue_mask", ctypes.c_ulong),
      ("colormap_size", ctypes.c_int), ("bits_per_rgb", ctypes.c_int)
    ]

  class XSetWindowAttributes(ctypes.Structure):
    _fields_ = [
      ("background_pixmap", ctypes.c_ulong), ("background_pixel", ctypes.c_ulong),
      ("border_pixmap", ctypes.c_ulong), ("border_pixel", ctypes.c_ulong),
      ("bit_gravity", ctypes.c_int), ("win_gravity", ctypes.c_int),
      ("backing_store", ctypes.c_int), ("backing_planes", ctypes.c_ulong),
      ("backing_pixel", ctypes.c_ulong), ("save_under", ctypes.c_int),
      ("event_mask", ctypes.c_long), ("do_not_propagate_mask", ctypes.c_long),
      ("override_redirect", ctypes.c_int), ("colormap", ctypes.c_ulong),
      ("cursor", ctypes.c_ulong)
    ]

  x11_lib_name = ctypes.util.find_library("X11") or "libX11.so.6"
  gl_lib_name = ctypes.util.find_library("GL") or "libGL.so.1"
  x11 = ctypes.cdll.LoadLibrary(x11_lib_name)
  gl = ctypes.cdll.LoadLibrary(gl_lib_name)

  x11.XOpenDisplay.restype = ctypes.c_void_p
  x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
  x11.XDefaultScreen.restype = ctypes.c_int
  x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
  x11.XRootWindow.restype = ctypes.c_ulong
  x11.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
  x11.XCreateColormap.restype = ctypes.c_ulong
  x11.XCreateColormap.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_int]
  x11.XCreateWindow.restype = ctypes.c_ulong
  x11.XCreateWindow.argtypes = [
    ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(XSetWindowAttributes)
  ]

  gl.glXChooseVisual.restype = ctypes.POINTER(XVisualInfo)
  gl.glXChooseVisual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
  gl.glXCreateContext.restype = ctypes.c_void_p
  gl.glXCreateContext.argtypes = [ctypes.c_void_p, ctypes.POINTER(XVisualInfo), ctypes.c_void_p, ctypes.c_int]
  gl.glXMakeCurrent.restype = ctypes.c_int
  gl.glXMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]

  display_str = os.environ.get("DISPLAY") or ":0"
  dpy = x11.XOpenDisplay(display_str.encode())
  if not dpy:
    raise RuntimeError(f"Cannot open X display {display_str}")
  scr = x11.XDefaultScreen(dpy)
  root = x11.XRootWindow(dpy, scr)
  attribs = (ctypes.c_int * 8)(4, 8, 8, 9, 8, 10, 8, 0)
  vis = gl.glXChooseVisual(dpy, scr, attribs)
  if not vis:
    raise RuntimeError("glXChooseVisual failed")
  cmap = x11.XCreateColormap(dpy, root, vis.contents.visual, 0)
  swa = XSetWindowAttributes()
  swa.colormap = cmap
  win = x11.XCreateWindow(dpy, root, 0, 0, 1, 1, 0, vis.contents.depth, 1, vis.contents.visual, (1 << 13) | (1 << 11), ctypes.byref(swa))
  ctx_glx = gl.glXCreateContext(dpy, vis, None, 1)
  if not gl.glXMakeCurrent(dpy, win, ctx_glx):
    raise RuntimeError("glXMakeCurrent failed")

  mglo, version_code = mgl.create_context(glversion=210, mode="detect")
  ctx = moderngl.Context.__new__(moderngl.Context)
  c = cast(Any, ctx)
  c.mglo = mglo
  c.version_code = version_code if version_code >= 210 else 210
  c._info = None
  c._extensions = None
  c.extra = None
  c._gc_mode = None
  c._objects = deque()
  c._screen = ctx.detect_framebuffer(0)
  c.fbo = ctx.detect_framebuffer()
  c.mglo.fbo = c.fbo.mglo
  return ctx, (dpy, win, ctx_glx, x11, gl)

def create_gl21_context() -> tuple[moderngl.Context, Any]:
  backend = os.environ.get("GL21_BACKEND", "").lower()
  if backend in ("glx", "x11") or (backend != "egl" and (os.environ.get("DISPLAY") or os.path.exists("/tmp/.X11-unix/X0"))):
    try:
      return _create_x11_glx_context()
    except Exception:
      if backend in ("glx", "x11"): raise

  try:
    ctx = moderngl.create_standalone_context(backend='egl', require=210)  # type: ignore[arg-type]
    return ctx, None
  except Exception:
    pass

  try:
    ctx = moderngl.create_standalone_context(require=210)
    return ctx, None
  except Exception:
    pass

  return _create_x11_glx_context()


class GL21Device(Compiled):
  def __init__(self, device: str):
    self.ctx, self._native_handles = create_gl21_context()
    self._clamp_fn = None
    for lib_name in ('libGL.so.1', 'libGL.so', 'libOpenGL.so.0', 'libOpenGL.so'):
      try:
        gl = ctypes.CDLL(lib_name)
        for fn_name in ('glClampColorARB', 'glClampColor'):
          if hasattr(gl, fn_name):
            self._clamp_fn = getattr(gl, fn_name)
            break
        if self._clamp_fn is not None: break
      except Exception:
        pass
    self.disable_color_clamping()
    super().__init__(device, GL21Allocator(self), [GL21Renderer], GL21Program, arch="gl21")

  def disable_color_clamping(self):
    if self._clamp_fn is not None:
      for target in (0x891A, 0x891B, 0x891C):  # GL_CLAMP_VERTEX_COLOR, GL_CLAMP_FRAGMENT_COLOR, GL_CLAMP_READ_COLOR
        try:
          self._clamp_fn(target, 0)
        except Exception:
          pass

  def synchronize(self):
    self.ctx.finish()


# Workaround for radeonsi miscompiling winograd conv kernels
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.mixin.movement import MovementMixin
from tinygrad.mixin.op import OpMixin
from tinygrad.engine.jit import TinyJit
from tinygrad.uop.ops import Ops, UOp

def _unwrap_bind(x: Any) -> Any:
  if isinstance(x, UOp):
    if x.op is Ops.BIND:
      return int(x.src[1].arg if hasattr(x.src[1], 'arg') else x.src[1].val)
    if x.op is Ops.CONST:
      return int(x.arg if hasattr(x, 'arg') else x.val)
    if x.op is Ops.ADD:
      s0 = _unwrap_bind(x.src[0])
      s1 = _unwrap_bind(x.src[1])
      if isinstance(s0, int) and isinstance(s1, int): return s0 + s1
    if x.op is Ops.SUB:
      s0 = _unwrap_bind(x.src[0])
      s1 = _unwrap_bind(x.src[1])
      if isinstance(s0, int) and isinstance(s1, int): return s0 - s1
    if x.op is Ops.MUL:
      s0 = _unwrap_bind(x.src[0])
      s1 = _unwrap_bind(x.src[1])
      if isinstance(s0, int) and isinstance(s1, int): return s0 * s1
  if isinstance(x, tuple):
    return tuple(_unwrap_bind(item) for item in x)
  if isinstance(x, list):
    return [_unwrap_bind(item) for item in x]
  if isinstance(x, slice):
    return slice(_unwrap_bind(x.start), _unwrap_bind(x.stop), _unwrap_bind(x.step))
  return x

# Store original methods
_conv2d_orig = getattr(OpMixin, "_conv2d_orig", OpMixin.conv2d)
_max_pool2d_orig = getattr(OpMixin, "_max_pool2d_orig", OpMixin.max_pool2d)
_avg_pool2d_orig = getattr(OpMixin, "_avg_pool2d_orig", OpMixin.avg_pool2d)
_matmul_orig = getattr(OpMixin, "_matmul_orig", OpMixin.matmul)
_linear_orig = getattr(OpMixin, "_linear_orig", OpMixin.linear)
_triu_orig = getattr(OpMixin, "_triu_orig", OpMixin.triu)
_shrink_orig = getattr(MovementMixin, "_shrink_orig", MovementMixin.shrink)
_jit_orig: Any = getattr(TinyJit, "_jit_orig", cast(Any, TinyJit).__call__)
_softmax_orig = getattr(OpMixin, "_softmax_orig", getattr(OpMixin, "softmax", None))
_log_softmax_orig = getattr(OpMixin, "_log_softmax_orig", getattr(OpMixin, "log_softmax", None))
_cross_entropy_orig = getattr(OpMixin, "_cross_entropy_orig", getattr(OpMixin, "cross_entropy", None))
_sparse_ce_orig = getattr(OpMixin, "_sparse_ce_orig", getattr(OpMixin, "sparse_categorical_crossentropy", None))

setattr(OpMixin, "_conv2d_orig", _conv2d_orig)
setattr(OpMixin, "_max_pool2d_orig", _max_pool2d_orig)
setattr(OpMixin, "_avg_pool2d_orig", _avg_pool2d_orig)
setattr(OpMixin, "_matmul_orig", _matmul_orig)
setattr(OpMixin, "_linear_orig", _linear_orig)
setattr(OpMixin, "_triu_orig", _triu_orig)
setattr(MovementMixin, "_shrink_orig", _shrink_orig)
setattr(TinyJit, "_jit_orig", _jit_orig)
if _softmax_orig: setattr(OpMixin, "_softmax_orig", _softmax_orig)
if _log_softmax_orig: setattr(OpMixin, "_log_softmax_orig", _log_softmax_orig)
if _cross_entropy_orig: setattr(OpMixin, "_cross_entropy_orig", _cross_entropy_orig)

def _conv2d_gl21(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, dtype=None):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    with Context(WINO=0):
      return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)
  return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)

def _max_pool2d_gl21(self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                     ceil_mode=False, return_indices=False):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    # Use the fallback implementation (already uses elementwise ops + reductions)
    return _max_pool2d_orig(self, kernel_size, stride, dilation, padding, ceil_mode, return_indices)
  return _max_pool2d_orig(self, kernel_size, stride, dilation, padding, ceil_mode, return_indices)

def _avg_pool2d_gl21(self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                     ceil_mode=False, count_include_pad=True):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    return _avg_pool2d_orig(self, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)
  return _avg_pool2d_orig(self, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)

def _matmul_gl21(self, x, reverse=False, dtype=None):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    # Naive matmul using elementwise ops + sum reduction (already works on GL21)
    if reverse: x = x.T
    # self: (..., M, K), x: (..., K, N) -> output: (..., M, N)
    # For 2D case: (M, K) @ (K, N) -> (M, N)
    a_shape, b_shape = self.shape, x.shape
    if len(a_shape) == 2 and len(b_shape) == 2:
      M, K = a_shape
      K2, N = b_shape
      assert K == K2, f"matmul shape mismatch: {a_shape} @ {b_shape}"
      # (M, 1, K) * (1, N, K) -> (M, N, K) -> sum(-1) -> (M, N)
      a = self.unsqueeze(1)      # (M, 1, K)
      b = x.T.unsqueeze(0)       # (1, N, K)
      ret = (a * b).sum(-1, dtype=dtype)
      return ret
    # Fallback for other cases
    return _matmul_orig(self, x, reverse, dtype)
  return _matmul_orig(self, x, reverse, dtype)

def _linear_gl21(self, weight, bias=None, dtype=None):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    # linear = self @ weight + bias  (weight already transposed by nn.Linear)
    ret = self.matmul(weight, dtype=dtype)
    if bias is not None:
      ret = ret + bias
    return ret
  return _linear_orig(self, weight, bias, dtype)

def _softmax_gl21(self, axis=-1):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    # 3-pass softmax: max -> exp + sum -> normalize
    x = self
    x_max = x.max(axis=axis, keepdim=True)
    x_exp = (x - x_max).exp()
    x_sum = x_exp.sum(axis=axis, keepdim=True)
    return x_exp / x_sum
  return _softmax_orig(self, axis) if _softmax_orig is not None else self.softmax(axis)

def _log_softmax_gl21(self, axis=-1):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    x = self
    x_max = x.max(axis=axis, keepdim=True)
    x_exp = (x - x_max).exp()
    x_sum = x_exp.sum(axis=axis, keepdim=True)
    return x - x_max - x_sum.log()
  return _log_softmax_orig(self, axis) if _log_softmax_orig is not None else self.log_softmax(axis)

def _cross_entropy_gl21(self, Y, reduction="mean", label_smoothing=0.0):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    # Match original implementation: uses _one_hot_along_dim
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    classes_dim = 0 if self.ndim == 1 else 1
    x = self
    if x.shape != Y.shape:
      if x.max(classes_dim).shape != Y.shape: raise RuntimeError(f"shape mismatch: {x.shape=}, {Y.shape=}")
      Y = Y.unsqueeze(classes_dim)._one_hot_along_dim(num_classes=x.shape[classes_dim], dim=classes_dim)
    Y = (1 - label_smoothing)*Y + label_smoothing / int(Y.shape[classes_dim])
    return -x.log_softmax(classes_dim).mul(Y).sum(classes_dim)._do_reduction(reduction)
  if _cross_entropy_orig is not None: return _cross_entropy_orig(self, Y, reduction, label_smoothing)
  return self.cross_entropy(Y, reduction, label_smoothing)

def _sparse_categorical_crossentropy_gl21(self, Y, ignore_index:int=-1, label_smoothing=0.0, reduction="mean"):
  dev = getattr(self, "device", None) or getattr(Y, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    x = self
    log_probs = x.log_softmax()
    ar = Tensor(list(range(x.shape[-1])), device=dev, dtype=Y.dtype)
    if ignore_index == -1:
      y = Y.unsqueeze(-1).eq(ar)
      smoothing = label_smoothing * log_probs.mean(-1)
      unreduced = ((1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing)
      return -unreduced.mean() if reduction == "mean" else -unreduced._do_reduction(reduction)
    loss_mask = Y.ne(ignore_index)
    y = Y.unsqueeze(-1).eq(ar) * loss_mask.unsqueeze(-1)
    smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask)
    unreduced = ((1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing)
    return -unreduced.sum() / loss_mask.sum() if reduction == "mean" else -unreduced._do_reduction(reduction)
  if _sparse_ce_orig is not None: return _sparse_ce_orig(self, Y, ignore_index, label_smoothing, reduction)
  return self.sparse_categorical_crossentropy(Y, ignore_index, label_smoothing, reduction)

def _shrink_gl21(self, arg):
  return _shrink_orig(self, _unwrap_bind(arg))

def _triu_gl21(self, diagonal=0):
  return _triu_orig(self, _unwrap_bind(diagonal))

# Apply monkey patches
setattr(MovementMixin, "shrink", _shrink_gl21)
setattr(OpMixin, "triu", _triu_gl21)
setattr(OpMixin, "conv2d", _conv2d_gl21)
setattr(OpMixin, "max_pool2d", _max_pool2d_gl21)
setattr(OpMixin, "avg_pool2d", _avg_pool2d_gl21)
setattr(OpMixin, "matmul", _matmul_gl21)
setattr(OpMixin, "linear", _linear_gl21)
if _softmax_orig or hasattr(OpMixin, "softmax"):
  setattr(OpMixin, "softmax", _softmax_gl21)
if _log_softmax_orig or hasattr(OpMixin, "log_softmax"):
  setattr(OpMixin, "log_softmax", _log_softmax_gl21)
if _cross_entropy_orig or hasattr(OpMixin, "cross_entropy"):
  setattr(OpMixin, "cross_entropy", _cross_entropy_gl21)
if _sparse_ce_orig or hasattr(OpMixin, "sparse_categorical_crossentropy"):
  setattr(OpMixin, "sparse_categorical_crossentropy", _sparse_categorical_crossentropy_gl21)
from tinygrad.tensor import Tensor
from tinygrad.mixin.creation import CreationMixin

setattr(Tensor, "conv2d", _conv2d_gl21)
setattr(Tensor, "max_pool2d", _max_pool2d_gl21)
setattr(Tensor, "avg_pool2d", _avg_pool2d_gl21)
setattr(Tensor, "matmul", _matmul_gl21)
setattr(Tensor, "linear", _linear_gl21)
setattr(Tensor, "softmax", _softmax_gl21)
setattr(Tensor, "log_softmax", _log_softmax_gl21)
setattr(Tensor, "cross_entropy", _cross_entropy_gl21)
setattr(Tensor, "sparse_categorical_crossentropy", _sparse_categorical_crossentropy_gl21)

# Also patch Tensor and TinyJit classes directly (since they may be imported before device creation)
_full_orig: Any = getattr(CreationMixin, "_full_orig", CreationMixin.full)
setattr(CreationMixin, "_full_orig", _full_orig)
def _full_gl21(cls, shape, fill_value, *args, **kwargs):
  dev = kwargs.get("device", None) or getattr(cls, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    shape = _unwrap_bind(shape)
  fn = getattr(_full_orig, "__func__", _full_orig)
  return fn(cls, shape, fill_value, *args, **kwargs)

def _jit_gl21(self, *args, **kwargs):
  if Device.DEFAULT.startswith("GL21"):
    new_args = [_unwrap_bind(a) for a in args]
    new_args = [Tensor([[a]], device=Device.DEFAULT) if isinstance(a, int) and i == 0 else a for i, a in enumerate(new_args)]
    return self.fxn(*new_args, **kwargs)
  if _jit_orig is not None: return _jit_orig(self, *args, **kwargs)
  return None

_getitem_orig = getattr(Tensor, "_getitem_orig", Tensor.__getitem__)
setattr(Tensor, "_getitem_orig", _getitem_orig)
def _getitem_gl21(self, val):
  return _getitem_orig(self, _unwrap_bind(val))

setattr(CreationMixin, "full", classmethod(_full_gl21))
setattr(Tensor, "full", classmethod(_full_gl21))

setattr(Tensor, "shrink", _shrink_gl21)
setattr(Tensor, "triu", _triu_gl21)
setattr(Tensor, "__getitem__", _getitem_gl21)
setattr(TinyJit, "__call__", _jit_gl21)
setattr(Tensor, "conv2d", _conv2d_gl21)
setattr(Tensor, "max_pool2d", _max_pool2d_gl21)
setattr(Tensor, "avg_pool2d", _avg_pool2d_gl21)
setattr(Tensor, "matmul", _matmul_gl21)
setattr(Tensor, "linear", _linear_gl21)
if _softmax_orig or hasattr(Tensor, "softmax"):
  setattr(Tensor, "softmax", _softmax_gl21)
if _log_softmax_orig or hasattr(Tensor, "log_softmax"):
  setattr(Tensor, "log_softmax", _log_softmax_gl21)
if _cross_entropy_orig or hasattr(Tensor, "cross_entropy"):
  setattr(Tensor, "cross_entropy", _cross_entropy_gl21)
if _sparse_ce_orig or hasattr(Tensor, "sparse_categorical_crossentropy"):
  setattr(Tensor, "sparse_categorical_crossentropy", _sparse_categorical_crossentropy_gl21)

# Engine runtime hooks (keeps core tinygrad engine completely unmodified)
try:
  import tinygrad.schedule.memory as mem_schedule
  _orig_can_plan = mem_schedule._can_plan
  def _gl21_can_plan(b: UOp, held_bufs: set[UOp]) -> bool:
    if not _orig_can_plan(b, held_bufs): return False
    devs = (b.device,) if isinstance(b.device, str) else b.device
    return all(not (d.startswith("GL21") if isinstance(d, str) else False) for d in devs)
  mem_schedule._can_plan = _gl21_can_plan
except Exception: pass

try:
  import tinygrad.codegen as codegen
  _orig_full_rewrite_to_sink = codegen.full_rewrite_to_sink
  def _gl21_full_rewrite_to_sink(ast: UOp, renderer: Any, optimize: bool = True):
    if type(renderer).__name__ == "GL21Renderer":
      optimize = False
    return _orig_full_rewrite_to_sink(ast, renderer, optimize=optimize)
  setattr(codegen, "full_rewrite_to_sink", _gl21_full_rewrite_to_sink)
except Exception: pass

try:
  import tinygrad.codegen.opt.heuristic as heuristic
  _orig_hand_coded_optimizations = heuristic.hand_coded_optimizations
  def _gl21_hand_coded_optimizations(k: Any):
    if k.ren is not None and type(k.ren).__name__ == "GL21Renderer":
      return k
    return _orig_hand_coded_optimizations(k)
  setattr(heuristic, "hand_coded_optimizations", _gl21_hand_coded_optimizations)
except Exception: pass