from typing import Any, cast
import struct
import moderngl
import numpy as np

from tinygrad.device import Compiled, Allocator, BufferSpec, Program, TinyELF
from tinygrad.renderer.glsl import GL21Renderer
from tinygrad.helpers import round_up, prod
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
        derived_name = name
        if derived_name is None:
          shape_suffix = '_'.join([str(x) for x in shape]) if shape else ''
          derived_name = f"data{slot}_" + shape_suffix
        self.texture_slots.append((derived_name, slot, dt, shape))
      else:  # ALU scalar uniform param (slot corresponds to index in vals)
        derived_name = f"data{slot}_"
        self.uniform_params.append((derived_name, dt, shape))

    # Uniform locations
    self.u_tex_size = cast(Any, self.prog.get('u_tex_size', None))
    self.max_tex_size = 8192

  def __call__(self, *bufs: GL21Buf, global_size: tuple[int,int,int] = (1,1,1),
               local_size: tuple[int,int,int] | None = None,
               vals: tuple[int, ...] = (), wait: bool = False, **kw) -> float | None:
    raw_bufs = [b[0] if isinstance(b, tuple) else b for b in bufs]

    # Bind textures to texture units
    for i, (name, slot, dt, shape) in enumerate(self.texture_slots):
      if i < len(raw_bufs):
        buf = raw_bufs[i]
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

    # Check if output texture is also used as an input (read-write conflict)
    # If so, create a temporary output texture and copy result back
    input_textures = set(raw_bufs[1:])

    use_temp_output = output_tex in input_textures
    if use_temp_output:
      total = prod(global_size)
      if local_size:
        total *= prod(local_size)
      w = min(total, self.max_tex_size)
      h = max(1, (total + w - 1) // w)
      temp_output_tex = self.dev.ctx.texture((w, h), 4, dtype='f4')
      temp_output_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
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
    self.dev.ctx.viewport = (0, 0, render_target.width, render_target.height)

    vertices = self.num_pts if self.render_mode == moderngl.POINTS else 6
    if wait:
      with self.dev.ctx.query(time=True) as q:
        self.vao.render(self.render_mode, vertices=vertices)
        self.dev.ctx.finish()
      if use_temp_output:
        self.dev.ctx.copy_framebuffer(output_tex, fbo)
        temp_output_tex.release()
      return q.elapsed / 1e9

    self.vao.render(self.render_mode, vertices=vertices)
    self.dev.ctx.finish()
    if use_temp_output:
      self.dev.ctx.copy_framebuffer(output_tex, fbo)
      temp_output_tex.release()
    return None


class GL21Allocator(Allocator['GL21Device']):
  def _alloc(self, size: int, options: BufferSpec) -> moderngl.Texture:
    # OpenGL 2.1: allocate as RGBA32F texture
    # Minimum allocation size is 16 bytes (1 RGBA32F texel)
    num_floats = round_up(size, 16) // 4  # 16-byte aligned, 4 floats per texel
    # Use 1D texture layout: width = min(num_floats, 8192), height = ceil(num_floats/width)
    width = min(num_floats, 8192)
    height = max(1, (num_floats + width - 1) // width)

    tex = self.dev.ctx.texture((width, height), 4, dtype='f4')
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.swizzle = 'RGBA'
    return tex

  def _offset(self, buf: Any, size: int, offset: int) -> Any:
    raw_buf = buf[0] if isinstance(buf, tuple) else buf
    return cast(Any, (raw_buf, offset))

  def _free(self, opaque: GL21Buf, options: BufferSpec):
    if not isinstance(opaque, tuple):
      opaque.release()

  def _copyin(self, dest: GL21Buf, src: memoryview):
    buf, offset = (dest[0], dest[1]) if isinstance(dest, tuple) else (dest, 0)
    if src.format == '?' or getattr(src, 'itemsize', 1) == 1 and src.nbytes % 4 != 0:
      data = np.frombuffer(src, dtype=np.bool_ if src.format == '?' else np.uint8).astype(np.float32)
    elif getattr(src, 'itemsize', 4) == 8:
      data = np.frombuffer(src, dtype=np.float64 if src.format in ('d', 'f8') else np.int64).astype(np.float32)
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
    data = buf.read()
    floats = np.frombuffer(data, dtype=np.float32)[0::4]
    float_offset = offset // 4
    if dest.format == '?' or dest.nbytes % 4 != 0:
      num_elements = dest.nbytes
      converted = (floats[float_offset : float_offset + num_elements] != 0).astype(np.bool_).tobytes()
    else:
      needed_floats = dest.nbytes // 4
      converted = floats[float_offset : float_offset + needed_floats].tobytes()
    dest[:] = converted[:dest.nbytes]


class GL21Device(Compiled):
  def __init__(self, device: str):
    # Create OpenGL 2.1 context via EGL
    self.ctx = moderngl.create_standalone_context(backend='egl', require=210)  # type: ignore[arg-type]
    super().__init__(device, GL21Allocator(self), [GL21Renderer], GL21Program, arch="gl21")

  def synchronize(self):
    self.ctx.finish()


# Workaround for radeonsi miscompiling winograd conv kernels
from tinygrad.device import Device, canonicalize_device
from tinygrad.helpers import Context, argfix
from tinygrad.mixin.movement import MovementMixin
from tinygrad.mixin.op import OpMixin
from tinygrad.mixin.rand import RandMixin
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
_rand_orig = getattr(RandMixin, "_rand_orig", getattr(RandMixin, "rand", None))
_randn_orig = getattr(RandMixin, "_randn_orig", getattr(RandMixin, "randn", None))
_uniform_orig = getattr(RandMixin, "_uniform_orig", getattr(RandMixin, "uniform", None))
_normal_orig = getattr(RandMixin, "_normal_orig", getattr(RandMixin, "normal", None))

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
if _rand_orig: setattr(RandMixin, "_rand_orig", _rand_orig)
if _randn_orig: setattr(RandMixin, "_randn_orig", _randn_orig)
if _uniform_orig: setattr(RandMixin, "_uniform_orig", _uniform_orig)
if _normal_orig: setattr(RandMixin, "_normal_orig", _normal_orig)

def _conv2d_gl21(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, dtype=None):
  dev = getattr(self, "device", None) or Device.DEFAULT
  if str(dev).startswith("GL21") or Device.DEFAULT.startswith("GL21"):
    with Context(WINO=0):
      self_real = self.realize()
      buf_in = self_real.uop.buffer
      if buf_in is not None:
        self.uop = UOp.from_buffer(buf_in).reshape(self.shape)
        self_real = self
      ret = _conv2d_orig(self_real, weight, bias, groups, stride, dilation, padding, dtype)
      ret = ret.realize()
      buf_out = ret.uop.buffer
      if buf_out is not None:
        ret.uop = UOp.from_buffer(buf_out).reshape(ret.shape)
      return ret
  return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)

def _max_pool2d_gl21(self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                     ceil_mode=False, return_indices=False):
  if Device.DEFAULT.startswith("GL21"):
    # Use the fallback implementation (already uses elementwise ops + reductions)
    return _max_pool2d_orig(self, kernel_size, stride, dilation, padding, ceil_mode, return_indices)
  return _max_pool2d_orig(self, kernel_size, stride, dilation, padding, ceil_mode, return_indices)

def _avg_pool2d_gl21(self, kernel_size:tuple[int, ...]=(2,2), stride=None, dilation=1, padding:int|tuple[int, ...]=0,
                     ceil_mode=False, count_include_pad=True):
  if Device.DEFAULT.startswith("GL21"):
    return _avg_pool2d_orig(self, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)
  return _avg_pool2d_orig(self, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)

def _matmul_gl21(self, x, reverse=False, dtype=None):
  if Device.DEFAULT.startswith("GL21"):
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
  if Device.DEFAULT.startswith("GL21"):
    # linear = self @ weight + bias  (weight already transposed by nn.Linear)
    ret = self.matmul(weight, dtype=dtype)
    if bias is not None:
      ret = ret + bias
    return ret
  return _linear_orig(self, weight, bias, dtype)

def _softmax_gl21(self, axis=-1):
  if Device.DEFAULT.startswith("GL21"):
    # 3-pass softmax: max -> exp + sum -> normalize
    x = self
    x_max = x.max(axis=axis, keepdim=True)
    x_exp = (x - x_max).exp()
    x_sum = x_exp.sum(axis=axis, keepdim=True)
    return x_exp / x_sum
  return _softmax_orig(self, axis) if _softmax_orig is not None else self.softmax(axis)

def _log_softmax_gl21(self, axis=-1):
  if Device.DEFAULT.startswith("GL21"):
    x = self
    x_max = x.max(axis=axis, keepdim=True)
    x_exp = (x - x_max).exp()
    x_sum = x_exp.sum(axis=axis, keepdim=True)
    return x - x_max - x_sum.log()
  return _log_softmax_orig(self, axis) if _log_softmax_orig is not None else self.log_softmax(axis)

def _cross_entropy_gl21(self, Y, reduction="mean", label_smoothing=0.0):
  if Device.DEFAULT.startswith("GL21"):
    # Match original implementation: uses _one_hot_along_dim
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    classes_dim = 0 if self.ndim == 1 else 1
    if self.shape != Y.shape:
      if self.max(classes_dim).shape != Y.shape: raise RuntimeError(f"shape mismatch: {self.shape=}, {Y.shape=}")
      Y = Y.unsqueeze(classes_dim)._one_hot_along_dim(num_classes=self.shape[classes_dim], dim=classes_dim)
    Y = (1 - label_smoothing)*Y + label_smoothing / int(Y.shape[classes_dim])
    return -self.log_softmax(classes_dim).mul(Y).sum(classes_dim)._do_reduction(reduction)
  if _cross_entropy_orig is not None: return _cross_entropy_orig(self, Y, reduction, label_smoothing)
  return self.cross_entropy(Y, reduction, label_smoothing)

def _rand_gl21(cls, *shape, device=None, dtype=None, **kwargs):
  dev = canonicalize_device(device or Device.DEFAULT)
  if str(dev).startswith("GL21"):
    shape = argfix(*shape)
    dt = dtype or dtypes.default_float
    data = np.random.rand(*shape).astype(np.float32)
    return cls(data, device=dev, dtype=dt)
  target_cls = cls if issubclass(cls, Tensor) else Tensor
  if _rand_orig is not None: return _rand_orig.__func__(target_cls, *shape, device=device, dtype=dtype, **kwargs)
  return target_cls.rand(*shape, device=device, dtype=dtype, **kwargs)

def _randn_gl21(cls, *shape, device=None, dtype=None, **kwargs):
  dev = canonicalize_device(device or Device.DEFAULT)
  if str(dev).startswith("GL21"):
    shape = argfix(*shape)
    dt = dtype or dtypes.default_float
    data = np.random.randn(*shape).astype(np.float32)
    return cls(data, device=dev, dtype=dt)
  target_cls = cls if issubclass(cls, Tensor) else Tensor
  if _randn_orig is not None: return _randn_orig.__func__(target_cls, *shape, device=device, dtype=dtype, **kwargs)
  return target_cls.randn(*shape, device=device, dtype=dtype, **kwargs)

def _uniform_gl21(cls, *shape, low=0.0, high=1.0, dtype=None, device=None, **kwargs):
  dev = canonicalize_device(device or Device.DEFAULT)
  if str(dev).startswith("GL21"):
    shape = argfix(*shape)
    dt = dtype or dtypes.default_float
    data = np.random.uniform(low=low, high=high, size=shape).astype(np.float32)
    return cls(data, device=dev, dtype=dt)
  target_cls = cls if issubclass(cls, Tensor) else Tensor
  if _uniform_orig is not None: return _uniform_orig.__func__(target_cls, *shape, low=low, high=high, dtype=dtype, device=device, **kwargs)
  return target_cls.uniform(*shape, low=low, high=high, dtype=dtype, device=device, **kwargs)

def _normal_gl21(cls, *shape, mean=0.0, std=1.0, dtype=None, device=None, **kwargs):
  dev = canonicalize_device(device or Device.DEFAULT)
  if str(dev).startswith("GL21"):
    shape = argfix(*shape)
    dt = dtype or dtypes.default_float
    data = np.random.normal(loc=mean, scale=std, size=shape).astype(np.float32)
    return cls(data, device=dev, dtype=dt)
  target_cls = cls if issubclass(cls, Tensor) else Tensor
  if _normal_orig is not None: return _normal_orig.__func__(target_cls, *shape, mean=mean, std=std, dtype=dtype, device=device, **kwargs)
  return target_cls.normal(*shape, mean=mean, std=std, dtype=dtype, device=device, **kwargs)

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
if _rand_orig: setattr(RandMixin, "rand", classmethod(_rand_gl21))
if _randn_orig: setattr(RandMixin, "randn", classmethod(_randn_gl21))
if _uniform_orig: setattr(RandMixin, "uniform", classmethod(_uniform_gl21))
if _normal_orig: setattr(RandMixin, "normal", classmethod(_normal_gl21))

# Also patch Tensor and TinyJit classes directly (since they may be imported before device creation)
from tinygrad.tensor import Tensor
from tinygrad.mixin.creation import CreationMixin
_full_orig: Any = getattr(CreationMixin, "_full_orig", CreationMixin.full)
setattr(CreationMixin, "_full_orig", _full_orig)
def _full_gl21(cls, shape, fill_value, *args, **kwargs):
  if Device.DEFAULT.startswith("GL21"):
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
if _rand_orig: setattr(Tensor, "rand", classmethod(_rand_gl21))
if _randn_orig: setattr(Tensor, "randn", classmethod(_randn_gl21))
if _uniform_orig: setattr(Tensor, "uniform", classmethod(_uniform_gl21))
if _normal_orig: setattr(Tensor, "normal", classmethod(_normal_gl21))

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