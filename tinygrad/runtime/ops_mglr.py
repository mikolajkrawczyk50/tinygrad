import os, struct, moderngl
from typing import Any
from tinygrad.device import Compiled, Allocator, BufferSpec, Program, TinyELF
from tinygrad.helpers import round_up, prod
from tinygrad.renderer.glsl import MGLRasterRenderer, MGLRASTER_MAGIC

class MGLRasterProgram(Program['MGLRDevice']):
  def __init__(self, dev:'MGLRDevice', obj:TinyELF):
    self.dev, self.name = dev, obj.name
    assert obj.lib.startswith(MGLRASTER_MAGIC), f"MGLR runtime requires a raster lib (magic {MGLRASTER_MAGIC!r}), got {obj.lib[:8]!r}"
    vs_len, w, h = struct.unpack("<III", obj.lib[len(MGLRASTER_MAGIC):len(MGLRASTER_MAGIC)+12])
    vs = obj.lib[len(MGLRASTER_MAGIC)+12:len(MGLRASTER_MAGIC)+12+vs_len].decode()
    fs = obj.lib[len(MGLRASTER_MAGIC)+12+vs_len:].decode()
    self.prg = dev.ctx.program(vertex_shader=vs, fragment_shader=fs)
    self.vao = dev.ctx.vertex_array(self.prg, [])
    self.global_slots = [slot for _, slot, _, _ in obj.signature]
    self.out_shape = obj.signature[0][3] if len(obj.signature) else ()
    self.viewport: tuple[int, int]|None = (w, h) if w and h else None
    self.u_size: Any = self.prg.get('u_size', None)
    self.alu_dtypes = [dt for _, _, dt, _ in obj.signature]
    self.ubo: moderngl.Buffer|None = None
    self.has_loops = "Lidx" in fs

  def _viewport(self, global_size:tuple[int,int,int], local_size:tuple[int,int,int]|None=None) -> tuple[int, int]:
    if self.viewport is not None: return self.viewport
    ls = local_size if local_size is not None else (1, 1, 1)
    n = prod(global_size) * prod(ls)
    if n <= 1:
      if not self.has_loops and len(self.out_shape) > 0:
        n = prod(self.out_shape)
      else:
        return (1, 1)
    w = max(1, min(n, 4096))
    h = max(1, (n + w - 1) // w)
    return (w, h)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]|None=None,
               vals:tuple[int, ...]=(), wait=False, **kw) -> float|None:
    for i, b in enumerate(bufs):
      if isinstance(b, tuple): b[0].bind_to_storage_buffer(self.global_slots[i], offset=b[1])
      else: b.bind_to_storage_buffer(self.global_slots[i])
    if len(vals) and vals[0] is not None and all(v is not None for v in vals):
      n_alu = len(vals)
      self.alu_dtypes = self.alu_dtypes[-n_alu:] if len(self.alu_dtypes) >= n_alu else self.alu_dtypes
      if self.ubo is None or self.ubo.size < 4*n_alu: self.ubo = self.dev.ctx.buffer(reserve=4*n_alu)
      fmt = "<" + "".join("i" if dt.name == "int" else "I" if dt.name == "unsigned int" else
                          "f" if dt.name == "float" else "i" for dt in self.alu_dtypes)
      self.ubo.write(struct.pack(fmt, *vals))
      self.ubo.bind_to_uniform_block(0)
    w, h = self._viewport(global_size, local_size)
    if (w, h) not in self.dev.fbo_cache:
      tex = self.dev.ctx.texture((w, h), 4)
      self.dev.fbo_cache[(w, h)] = self.dev.ctx.framebuffer(color_attachments=[tex])
    self.dev.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT)
    self.dev.fbo_cache[(w, h)].use()
    self.dev.ctx.viewport = (0, 0, w, h)
    if self.u_size is not None: self.u_size.value = (w, h)
    if wait:
      with self.dev.ctx.query(time=True) as q:
        self.vao.render(moderngl.TRIANGLES, vertices=6)
      self.dev.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT)
      self.dev.ctx.finish()
      return q.elapsed / 1e9
    self.vao.render(moderngl.TRIANGLES, vertices=6)
    self.dev.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT)
    self.dev.ctx.finish()
    return None

MGLRBuf = moderngl.Buffer | tuple[moderngl.Buffer, int]

class MGLRAllocator(Allocator['MGLRDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> moderngl.Buffer:
    # OpenGL buffers have to be 4-byte aligned (same as MGL)
    return self.dev.ctx.buffer(reserve=round_up(size, 4))

  def _free(self, opaque:moderngl.Buffer, options:BufferSpec): opaque.release()

  def _offset(self, buf:moderngl.Buffer, size:int, offset:int) -> tuple[moderngl.Buffer, int]:
    return buf, offset

  def _copyin(self, dest:MGLRBuf, src:memoryview):
    buf, offset = (dest[0], dest[1]) if isinstance(dest, tuple) else (dest, 0)
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    buf.write(padded_src if src.nbytes % 4 else src, offset=offset)

  def _copyout(self, dest:memoryview, src:MGLRBuf):
    buf, offset = (src[0], src[1]) if isinstance(src, tuple) else (src, 0)
    dest[:] = buf.read(size=dest.nbytes, offset=offset)

class MGLRDevice(Compiled):
  def __init__(self, device:str):
    self.ctx = moderngl.create_standalone_context(backend=os.getenv("MGL_BACKEND", "egl"), require=430)  # type: ignore[arg-type]
    self.fbo_cache: dict[tuple[int, int], moderngl.Framebuffer] = {}
    super().__init__(device, MGLRAllocator(self), [MGLRasterRenderer], MGLRasterProgram, arch="gl430")

  def synchronize(self): self.ctx.finish()

# radeonsi (Mesa 26.1) miscompiles winograd conv kernels, so skip the winograd path on MGL and
# always build the direct conv graph (correct, just not as fast). confined to this backend module.
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.mixin.op import OpMixin
_conv2d_orig = getattr(OpMixin, "_conv2d_orig", OpMixin.conv2d)
setattr(OpMixin, "_conv2d_orig", _conv2d_orig)
def _conv2d_mgl(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, dtype=None):
  if Device.DEFAULT.startswith("MGL"):
    with Context(WINO=0): return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)
  return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)
setattr(OpMixin, "conv2d", _conv2d_mgl)