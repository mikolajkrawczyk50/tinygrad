import os, moderngl, struct
from tinygrad.device import Compiled, Allocator, BufferSpec, Program, TinyELF
from tinygrad.renderer.glsl import MGLRenderer
from tinygrad.helpers import round_up

MGLBuf = moderngl.Buffer | tuple[moderngl.Buffer, int]

class MGLProgram(Program['MGLDevice']):
  def __init__(self, dev:'MGLDevice', obj:TinyELF):
    self.dev, self.name = dev, obj.name
    self.prg = dev.ctx.compute_shader(obj.lib.decode())
    # signature = (name, slot, dtype, shape) for globals then ALU vars; globals keep their slot as SSBO binding
    self.global_slots = [slot for _, slot, _, _ in obj.signature]
    # ALU params (symbolic vars) are the trailing entries of the signature; their dtypes drive UBO packing
    self.alu_dtypes = [dt for _, _, dt, _ in obj.signature]
    self.ubo:moderngl.Buffer|None = None

  def __call__(self, *bufs:MGLBuf, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]|None=None,
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
    self.dev.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT)
    if wait:
      with self.dev.ctx.query(time=True) as q:
        self.prg.run(*global_size)
      self.dev.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT)
      self.dev.ctx.finish()
      return q.elapsed / 1e9
    self.prg.run(*global_size)
    self.dev.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT)
    self.dev.ctx.finish()
    return None

class MGLAllocator(Allocator['MGLDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> moderngl.Buffer:
    # OpenGL buffers have to be 4-byte aligned
    return self.dev.ctx.buffer(reserve=round_up(size, 4))

  def _free(self, opaque:moderngl.Buffer, options:BufferSpec): opaque.release()

  def _offset(self, buf:moderngl.Buffer, size:int, offset:int) -> tuple[moderngl.Buffer, int]:
    return buf, offset

  def _copyin(self, dest:MGLBuf, src:memoryview):
    buf, offset = (dest[0], dest[1]) if isinstance(dest, tuple) else (dest, 0)
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    buf.write(padded_src if src.nbytes % 4 else src, offset=offset)

  def _copyout(self, dest:memoryview, src:MGLBuf):
    buf, offset = (src[0], src[1]) if isinstance(src, tuple) else (src, 0)
    dest[:] = buf.read(size=dest.nbytes, offset=offset)

class MGLDevice(Compiled):
  def __init__(self, device:str):
    self.ctx = moderngl.create_standalone_context(backend=os.getenv("MGL_BACKEND", "egl"), require=430)  # type: ignore[arg-type]
    super().__init__(device, MGLAllocator(self), [MGLRenderer], MGLProgram, arch="gl430")

  def synchronize(self): self.ctx.finish()

# radeonsi (Mesa 26.1) miscompiles winograd conv kernels, so skip the winograd path on MGL and
# always build the direct conv graph (correct, just not as fast). confined to this backend module.
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.mixin.op import OpMixin
_conv2d_orig = OpMixin.conv2d
def _conv2d_mgl(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, dtype=None):
  if Device.DEFAULT.startswith("MGL"):
    with Context(WINO=0): return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)
  return _conv2d_orig(self, weight, bias, groups, stride, dilation, padding, dtype)
setattr(OpMixin, "conv2d", _conv2d_mgl)