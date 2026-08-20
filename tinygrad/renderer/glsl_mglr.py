import struct
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, Ops
from tinygrad.device import Compiler
from tinygrad.renderer.glsl import MGLRenderer, hoist_complex_float

MGLRASTER_MAGIC = b"MGLR2"

class MGLRasterCompiler(Compiler):
  """Compiler for the MGLR raster backend. Packs vertex+fragment GLSL into a single lib blob."""
  def __init__(self):
    # NOTE: no disk cache, the lib is a custom packed format keyed on nothing else
    super().__init__(cachekey=None)
  def compile(self, src:str) -> bytes:
    assert "//MGLRASTER" in src and "//MGLFRAGMENT" in src, "MGLR raster kernels need a //MGLRASTER (vertex) and a //MGLFRAGMENT (fragment) section"
    # optional //MGLRVIEWPORT <W> <H> header overrides the viewport derived from the output shape
    w, h = 0, 0
    if src.startswith("//MGLRVIEWPORT"):
      line, _, src = src.partition("\n")
      w, h = map(int, line.split()[1:3])
    sections = src.split("//MGLRASTER")[1].split("//MGLFRAGMENT")
    assert len(sections) == 2, "MGLR raster kernel needs exactly one //MGLRASTER (vertex) and one //MGLFRAGMENT (fragment) section"
    vs, fs = sections[0].strip(), sections[1].strip()
    return MGLRASTER_MAGIC + struct.pack("<III", len(vs), w, h) + vs.encode() + fs.encode()

DEFAULT_VERTEX_SHADER = """#version 430 core
const vec2 verts[6] = vec2[6](
  vec2(-1, -1), vec2(1, -1), vec2(-1, 1),
  vec2(-1, 1), vec2(1, -1), vec2(1, 1)
);
void main() { gl_Position = vec4(verts[gl_VertexID], 0, 1); }
"""

def _render_dim(u: UOp, param_map: dict[UOp, str]) -> tuple[int | None, str]:
  try:
    simp = u.ssimplify()
    if isinstance(simp, int): return simp, str(simp)
    if hasattr(simp, 'val') and isinstance(simp.val, int): return simp.val, str(simp.val)
    if hasattr(simp, 'arg') and isinstance(simp.arg, int): return simp.arg, str(simp.arg)
  except Exception: pass
  if u in param_map: return None, param_map[u]
  if u.op is Ops.CONST: return int(u.arg), str(u.arg)
  if u.op is Ops.PARAM:
    name = param_map.get(u)
    if name is not None: return None, name
    if hasattr(u.arg, 'name'): return None, u.arg.name
    return None, f"data{u.arg.slot}_"
  if u.op is Ops.ADD:
    v0, s0 = _render_dim(u.src[0], param_map)
    v1, s1 = _render_dim(u.src[1], param_map)
    return (v0 + v1 if v0 is not None and v1 is not None else None), f"({s0}+{s1})"
  if u.op is Ops.MUL:
    v0, s0 = _render_dim(u.src[0], param_map)
    v1, s1 = _render_dim(u.src[1], param_map)
    return (v0 * v1 if v0 is not None and v1 is not None else None), f"({s0}*{s1})"
  return None, param_map.get(u, str(u))

class MGLRasterRenderer(MGLRenderer):
  compiler = MGLRasterCompiler()
  has_local = True
  has_shared = False
  shared_max = 0
  barrier = ""
  global_max = (16384, 16384, 16384)
  local_max = (1024, 1024, 64)
  code_for_workitem = {
    "g": lambda x: f"_gidx{x}",
    "l": lambda x: f"_lidx{x}",
    "i": lambda x: f"_idx{x}",
  }

  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[UOp,bool]]], uops:list[UOp], prefix=None) -> str:
    param_map = {p: name for name, (p, _) in bufs}
    dims: dict[tuple[str, int], tuple[int | None, str]] = {}
    for u in uops:
      if u.op is Ops.SPECIAL:
        kind = u.arg[0]
        idx = int(u.arg[4:] if u.arg.startswith(('gidx', 'lidx')) else u.arg[3:] if u.arg.startswith('idx') else u.arg[-1])
        dims[(kind, idx)] = _render_dim(u.src[0], param_map)

    l0_v, l0_s = dims.get(('l', 0), (1, "1"))
    l1_v, l1_s = dims.get(('l', 1), (1, "1"))
    l2_v, l2_s = dims.get(('l', 2), (1, "1"))
    g0_v, g0_s = dims.get(('g', 0), (1, "1"))
    g1_v, g1_s = dims.get(('g', 1), (1, "1"))
    g2_v, g2_s = dims.get(('g', 2), (1, "1"))
    i0_v, i0_s = dims.get(('i', 0), (1, "1"))
    i1_v, i1_s = dims.get(('i', 1), (1, "1"))
    i2_v, i2_s = dims.get(('i', 2), (1, "1"))

    has_gl = any(k[0] in ('g', 'l') for k in dims)
    has_i = any(k[0] == 'i' for k in dims)

    if has_gl:
      relevant_dims = [(l0_v, l0_s), (l1_v, l1_s), (l2_v, l2_s), (g0_v, g0_s), (g1_v, g1_s), (g2_v, g2_s)]
    elif has_i:
      relevant_dims = [(i0_v, i0_s), (i1_v, i1_s), (i2_v, i2_s)]
    else:
      relevant_dims = [(1, "1")]

    is_static = all(v is not None for v, _ in relevant_dims)

    if is_static:
      total_n = 1
      for v, _ in relevant_dims: total_n *= v  # type: ignore[operator]
      w_grid = max(1, min(total_n, 4096))
      h_grid = max(1, (total_n + w_grid - 1) // w_grid)
      header = f"//MGLRVIEWPORT {w_grid} {h_grid}\n"
      flat_id_code = f"int(gl_FragCoord.y) * {w_grid} + int(gl_FragCoord.x)"
      bounds_check = [f"  if (_flat_id >= {total_n}) return;"] if total_n > 1 else []
    else:
      total_n_factors = [s for _, s in relevant_dims if s != "1"]
      total_n_str = "*".join(total_n_factors) if len(total_n_factors) else "1"
      header = ""
      flat_id_code = "int(gl_FragCoord.y) * u_size.x + int(gl_FragCoord.x)"
      bounds_check = [f"  if (_flat_id >= ({total_n_str})) return;"]

    ssbos: list[tuple[str, UOp]] = []
    alus: list[tuple[str, UOp]] = []
    for name,(u,_) in bufs: (ssbos if u.addrspace != AddrSpace.ALU else alus).append((name, u))
    frag = "#version 430 core\n"
    frag += "uniform ivec2 u_size;\n"
    frag += "#define INFINITY uintBitsToFloat(0x7f800000u)\n"
    frag += "#define NAN uintBitsToFloat(0x7fc00000u)\n"
    for name, u in ssbos:
      tname = 'uint' if u.dtype.itemsize < 4 else self.type_map.get(u.dtype, u.dtype.name)
      frag += f"layout(std430, binding={u.arg.slot}) coherent buffer SSBO{u.arg.slot} {{ {tname} {name}[]; }};\n"
    if len(alus):
      frag += "layout(std140, binding=0) uniform UBO { "
      frag += "; ".join(f"{self.type_map.get(u.dtype, u.dtype.name)} {name}" for name,u in alus)
      frag += "; };\n"
    frag += "layout(location = 0) out vec4 _out_color;\n"

    coord_decls = [f"  int _flat_id = {flat_id_code};"] + bounds_check + ["  int _rem = _flat_id;"]
    if has_i:
      if i0_v == 1: coord_decls.append("  int _idx0 = 0;")
      else: coord_decls.append(f"  int _idx0 = _rem % ({i0_s}); _rem /= ({i0_s});")
      if i1_v == 1: coord_decls.append("  int _idx1 = 0;")
      else: coord_decls.append(f"  int _idx1 = _rem % ({i1_s}); _rem /= ({i1_s});")
      coord_decls.append("  int _idx2 = _rem;")
      coord_decls.extend(["  int _gidx0 = _idx0;", "  int _gidx1 = _idx1;", "  int _gidx2 = _idx2;",
                          "  int _lidx0 = 0;", "  int _lidx1 = 0;", "  int _lidx2 = 0;"])
    else:
      if l0_v == 1: coord_decls.append("  int _lidx0 = 0;")
      else: coord_decls.append(f"  int _lidx0 = _rem % ({l0_s}); _rem /= ({l0_s});")
      if l1_v == 1: coord_decls.append("  int _lidx1 = 0;")
      else: coord_decls.append(f"  int _lidx1 = _rem % ({l1_s}); _rem /= ({l1_s});")
      if l2_v == 1: coord_decls.append("  int _lidx2 = 0;")
      else: coord_decls.append(f"  int _lidx2 = _rem % ({l2_s}); _rem /= ({l2_s});")
      if g0_v == 1: coord_decls.append("  int _gidx0 = 0;")
      else: coord_decls.append(f"  int _gidx0 = _rem % ({g0_s}); _rem /= ({g0_s});")
      if g1_v == 1: coord_decls.append("  int _gidx1 = 0;")
      else: coord_decls.append(f"  int _gidx1 = _rem % ({g1_s}); _rem /= ({g1_s});")
      coord_decls.append("  int _gidx2 = _rem;")
      coord_decls.append(f"  int _idx0 = _gidx0 * ({l0_s}) + _lidx0;")
      coord_decls.append(f"  int _idx1 = _gidx1 * ({l1_s}) + _lidx1;")
      coord_decls.append(f"  int _idx2 = _gidx2 * ({l2_s}) + _lidx2;")

    body = ["  _out_color = vec4(0.0);"] + coord_decls + hoist_complex_float(kernel)
    frag += "\nvoid main() {\n" + "\n".join(body) + "\n}\n"
    return f"{header}//MGLRASTER\n{DEFAULT_VERTEX_SHADER}//MGLFRAGMENT\n{frag}"