import struct
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, AxisType
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite
from tinygrad.helpers import strip_parens, prod
from tinygrad.device import Compiler

def _extract_hi_lo_str(ctx, u: UOp) -> tuple[str, str]:
  if u.op is Ops.CONST:
    val = int(u.arg)
    hi_val = (val >> 32) & 0xFFFFFFFF
    lo_val = val & 0xFFFFFFFF
    return f"vec2({hi_val & 0xFFFF}.0, {(hi_val >> 16) & 0xFFFF}.0)", f"vec2({lo_val & 0xFFFF}.0, {(lo_val >> 16) & 0xFFFF}.0)"
  if u.op is Ops.OR:
    if u.src[0].op is Ops.SHL: hi, lo = u.src[0].src[0], u.src[1]
    elif u.src[1].op is Ops.SHL: hi, lo = u.src[1].src[0], u.src[0]
    else: hi, lo = u.src[0], u.src[1]
  else:
    hi, lo = u.alu(Ops.SHR, u.const_like(32)), u.cast(dtypes.uint32)
  if hi.op is Ops.CAST: hi = hi.src[0]
  if lo.op is Ops.CAST: lo = lo.src[0]
  def to_vec2_str(x):
    if x.op is Ops.CONST:
      v = int(x.arg)
      return f"vec2({v & 0xFFFF}.0, {(v >> 16) & 0xFFFF}.0)"
    if x.dtype in (dtypes.uint32, dtypes.int32, dtypes.uint64, dtypes.int64):
      return str(ctx[x])
    return f"_to_u32({ctx[x]})"
  return to_vec2_str(hi), to_vec2_str(lo)

def _as_float(ctx, u: UOp) -> str:
  if u.dtype in (dtypes.uint32, dtypes.int32, dtypes.uint64, dtypes.int64):
    if u.op is Ops.CONST: return f"{int(u.arg)}.0"
    return f"({ctx[u]}.x + {ctx[u]}.y * 65536.0)"
  return str(ctx[u])

def _match_paren(s:str, i:int) -> tuple[int, str]:
  depth = 0
  for j in range(i, len(s)):
    if s[j] == '(': depth += 1
    elif s[j] == ')':
      depth -= 1
      if depth == 0: return j, s[i+1:j]
  return len(s)-1, s[i+1:]

def _strip_casts(expr:str) -> str:
  out, i = [], 0
  prefixes = ('int(', 'uint(', 'bool(', 'floatBitsToInt(', 'floatBitsToUint(')
  while i < len(expr):
    matched = False
    for p in prefixes:
      if expr.startswith(p, i):
        paren_start = i + len(p) - 1
        j, _ = _match_paren(expr, paren_start)
        out.append('0')
        i = j + 1
        matched = True
        break
    if not matched:
      out.append(expr[i])
      i += 1
  return ''.join(out)

def _strip_outer_parens(s:str) -> str:
  s = s.strip()
  while s.startswith('(') and s.endswith(')'):
    j, _ = _match_paren(s, 0)
    if j == len(s) - 1: s = s[1:-1].strip()
    else: break
  return s

def _has_float_lit(s:str) -> bool:
  for i, c in enumerate(s):
    if c in '.fF':
      if (i > 0 and s[i-1].isdigit()) or (i + 1 < len(s) and s[i+1].isdigit()): return True
  return False

def _is_complex_float(inner:str) -> bool:
  stripped = _strip_outer_parens(_strip_casts(inner))
  if not _has_float_lit(stripped): return False
  depth = bdepth = 0
  for c in stripped:
    if c == '(': depth += 1
    elif c == ')': depth -= 1
    elif c == '[': bdepth += 1
    elif c == ']': bdepth -= 1
    elif c == ',' and depth == 0 and bdepth == 0: return False
    elif c in '+-*/' and depth == 0 and bdepth == 0: return True
  return False

def _has_top_mul(inner:str) -> bool:
  depth = bdepth = 0
  for c in inner:
    if c == '(': depth += 1
    elif c == ')': depth -= 1
    elif c == '[': bdepth += 1
    elif c == ']': bdepth -= 1
    elif c == ',' and depth == 0 and bdepth == 0: return False
    elif c == '*' and depth == 0 and bdepth == 0: return True
  return False

def _hoist_expr(expr:str, tmp:list[str], counter:list[int]) -> str:
  out, i = [], 0
  while i < len(expr):
    if expr[i] == '(':
      j, inner = _match_paren(expr, i)
      is_func_call = i > 0 and (expr[i-1].isalnum() or expr[i-1] == '_')
      mul_operand = (i > 0 and expr[i-1] == '*') or (j+1 < len(expr) and expr[j+1] == '*')
      if not is_func_call and _is_complex_float(inner) and (mul_operand or _has_top_mul(inner)):
        name = f"_ht{counter[0]}"
        counter[0] += 1
        hoisted_inner = _hoist_expr(inner, tmp, counter)
        if _is_complex_float(hoisted_inner):
          tmp.append(f"float {name} = ({hoisted_inner.replace('+-', '-')});")
          out.append('(' + name + ')')
        else:
          out.append('(' + hoisted_inner + ')')
      else:
        out.append('(' + _hoist_expr(inner, tmp, counter) + ')')
      i = j + 1
      continue
    out.append(expr[i])
    i += 1
  return ''.join(out)

def _fix_acc_chain(stmt:str) -> str:
  if '=' not in stmt: return stmt
  lhs, rhs = stmt.split('=', 1)
  lhs, rhs = lhs.strip(), rhs.strip()
  if not (lhs.startswith('buf') and '[' in lhs and lhs.endswith(']')): return stmt
  if not (rhs.startswith('(') and rhs.endswith(')')): return stmt
  inner = rhs[1:-1]
  terms: list[str] = []
  cur, depth, bdepth = '', 0, 0
  for c in inner:
    if c == '(': depth += 1
    elif c == ')': depth -= 1
    elif c == '[': bdepth += 1
    elif c == ']': bdepth -= 1
    if c == '+' and depth == 0 and bdepth == 0 and cur:
      terms.append(cur)
      cur = c
    else: cur += c
  terms.append(cur)
  acc_idx = None
  for i, t in enumerate(terms):
    if i == 0: continue
    if t.lstrip('+').strip() == lhs: acc_idx = i
  if acc_idx is None or acc_idx == len(terms) - 1: return stmt
  acc = terms.pop(acc_idx)
  terms.append(acc)
  return f"{lhs}= ({''.join(terms)})"

# radeonsi (Mesa 26.1) miscompiles nested float arithmetic in loop bodies; hoisting complex
# MUL operands to named locals and normalizing "+-0.0f" to "-0.0f" keeps each generated kernel correct.
def hoist_complex_float(kernel:list[str]) -> list[str]:
  out, counter = [], [0]
  for line in kernel:
    stripped = line.strip()
    if stripped.endswith(';') and not stripped.startswith(('for ', 'if ', 'while ')):
      depth = bdepth = 0
      cur = ''
      for c in stripped:
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == '[': bdepth += 1
        elif c == ']': bdepth -= 1
        if c == ';' and depth == 0 and bdepth == 0:
          stmt = cur.strip()
          if stmt:
            tmp:list[str] = []
            new = _fix_acc_chain(_hoist_expr(stmt, tmp, counter).replace('+-', '-'))
            out.extend(tmp)
            out.append(new + ';')
          cur = ''
          continue
        cur += c
      stmt = cur.strip()
      if stmt:
        tmp = []
        new = _fix_acc_chain(_hoist_expr(stmt, tmp, counter).replace('+-', '-'))
        out.extend(tmp)
        out.append(new + ';')
    else: out.append(stripped.replace('+-', '-'))
  return out

def _bitcast(ctx, x:UOp) -> str:
  fxn = {(dtypes.uint32, dtypes.float): "floatBitsToUint", (dtypes.float, dtypes.uint32): "uintBitsToFloat",
         (dtypes.int32, dtypes.float): "floatBitsToInt", (dtypes.float, dtypes.int32): "intBitsToFloat"}.get((x.dtype, x.src[0].dtype))
  if fxn: return f"{fxn}({ctx[x.src[0]]})"
  val = ctx[x.src[0]]
  if x.src[0].dtype.itemsize < 4:
    val = f"({val}&{0xFF if x.src[0].dtype.itemsize == 1 else 0xFFFF}u)"
  if x.dtype == dtypes.int8: return f"((int({val})<<24)>>24)"
  if x.dtype == dtypes.int16: return f"((int({val})<<16)>>16)"
  if x.dtype in (dtypes.uint8, dtypes.uint16): return f"uint({val})"
  return f"{ctx.type_map.get(x.dtype, x.dtype.name)}({val})"

def is_packed(x:UOp) -> bool:
  if x.op is Ops.LOAD: dt, addrspace = x.dtype, x.src[0].addrspace
  elif x.op is Ops.STORE: dt, addrspace = x.src[1].dtype, x.src[0].addrspace
  elif x.op is Ops.INDEX: dt, addrspace = x.dtype, x.src[0].addrspace
  else: dt, addrspace = x.dtype, getattr(x, 'addrspace', AddrSpace.GLOBAL)
  return dt.itemsize < 4 and dt != dtypes.half and addrspace != AddrSpace.REG

def sign_extend(val:UOp, sext_am:int):
  return (UOp.where((val >> (sext_am - 1)) > 0, UOp.const(0xffffffff << sext_am, dtypes.uint32), UOp.const(0, dtypes.uint32)) \
        | val.bitcast(dtypes.uint32)).bitcast(dtypes.int)

# store for a sub-4-byte var: atomicAnd(loc, wmask) + atomicAdd(loc, var<<shift)
def packed_store(bidx:UOp, var:UOp, gate:UOp|None=None):
  elems, mask = 4//var.dtype.itemsize, 0xFF if var.dtype.itemsize == 1 else 0xFFFF
  uidx = bidx.src[1].cast(dtypes.uint32)
  shift_am, div_idx = (uidx % elems) * (8*var.dtype.itemsize), (uidx // elems).cast(dtypes.int)
  if var.dtype == dtypes.bool: var = var.cast(dtypes.int32)
  new_v, wmask = (var.cast(dtypes.uint32) & mask) << shift_am, ((mask << shift_am) ^ 0xFFFFFFFF).cast(dtypes.uint32)
  idx = UOp(Ops.INDEX, src=(bidx.src[0], div_idx))
  buf = UOp.load(idx, *((UOp.const(0, dtypes.uint32), gate) if gate is not None else ()), dtype=dtypes.uint32)
  return UOp.store(idx, (buf & wmask) | new_v, *((gate,) if gate is not None else ()))

# load for a sub-4-byte var
def packed_load(root:UOp, bidx:UOp, dtype, var:UOp|None=None, gate:UOp|None=None):
  elems, mask = 4//dtype.itemsize, 0xFF if dtype.itemsize == 1 else 0xFFFF
  uidx = bidx.src[1].cast(dtypes.uint32)
  shift_am, div_idx = (uidx % elems) * (8*dtype.itemsize), (uidx // elems).cast(dtypes.int)
  idx = UOp(Ops.INDEX, src=(bidx.src[0], div_idx))
  load = UOp.load(idx, *((var, gate) if var is not None and gate is not None else root.src[1:]), dtype=dtypes.uint32, arg=root.arg)
  val = (load.cast(dtypes.uint32) >> shift_am) & mask
  return sign_extend(val, 8*dtype.itemsize).cast(dtype) if dtype in (dtypes.int8, dtypes.int16) else val.cast(dtype)

glsl_matcher = PatternMatcher([
  (UPat.load(UPat.var("b"), UPat.var("c"), UPat.var("gate"), name="l"),
   lambda l,b,c,gate: packed_load(l,b,l.dtype,c.cast(dtypes.uint32),gate) if is_packed(l) else None),
  (UPat.load(UPat.var("b"), name='l'), lambda l,b: packed_load(l,b,l.dtype) if is_packed(l) else None),
  (UPat.store(UPat.var("b"), UPat.var("var"), UPat.var("gate"), name="s"),
   lambda b,var,gate,s: packed_store(b,var,gate) if is_packed(s) else None),
  (UPat.store(UPat.var("b"), UPat.var("var"), name="s"), lambda b,var,s: packed_store(b,var) if is_packed(s) else None),
])

def _render_store(ctx, b, v) -> str:
  if is_packed(b):
    if v.op is Ops.OR and len(v.src[0].src) > 1:
      return f"atomicAnd({ctx[b]},{ctx[v.src[0].src[1]]});\n  atomicAdd({ctx[b]},{ctx[v.src[1]]});"
    elif v.op is Ops.AND and len(v.src) > 1:
      return f"atomicAnd({ctx[b]},{ctx[v.src[1]]});"
  return f"{ctx[b]} = {ctx[v]};"

class MGLRenderer(CStyleLanguage):
  supports_float4 = False
  global_max = (65535, 65535, 65535)
  local_max = (1024, 1024, 64)
  code_for_workitem = {"g": lambda x: f"int(gl_WorkGroupID.{'xyz'[int(x)]})", "l": lambda x: f"int(gl_LocalInvocationID.{'xyz'[int(x)]})",
                       "i": lambda x: f"int(gl_GlobalInvocationID.{'xyz'[int(x)]})"}
  type_map = { dtypes.float: "float", dtypes.int32: "int", dtypes.uint32: "uint", dtypes.bool: "bool",
                 dtypes.int8: "int", dtypes.uint8: "uint", dtypes.int16: "int", dtypes.uint16: "uint" }
  infinity = "INFINITY"
  nan = "NAN"
  barrier = "barrier();\n  memoryBarrierShared();"
  extra_matcher: PatternMatcher | None = glsl_matcher
  code_for_op = {**CStyleLanguage.code_for_op, Ops.CMOD: lambda a,b,dtype: f"(({a})-({a})/({b})*({b}))"}

  string_rewrite = PatternMatcher([
    (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "true" if x.val else "false"),
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: _bitcast(ctx, x)),
    (UPat((Ops.CMPNE, Ops.CMPEQ), src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"({ctx[a]}{'!=' if x.op is Ops.CMPNE else '=='}bool({ctx[b]}))"),
    (UPat((Ops.CMPNE, Ops.CMPEQ), src=(UPat.var("a"), UPat.var("b", dtypes.bool)), name="x"),
     lambda ctx,a,b,x: f"(bool({ctx[a]}){'!=' if x.op is Ops.CMPNE else '=='}{ctx[b]})"),
    (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"(int({ctx[a]})<int({ctx[b]}))"),
    (UPat(Ops.CMPLT, src=(UPat.var("a"), UPat.var("b", dtypes.bool)), name="x"),
     lambda ctx,a,b,x: f"(int({ctx[a]})<int({ctx[b]}))"),
    (UPat(Ops.AND, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"({ctx[a]}&&{ctx[b]})"),
    (UPat(Ops.OR, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"({ctx[a]}||{ctx[b]})"),
    (UPat(Ops.XOR, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"(({ctx[a]})!=({ctx[b]}))"),
    (UPat.load(UPat.var("b"), UPat.var("v"), UPat.var("gate")),
     lambda ctx,b,v,gate: f"({ctx[gate]}?{ctx[b]}:{ctx[v]})"),
    (UPat.load(UPat.var("b")), lambda ctx,b: ctx[b]),
    (UPat.store(UPat.var("b"), UPat.var("v")), lambda ctx,b,v: _render_store(ctx,b,v)),
    (UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx"))),
     lambda ctx,b,idx: f"{ctx[b]}[{strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}]"),
  ]) + base_rewrite

  def render_type(self, u:UOp) -> str:
    return "bool" if u.dtype == dtypes.bool else self.type_map.get(u.dtype, u.dtype.name)

  def render_cast(self, u:UOp, val:str) -> str:
    return f"bool({val})" if u.dtype == dtypes.bool else f"{self.type_map.get(u.dtype, u.dtype.name)}({val})"

  def render_buffer(self, x:UOp):
    if x.addrspace == AddrSpace.LOCAL:
      return f"shared {'uint' if x.dtype.itemsize < 4 else self.type_map.get(x.dtype, x.dtype.name)} {self[x]}[{x.max_numel()}];"
    return super().render_buffer(x)

  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[UOp,bool]]], uops:list[UOp], prefix=None) -> str:
    local_dims = [u.src[0].ssimplify() for u in sorted([u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == 'l'], key=lambda u: u.arg)]
    while len(local_dims) < 3: local_dims.append(1)
    ssbos: list[tuple[str, UOp]] = []
    alus: list[tuple[str, UOp]] = []
    for name,(u,_) in bufs: (ssbos if u.addrspace != AddrSpace.ALU else alus).append((name, u))
    prg = "#version 430 core\n"
    prg += f"layout(local_size_x={local_dims[0]}, local_size_y={local_dims[1]}, local_size_z={local_dims[2]}) in;\n"
    prg += "#define INFINITY uintBitsToFloat(0x7f800000u)\n"
    prg += "#define NAN uintBitsToFloat(0x7fc00000u)\n"
    for name, u in ssbos:
      tname = 'uint' if u.dtype.itemsize < 4 else self.type_map.get(u.dtype, u.dtype.name)
      prg += f"layout(std430, binding={u.arg.slot}) coherent buffer SSBO{u.arg.slot} {{ {tname} {name}[]; }};\n"
    if len(alus):
      prg += "layout(std140, binding=0) uniform UBO { "
      prg += "; ".join(f"{self.type_map.get(u.dtype, u.dtype.name)} {name}" for name,u in alus)
      prg += "; };\n"
    return prg + "\nvoid main() {\n" + "\n".join(hoist_complex_float(kernel)) + "\n}\n"

  def supported_dtypes(self):
    return {dtypes.float, dtypes.int32, dtypes.uint32, dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16}

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


# =============================================================================
# OpenGL 2.1 / GLSL 1.20 Renderer (no compute shaders, SSBOs, UBOs)
# =============================================================================

GL21_VERTEX_SHADER = """#version 120
attribute vec2 in_pos;
varying vec2 v_texcoord;
void main() {
    v_texcoord = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

class GL21Compiler(Compiler):
  """Compiler for GL21 backend. Outputs separate vertex/fragment GLSL 1.20 sources."""
  def __init__(self):
    super().__init__(cachekey="gl21_v2")
  def compile(self, src: str) -> bytes:
    import struct
    if src.startswith("//GL21_SCATTER\n"):
      _, rest = src.split("//GL21_SCATTER\n", 1)
      total_n_str, rest = rest.split("\n", 1)
      total_n = int(total_n_str)
      vs, fs = rest.split("//GL21_FRAGMENT\n", 1)
      render_mode = 1  # GL_POINTS
    else:
      vs = GL21_VERTEX_SHADER
      fs = src
      render_mode = 0  # GL_TRIANGLES
      total_n = 0
    return struct.pack("<IIII", len(vs), len(fs), render_mode, total_n) + vs.encode() + fs.encode()


class GL21Renderer(MGLRenderer):
  """GLSL 1.20 renderer for OpenGL 2.1 - uses textures instead of SSBOs."""
  compiler = GL21Compiler()
  has_local = False
  has_shared = False
  can_upcast = False  # one fragment = one element; UPCAST would need multiple outputs per fragment
  shared_max = 0
  barrier = ""
  global_max = (8192, 8192, 1)  # Limited by max texture size
  local_max = (1, 1, 1)
  supports_float4 = False
  extra_matcher: PatternMatcher | None = None

  # GLSL 1.20 type mapping - no uint/int, use float for storage
  type_map = {
    dtypes.float: "float",
    dtypes.int32: "float",   # Emulated via float
    dtypes.uint32: "float",  # Emulated via float
    dtypes.int64: "float",   # Emulated via float
    dtypes.uint64: "float",  # Emulated via float
    dtypes.bool: "float",    # 0.0 or 1.0
    dtypes.int8: "float",
    dtypes.uint8: "float",
    dtypes.int16: "float",
    dtypes.uint16: "float",
  }

  # Work item mapping - compute from gl_FragCoord in fragment shader
  code_for_workitem = {
    "g": lambda x: f"_gidx{x}",
    "l": lambda x: f"_lidx{x}",
    "i": lambda x: f"_idx{x}",
  }

  # Integer ops emulated via float (GLSL 1.20 has no integer support)
  code_for_op = {
    **MGLRenderer.code_for_op,
    Ops.TRUNC: lambda a, dtype: f"(floor(abs({a})) * sign({a}))",
    Ops.CMOD: lambda a, b, dtype: f"mod({a}, {b})",
    Ops.FDIV: lambda a, b, dtype: f"({a}/{b})",
    Ops.CDIV: lambda a, b, dtype: f"floor({a}/{b})",
    Ops.SHR: lambda a, b, dtype: f"floor({a} * exp2(-{b}))",
    Ops.SHL: lambda a, b, dtype: f"floor({a} * exp2({b}))",
    Ops.AND: lambda a, b, dtype: f"_bit_and({a}, {b})",
    Ops.OR: lambda a, b, dtype: f"_bit_or({a}, {b})",
    Ops.XOR: lambda a, b, dtype: f"_bit_xor({a}, {b})",
    Ops.THREEFRY: lambda a, b, dtype: f"_threefry_u32_0({a}, {b})",
  }

  # GLSL 1.20 string rewrites
  string_rewrite = PatternMatcher([
    # Threefry rewrite
    # Threefry rewrite
    (UPat(Ops.CAST, dtype=(dtypes.uint32, dtypes.uint64),
          src=(UPat(Ops.SHR, src=(UPat(Ops.THREEFRY, src=(UPat.var('x'), UPat.var('k'))), UPat())),)),
     lambda ctx, x, k: (c:=_extract_hi_lo_str(ctx, x), key:=_extract_hi_lo_str(ctx, k),
                        f"_threefry_u32_1({c[1]}, {c[0]}, {key[1]}, {key[0]})")[-1]),
    (UPat(Ops.CAST, dtype=(dtypes.uint32, dtypes.uint64),
          src=(UPat(Ops.THREEFRY, src=(UPat.var('x'), UPat.var('k'))),)),
     lambda ctx, x, k: (c:=_extract_hi_lo_str(ctx, x), key:=_extract_hi_lo_str(ctx, k),
                        f"_threefry_u32_0({c[1]}, {c[0]}, {key[1]}, {key[0]})")[-1]),
    (UPat(Ops.SHR, src=(UPat(Ops.THREEFRY, src=(UPat.var('x'), UPat.var('k'))), UPat()), name='s'),
     lambda ctx, x, k, s: (c:=_extract_hi_lo_str(ctx, x), key:=_extract_hi_lo_str(ctx, k),
                           f"_threefry_u32_1({c[1]}, {c[0]}, {key[1]}, {key[0]})")[-1]),
    (UPat(Ops.THREEFRY, src=(UPat.var('x'), UPat.var('k')), name='tf'),
     lambda ctx, x, k, tf: (c:=_extract_hi_lo_str(ctx, x), key:=_extract_hi_lo_str(ctx, k),
                            f"_threefry_u32_0({c[1]}, {c[0]}, {key[1]}, {key[0]})")[-1]),

    # 32-bit integer arithmetic (vec2) for uint32/uint64
    (UPat(Ops.CONST, dtype=(dtypes.uint32, dtypes.uint64), name="x"),
     lambda ctx,x: f"vec2({int(x.val) & 0xFFFF}.0, {(int(x.val) >> 16) & 0xFFFF}.0)"),
    (UPat(Ops.ADD, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("a"), UPat.var("b"))),
     lambda ctx,a,b: f"_u32_add({ctx[a]}, {ctx[b]})"),
    (UPat(Ops.SUB, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("a"), UPat.var("b"))),
     lambda ctx,a,b: f"_u32_sub({ctx[a]}, {ctx[b]})"),
    (UPat(Ops.SHR, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("a"), UPat.var("b"))),
     lambda ctx,a,b: f"_u32_shr({ctx[a]}, {ctx[b]}.x)"),
    (UPat(Ops.SHL, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("a"), UPat.var("b"))),
     lambda ctx,a,b: f"_u32_shl({ctx[a]}, {ctx[b]}.x)"),
    (UPat(Ops.AND, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("a"), UPat.var("b"))),
     lambda ctx,a,b: f"_u32_and({ctx[a]}, {ctx[b]})"),
    (UPat(Ops.OR, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("a"), UPat.var("b"))),
     lambda ctx,a,b: f"_u32_or({ctx[a]}, {ctx[b]})"),
    (UPat(Ops.XOR, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("a"), UPat.var("b"))),
     lambda ctx,a,b: f"_u32_xor({ctx[a]}, {ctx[b]})"),
    (UPat(Ops.CMPLT, dtype=dtypes.bool, src=(UPat.var("a", (dtypes.uint32, dtypes.uint64)), UPat.var("b", (dtypes.uint32, dtypes.uint64)))),
     lambda ctx,a,b: f"float(_u32_cmplt({ctx[a]}, {ctx[b]}))"),
    (UPat(Ops.CMPEQ, dtype=dtypes.bool, src=(UPat.var("a", (dtypes.uint32, dtypes.uint64)), UPat.var("b", (dtypes.uint32, dtypes.uint64)))),
     lambda ctx,a,b: f"float(({ctx[a]}.x == {ctx[b]}.x) && ({ctx[a]}.y == {ctx[b]}.y))"),
    (UPat(Ops.CMPNE, dtype=dtypes.bool, src=(UPat.var("a", (dtypes.uint32, dtypes.uint64)), UPat.var("b", (dtypes.uint32, dtypes.uint64)))),
     lambda ctx,a,b: f"float(({ctx[a]}.x != {ctx[b]}.x) || ({ctx[a]}.y != {ctx[b]}.y))"),
    (UPat(Ops.WHERE, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("c"), UPat.var("t"), UPat.var("f"))),
     lambda ctx,c,t,f: f"(bool({ctx[c]}) ? {ctx[t]} : {ctx[f]})"),

    # Constants
    (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "1.0" if x.val else "0.0"),
    (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: "INFINITY" if x.val == float('inf') else (
      "-INFINITY" if x.val == float('-inf') else ("NAN" if x.val != x.val else f"{x.val}"))),
    (UPat(Ops.CONST, dtype=(dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16), name="x"),
     lambda ctx,x: f"{x.val}.0"),

    # Bitcast - float bitcast from uint mantissa emulation and identity for int<->uint
    (UPat(Ops.BITCAST, dtype=dtypes.float, src=(UPat.var("a", (dtypes.uint32, dtypes.uint64)),)),
     lambda ctx,a: f"((({ctx[a]}.x + mod({ctx[a]}.y, 128.0) * 65536.0) / 8388608.0) + 1.0)"),
    (UPat(Ops.BITCAST, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("a", (dtypes.uint32, dtypes.uint64)),)),
     lambda ctx,a: ctx[a]),
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: ctx[x.src[0]]),

    # Cast to integer dtypes (float truncation) and bool
    (UPat(Ops.CAST, dtype=(dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16), name="x"),
     lambda ctx,x: f"(floor(abs({ctx[x.src[0]]})) * sign({ctx[x.src[0]]}))"),
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"),
     lambda ctx,x: f"float(({ctx[x.src[0]]}) != 0.0)"),

    # Boolean ops via float
    (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"float(({ctx[a]})!=({ctx[b]}))"),
    (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"float(({ctx[a]})==({ctx[b]}))"),
    (UPat(Ops.AND, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"({ctx[a]}*{ctx[b]})"),
    (UPat(Ops.OR, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"min(1.0, {ctx[a]}+{ctx[b]})"),
    (UPat(Ops.XOR, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"abs({ctx[a]}-{ctx[b]})"),
    (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.bool), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"float(({ctx[a]})<({ctx[b]}))"),
    (UPat((Ops.CMPNE, Ops.CMPEQ, Ops.CMPLT),
          src=(UPat.var("a"), UPat.var("b")), name="x"),
     lambda ctx,a,b,x: f"float(({ctx[a]}){ {Ops.CMPNE:'!=',Ops.CMPEQ:'==',Ops.CMPLT:'<'}[x.op] }({ctx[b]}))"),

    # WHERE (ternary) - use ?: operator (supported in GLSL 1.20)
    (UPat(Ops.WHERE, src=(UPat.var("c"), UPat.var("t"), UPat.var("f")), name="x"),
     lambda ctx,c,t,f,x: f"(bool({ctx[c]}) ? {ctx[t]} : {ctx[f]})"),

    # Load/Store via texture2D with index-derived coordinates (GLOBAL buffers = textures)
    (UPat(Ops.LOAD, dtype=(dtypes.uint32, dtypes.uint64),
          src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("c")))), name="bidx"),), name="x"),
     lambda ctx,b,a,c,bidx,x: (f"texture2D({ctx[b]}, _coord_add(float({ctx[a]}), float({ctx[c]}), {ctx[b]}_tex_size)).rg"
                               if b.addrspace == AddrSpace.GLOBAL else f"({ctx[b]}[int(({ctx[a]})+({ctx[c]}))])")),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("c")))), name="bidx"),), name="x"),
     lambda ctx,b,a,c,bidx,x: (f"_tex_read(texture2D({ctx[b]}, _coord_add(float({ctx[a]}), float({ctx[c]}), {ctx[b]}_tex_size)))"
                               if b.addrspace == AddrSpace.GLOBAL else f"({ctx[b]}[int(({ctx[a]})+({ctx[c]}))])")),
    (UPat(Ops.LOAD, dtype=(dtypes.uint32, dtypes.uint64),
          src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), name="bidx"),), name="x"),
     lambda ctx,b,idx,bidx,x: (f"texture2D({ctx[b]}, _coord(float({ctx[idx]}), {ctx[b]}_tex_size)).rg"
                               if b.addrspace == AddrSpace.GLOBAL else f"({ctx[b]}[int({ctx[idx]})])")),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), name="bidx"),), name="x"),
     lambda ctx,b,idx,bidx,x: (f"_tex_read(texture2D({ctx[b]}, _coord(float({ctx[idx]}), {ctx[b]}_tex_size)))"
                               if b.addrspace == AddrSpace.GLOBAL else f"({ctx[b]}[int({ctx[idx]})])")),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), name="bidx"), UPat.var("v"), UPat.var("gate")), name="x"),
     lambda ctx,b,idx,bidx,v,gate,x: f"(bool({ctx[gate]}) ? " + (
       f"_tex_read(texture2D({ctx[b]}, _coord(float({ctx[idx]}), {ctx[b]}_tex_size)))" if b.addrspace == AddrSpace.GLOBAL else (
         f"({ctx[b]}[int({ctx[idx]})])")) + f" : {ctx[v]})"),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), name="bidx"),
                          UPat.var("v", (dtypes.uint32, dtypes.uint64))), name="x"),
     lambda ctx,b,idx,bidx,v,x: f"gl_FragColor.rg = {ctx[v]};" if b.addrspace == AddrSpace.GLOBAL else f"{ctx[b]}[int({ctx[idx]})] = {ctx[v]};"),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), name="bidx"), UPat.var("v")), name="x"),
     lambda ctx,b,idx,bidx,v,x: f"gl_FragColor.r = {ctx[v]};" if b.addrspace == AddrSpace.GLOBAL else f"{ctx[b]}[int({ctx[idx]})] = {ctx[v]};"),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), name="bidx"),
                          UPat.var("v", (dtypes.uint32, dtypes.uint64)), UPat.var("gate")), name="x"),
     lambda ctx,b,idx,bidx,v,gate,x: f"if (bool({ctx[gate]})) " + (
       f"gl_FragColor.rg = {ctx[v]};" if b.addrspace == AddrSpace.GLOBAL else f"{ctx[b]}[int({ctx[idx]})] = {ctx[v]};")),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), name="bidx"), UPat.var("v"), UPat.var("gate")), name="x"),
     lambda ctx,b,idx,bidx,v,gate,x: f"if (bool({ctx[gate]})) " + (
       f"gl_FragColor.r = {ctx[v]};" if b.addrspace == AddrSpace.GLOBAL else f"{ctx[b]}[int({ctx[idx]})] = {ctx[v]};")),

    # IF condition - must be scalar bool in GLSL 1.20
    (UPat(Ops.IF, name="x"), lambda ctx,x: f"if (bool({ctx[x.src[0]]})) {{"),

    # Fallback LOAD/STORE
    (UPat(Ops.LOAD, dtype=(dtypes.uint32, dtypes.uint64), src=(UPat.var("bidx"),), name="x"),
     lambda ctx,bidx,x: f"texture2D({ctx[bidx]}, tex_coord).rg"),
    (UPat(Ops.LOAD, src=(UPat.var("bidx"),), name="x"),
     lambda ctx,bidx,x: f"_tex_read(texture2D({ctx[bidx]}, tex_coord))"),
    (UPat(Ops.STORE, src=(UPat.var("bidx"), UPat.var("v", (dtypes.uint32, dtypes.uint64))), name="x"),
     lambda ctx,bidx,v,x: f"gl_FragColor.rg = {ctx[v]};"),
    (UPat(Ops.STORE, src=(UPat.var("bidx"), UPat.var("v")), name="x"),
     lambda ctx,bidx,v,x: f"gl_FragColor.r = {ctx[v]};"),

    # Index arithmetic - return buffer name
    (UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx")), name="x"),
     lambda ctx,b,idx,x: ctx[b]),

    # RANGE/END handling
    # Output ranges (WEAK/LOOP/GLOBAL) -> bind to fragment position
    # Reduce ranges (REDUCE/GROUP_REDUCE/UNROLL) -> emit real for-loop
    (UPat(Ops.RANGE, dtypes.void, name="x"),
     lambda ctx,x: f"float {ctx[x]} = float(_idx{ctx.output_range_map.get(x.arg[0], 0)});" if (
       x.arg[-1] in (AxisType.WEAK, AxisType.LOOP, AxisType.GLOBAL) and hasattr(ctx, 'output_range_map')) else ""),
    (UPat(Ops.RANGE, name="x"),
     lambda ctx,x: f"float {ctx[x]} = float(_idx{ctx.output_range_map.get(x.arg[0], 0)});" if (
       x.arg[-1] in (AxisType.WEAK, AxisType.LOOP, AxisType.GLOBAL) and hasattr(ctx, 'output_range_map')) else (
       f"for (float {ctx[x]} = 0.0; {ctx[x]} < {ctx[x.src[0]]}; {ctx[x]} += 1.0) {{" if (
         x.arg[-1] in (AxisType.REDUCE, AxisType.GROUP_REDUCE, AxisType.UNROLL)) else "")),
    (UPat(Ops.END, src=(UPat(), UPat(Ops.RANGE)), name="x"),
     lambda ctx,x: "" if x.src[1].arg[-1] in (AxisType.WEAK, AxisType.LOOP, AxisType.GLOBAL) else "}"),
    (UPat(Ops.END, src=(UPat(), UPat(Ops.RANGE), UPat()), name="x"),
     lambda ctx,x: "" if x.src[1].arg[-1] in (AxisType.WEAK, AxisType.LOOP, AxisType.GLOBAL) else "}"),
  ]) + base_rewrite

  def render_type(self, u: UOp) -> str:
    if u.dtype in (dtypes.uint32, dtypes.uint64):
      return "vec2"
    return "float"

  def render_cast(self, u: UOp, val: str) -> str:
    if u.dtype in (dtypes.uint32, dtypes.uint64):
      if u.src[0].dtype in (dtypes.uint32, dtypes.uint64):
        return val
      return f"_to_u32({val})"
    elif u.dtype == dtypes.float:
      if u.src[0].dtype in (dtypes.uint32, dtypes.uint64):
        return f"({val}.x + {val}.y * 65536.0)"
    return val

  def render(self, uops: list[UOp]) -> str:
    # Compute output_range_map BEFORE _render (which does pattern matching)
    output_ranges: list[UOp] = []
    for u in uops:
      if u.op is Ops.STORE and u.src[0].op is Ops.INDEX:
        idx_expr = u.src[0].src[1]
        def check_ranges(expr):
          if expr.op is Ops.RANGE and expr.arg[-1] in (AxisType.WEAK, AxisType.LOOP, AxisType.GLOBAL):
            output_ranges.append(expr)
          for r in expr.backward_slice:
            if r.op is Ops.RANGE and r.arg[-1] in (AxisType.WEAK, AxisType.LOOP, AxisType.GLOBAL):
              output_ranges.append(r)
        check_ranges(idx_expr)
    output_ranges = sorted(set(output_ranges), key=lambda r: r.arg[0])
    self.output_range_map = {r.arg[0]: i for i, r in enumerate(output_ranges)}
    self.output_range_sizes = [int(r.src[0].val) if isinstance(r.src[0], UOp) and r.src[0].op is Ops.CONST else r.vmax+1 for r in output_ranges]

    return super().render(uops)

  def render_kernel(self, function_name: str, kernel: list[str],
                    bufs: list[tuple[str, tuple[UOp, bool]]], uops: list[UOp], prefix=None) -> str:
    """Generate GLSL 1.20 fragment shader with texture-based buffers."""

    # Separate buffers into textures (SSBOs) and uniforms (ALU)
    textures: list[tuple[str, UOp, bool]] = []
    uniforms: list[tuple[str, UOp, bool]] = []
    for name, (u, is_global) in bufs:
      if u.addrspace == AddrSpace.ALU:
        uniforms.append((name, u, is_global))
      else:
        textures.append((name, u, is_global))

    # Compute per-buffer texture sizes
    # Buffer layout: width = min(numel, 8192), height = ceil(numel/width)
    buffer_tex_sizes: dict[str, tuple[int, int]] = {}
    for name, u, _ in textures:
      numel = u.max_numel()
      w = min(numel, 8192)
      h = (numel + w - 1) // w
      buffer_tex_sizes[name] = (w, h)
    self.buffer_tex_sizes = buffer_tex_sizes

    # Total number of elements
    if hasattr(self, 'output_range_sizes') and self.output_range_sizes:
      total_n = prod(self.output_range_sizes)
    elif textures:
      total_n = textures[0][1].max_numel()
    else:
      total_n = 1

    flat_id_code = "int(gl_FragCoord.y) * int(u_tex_size.x) + int(gl_FragCoord.x)"
    bounds_check = [f"  if (_flat_id >= {total_n}) discard;"] if total_n > 1 else []

    # Build fragment shader
    frag = "#version 120\n"
    frag += "varying vec2 v_texcoord;\n"
    frag += "uniform vec2 u_tex_size;\n"
    frag += "#define INFINITY 1e30\n"
    frag += "#define NAN (0.0/0.0)\n"
    frag += "#define trunc(x) (floor(abs(x)) * sign(x))\n"

    # Bitwise emulation functions for GLSL 1.20
    frag += "float _bit_and(float a, float b) {\n"
    frag += "  if (a < 0.0) a = floor(a + 4294967296.0);\n"
    frag += "  if (b < 0.0) b = floor(b + 4294967296.0);\n"
    frag += "  float r = 0.0; float p = 1.0;\n"
    frag += "  for (int i = 0; i < 24; i++) {\n"
    frag += "    if (mod(a, 2.0) >= 1.0 && mod(b, 2.0) >= 1.0) r += p;\n"
    frag += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
    frag += "    if (a == 0.0 || b == 0.0) break;\n"
    frag += "  }\n"
    frag += "  return r;\n"
    frag += "}\n"
    frag += "float _bit_or(float a, float b) {\n"
    frag += "  if (a < 0.0) a = floor(a + 4294967296.0);\n"
    frag += "  if (b < 0.0) b = floor(b + 4294967296.0);\n"
    frag += "  float r = 0.0; float p = 1.0;\n"
    frag += "  for (int i = 0; i < 24; i++) {\n"
    frag += "    if (mod(a, 2.0) >= 1.0 || mod(b, 2.0) >= 1.0) r += p;\n"
    frag += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
    frag += "    if (a == 0.0 && b == 0.0) break;\n"
    frag += "  }\n"
    frag += "  return r;\n"
    frag += "}\n"
    frag += "float _bit_xor(float a, float b) {\n"
    frag += "  if (a < 0.0) a = floor(a + 4294967296.0);\n"
    frag += "  if (b < 0.0) b = floor(b + 4294967296.0);\n"
    frag += "  float r = 0.0; float p = 1.0;\n"
    frag += "  for (int i = 0; i < 24; i++) {\n"
    frag += "    float ma = mod(a, 2.0); float mb = mod(b, 2.0);\n"
    frag += "    if ((ma >= 1.0) != (mb >= 1.0)) r += p;\n"
    frag += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
    frag += "    if (a == 0.0 && b == 0.0) break;\n"
    frag += "  }\n"
    frag += "  return r;\n"
    frag += "}\n"
    frag += "float _tex_read(vec4 t) {\n"
    frag += "  return t.r;\n"
    frag += "}\n"
    frag += "vec2 _to_u32(float v) {\n"
    frag += "  return (v >= 65536.0) ? vec2(mod(v, 65536.0), floor(v / 65536.0)) : vec2(v, 0.0);\n"
    frag += "}\n"
    frag += "vec2 _u32_add(vec2 a, vec2 b) {\n"
    frag += "  float s_lo = a.x + b.x;\n"
    frag += "  float c = floor(s_lo / 65536.0);\n"
    frag += "  float r_lo = s_lo - c * 65536.0;\n"
    frag += "  float s_hi = a.y + b.y + c;\n"
    frag += "  float c_hi = floor(s_hi / 65536.0);\n"
    frag += "  float r_hi = s_hi - c_hi * 65536.0;\n"
    frag += "  return vec2(r_lo, r_hi);\n"
    frag += "}\n"
    frag += "vec2 _u32_sub(vec2 a, vec2 b) {\n"
    frag += "  float b_lo = (a.x < b.x) ? 1.0 : 0.0;\n"
    frag += "  float r_lo = a.x - b.x + b_lo * 65536.0;\n"
    frag += "  float r_hi = mod(a.y - b.y - b_lo + 65536.0, 65536.0);\n"
    frag += "  return vec2(r_lo, r_hi);\n"
    frag += "}\n"
    frag += "bool _u32_cmplt(vec2 a, vec2 b) {\n"
    frag += "  return (a.y < b.y) || (a.y == b.y && a.x < b.x);\n"
    frag += "}\n"
    frag += "float _and16(float a, float b) {\n"
    frag += "  float r = 0.0; float p = 1.0;\n"
    frag += "  for (int i = 0; i < 16; i++) {\n"
    frag += "    float ma = mod(a, 2.0); float mb = mod(b, 2.0);\n"
    frag += "    if (ma >= 1.0 && mb >= 1.0) r += p;\n"
    frag += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
    frag += "    if (a == 0.0 || b == 0.0) break;\n"
    frag += "  }\n"
    frag += "  return r;\n"
    frag += "}\n"
    frag += "float _or16(float a, float b) {\n"
    frag += "  float r = 0.0; float p = 1.0;\n"
    frag += "  for (int i = 0; i < 16; i++) {\n"
    frag += "    float ma = mod(a, 2.0); float mb = mod(b, 2.0);\n"
    frag += "    if (ma >= 1.0 || mb >= 1.0) r += p;\n"
    frag += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
    frag += "    if (a == 0.0 && b == 0.0) break;\n"
    frag += "  }\n"
    frag += "  return r;\n"
    frag += "}\n"
    frag += "vec2 _u32_and(vec2 a, vec2 b) {\n"
    frag += "  return vec2(_and16(a.x, b.x), _and16(a.y, b.y));\n"
    frag += "}\n"
    frag += "vec2 _u32_or(vec2 a, vec2 b) {\n"
    frag += "  return vec2(_or16(a.x, b.x), _or16(a.y, b.y));\n"
    frag += "}\n"
    frag += "vec2 _u32_shl(vec2 a, float s) {\n"
    frag += "  if (s == 0.0) return a;\n"
    frag += "  if (s < 16.0) {\n"
    frag += "    float p2 = exp2(s); float p2_c = exp2(16.0 - s);\n"
    frag += "    float r_lo = mod(a.x * p2, 65536.0);\n"
    frag += "    float r_hi = mod(a.y * p2 + floor(a.x / p2_c), 65536.0);\n"
    frag += "    return vec2(r_lo, r_hi);\n"
    frag += "  } else {\n"
    frag += "    float s_prime = s - 16.0; float p2 = exp2(s_prime);\n"
    frag += "    return vec2(0.0, mod(a.x * p2, 65536.0));\n"
    frag += "  }\n"
    frag += "}\n"
    frag += "vec2 _u32_shr(vec2 a, float s) {\n"
    frag += "  if (s == 0.0) return a;\n"
    frag += "  if (s < 16.0) {\n"
    frag += "    float p2 = exp2(s); float p2_c = exp2(16.0 - s);\n"
    frag += "    float r_lo = floor(a.x / p2) + mod(a.y, p2) * p2_c;\n"
    frag += "    float r_hi = floor(a.y / p2);\n"
    frag += "    return vec2(r_lo, r_hi);\n"
    frag += "  } else {\n"
    frag += "    float s_prime = s - 16.0; float p2 = exp2(s_prime);\n"
    frag += "    return vec2(floor(a.y / p2), 0.0);\n"
    frag += "  }\n"
    frag += "}\n"
    frag += "vec2 _u32_xor(vec2 a, vec2 b) {\n"
    frag += "  vec2 r = vec2(0.0, 0.0);\n"
    frag += "  float p = 1.0;\n"
    frag += "  for (int i = 0; i < 16; i++) {\n"
    frag += "    vec2 ma = mod(a, 2.0); vec2 mb = mod(b, 2.0);\n"
    frag += "    r += vec2((ma.x >= 1.0) != (mb.x >= 1.0) ? p : 0.0, (ma.y >= 1.0) != (mb.y >= 1.0) ? p : 0.0);\n"
    frag += "    a = floor(a * 0.5); b = floor(b * 0.5); p *= 2.0;\n"
    frag += "    if (a.x == 0.0 && a.y == 0.0 && b.x == 0.0 && b.y == 0.0) break;\n"
    frag += "  }\n"
    frag += "  return r;\n"
    frag += "}\n"
    frag += "vec2 _u32_rotl(vec2 a, float r) {\n"
    frag += "  if (r == 16.0) return vec2(a.y, a.x);\n"
    frag += "  if (r < 16.0) {\n"
    frag += "    float p2 = exp2(r); float p2_c = exp2(16.0 - r);\n"
    frag += "    float hi_x = floor(a.x / p2_c); float lo_x = a.x - hi_x * p2_c;\n"
    frag += "    float hi_y = floor(a.y / p2_c); float lo_y = a.y - hi_y * p2_c;\n"
    frag += "    return vec2(lo_x * p2 + hi_y, lo_y * p2 + hi_x);\n"
    frag += "  } else {\n"
    frag += "    float r_prime = r - 16.0;\n"
    frag += "    float p2 = exp2(r_prime); float p2_c = exp2(16.0 - r_prime);\n"
    frag += "    float hi_x = floor(a.x / p2_c); float lo_x = a.x - hi_x * p2_c;\n"
    frag += "    float hi_y = floor(a.y / p2_c); float lo_y = a.y - hi_y * p2_c;\n"
    frag += "    return vec2(lo_y * p2 + hi_x, lo_x * p2 + hi_y);\n"
    frag += "  }\n"
    frag += "}\n"
    frag += "vec4 _threefry2x32(vec2 x0, vec2 x1, vec2 k0, vec2 k1) {\n"
    frag += "  vec2 c_const = vec2(7130.0, 7121.0);\n"
    frag += "  vec2 ks0 = k1;\n"
    frag += "  vec2 ks1 = _u32_xor(_u32_xor(k0, k1), c_const);\n"
    frag += "  vec2 ks2 = k0;\n"
    frag += "  vec2 xr0 = _u32_add(x0, ks2);\n"
    frag += "  vec2 xr1 = _u32_add(x1, ks0);\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 13.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 15.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 26.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 6.0));\n"
    frag += "  xr0 = _u32_add(xr0, ks0); xr1 = _u32_add(xr1, _u32_add(ks1, vec2(1.0, 0.0)));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 17.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 29.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 16.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 24.0));\n"
    frag += "  xr0 = _u32_add(xr0, ks1); xr1 = _u32_add(xr1, _u32_add(ks2, vec2(2.0, 0.0)));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 13.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 15.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 26.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 6.0));\n"
    frag += "  xr0 = _u32_add(xr0, ks2); xr1 = _u32_add(xr1, _u32_add(ks0, vec2(3.0, 0.0)));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 17.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 29.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 16.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 24.0));\n"
    frag += "  xr0 = _u32_add(xr0, ks0); xr1 = _u32_add(xr1, _u32_add(ks1, vec2(4.0, 0.0)));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 13.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 15.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 26.0));\n"
    frag += "  xr0 = _u32_add(xr0, xr1); xr1 = _u32_xor(xr0, _u32_rotl(xr1, 6.0));\n"
    frag += "  xr0 = _u32_add(xr0, ks1); xr1 = _u32_add(xr1, _u32_add(ks2, vec2(5.0, 0.0)));\n"
    frag += "  return vec4(xr0.x, xr0.y, xr1.x, xr1.y);\n"
    frag += "}\n"
    frag += "vec2 _threefry_u32_0(vec2 c0, vec2 c1, vec2 k0, vec2 k1) {\n"
    frag += "  vec4 tf = _threefry2x32(c0, c1, k0, k1);\n"
    frag += "  return tf.xy;\n"
    frag += "}\n"
    frag += "vec2 _threefry_u32_1(vec2 c0, vec2 c1, vec2 k0, vec2 k1) {\n"
    frag += "  vec4 tf = _threefry2x32(c0, c1, k0, k1);\n"
    frag += "  return tf.zw;\n"
    frag += "}\n"

    # Texture uniforms (sampler2D)
    for i, (name, u, _) in enumerate(textures):
      frag += f"uniform sampler2D {name};\n"
      # Per-buffer texture size constants
      w, h = buffer_tex_sizes[name]
      frag += f"const vec2 {name}_tex_size = vec2({float(w)}, {float(h)});\n"

    # ALU uniforms (individual float uniforms for each param)
    if uniforms:
      for name, u, _ in uniforms:
        frag += f"uniform float {name};\n"

    # Helper function to compute texture coordinate from flat index
    frag += "vec2 _coord(float i, vec2 sz) {\n"
    frag += "  float y = floor((i + 0.1) / sz.x);\n"
    frag += "  float x = floor(i - y * sz.x + 0.1);\n"
    frag += "  return vec2((x + 0.5) / sz.x, (y + 0.5) / sz.y);\n"
    frag += "}\n"
    frag += "vec2 _coord_add(float a, float b, vec2 sz) {\n"
    frag += "  float ya = floor((a + 0.1) / sz.x);\n"
    frag += "  float rema = floor(a - ya * sz.x + 0.1);\n"
    frag += "  float yb = floor((b + 0.1) / sz.x);\n"
    frag += "  float remb = floor(b - yb * sz.x + 0.1);\n"
    frag += "  float rem = rema + remb;\n"
    frag += "  float yextra = floor((rem + 0.1) / sz.x);\n"
    frag += "  float y = ya + yb + yextra;\n"
    frag += "  float x = floor(rem - yextra * sz.x + 0.1);\n"
    frag += "  return vec2((x + 0.5) / sz.x, (y + 0.5) / sz.y);\n"
    frag += "}\n"

    # Coordinate decoding - compute logical output indices from flat_id
    coord_decls = [
      f"  int _flat_id = {flat_id_code};",
    ] + bounds_check

    # Decompose flat_id by logical output range sizes in reverse (innermost to outermost)
    if hasattr(self, 'output_range_sizes') and self.output_range_sizes:
      rem = "_flat_id"
      n_ranges = len(self.output_range_sizes)
      idx_stmts: dict[int, str] = {}
      for i in reversed(range(n_ranges)):
        size = float(self.output_range_sizes[i])
        idx_stmts[i] = f"  int _idx{i} = int(mod(float({rem}), {size}));"
        rem = f"int(float({rem}) / {size})"
      for i in range(n_ranges):
        coord_decls.append(idx_stmts[i])
      for i in range(n_ranges, 3):
        coord_decls.append(f"  int _idx{i} = 0;")
    else:
      # Fallback: use texture size
      coord_decls.append("  int _idx0 = int(mod(float(_flat_id), u_tex_size.x));")
      coord_decls.append("  int _idx1 = int(float(_flat_id) / u_tex_size.x);")
      coord_decls.append("  int _idx2 = 0;")

    coord_decls.extend(["  int _gidx0 = _idx0;", "  int _gidx1 = _idx1;", "  int _gidx2 = _idx2;",
                        "  int _lidx0 = 0;", "  int _lidx1 = 0;", "  int _lidx2 = 0;"])

    # Texture coordinates from gl_FragCoord (for output write)
    coord_setup = [
      "  vec2 tex_coord = vec2(",
      "    gl_FragCoord.x / u_tex_size.x,",
      "    gl_FragCoord.y / u_tex_size.y",
      "  );",
    ]

    # Check if there is a store to an indexed position that requires point rasterization (scatter write)
    is_scatter = False
    dst_idx_glsl = None
    if textures and textures[0][1].max_numel() > total_n:
      dst_buf_uop = textures[0][1]
      for u in uops:
        if u.op is Ops.STORE and u.src[0].op is Ops.INDEX and u.src[0].src[0] is dst_buf_uop:
          def _uop_to_glsl(x: UOp) -> str:
            if x.op is Ops.CONST: return f"{float(x.arg)}"
            elif x.op is Ops.RANGE: return f"float(_idx{self.output_range_map.get(x.arg[0], 0)})"
            elif x.op is Ops.PARAM: return f"data{x.arg.slot}_"
            elif x.op is Ops.CAST: return _uop_to_glsl(x.src[0])
            elif x.op is Ops.NEG: return f"(-{_uop_to_glsl(x.src[0])})"
            elif x.op is Ops.ADD: return f"({_uop_to_glsl(x.src[0])} + {_uop_to_glsl(x.src[1])})"
            elif x.op is Ops.SUB: return f"({_uop_to_glsl(x.src[0])} - {_uop_to_glsl(x.src[1])})"
            elif x.op is Ops.MUL: return f"({_uop_to_glsl(x.src[0])} * {_uop_to_glsl(x.src[1])})"
            elif x.op is Ops.AND: return f"_bit_and({_uop_to_glsl(x.src[0])}, {_uop_to_glsl(x.src[1])})"
            elif x.op is Ops.OR: return f"_bit_or({_uop_to_glsl(x.src[0])}, {_uop_to_glsl(x.src[1])})"
            elif x.op is Ops.XOR: return f"_bit_xor({_uop_to_glsl(x.src[0])}, {_uop_to_glsl(x.src[1])})"
            elif x.op is Ops.SHL: return f"floor({_uop_to_glsl(x.src[0])} * exp2({_uop_to_glsl(x.src[1])}))"
            elif x.op is Ops.SHR: return f"floor({_uop_to_glsl(x.src[0])} * exp2(-{_uop_to_glsl(x.src[1])}))"
            elif x.op is Ops.CMOD: return f"mod({_uop_to_glsl(x.src[0])}, {_uop_to_glsl(x.src[1])})"
            elif x.op in (Ops.CDIV, Ops.FDIV): return f"floor({_uop_to_glsl(x.src[0])} / {_uop_to_glsl(x.src[1])})"
            return "0.0"
          is_scatter = True
          dst_idx_glsl = _uop_to_glsl(u.src[0].src[1])
          break

    if is_scatter:
      vs = "#version 120\n"
      vs += "attribute float in_idx;\n"
      vs += "uniform vec2 u_tex_size;\n"
      vs += "varying float v_flat_id;\n"
      vs += "#define trunc(x) (floor(abs(x)) * sign(x))\n"
      vs += "float _bit_and(float a, float b) {\n"
      vs += "  float r = 0.0; float p = 1.0;\n"
      vs += "  for (int i = 0; i < 24; i++) {\n"
      vs += "    if (mod(a, 2.0) >= 1.0 && mod(b, 2.0) >= 1.0) r += p;\n"
      vs += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
      vs += "    if (a == 0.0 || b == 0.0) break;\n"
      vs += "  }\n"
      vs += "  return r;\n"
      vs += "}\n"
      vs += "float _bit_or(float a, float b) {\n"
      vs += "  float r = 0.0; float p = 1.0;\n"
      vs += "  for (int i = 0; i < 24; i++) {\n"
      vs += "    if (mod(a, 2.0) >= 1.0 || mod(b, 2.0) >= 1.0) r += p;\n"
      vs += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
      vs += "    if (a == 0.0 && b == 0.0) break;\n"
      vs += "  }\n"
      vs += "  return r;\n"
      vs += "}\n"
      vs += "float _bit_xor(float a, float b) {\n"
      vs += "  float r = 0.0; float p = 1.0;\n"
      vs += "  for (int i = 0; i < 24; i++) {\n"
      vs += "    float ma = mod(a, 2.0); float mb = mod(b, 2.0);\n"
      vs += "    if ((ma >= 1.0) != (mb >= 1.0)) r += p;\n"
      vs += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
      vs += "    if (a == 0.0 && b == 0.0) break;\n"
      vs += "  }\n"
      vs += "  return r;\n"
      vs += "}\n"
      if uniforms:
        for name, u, _ in uniforms:
          vs += f"uniform float {name};\n"
      vs_coord_decls = ["  int _flat_id = int(in_idx);"] + [c for c in coord_decls[1:] if "discard;" not in c]
      vs += "\nvoid main() {\n"
      vs += "  v_flat_id = in_idx;\n"
      vs += "\n".join(vs_coord_decls) + "\n"
      vs += f"  float dst_flat = {dst_idx_glsl};\n"
      vs += "  float dst_y = floor((dst_flat + 0.1) / u_tex_size.x);\n"
      vs += "  float dst_x = dst_flat - dst_y * u_tex_size.x;\n"
      vs += "  gl_Position = vec4((dst_x + 0.5) / u_tex_size.x * 2.0 - 1.0, (dst_y + 0.5) / u_tex_size.y * 2.0 - 1.0, 0.0, 1.0);\n"
      vs += "  gl_PointSize = 1.0;\n"
      vs += "}\n"

      frag = "#version 120\n"
      frag += "varying float v_flat_id;\n"
      frag += "uniform vec2 u_tex_size;\n"
      frag += "#define INFINITY 1e30\n"
      frag += "#define NAN (0.0/0.0)\n"
      frag += "#define trunc(x) (floor(abs(x)) * sign(x))\n"
      frag += "float _bit_and(float a, float b) {\n"
      frag += "  float r = 0.0; float p = 1.0;\n"
      frag += "  for (int i = 0; i < 24; i++) {\n"
      frag += "    if (mod(a, 2.0) >= 1.0 && mod(b, 2.0) >= 1.0) r += p;\n"
      frag += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
      frag += "    if (a == 0.0 || b == 0.0) break;\n"
      frag += "  }\n"
      frag += "  return r;\n"
      frag += "}\n"
      frag += "float _bit_or(float a, float b) {\n"
      frag += "  float r = 0.0; float p = 1.0;\n"
      frag += "  for (int i = 0; i < 24; i++) {\n"
      frag += "    if (mod(a, 2.0) >= 1.0 || mod(b, 2.0) >= 1.0) r += p;\n"
      frag += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
      frag += "    if (a == 0.0 && b == 0.0) break;\n"
      frag += "  }\n"
      frag += "  return r;\n"
      frag += "}\n"
      frag += "float _bit_xor(float a, float b) {\n"
      frag += "  float r = 0.0; float p = 1.0;\n"
      frag += "  for (int i = 0; i < 24; i++) {\n"
      frag += "    float ma = mod(a, 2.0); float mb = mod(b, 2.0);\n"
      frag += "    if ((ma >= 1.0) != (mb >= 1.0)) r += p;\n"
      frag += "    a = floor(a / 2.0); b = floor(b / 2.0); p *= 2.0;\n"
      frag += "    if (a == 0.0 && b == 0.0) break;\n"
      frag += "  }\n"
      frag += "  return r;\n"
      frag += "}\n"
      frag += "float _tex_read(vec4 c) {\n"
      frag += "  return (abs(c.g) > 0.0) ? (c.r + c.g * 65536.0) : c.r;\n"
      frag += "}\n"
      for i, (name, u, _) in enumerate(textures):
        frag += f"uniform sampler2D {name};\n"
        w, h = buffer_tex_sizes[name]
        frag += f"const vec2 {name}_tex_size = vec2({w}.0, {h}.0);\n"
      if uniforms:
        for name, u, _ in uniforms:
          frag += f"uniform float {name};\n"
      frag += "vec2 _coord(float i, vec2 sz) {\n"
      frag += "  float y = floor((i + 0.1) / sz.x);\n"
      frag += "  float x = floor(i - y * sz.x + 0.1);\n"
      frag += "  return vec2((x + 0.5) / sz.x, (y + 0.5) / sz.y);\n"
      frag += "}\n"
      frag += "vec2 _coord_add(float a, float b, vec2 sz) {\n"
      frag += "  float ya = floor((a + 0.1) / sz.x);\n"
      frag += "  float rema = floor(a - ya * sz.x + 0.1);\n"
      frag += "  float yb = floor((b + 0.1) / sz.x);\n"
      frag += "  float remb = floor(b - yb * sz.x + 0.1);\n"
      frag += "  float rem = rema + remb;\n"
      frag += "  float yextra = floor((rem + 0.1) / sz.x);\n"
      frag += "  float y = ya + yb + yextra;\n"
      frag += "  float x = floor(rem - yextra * sz.x + 0.1);\n"
      frag += "  return vec2((x + 0.5) / sz.x, (y + 0.5) / sz.y);\n"
      frag += "}\n"
      frag_coord_decls = ["  int _flat_id = int(v_flat_id);"] + coord_decls[1:]
      body = frag_coord_decls + coord_setup + hoist_complex_float(kernel)
      frag += "\nvoid main() {\n" + "\n".join(body) + "\n}\n"
      return f"//GL21_SCATTER\n{total_n}\n{vs}//GL21_FRAGMENT\n{frag}"

    # Body: coord_decls first, then coord_setup, then kernel
    body = coord_decls + coord_setup + hoist_complex_float(kernel)
    frag += "\nvoid main() {\n" + "\n".join(body) + "\n}\n"

    return frag

  def supported_dtypes(self):
    return super().supported_dtypes() | {dtypes.int64, dtypes.uint64}