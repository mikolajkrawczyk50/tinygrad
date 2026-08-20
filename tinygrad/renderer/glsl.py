from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite
from tinygrad.helpers import strip_parens

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