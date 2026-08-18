#!/usr/bin/env python3
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tinygrad import Tensor, Context, Variable
from examples.gpt2 import GPT2

def run_gpt2_gl21(prompt: str = "The capital of France is", count: int = 10):
  with Context(DEV="GL21"):
    gpt2 = GPT2.build("gpt2")
    gpt2.model.allpos = Tensor.arange(0, 1024).to("GL21").reshape(1, -1).realize()

    tokens = gpt2.tokenizer.encode(prompt)
    print(f"Prompt: {prompt!r}")

    for step in range(count):
      cur_toks = tokens if step == 0 else [tokens[-1]]
      t = Tensor([cur_toks], device="GL21")
      pos = Variable("start_pos", 0, 128).bind(0 if step == 0 else len(tokens) - 1)
      tok = int(gpt2.model(t, pos, 0.0).numpy().flatten()[0])
      tokens.append(tok)
      print(f"Token {step + 1}: {gpt2.tokenizer.decode([tok])!r}")

    output_text = gpt2.tokenizer.decode(tokens)
    print(f"\nGenerated text:\n{output_text}")
    return output_text

if __name__ == "__main__":
  user_prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
  run_gpt2_gl21(user_prompt)
