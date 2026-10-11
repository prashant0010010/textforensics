"""Write a ~30,000-word synthetic document (seeded) to examples/long_synthetic_30000_words.txt."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.synthetic import make_text  # noqa: E402

out = os.path.join(ROOT, "examples", "long_synthetic_30000_words.txt")
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(make_text(2024, 30000, para_words=(80, 220)) + "\n")
print("wrote", out)
