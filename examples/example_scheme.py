"""TOY green-list rule used only to show the interface expected by watermark/green_list.py.

It is NOT the rule of any real watermarking system. A real test needs the exact rule (and the tokenizer) used by the generator.
"""
import hashlib


def is_green(prev_token_id: int, token_id: int) -> bool:
    h = hashlib.sha256(f"toy-key:{prev_token_id}:{token_id}".encode()).digest()
    return h[0] < 64          # ~25% of (prev, token) pairs are "green" (gamma = 0.25)
