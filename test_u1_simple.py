#!/usr/bin/env python3
import sys
print("[TEST] Script started", flush=True, file=sys.stderr)
print("[TEST] Script started", flush=True)

if __name__ == "__main__":
    print("[TEST] __main__ block executed", flush=True, file=sys.stderr)
    print("[TEST] __main__ block executed", flush=True)
    print("Arguments:", sys.argv, flush=True)

