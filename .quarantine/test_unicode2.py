#!/usr/bin/env python3
# Test Unicode conversion
s = 'p → (q ∧ r)'
print('Original:', repr(s))
print('Chars:', [ord(c) for c in s])
print('After replace spaces:', repr(s.replace(' ', '')))

from backend.logic.canon import _to_ascii
print('ASCII:', repr(_to_ascii(s)))
