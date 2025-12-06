#!/usr/bin/env python3
# Test Unicode conversion
s = 'p → (q ∧ r)'
print('Original:', repr(s))
print('Chars:', [ord(c) for c in s])
print('After replace spaces:', repr(s.replace(' ', '')))

from backend.logic.canon import _to_ascii
print('ASCII:', repr(_to_ascii(s)))

# Test parsing
from backend.logic.canon import _parse
try:
    n = _parse(_to_ascii(s))
    print('Parsed successfully')
except Exception as e:
    print('Parse error:', e)

# Test full normalization
from backend.logic.canon import normalize
try:
    result = normalize(s)
    print('Normalized:', repr(result))
except Exception as e:
    print('Normalize error:', e)
