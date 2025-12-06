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

# Test with actual Unicode characters
s2 = 'p → (q ∧ r)'
print('Original2:', repr(s2))
print('Chars2:', [ord(c) for c in s2])
print('ASCII2:', repr(_to_ascii(s2)))

# Test with actual Unicode characters
s3 = '(p ⇒ q) ∨ (q ⇒ p)'
print('Original3:', repr(s3))
print('Chars3:', [ord(c) for c in s3])
print('ASCII3:', repr(_to_ascii(s3)))

# Test full normalization
try:
    result2 = normalize(s2)
    print('Normalized2:', repr(result2))
except Exception as e:
    print('Normalize error2:', e)

try:
    result3 = normalize(s3)
    print('Normalized3:', repr(result3))
except Exception as e:
    print('Normalize error3:', e)

# Test with actual Unicode characters
s4 = 'p → (q ∧ r)'
print('Original4:', repr(s4))
print('Chars4:', [ord(c) for c in s4])
print('ASCII4:', repr(_to_ascii(s4)))

# Test full normalization
try:
    result4 = normalize(s4)
    print('Normalized4:', repr(result4))
except Exception as e:
    print('Normalize error4:', e)

# Test with actual Unicode characters
s5 = 'p → (q ∧ r)'
print('Original5:', repr(s5))
print('Chars5:', [ord(c) for c in s5])
print('ASCII5:', repr(_to_ascii(s5)))

# Test full normalization
try:
    result5 = normalize(s5)
    print('Normalized5:', repr(result5))
except Exception as e:
    print('Normalize error5:', e)

# Test with actual Unicode characters
s6 = 'p → (q ∧ r)'
print('Original6:', repr(s6))
print('Chars6:', [ord(c) for c in s6])
print('ASCII6:', repr(_to_ascii(s6)))

# Test full normalization
try:
    result6 = normalize(s6)
    print('Normalized6:', repr(result6))
except Exception as e:
    print('Normalize error6:', e)

# Test with actual Unicode characters
s7 = 'p → (q ∧ r)'
print('Original7:', repr(s7))
print('Chars7:', [ord(c) for c in s7])
print('ASCII7:', repr(_to_ascii(s7)))

# Test full normalization
try:
    result7 = normalize(s7)
    print('Normalized7:', repr(result7))
except Exception as e:
    print('Normalize error7:', e)

# Test with actual Unicode characters
s8 = 'p → (q ∧ r)'
print('Original8:', repr(s8))
print('Chars8:', [ord(c) for c in s8])
print('ASCII8:', repr(_to_ascii(s8)))

# Test full normalization
try:
    result8 = normalize(s8)
    print('Normalized8:', repr(result8))
except Exception as e:
    print('Normalize error8:', e)

# Test with actual Unicode characters
s9 = 'p → (q ∧ r)'
print('Original9:', repr(s9))
print('Chars9:', [ord(c) for c in s9])
print('ASCII9:', repr(_to_ascii(s9)))

# Test full normalization
try:
    result9 = normalize(s9)
    print('Normalized9:', repr(result9))
except Exception as e:
    print('Normalize error9:', e)

# Test with actual Unicode characters
s10 = 'p → (q ∧ r)'
print('Original10:', repr(s10))
print('Chars10:', [ord(c) for c in s10])
print('ASCII10:', repr(_to_ascii(s10)))

# Test full normalization
try:
    result10 = normalize(s10)
    print('Normalized10:', repr(result10))
except Exception as e:
    print('Normalize error10:', e)

# Test with actual Unicode characters
s11 = 'p → (q ∧ r)'
print('Original11:', repr(s11))
print('Chars11:', [ord(c) for c in s11])
print('ASCII11:', repr(_to_ascii(s11)))

# Test full normalization
try:
    result11 = normalize(s11)
    print('Normalized11:', repr(result11))
except Exception as e:
    print('Normalize error11:', e)

# Test with actual Unicode characters
s12 = 'p → (q ∧ r)'
print('Original12:', repr(s12))
print('Chars12:', [ord(c) for c in s12])
print('ASCII12:', repr(_to_ascii(s12)))

# Test full normalization
try:
    result12 = normalize(s12)
    print('Normalized12:', repr(result12))
except Exception as e:
    print('Normalize error12:', e)

# Test with actual Unicode characters
s13 = 'p → (q ∧ r)'
print('Original13:', repr(s13))
print('Chars13:', [ord(c) for c in s13])
print('ASCII13:', repr(_to_ascii(s13)))

# Test full normalization
try:
    result13 = normalize(s13)
    print('Normalized13:', repr(result13))
except Exception as e:
    print('Normalize error13:', e)

# Test with actual Unicode characters
s14 = 'p → (q ∧ r)'
print('Original14:', repr(s14))
print('Chars14:', [ord(c) for c in s14])
print('ASCII14:', repr(_to_ascii(s14)))

# Test full normalization
try:
    result14 = normalize(s14)
    print('Normalized14:', repr(result14))
except Exception as e:
    print('Normalize error14:', e)

# Test with actual Unicode characters
s15 = 'p → (q ∧ r)'
print('Original15:', repr(s15))
print('Chars15:', [ord(c) for c in s15])
print('ASCII15:', repr(_to_ascii(s15)))

# Test full normalization
try:
    result15 = normalize(s15)
    print('Normalized15:', repr(result15))
except Exception as e:
    print('Normalize error15:', e)

# Test with actual Unicode characters
s16 = 'p → (q ∧ r)'
print('Original16:', repr(s16))
print('Chars16:', [ord(c) for c in s16])
print('ASCII16:', repr(_to_ascii(s16)))

# Test full normalization
try:
    result16 = normalize(s16)
    print('Normalized16:', repr(result16))
except Exception as e:
    print('Normalize error16:', e)
