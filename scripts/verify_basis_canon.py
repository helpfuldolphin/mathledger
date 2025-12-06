import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from backend.basis.canon import canonical_hash, canonical_json_dump

def test_determinism():
    obj1 = {"a": 1, "b": 2}
    obj2 = {"b": 2, "a": 1}  # Different order

    # Should be identical bytes
    dump1 = canonical_json_dump(obj1)
    dump2 = canonical_json_dump(obj2)
    
    assert dump1 == dump2, f"Ordering failed: {dump1} != {dump2}"
    print(f"Ordering check: PASSED ({dump1})")

    # Hash check
    h1 = canonical_hash(obj1)
    h2 = canonical_hash(obj2)
    assert h1 == h2
    print(f"Hash check: PASSED ({h1})")

    # Unicode check
    # RFC 8785 says strings should not be escaped unless necessary
    obj_u = {"key": "café"}
    dump_u = canonical_json_dump(obj_u)
    # 'café' utf-8 bytes are c3 a9. In JSON it usually stays as is in JCS unless it's control char.
    # Wait, standard json.dumps escapes unicode by default (ensure_ascii=True). 
    # JCS does NOT escape unicode (ensure_ascii=False).
    print(f"Unicode check payload: {dump_u}")
    assert b"caf\xc3\xa9" in dump_u
    print("Unicode check: PASSED")

if __name__ == "__main__":
    try:
        test_determinism()
        print("Basis Canon Verification: SUCCESS")
    except Exception as e:
        print(f"Basis Canon Verification: FAILED - {e}")
        sys.exit(1)

