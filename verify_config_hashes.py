
import yaml
import sys
import os
import hashlib

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from normalization.canon import normalize
    from substrate.crypto.hashing import sha256_hex, DOMAIN_STMT
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please run this script from the root of the mathledger project.")
    # Fallback for normalize
    print("Using a dummy normalize function. Hashes will likely be incorrect.")
    def normalize(s: str) -> str:
        return s.replace(" ", "").replace("(", "").replace(")", "")
    DOMAIN_STMT = b'\x02'
    def sha256_hex(data, domain):
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(domain + data).hexdigest()


def canonical_bytes(s: str) -> bytes:
    """Encode the normalized representation of ``s`` as canonical ASCII bytes."""
    if s is None:
        s = ""
    normalized = normalize(s)
    try:
        return normalized.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"normalized statement is not ASCII-clean: {normalized!r}") from exc

def sha256_statement(s: str) -> str:
    """Compute SHA-256 hash of statement with domain separation."""
    return sha256_hex(canonical_bytes(s), domain=DOMAIN_STMT)


def main():
    file_path = 'config/curriculum_uplift_phase2.yaml'
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    formulas_in_pool = {}
    for entry in data.get('systems', [])[0].get('slices', [])[0].get('formula_pool_entries', []):
        formulas_in_pool[entry['formula']] = entry['hash']

    all_ok = True
    
    print("Verifying hashes in formula_pool_entries...")
    for slice_data in data.get('systems', [])[0].get('slices', []):
        print(f"--- Slice: {slice_data['name']} ---")
        if 'formula_pool_entries' in slice_data:
            for entry in slice_data['formula_pool_entries']:
                formula = entry['formula']
                expected_hash = entry['hash']
                actual_hash = sha256_statement(formula)
                if expected_hash != actual_hash:
                    all_ok = False
                    print(f"  [MISMATCH] Formula: {formula}")
                    print(f"    Expected: {expected_hash}")
                    print(f"    Actual:   {actual_hash}")
                else:
                    print(f"  [OK] Formula: {formula}")
        
        if 'success_metric' in slice_data:
            metric = slice_data['success_metric']
            if 'target_hashes' in metric:
                print("\n  Verifying target_hashes...")
                for h in metric['target_hashes']:
                    found = False
                    for entry in slice_data['formula_pool_entries']:
                        if entry['hash'] == h:
                            found = True
                            print(f"    [OK] Hash {h[:10]}... found in formula pool.")
                            break
                    if not found:
                        all_ok = False
                        print(f"    [MISSING] Hash {h[:10]}... not found in formula pool.")
            
            if 'chain_target_hash' in metric:
                print("\n  Verifying chain_target_hash...")
                h = metric['chain_target_hash']
                found = False
                for entry in slice_data['formula_pool_entries']:
                    if entry['hash'] == h:
                        found = True
                        print(f"    [OK] Hash {h[:10]}... found in formula pool.")
                        break
                if not found:
                    all_ok = False
                    print(f"    [MISSING] Hash {h[:10]}... not found in formula pool.")

            if 'required_goal_hashes' in metric:
                print("\n  Verifying required_goal_hashes...")
                for h in metric['required_goal_hashes']:
                    found = False
                    for entry in slice_data['formula_pool_entries']:
                        if entry['hash'] == h:
                            found = True
                            print(f"    [OK] Hash {h[:10]}... found in formula pool.")
                            break
                    if not found:
                        all_ok = False
                        print(f"    [MISSING] Hash {h[:10]}... not found in formula pool.")


    if all_ok:
        print("\nVerification complete. All hashes are correct.")
    else:
        print("\nVerification failed. Some hashes are incorrect or missing.")

if __name__ == "__main__":
    main()
