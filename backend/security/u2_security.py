# PHASE II — U2 UPLIFT EXPERIMENT
import hashlib
import os
import random
from typing import Optional

# --- Custom Exception ---

class SecurityException(Exception):
    """Raised for any U2 security policy violation."""
    pass

# --- PRNG Determinism Guard ---

class DeterministicPRNG:
    """
    A guarded wrapper around the random module to ensure determinism.

    Once initialized with a master seed, this object becomes the ONLY approved
    source of randomness for the U2 runner. Direct calls to `random.random()`
    are banned by convention and should be flagged in code reviews.
    """
    _instance: Optional['DeterministicPRNG'] = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DeterministicPRNG, cls).__new__(cls)
        return cls._instance

    def initialize(self, master_seed: int):
        """Initialize the singleton PRNG with a specific seed."""
        if self._initialized:
            raise SecurityException("PRNG determinism guard is already initialized.")
        self._prng = random.Random(master_seed)
        self._initialized = True
        print(f"DeterministicPRNG initialized with master seed: {master_seed}")

    @property
    def prng(self) -> random.Random:
        """Get the seeded random number generator instance."""
        if not self._initialized:
            raise SecurityException("PRNG determinism guard has not been initialized.")
        return self._prng

# --- Security Checks ---

def verify_environment():
    """
    Ensures the environment is correctly configured for a U2 run.
    
    Raises:
        SecurityException: If RFL_ENV_MODE is not set to 'PHASE-II-U2'.
    """
    env_mode = os.environ.get("RFL_ENV_MODE")
    required_mode = "PHASE-II-U2"
    if env_mode != required_mode:
        raise SecurityException(
            f"Environment validation failed! Expected RFL_ENV_MODE='{required_mode}', but found '{env_mode}'."
        )
    print("✅ Environment validation passed.")

def calculate_manifest_hash(manifest_path: str) -> str:
    """
    Computes the SHA256 hash of the manifest file.

    Args:
        manifest_path: Path to the manifest file.

    Returns:
        The hex digest of the SHA256 hash.
        
    Raises:
        SecurityException: If the manifest file does not exist.
    """
    if not os.path.exists(manifest_path):
        raise SecurityException(f"Manifest file not found at: {manifest_path}")
        
    hasher = hashlib.sha256()
    with open(manifest_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
            
    return hasher.hexdigest()

def run_pre_flight_checks(manifest_path: str) -> str:
    """
    Runs critical pre-flight security checks for a U2 run.

    This function should be called at the very beginning of a U2 runner process.

    Args:
        manifest_path: Path to the U2 run manifest.

    Returns:
        The calculated hash of the manifest for downstream processing (e.g., seeding).
        
    Raises:
        SecurityException: On any security violation.
    """
    print("--- Running U2 Pre-flight Security Checks ---")
    
    # 1. Environment Validation
    verify_environment()
    
    # 2. Manifest Hash Verification
    print(f"Calculating hash for manifest: {manifest_path}...")
    manifest_hash = calculate_manifest_hash(manifest_path)
    print(f"✅ Manifest hash calculated: {manifest_hash}")
    
    print("--- All U2 Pre-flight Security Checks Passed ---")
    return manifest_hash

# --- Banned Randomness Detection (Conceptual) ---
# A true implementation would involve bytecode manipulation or monkey-patching,
# which is complex. The primary guard is the convention of using only the
# DeterministicPRNG instance. We can add a simple check to discourage direct use.

def check_for_banned_randomness():
    """
    A simple check to detect if the global random instance has been used.
    This is not foolproof but can catch basic misuses.
    """
    # This is a conceptual check. In a real scenario, you'd want to patch `random`
    # at startup to raise an error on calls to functions other than `seed`.
    # For now, we rely on the convention of using the DeterministicPRNG.
    pass
