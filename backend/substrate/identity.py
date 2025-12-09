# PHASE II — NOT USED IN PHASE I
#
# This module defines the Substrate Identity Envelope schema and canonical
# serialization rules as per Operation Substrate-SEAL.

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class SubstrateIdentityEnvelope:
    """
    A cryptographically verifiable record of a single substrate execution.
    This is the fundamental atom of the Identity Ledger.
    """
    substrate_name: str
    version_hash: str
    spec_version: str
    execution_input: Dict[str, Any]
    execution_output: Dict[str, Any]
    forbidden_behavior_audit: Dict[str, Any]
    determinism_signature: str = field(init=False) # Calculated after initialization

    def to_dict(self) -> Dict[str, Any]:
        """Converts the envelope to a dictionary for serialization."""
        return self.__dict__

    def _get_canonical_representation(self, for_signing: bool = False) -> str:
        """
        Produces a canonical JSON string representation of the envelope.
        
        Args:
            for_signing: If True, excludes the determinism_signature field.
        """
        data_to_serialize = self.to_dict()
        if for_signing:
            data_to_serialize.pop("determinism_signature", None)
        
        return json.dumps(
            data_to_serialize,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False
        )

    def sign(self) -> None:
        """
        Calculates and sets the determinism_signature for the envelope.
        The signature is a SHA-256 hash of the canonical representation of
        all other fields in the envelope.
        """
        canonical_string = self._get_canonical_representation(for_signing=True)
        self.determinism_signature = hashlib.sha256(
            canonical_string.encode('utf-8')
        ).hexdigest()

    def verify_signature(self) -> bool:
        """
        Verifies the integrity of the envelope by re-calculating the signature.
        
        Returns:
            True if the signature is valid, False otherwise.
        """
        if not hasattr(self, 'determinism_signature'):
            return False
        
        expected_signature = self.determinism_signature
        
        canonical_string = self._get_canonical_representation(for_signing=True)
        calculated_signature = hashlib.sha256(
            canonical_string.encode('utf-8')
        ).hexdigest()
            
        return expected_signature == calculated_signature

def get_source_file_hash(file_path: str) -> str:
    """Computes the SHA-256 hash of a source file."""
    from pathlib import Path
    path = Path(file_path)
    if not path.is_file():
        return "ERROR_FILE_NOT_FOUND"
    
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()
