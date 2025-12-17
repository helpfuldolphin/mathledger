# backend/governance_signal/model.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class CryptographicMetadata:
    """Represents the cryptographic metadata for a signal."""
    signature: str
    publicKey: str
    signingAlgorithm: str

@dataclass
class GovernanceSignal:
    """Represents a Governance Signal, conforming to the JSON schema."""
    signalId: str
    originatorId: str
    timestamp: str
    semanticType: str
    severity: int
    ttl: int
    cryptographicMetadata: CryptographicMetadata
    isMonotonic: bool = False
    payload: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceSignal":
        """Creates a GovernanceSignal instance from a dictionary."""
        crypto_meta_data = data.get("cryptographicMetadata", {})
        crypto_meta = CryptographicMetadata(
            signature=crypto_meta_data.get("signature"),
            publicKey=crypto_meta_data.get("publicKey"),
            signingAlgorithm=crypto_meta_data.get("signingAlgorithm"),
        )
        return cls(
            signalId=data.get("signalId"),
            originatorId=data.get("originatorId"),
            timestamp=data.get("timestamp"),
            semanticType=data.get("semanticType"),
            severity=data.get("severity"),
            ttl=data.get("ttl"),
            isMonotonic=data.get("isMonotonic", False),
            payload=data.get("payload"),
            cryptographicMetadata=crypto_meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the GovernanceSignal instance to a dictionary."""
        return {
            "signalId": self.signalId,
            "originatorId": self.originatorId,
            "timestamp": self.timestamp,
            "semanticType": self.semanticType,
            "severity": self.severity,
            "ttl": self.ttl,
            "isMonotonic": self.isMonotonic,
            "payload": self.payload,
            "cryptographicMetadata": {
                "signature": self.cryptographicMetadata.signature,
                "publicKey": self.cryptographicMetadata.publicKey,
                "signingAlgorithm": self.cryptographicMetadata.signingAlgorithm,
            },
        }
