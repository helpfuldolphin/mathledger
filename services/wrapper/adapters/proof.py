"""POA/Proof Service Client - verifies theorem statements (prove or abstain)."""
import httpx
from os import getenv
from typing import Optional


class ProofClient:
    """Client for interacting with POA/Proof service."""

    def __init__(self):
        self.base_url = getenv("PROOF_BASE_URL", "http://127.0.0.1:6000")
        self.timeout = 60.0  # Proof attempts may take longer

    async def verify_theorem(self, statement: str, context: Optional[dict] = None) -> dict:
        """
        Verify a theorem statement via POA/Proof service.

        Args:
            statement: The theorem statement to verify
            context: Optional context (dependencies, file path, etc.)

        Returns:
            {
                "status": "PROVED" | "ABSTAIN",
                "outline": str,  # Proof outline or reason for abstaining
                "source_refs": [str]  # References to lemmas/axioms used
            }
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {"statement": statement}
                if context:
                    payload["context"] = context

                resp = await client.post(
                    f"{self.base_url}/verify",
                    json=payload
                )
                resp.raise_for_status()
                data = resp.json()

                # Normalize response
                return {
                    "status": data.get("status", "ABSTAIN"),
                    "outline": data.get("outline", ""),
                    "source_refs": data.get("source_refs", [])
                }

        except httpx.ConnectError:
            # POA service not available
            return {
                "status": "ABSTAIN",
                "outline": "POA service unavailable (connection refused)",
                "source_refs": []
            }
        except httpx.TimeoutException:
            # Proof attempt timed out
            return {
                "status": "ABSTAIN",
                "outline": f"Proof timeout after {self.timeout}s",
                "source_refs": []
            }
        except httpx.HTTPStatusError as e:
            # POA returned error (e.g., 400 for invalid statement)
            return {
                "status": "ABSTAIN",
                "outline": f"POA error: {e.response.status_code} - {e.response.text}",
                "source_refs": []
            }
        except Exception as e:
            # Unexpected error
            return {
                "status": "ABSTAIN",
                "outline": f"Unexpected error: {str(e)}",
                "source_refs": []
            }

    async def health(self) -> dict:
        """Check POA service health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return {"status": "unavailable"}


# Singleton instance
_proof_client = None

def get_proof_client() -> ProofClient:
    """Get or create Proof client singleton."""
    global _proof_client
    if _proof_client is None:
        _proof_client = ProofClient()
    return _proof_client
