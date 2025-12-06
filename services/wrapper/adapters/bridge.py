"""Bridge API Client - communicates with Manus' Bridge conduit for file access."""
import httpx
from os import getenv
from typing import Optional


class BridgeClient:
    """Client for interacting with the Bridge API (file read/list operations)."""

    def __init__(self):
        self.base_url = getenv("BRIDGE_BASE_URL", "http://127.0.0.1:5055")
        self.token = getenv("BRIDGE_TOKEN", "")
        self.timeout = 30.0  # seconds

    def _headers(self) -> dict:
        """Build headers with X-Token auth."""
        return {"X-Token": self.token} if self.token else {}

    async def health(self) -> dict:
        """Check Bridge API health."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/health", headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def list_files(self, path: str = ".", pattern: str = "*") -> list[str]:
        """
        List files via Bridge /list endpoint.

        Args:
            path: Directory to list (relative to repo root)
            pattern: Glob pattern (e.g., "*.lean", "*.py")

        Returns:
            List of file paths
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/list",
                headers=self._headers(),
                params={"path": path, "pattern": pattern}
            )
            resp.raise_for_status()
            data = resp.json()
            # Bridge returns {"files": [...]} or similar
            return data.get("files", []) if isinstance(data, dict) else data

    async def read_file(self, file_path: str, offset: int = 0, limit: Optional[int] = None) -> dict:
        """
        Read file contents via Bridge /read endpoint.

        Args:
            file_path: Path to file (relative to repo root)
            offset: Line offset (0-indexed)
            limit: Max lines to read (None = all)

        Returns:
            {"content": str, "lines": int, "path": str}
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            params = {"file_path": file_path}
            if offset > 0:
                params["offset"] = offset
            if limit is not None:
                params["limit"] = limit

            resp = await client.get(
                f"{self.base_url}/read",
                headers=self._headers(),
                params=params
            )
            resp.raise_for_status()
            data = resp.json()

            # Normalize response format
            if isinstance(data, dict):
                return {
                    "content": data.get("content", ""),
                    "lines": data.get("lines", 0),
                    "path": file_path
                }
            else:
                # If Bridge returns raw string, wrap it
                return {"content": str(data), "lines": len(str(data).splitlines()), "path": file_path}

    async def search_theorems(self, paths: list[str] = None) -> list[dict]:
        """
        Search for theorem definitions in source files.

        This is a higher-level helper that scans files for theorem-like patterns.
        Default paths: ["lean4/", "python/"]

        Returns:
            List of {id, label, file, line, statement}
        """
        if paths is None:
            paths = ["lean4/", "python/"]

        theorems = []

        for base_path in paths:
            try:
                # List all relevant files
                if "lean4" in base_path:
                    files = await self.list_files(base_path, "*.lean")
                elif "python" in base_path:
                    files = await self.list_files(base_path, "*.py")
                else:
                    files = await self.list_files(base_path, "*")

                # Parse each file for theorem patterns
                for file_path in files[:10]:  # Limit to first 10 files for demo
                    try:
                        file_data = await self.read_file(file_path)
                        content = file_data["content"]

                        # Simple regex-like search for theorem declarations
                        # Lean: "theorem <name> :"
                        # Python: "def test_<name>" or class with "Theorem" in name
                        lines = content.splitlines()
                        for i, line in enumerate(lines):
                            if "theorem " in line.lower() or "def test_" in line:
                                # Extract theorem name and statement
                                theorem_id = f"T{len(theorems) + 1}"
                                label = self._extract_name(line)
                                statement = line.strip()

                                theorems.append({
                                    "id": theorem_id,
                                    "label": label,
                                    "file": file_path,
                                    "line": i + 1,
                                    "statement": statement,
                                    "proof_status": "PENDING"  # Default; POA will update
                                })

                                if len(theorems) >= 20:  # Cap at 20 for demo
                                    return theorems
                    except Exception as e:
                        # Skip files that fail to parse
                        continue

            except Exception as e:
                # Skip paths that fail to list
                continue

        return theorems

    def _extract_name(self, line: str) -> str:
        """Extract theorem/function name from a line."""
        # Remove common prefixes
        line = line.strip()
        for prefix in ["theorem ", "def ", "test_", "class "]:
            if prefix in line.lower():
                start = line.lower().find(prefix) + len(prefix)
                rest = line[start:].strip()
                # Take first word (up to space, paren, colon)
                for delim in [" ", "(", ":", "{"]:
                    if delim in rest:
                        rest = rest[:rest.index(delim)]
                        break
                return rest if rest else "Unknown"
        return line[:40]  # Fallback: first 40 chars


# Singleton instance
_bridge_client = None

def get_bridge_client() -> BridgeClient:
    """Get or create Bridge client singleton."""
    global _bridge_client
    if _bridge_client is None:
        _bridge_client = BridgeClient()
    return _bridge_client
