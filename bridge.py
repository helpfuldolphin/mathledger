from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import subprocess
import json
from pathlib import Path

app = FastAPI()

# Configuration
ROOT = Path(__file__).parent.resolve()
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN")

def check_token(x_token: str):
    if not BRIDGE_TOKEN or x_token != BRIDGE_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")

class ListRequest(BaseModel):
    path: str

class ReadRequest(BaseModel):
    path: str

class RunRequest(BaseModel):
    name: str

# Resolve repo Python (critical for pytest)
def _resolve_repo_python():
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return "python"

REPO_PYTHON = _resolve_repo_python()

SAFE_CMDS = {
    "git status": ["git", "-C", str(ROOT), "status", "--porcelain", "-b"],
    "pytest quick": [REPO_PYTHON, "-m", "pytest", "-q", "-k", "not integration"],
    "ls repo root": ["dir", str(ROOT)],
}

@app.get("/health")
def health(x_token: str = Header(None, alias="X-Token")):
    check_token(x_token)
    return {"ok": True, "root": str(ROOT), "readonly": True}

@app.post("/list")
def list_files(request: ListRequest, x_token: str = Header(None, alias="X-Token")):
    check_token(x_token)
    path = ROOT / request.path if request.path else ROOT
    if not path.exists():
        raise HTTPException(status_code=404, detail="path not found")

    if path.is_dir():
        entries = [item.name for item in path.iterdir()]
        return {"type": "dir", "entries": entries}
    else:
        return {"type": "file", "size": path.stat().st_size}

@app.post("/read")
def read_file(request: ReadRequest, x_token: str = Header(None, alias="X-Token")):
    check_token(x_token)
    file_path = ROOT / request.path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="file not found")

    try:
        text = file_path.read_text(encoding="utf-8")
        return {"path": request.path, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
def run_command(request: RunRequest, x_token: str = Header(None, alias="X-Token")):
    check_token(x_token)

    if request.name not in SAFE_CMDS:
        raise HTTPException(status_code=400, detail="command not allowed")

    cmd = SAFE_CMDS[request.name]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300
        )
        return {
            "cmd": cmd,
            "code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="command timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
