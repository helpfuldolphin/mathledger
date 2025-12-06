from backend.orchestrator.app import app
print([r.path for r in app.router.routes if "/ui/parents" in getattr(r,"path","") or "/ui/proofs" in getattr(r,"path","")])
