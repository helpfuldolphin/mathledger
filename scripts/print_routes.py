from backend.orchestrator.app import app
for r in app.router.routes:
    try:
        print(r.path)
    except Exception as e:
        print(r)
