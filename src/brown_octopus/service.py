"""Optional HTTP service surface for agent-harness integrations."""


class OctopusService:
    """Concurrency-safe session coordinator around an initialized Octopus."""

    def __init__(self, octopus) -> None:
        self.octopus = octopus

    def health(self) -> dict:
        return {"status": "ok", "initialized": self.octopus.pipeline is not None}

    def ready(self) -> dict:
        ready = self.octopus.pipeline is not None
        return {"status": "ready" if ready else "not_ready", "ready": ready}

    def process(self, session_id: str, query: str) -> dict:
        return self.octopus.process(query, session_id=session_id)

    def reset(self, session_id: str) -> None:
        self.octopus.reset_session(session_id)

    def delete(self, session_id: str) -> None:
        self.octopus.delete_session(session_id)


def create_app(octopus=None):
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install the 'service' extra to expose HTTP endpoints") from exc

    instance = octopus or Octopus()
    service = OctopusService(instance)
    app = FastAPI(title="Brown Octopus", version="0.4.7")

    @app.get("/health")
    def health():
        return service.health()

    @app.get("/ready")
    def ready():
        return service.ready()

    @app.post("/v1/capabilities")
    def capabilities(payload: dict):
        return service.process(payload.get("session_id", "default"), payload["query"])

    @app.post("/v1/sessions/{session_id}/reset")
    def reset(session_id: str):
        service.reset(session_id)
        return {"status": "reset"}

    return app
