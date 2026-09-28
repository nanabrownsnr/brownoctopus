"""Uvicorn entry point for the optional HTTP deployment."""

from brown_octopus.octopus import Octopus
from brown_octopus.service import create_app


octopus = Octopus()
app = create_app(octopus)


@app.on_event("startup")
async def initialize_octopus() -> None:
    await octopus.initialize()
