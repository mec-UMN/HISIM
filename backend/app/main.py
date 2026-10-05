from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from app.api.routes import router
from app.services.paths import WEBUI_DIR


class RevalidatingStaticFiles(StaticFiles):
    """Serve the UI with ``Cache-Control: no-cache``: browsers keep a copy but re-check it on every load,
    so a rebuilt stylesheet or script shows up immediately instead of after a hard refresh."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="HISIM GUI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def _recover_after_restart() -> None:
    """A previous server process may have died mid-run: fail the runs it left queued or running."""
    from app.services.hisim_runner import recover_interrupted_runs

    recover_interrupted_runs()


if WEBUI_DIR.exists():
    # The vanilla HTML/CSS/JS workbench is served straight from the API server,
    # so a single `uvicorn app.main:app` command brings the whole GUI up.
    app.mount("/app", RevalidatingStaticFiles(directory=str(WEBUI_DIR), html=True), name="webui")

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/app/")
