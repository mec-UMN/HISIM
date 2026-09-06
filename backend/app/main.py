from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.paths import WEBUI_DIR


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


if WEBUI_DIR.exists():
    # The vanilla HTML/CSS/JS workbench is served straight from the API server,
    # so a single `uvicorn app.main:app` command brings the whole GUI up.
    app.mount("/app", StaticFiles(directory=str(WEBUI_DIR), html=True), name="webui")

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/app/")
