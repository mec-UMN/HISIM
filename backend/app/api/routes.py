from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response

from app.schemas.run import RunCreate, RunRecord, SweepCreate
from app.services.hisim_runner import (
    artifact_path,
    backfill_record,
    available_models,
    create_run,
    create_sweep,
    default_config_from_file,
    model_info,
    run_artifact_tree,
    run_logs,
    run_result,
)
from app.services.knobs import knob_schema
from app.services.mapping_files import (
    mapping_file_path,
    mapping_files_overview,
    mapping_files_status,
    input_file_template,
    preferred_mapping_file_path,
    write_mapping_file,
)
from app.services.plotly_data import available_plots, plot_payload
from app.services.store import delete_record, list_records, load_record


router = APIRouter()


# --------------------------------------------------------------------------- #
#  meta
# --------------------------------------------------------------------------- #
@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models")
def models() -> dict[str, Any]:
    return {"models": available_models(), "details": model_info()}


@router.get("/knobs")
def knobs() -> dict[str, Any]:
    return knob_schema()


@router.get("/config/defaults")
def config_defaults() -> dict[str, object]:
    return default_config_from_file()


# --------------------------------------------------------------------------- #
#  runs
# --------------------------------------------------------------------------- #
@router.post("/runs", response_model=RunRecord)
def start_run(payload: RunCreate) -> RunRecord:
    try:
        return create_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sweeps", response_model=list[RunRecord])
def start_sweep(payload: SweepCreate) -> list[RunRecord]:
    try:
        return create_sweep(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs", response_model=list[RunRecord])
def runs() -> list[RunRecord]:
    return [backfill_record(record) for record in list_records()]


@router.get("/runs/{run_id}", response_model=RunRecord)
def run_detail(run_id: str) -> RunRecord:
    try:
        return backfill_record(load_record(run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@router.delete("/runs/{run_id}")
def run_delete(run_id: str) -> dict[str, str]:
    try:
        delete_record(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    return {"status": "deleted", "id": run_id}


@router.get("/runs/{run_id}/result")
def run_result_json(run_id: str) -> dict[str, Any]:
    try:
        return run_result(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No structured result for this run yet") from exc


@router.get("/runs/{run_id}/logs", response_class=PlainTextResponse)
def run_log(
    run_id: str,
    stream: str = Query("stdout", pattern="^(stdout|stderr)$"),
    tail: int | None = Query(None, ge=1, le=20000),
) -> str:
    return run_logs(run_id, stream=stream, tail=tail)


@router.get("/runs/{run_id}/artifacts")
def run_artifacts(run_id: str) -> dict[str, Any]:
    return {"artifacts": run_artifact_tree(run_id)}


@router.get("/runs/{run_id}/artifacts/file")
def run_artifact_file(run_id: str, path: str = Query(...)) -> FileResponse:
    try:
        return FileResponse(artifact_path(run_id, path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc


# --------------------------------------------------------------------------- #
#  input files — download the current map/spec (or a schema template before
#  the first run) for an AI model, or upload a hand-edited replacement into
#  uploaded_files/<model>/ for the next run.
# --------------------------------------------------------------------------- #
@router.get("/mapping-files/{aimodel}")
def mapping_files_get(aimodel: str) -> dict[str, Any]:
    return mapping_files_status(aimodel)


@router.get("/mapping-files/{aimodel}/overview")
def mapping_files_overview_get(aimodel: str) -> dict[str, Any]:
    try:
        return mapping_files_overview(aimodel)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/mapping-files/{aimodel}/{kind}")
def mapping_file_download(aimodel: str, kind: str) -> Response:
    try:
        path = preferred_mapping_file_path(aimodel, kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        filename, content = input_file_template(aimodel, kind)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return FileResponse(path, filename=path.name, media_type="text/csv")


@router.post("/mapping-files/{aimodel}/{kind}")
async def mapping_file_upload(aimodel: str, kind: str, file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        mapping_file_path(aimodel, kind)  # validates `kind` before touching disk
        content = await file.read()
        write_mapping_file(aimodel, kind, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return mapping_files_status(aimodel)


# --------------------------------------------------------------------------- #
#  pre-computed design-space sweeps shipped with the repo
# --------------------------------------------------------------------------- #
@router.get("/plots")
def plots() -> dict[str, object]:
    return {"plots": available_plots()}


@router.get("/plots/{plot_id}")
def plot(plot_id: str) -> dict[str, object]:
    try:
        return plot_payload(plot_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Plot not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Sweep data file missing") from exc
