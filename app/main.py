"""Garment attribute API (FastAPI).

The model loads at startup in a background thread: /health answers at once, /ready
stays 503 until it is in memory. Uploads are decoded in memory and not stored.

Run: uv run python run.py
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.attributes import describe
from app.config import MODEL, MODEL_REVISION, Settings, get_settings
from schemas.api import AnalyzeResponse, ErrorResponse
from services.images import ImageRejected, decode_upload

log = logging.getLogger("garment")


class ModelLike(Protocol):
    def predict(self, image: Image.Image, top_k: int = 3) -> dict[str, dict]: ...


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str, headers: dict | None = None) -> None:
        self.status, self.code, self.detail, self.headers = status, code, detail, headers


def _error(request: Request, status: int, code: str, detail: str, headers=None) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    body = ErrorResponse(error=code, detail=detail, request_id=rid).model_dump()
    return JSONResponse(body, status_code=status, headers=headers)


def load_model(settings: Settings) -> ModelLike:
    import torch

    from services.model import FashionClip

    torch.set_num_threads(settings.torch_threads)
    return FashionClip(MODEL, MODEL_REVISION, settings.models_dir)


def create_app(settings: Settings | None = None, model: ModelLike | None = None) -> FastAPI:
    settings = settings or get_settings()
    state = {"model": model, "loading": "ready" if model else "starting", "failed": False}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["model"] is None:

            def load() -> None:
                state["loading"] = "loading Fashion-CLIP"
                t0 = time.perf_counter()
                try:
                    state["model"] = load_model(settings)
                    state["loading"] = "ready"
                    log.info("model loaded in %.1f s", time.perf_counter() - t0)
                except Exception:
                    state["failed"] = True
                    state["loading"] = "model loading failed; see the server log"
                    log.exception("model loading failed")

            threading.Thread(target=load, name="model-loader", daemon=True).start()
        yield

    app = FastAPI(
        title="Garment attributes",
        version="0.2.0",
        summary="Zero-shot garment attributes (category, colour, pattern, sleeve, gender "
        "section) from a photo, with Fashion-CLIP.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error(request, exc.status, exc.code, exc.detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p not in ("body",))
        msg = first.get("msg", "invalid request")
        return _error(request, 422, "invalid_request", f"{where}: {msg}" if where else msg)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("unhandled error")
        return _error(request, 500, "internal_error", "Something went wrong; it was logged.")

    demo_dir = Path(settings.demo_dir)
    if (demo_dir / "samples").is_dir():
        app.mount("/samples", StaticFiles(directory=demo_dir / "samples"), name="samples")

    @app.get("/", include_in_schema=False)
    def demo():
        page = demo_dir / "index.html"
        return FileResponse(page) if page.is_file() else RedirectResponse("/docs")

    @app.get("/health", tags=["ops"])
    def health():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready(request: Request):
        """Readiness: the model is loaded."""
        if state["model"] is None:
            code = "startup_failed" if state["failed"] else "not_ready"
            return _error(request, 503, code, f"The service is starting ({state['loading']}).")
        return {"status": "ready", "model": f"{MODEL}@{MODEL_REVISION[:7]}"}

    def analyze(request: Request, file: UploadFile) -> AnalyzeResponse:
        t0 = time.perf_counter()
        model = state["model"]
        if model is None:
            raise ApiError(503, "not_ready", f"The service is starting ({state['loading']}).")
        limit = int(settings.max_upload_mb * 1024 * 1024)
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise ApiError(
                413, "too_large", f"Uploads are limited to {settings.max_upload_mb:g} MB."
            )
        try:
            decoded = decode_upload(data, settings.max_pixels, settings.min_side_px)
        except ImageRejected as exc:
            raise ApiError(exc.status, exc.code, exc.detail) from None
        t1 = time.perf_counter()
        if not lock.acquire(timeout=settings.busy_timeout_s):
            raise ApiError(
                503,
                "busy",
                "Another image is being analysed; try again shortly.",
                {"Retry-After": "5"},
            )
        try:
            attrs = model.predict(decoded.image, top_k=settings.top_k)
        finally:
            lock.release()
        t2 = time.perf_counter()
        title, description = describe({k: v["label"] for k, v in attrs.items()})
        return AnalyzeResponse(
            request_id=request.state.request_id,
            attributes=attrs,
            title=title,
            description=description,
            image={
                "format": decoded.format,
                "width": decoded.width,
                "height": decoded.height,
                "exif_orientation": decoded.exif_orientation,
            },
            timings={
                "decode_ms": round((t1 - t0) * 1000, 1),
                "model_ms": round((t2 - t1) * 1000, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
            model=f"{MODEL}@{MODEL_REVISION[:7]}",
            settings={"top_k": settings.top_k, "torch_threads": settings.torch_threads},
        )

    responses = {code: {"model": ErrorResponse} for code in (413, 415, 422, 503)}

    @app.post("/v1/analyze", response_model=AnalyzeResponse, responses=responses, tags=["analyze"])
    def analyze_v1(request: Request, file: UploadFile = File(..., description="Garment photo")):  # noqa: B008
        """Predict category, colour, pattern, sleeve and gender section, with scores."""
        return analyze(request, file)

    @app.post("/analyze", response_model=AnalyzeResponse, deprecated=True, tags=["analyze"])
    def analyze_legacy(request: Request, file: UploadFile = File(...)):  # noqa: B008
        """The January endpoint's path; same as /v1/analyze (the response is now richer)."""
        return analyze(request, file)

    return app
