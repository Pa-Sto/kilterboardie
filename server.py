import base64
import io
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw
import torch

from cvae_data import KilterRouteDataset
from cvae_generate import _grade_count_histograms, decode_route_with_priors
from cvae_model import KilterCVAE
from diffusion_model import GaussianDiffusion, KilterDiffusionUNet


ROOT = Path(__file__).resolve().parent
HOLD_MAP_PATH = ROOT / "ImageData" / "References" / "holds.json"
BOARD_IMAGE_PATH = ROOT / "ImageData" / "References" / "empty_board.png"
DATA_DIR = ROOT / "ImageData" / "50Degree" / "Export"
BUNDLE_DIR = ROOT / "inference_bundle"
STATIC_BUNDLE_PATH = BUNDLE_DIR / "static.npy"
COUNT_HIST_PATH = BUNDLE_DIR / "count_histograms.json"
OUTPUT_DIR = ROOT / "local_generated"
FEEDBACK_DIR = ROOT / "local_feedback"

COLORS = {
    "start": (0, 185, 90, 230),
    "finish": (206, 0, 145, 230),
    "hand": (0, 172, 199, 200),
    "foot": (255, 126, 30, 200),
}
MARKER_HALF_SIZE = 22


class GenerateRequest(BaseModel):
    grade: Optional[str] = None
    angle: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    requestId: Optional[str] = None


class FeedbackRequest(BaseModel):
    requestId: str
    grade: Optional[str] = None
    angle: Optional[str] = None
    model: Optional[str] = None
    suggestedGrade: Optional[str] = None
    userFeedback: Optional[str] = None
    createdAt: Optional[str] = None


app = FastAPI()


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _is_generation_enabled() -> bool:
    value = os.getenv("API_GENERATION_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


class GenerateGuard:
    def __init__(self) -> None:
        self.per_ip_limit = _env_int("RATE_LIMIT_PER_IP", 30)
        self.per_ip_window_seconds = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
        self.daily_limit = _env_int("DAILY_GENERATE_LIMIT", 20000)
        self.max_tracked_ips = _env_int("MAX_TRACKED_IPS", 10000)
        self._lock = Lock()
        self._window_by_ip: dict[str, deque[float]] = {}
        self._daily_key = datetime.utcnow().strftime("%Y-%m-%d")
        self._daily_count = 0

    def _rollover_day_if_needed(self) -> None:
        key = datetime.utcnow().strftime("%Y-%m-%d")
        if key != self._daily_key:
            self._daily_key = key
            self._daily_count = 0
            self._window_by_ip.clear()

    def _evict_stale_ips(self, cutoff: float) -> None:
        stale = []
        for ip, window in self._window_by_ip.items():
            while window and window[0] <= cutoff:
                window.popleft()
            if not window:
                stale.append(ip)
        for ip in stale:
            self._window_by_ip.pop(ip, None)

    def consume_generate(self, ip: str) -> None:
        now = time.time()
        cutoff = now - self.per_ip_window_seconds
        with self._lock:
            self._rollover_day_if_needed()
            if self._daily_count >= self.daily_limit:
                raise HTTPException(status_code=429, detail="Daily generation quota reached")

            if len(self._window_by_ip) > self.max_tracked_ips:
                self._evict_stale_ips(cutoff)
            if len(self._window_by_ip) > self.max_tracked_ips and ip not in self._window_by_ip:
                raise HTTPException(status_code=429, detail="Rate limit is busy, try again shortly")

            window = self._window_by_ip.setdefault(ip, deque())
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= self.per_ip_limit:
                raise HTTPException(status_code=429, detail="Too many requests, slow down")

            window.append(now)
            self._daily_count += 1


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


GENERATE_GUARD = GenerateGuard()


def _candidate_existing_path(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _latest_matching_path(patterns: list[str]) -> Optional[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(ROOT.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_checkpoint(env_name: str, fixed_paths: list[Path], patterns: list[str]) -> Path:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        env_path = Path(env_value)
        if not env_path.is_absolute():
            env_path = ROOT / env_path
        if env_path.exists():
            return env_path
        raise FileNotFoundError(f"{env_name} points to missing checkpoint: {env_path}")

    fixed = _candidate_existing_path(fixed_paths)
    if fixed is not None:
        return fixed

    latest = _latest_matching_path(patterns)
    if latest is not None:
        return latest

    raise FileNotFoundError(f"Could not resolve a checkpoint for {env_name}")


def _resolve_data_dir() -> Path:
    env_value = os.getenv("DATA_DIR", "").strip()
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
        raise FileNotFoundError(f"DATA_DIR points to missing directory: {path}")
    if DATA_DIR.exists():
        return DATA_DIR
    raise FileNotFoundError(f"Missing dataset directory at {DATA_DIR}")


def _resolve_static_bundle() -> Path:
    env_value = os.getenv("STATIC_BUNDLE_PATH", "").strip()
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
        raise FileNotFoundError(f"STATIC_BUNDLE_PATH points to missing file: {path}")
    if STATIC_BUNDLE_PATH.exists():
        return STATIC_BUNDLE_PATH
    raise FileNotFoundError(f"Missing static bundle at {STATIC_BUNDLE_PATH}")


def _resolve_count_hist_path() -> Optional[Path]:
    env_value = os.getenv("COUNT_HISTOGRAM_PATH", "").strip()
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
        raise FileNotFoundError(f"COUNT_HISTOGRAM_PATH points to missing file: {path}")
    if COUNT_HIST_PATH.exists():
        return COUNT_HIST_PATH
    return None


def _load_count_histograms(data_dir: Optional[Path], grade_min: int, grade_max: int) -> dict[int, dict[str, dict[int, int]]]:
    count_hist_path = _resolve_count_hist_path()
    if count_hist_path is not None:
        with count_hist_path.open("r") as f:
            payload = json.load(f)
        return {
            int(grade): {name: {int(k): int(v) for k, v in counts.items()} for name, counts in channels.items()}
            for grade, channels in payload.items()
        }
    if data_dir is None:
        raise FileNotFoundError("No dataset directory or count histogram bundle available")
    return _grade_count_histograms(str(data_dir), grade_min, grade_max)


def _infer_cvae_static_channels(ckpt: dict, config: dict) -> int:
    if "static_channels" in config:
        return int(config["static_channels"])
    weight = ckpt["model_state"]["enc_conv.0.weight"]
    emb_dim = int(config.get("emb_dim", 16))
    inferred = int(weight.shape[1] - 4 - emb_dim)
    return max(inferred, 1)


def _normalize_model_name(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    if "diff" in text:
        return "diffusion"
    return "cvae"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass
class CVAEModelBundle:
    checkpoint_path: Path
    model: KilterCVAE
    grade_min: int
    grade_max: int
    static_channels: int
    count_histograms: dict[int, dict[str, dict[int, int]]]


@dataclass
class DiffusionModelBundle:
    checkpoint_path: Path
    model: KilterDiffusionUNet
    diffusion: GaussianDiffusion
    grade_min: int
    grade_max: int
    static_channels: int
    count_histograms: dict[int, dict[str, dict[int, int]]]
    timesteps: int


class ModelState:
    def __init__(self) -> None:
        if not HOLD_MAP_PATH.exists():
            raise FileNotFoundError(f"Missing hold map at {HOLD_MAP_PATH}")
        if not BOARD_IMAGE_PATH.exists():
            raise FileNotFoundError(f"Missing board image at {BOARD_IMAGE_PATH}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with HOLD_MAP_PATH.open("r") as f:
            self.hold_map = json.load(f)
        try:
            self.data_dir: Optional[Path] = _resolve_data_dir()
        except FileNotFoundError:
            self.data_dir = None

        static_bundle = np.load(_resolve_static_bundle())
        if static_bundle.ndim != 3:
            raise RuntimeError(f"Expected static bundle with shape H x W x C, got {static_bundle.shape}")
        static_tensor = torch.from_numpy(static_bundle).float().permute(2, 0, 1).unsqueeze(0)
        self.full_static = static_tensor.to(self.device)
        self.full_static_channels = int(static_tensor.shape[1])
        self.cvae: Optional[CVAEModelBundle] = None
        self.diffusion: Optional[DiffusionModelBundle] = None

    def _static_for_model(self, static_channels: int, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        channels = min(static_channels, self.full_static_channels)
        static = self.full_static[:, :channels]
        if batch_size > 1:
            static = static.expand(batch_size, -1, -1, -1)
        hold_mask = self.full_static[:, 0:1]
        if batch_size > 1:
            hold_mask = hold_mask.expand(batch_size, -1, -1, -1)
        return static, hold_mask

    def get_cvae(self) -> CVAEModelBundle:
        if self.cvae is not None:
            return self.cvae

        checkpoint_path = _resolve_checkpoint(
            "CVAE_CHECKPOINT_PATH",
            [ROOT / "models" / "best.pt"],
            ["runs/cvae/*/best.pt"],
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        config = ckpt.get("config", {})
        grade_min = int(config.get("grade_min", 3))
        grade_max = int(config.get("grade_max", 13))
        static_channels = _infer_cvae_static_channels(ckpt, config)
        model = KilterCVAE(
            num_grades=int(config.get("num_grades", grade_max - grade_min + 1)),
            emb_dim=int(config.get("emb_dim", 16)),
            latent_dim=int(config.get("latent_dim", 64)),
            static_channels=static_channels,
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        self.cvae = CVAEModelBundle(
            checkpoint_path=checkpoint_path,
            model=model,
            grade_min=grade_min,
            grade_max=grade_max,
            static_channels=static_channels,
            count_histograms=_load_count_histograms(self.data_dir, grade_min, grade_max),
        )
        return self.cvae

    def get_diffusion(self) -> DiffusionModelBundle:
        if self.diffusion is not None:
            return self.diffusion

        checkpoint_path = _resolve_checkpoint(
            "DIFFUSION_CHECKPOINT_PATH",
            [ROOT / "models" / "diffusion_best.pt"],
            ["runs/diffusion*/**/best.pt"],
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        config = ckpt.get("config", {})
        grade_min = int(config.get("grade_min", 3))
        grade_max = int(config.get("grade_max", 13))
        static_channels = int(config.get("static_channels", self.full_static_channels))
        model = KilterDiffusionUNet(
            num_grades=int(config.get("num_grades", grade_max - grade_min + 1)),
            static_channels=static_channels,
            route_channels=int(config.get("route_channels", 4)),
            base_channels=int(config.get("base_channels", 64)),
            grade_emb_dim=int(config.get("grade_emb_dim", 32)),
            time_emb_dim=int(config.get("time_emb_dim", 128)),
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        diffusion = GaussianDiffusion(
            timesteps=int(config.get("timesteps", 200)),
            beta_start=float(config.get("beta_start", 1e-4)),
            beta_end=float(config.get("beta_end", 0.02)),
        )

        self.diffusion = DiffusionModelBundle(
            checkpoint_path=checkpoint_path,
            model=model,
            diffusion=diffusion,
            grade_min=grade_min,
            grade_max=grade_max,
            static_channels=static_channels,
            count_histograms=_load_count_histograms(self.data_dir, grade_min, grade_max),
            timesteps=int(config.get("timesteps", 200)),
        )
        return self.diffusion


STATE: Optional[ModelState] = None


def get_state() -> ModelState:
    global STATE
    if STATE is None:
        STATE = ModelState()
    return STATE


def parse_grade(value: Optional[str]) -> int:
    if value is None:
        return 6
    s = str(value)
    for token in s.replace("/", " ").split():
        if token.upper().startswith("V") and token[1:].isdigit():
            return int(token[1:])
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 6


def _full_matrix(state: ModelState, route: torch.Tensor) -> np.ndarray:
    route_np = route.cpu().numpy()[0]
    static_np = state.full_static.cpu().numpy()[0]
    full = np.concatenate([route_np, static_np], axis=0)
    return np.transpose(full, (1, 2, 0))


def sample_cvae_route(state: ModelState, grade_v: int, seed: Optional[int]) -> tuple[np.ndarray, str]:
    bundle = state.get_cvae()
    if grade_v < bundle.grade_min or grade_v > bundle.grade_max:
        raise ValueError(f"grade must be in [{bundle.grade_min}, {bundle.grade_max}]")
    if seed is not None:
        torch.manual_seed(seed)
    static, hold_mask = state._static_for_model(bundle.static_channels, batch_size=1)
    grade_idx = torch.tensor([grade_v - bundle.grade_min], dtype=torch.int64, device=state.device)
    grade_values = torch.tensor([grade_v], dtype=torch.int64, device=state.device)
    with torch.no_grad():
        logits = bundle.model.sample(grade_idx, static, n=1)
        probs = torch.sigmoid(logits)
        route = decode_route_with_priors(
            probs,
            hold_mask=hold_mask,
            grade_values=grade_values,
            count_histograms=bundle.count_histograms,
            seed=seed or 42,
            threshold=0.5,
            start_min=1,
            start_max=2,
            finish_min=1,
            finish_max=2,
            start_max_dist=8.0,
            finish_max_dist=8.0,
            foot_count_mode="median",
            foot_count_quantile=0.5,
        )
    return _full_matrix(state, route), _display_path(bundle.checkpoint_path)


def sample_diffusion_route(state: ModelState, grade_v: int, seed: Optional[int]) -> tuple[np.ndarray, str]:
    bundle = state.get_diffusion()
    if grade_v < bundle.grade_min or grade_v > bundle.grade_max:
        raise ValueError(f"grade must be in [{bundle.grade_min}, {bundle.grade_max}]")
    if seed is not None:
        torch.manual_seed(seed)
    static, hold_mask = state._static_for_model(bundle.static_channels, batch_size=1)
    grade_idx = torch.tensor([grade_v - bundle.grade_min], dtype=torch.int64, device=state.device)
    grade_values = torch.tensor([grade_v], dtype=torch.int64, device=state.device)
    with torch.no_grad():
        x0_pred = bundle.diffusion.sample(
            model=bundle.model,
            shape=(1, 4, static.shape[2], static.shape[3]),
            static=static,
            grade=grade_idx,
            device=state.device,
            hold_mask=hold_mask,
        )
        probs = ((x0_pred + 1.0) * 0.5).clamp(0.0, 1.0)
        probs = probs * hold_mask
        route = decode_route_with_priors(
            probs,
            hold_mask=hold_mask,
            grade_values=grade_values,
            count_histograms=bundle.count_histograms,
            seed=seed or 42,
            threshold=0.5,
            start_min=1,
            start_max=2,
            finish_min=1,
            finish_max=2,
            start_max_dist=8.0,
            finish_max_dist=8.0,
            foot_count_mode="median",
            foot_count_quantile=0.5,
        )
        route = route * hold_mask
    return _full_matrix(state, route), _display_path(bundle.checkpoint_path)


def render_overlay(hold_map: dict, route: np.ndarray) -> Image.Image:
    img = Image.open(BOARD_IMAGE_PATH).convert("RGBA")
    draw = ImageDraw.Draw(img)
    channels = {"start": 0, "finish": 1, "hand": 2, "foot": 3}
    active = {name: route[:, :, idx] > 0 for name, idx in channels.items()}

    for hold in hold_map["holds"]:
        r = int(hold["row"])
        c = int(hold["col"])
        x = float(hold["x"])
        y = float(hold["y"])
        for name, color in COLORS.items():
            if not active[name][r, c]:
                continue
            offset = 3 if name in ("start", "finish") else 0
            half = MARKER_HALF_SIZE + offset
            bbox = (x - half, y - half, x + half, y + half)
            draw.rectangle(bbox, outline=color, width=4)
    return img


@app.post("/generate")
def generate(req: GenerateRequest, request: Request):
    if not _is_generation_enabled():
        raise HTTPException(status_code=503, detail="Generation is temporarily disabled")

    GENERATE_GUARD.consume_generate(_client_ip(request))

    state = get_state()
    grade_v = parse_grade(req.grade)
    model_key = _normalize_model_name(req.model)
    request_id = req.requestId or f"local-{int(datetime.utcnow().timestamp())}"
    seed = req.seed if req.seed is not None else int(time.time() * 1000) % (2**31 - 1)
    started_at = time.perf_counter()

    try:
        if model_key == "diffusion":
            route, checkpoint_path = sample_diffusion_route(state, grade_v, seed)
        else:
            route, checkpoint_path = sample_cvae_route(state, grade_v, seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = OUTPUT_DIR / f"{request_id}.npy"
    image_path = OUTPUT_DIR / f"{request_id}.png"
    meta_path = OUTPUT_DIR / f"{request_id}.json"

    np.save(matrix_path, route)
    overlay = render_overlay(state.hold_map, route)
    overlay.save(image_path)

    created_at = datetime.utcnow().isoformat() + "Z"
    meta = {
        "request_id": request_id,
        "created_at": created_at,
        "model_key": model_key,
        "model_label": req.model or ("Diffusion" if model_key == "diffusion" else "KilterCVAE"),
        "grade_v": grade_v,
        "matrix_path": matrix_path.name,
        "image_path": image_path.name,
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "generation_seconds": round(time.perf_counter() - started_at, 3),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "requestId": request_id,
        "imageDataUrl": f"data:image/png;base64,{image_data}",
        "meta": meta,
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    entry_dir = FEEDBACK_DIR / req.requestId
    entry_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "request_id": req.requestId,
        "grade": req.grade,
        "angle": req.angle,
        "model": req.model,
        "suggested_grade": req.suggestedGrade,
        "user_feedback": req.userFeedback,
        "created_at": req.createdAt,
        "received_at": datetime.utcnow().isoformat() + "Z",
    }
    with (entry_dir / "feedback.json").open("w") as f:
        json.dump(payload, f, indent=2)

    return {"ok": True}


@app.get("/health")
def health():
    status = {"ok": True, "generation_enabled": _is_generation_enabled()}
    try:
        state = get_state()
        status["data_dir"] = _display_path(state.data_dir) if state.data_dir is not None else None
        status["static_bundle"] = _display_path(_resolve_static_bundle())
        status["available_models"] = {
            "cvae": _display_path(
                _resolve_checkpoint(
                    "CVAE_CHECKPOINT_PATH",
                    [ROOT / "models" / "best.pt"],
                    ["runs/cvae/*/best.pt"],
                )
            ),
            "diffusion": _display_path(
                _resolve_checkpoint(
                    "DIFFUSION_CHECKPOINT_PATH",
                    [ROOT / "models" / "diffusion_best.pt"],
                    ["runs/diffusion*/**/best.pt"],
                )
            ),
        }
    except FileNotFoundError as exc:
        status["ok"] = False
        status["detail"] = str(exc)
    return status
