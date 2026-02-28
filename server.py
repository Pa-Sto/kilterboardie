import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw
import torch

from cvae_model import KilterCVAE
from cvae_generate import _enforce_start_finish_counts


ROOT = Path(__file__).resolve().parent
HOLD_MAP_PATH = ROOT / "ImageData" / "References" / "holds.json"
BOARD_IMAGE_PATH = ROOT / "ImageData" / "References" / "empty_board.png"
CHECKPOINT_PATH = ROOT / "models" / "best.pt"
OUTPUT_DIR = ROOT / "local_generated"
FEEDBACK_DIR = ROOT / "local_feedback"

COLORS = {
    "start": (0, 185, 90, 230),
    "finish": (206, 0, 145, 230),
    "hand": (0, 172, 199, 200),
    "foot": (255, 126, 30, 200),
}
RING_RADIUS = 16


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ModelState:
    def __init__(self) -> None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Missing checkpoint at {CHECKPOINT_PATH}")
        if not HOLD_MAP_PATH.exists():
            raise FileNotFoundError(f"Missing hold map at {HOLD_MAP_PATH}")
        if not BOARD_IMAGE_PATH.exists():
            raise FileNotFoundError(f"Missing board image at {BOARD_IMAGE_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with HOLD_MAP_PATH.open("r") as f:
            self.hold_map = json.load(f)
        self.static = self._build_static(self.hold_map).to(self.device)

        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
        config = ckpt.get("config", {})
        self.grade_min = int(config.get("grade_min", 3))
        self.grade_max = int(config.get("grade_max", 13))
        self.model = KilterCVAE(
            num_grades=int(config.get("num_grades", self.grade_max - self.grade_min + 1)),
            emb_dim=int(config.get("emb_dim", 16)),
            latent_dim=int(config.get("latent_dim", 64)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @staticmethod
    def _build_static(hold_map: dict) -> torch.Tensor:
        rows = int(hold_map["rows"])
        cols = int(hold_map["cols"])
        holds = hold_map["holds"]
        presence = np.zeros((rows, cols), dtype=np.float32)
        size = np.zeros((rows, cols), dtype=np.float32)
        max_area = max(float(h.get("area_shape", 1.0)) for h in holds)
        for hold in holds:
            r = int(hold["row"])
            c = int(hold["col"])
            presence[r, c] = 1.0
            size[r, c] = float(hold.get("area_shape", 0.0)) / max_area
        return torch.from_numpy(np.stack([presence, size], axis=0))


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


def sample_route(state: ModelState, grade_v: int, seed: Optional[int]) -> np.ndarray:
    if grade_v < state.grade_min or grade_v > state.grade_max:
        raise ValueError(f"grade must be in [{state.grade_min}, {state.grade_max}]")
    if seed is not None:
        torch.manual_seed(seed)
    static = state.static.unsqueeze(0)
    grade_idx = torch.tensor([grade_v - state.grade_min], dtype=torch.int64, device=state.device)
    with torch.no_grad():
        logits = state.model.sample(grade_idx, static, n=1)
        probs = torch.sigmoid(logits)
        probs = _enforce_start_finish_counts(
            probs,
            hold_mask=static[:, 0:1],
            threshold=0.5,
            start_min=1,
            start_max=2,
            finish_min=1,
            finish_max=2,
            start_max_dist=8.0,
            finish_max_dist=8.0,
        )
        route = (probs >= 0.5).float()
    route_np = route.cpu().numpy()[0]
    static_np = static.cpu().numpy()[0]
    full = np.concatenate([route_np, static_np], axis=0)
    return np.transpose(full, (1, 2, 0))


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
            offset = 2 if name in ("start", "finish") else 0
            radius = RING_RADIUS + offset
            bbox = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(bbox, outline=color, width=3)
    return img


@app.post("/generate")
def generate(req: GenerateRequest):
    state = get_state()
    grade_v = parse_grade(req.grade)
    request_id = req.requestId or f"local-{int(datetime.utcnow().timestamp())}"
    seed = req.seed

    try:
        route = sample_route(state, grade_v, seed)
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
        "grade_v": grade_v,
        "matrix_path": str(matrix_path),
        "image_path": str(image_path),
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
    return {"ok": True}
