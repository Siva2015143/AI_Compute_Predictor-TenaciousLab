"""
backend/inference_engine.py
===========================
TenaciousLab Inference Engine (Production-Ready)

Manages model discovery, loading, inference, and caching.
"""

import os
import sys
import time
import json
import torch
import traceback
import threading
import numpy as np
from typing import Any, Dict, List, Optional

# ---------------------------------------------
# Root Path
# ---------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------
# Optional ONNX Runtime
# ---------------------------------------------
try:
    import onnxruntime as ort
    HAS_ONNX = True
except Exception:
    ort = None
    HAS_ONNX = False

# ---------------------------------------------
# Model Definition
# ---------------------------------------------
from trainer.train_regressor import MLPRegressor

# ---------------------------------------------
# Globals
# ---------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCK = threading.Lock()

def _find_pipeline_root() -> str:
    """Locate model training output directory."""
    candidates = [
        os.path.join(ROOT, "outputs"),
        os.path.join(ROOT, "pipeline_train_outputs"),
    ]
    for c in candidates:
        if os.path.exists(c) and any(os.path.isdir(os.path.join(c, d)) for d in os.listdir(c)):
            return c
    os.makedirs(candidates[0], exist_ok=True)
    return candidates[0]

PIPELINE_OUT = _find_pipeline_root()


# ---------------------------------------------
# Utility Functions
# ---------------------------------------------
def _is_scripted_model(path: str) -> bool:
    """Check if path contains a valid TorchScript model."""
    if not os.path.exists(path):
        return False
    try:
        torch.jit.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def _deterministic_demo_output(run_name: str, inputs: Any) -> Dict[str, Any]:
    """Fallback deterministic output."""
    arr = np.array(inputs, dtype=float)
    checksum = float(np.sum(arr)) * 0.001
    demo_value = (abs(hash(run_name)) % 97) / 100.0 + checksum
    result_shape = (1, arr.shape[0]) if arr.ndim == 1 else arr.shape
    demo_out = np.full(result_shape, demo_value).tolist()
    return {
        "ok": True,
        "result": demo_out,
        "meta": {
            "status": "demo_fallback",
            "warning": "Model failed to load; deterministic output."
        }
    }


def _predict_with_timeout(model_entry, inputs, timeout: float) -> Dict[str, Any]:
    """Thread-safe timeout wrapper for prediction."""
    result_container = {}

    def target():
        try:
            result_container["res"] = model_entry.predict(inputs)
        except Exception as e:
            result_container["res"] = {"ok": False, "error": str(e), "trace": traceback.format_exc()}

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return {"ok": False, "error": f"inference_timeout_after_{timeout}s"}
    return result_container.get("res", {"ok": False, "error": "unknown_error"})


# ---------------------------------------------
# Config Sanitization
# ---------------------------------------------
def _sanitize_model_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    allowed = {
        "input_dim": int,
        "output_dim": int,
        "hidden_sizes": list,
        "activation": str,
        "dropout": float,
    }
    out = {}
    for k, typ in allowed.items():
        if k in raw_cfg:
            v = raw_cfg[k]
            try:
                if typ is int:
                    out[k] = int(v)
                elif typ is float:
                    out[k] = float(v)
                elif typ is str:
                    out[k] = str(v)
                elif typ is list and isinstance(v, list):
                    out[k] = [int(x) for x in v]
            except Exception:
                continue
    return out


def _infer_io_from_state_dict(sd: Dict[str, Any]) -> Dict[str, int]:
    weights = [v for v in sd.values() if isinstance(v, torch.Tensor) and v.dim() == 2]
    if not weights:
        return {"input_dim": 1, "output_dim": 1}
    return {"input_dim": int(weights[0].shape[1]), "output_dim": int(weights[-1].shape[0])}


# ---------------------------------------------
# ModelEntry Class
# ---------------------------------------------
class ModelEntry:
    """Represents one model folder (run)."""

    def __init__(self, run_name: str, path: str):
        self.run_name = run_name
        self.path = path
        self.best_model_path: Optional[str] = None
        self.onnx_path: Optional[str] = None
        self.exported_scripted: Optional[str] = None
        self.model_obj: Optional[torch.nn.Module] = None
        self.onnx_session = None
        self.metadata: Dict[str, Any] = {}
        self.status = "unloaded"
        self.last_loaded = None
        self.load_error = None
        self._lock = threading.Lock()
        self._discover()

    def _discover(self):
        """Find model files in the folder."""
        candidates = [
            "exported_scripted.pt", "best_model_scripted.pt",
            "best_model.pt", "best_model.pth", "checkpoint.pt", "final_model.pt",
        ]
        for c in candidates:
            p = os.path.join(self.path, c)
            if os.path.exists(p):
                if _is_scripted_model(p):
                    self.best_model_path = p
                    break
                if not self.best_model_path:
                    self.best_model_path = p

        # ONNX
        onnx_files = [f for f in os.listdir(self.path) if f.endswith(".onnx")]
        if onnx_files:
            self.onnx_path = os.path.join(self.path, onnx_files[0])

        # Config
        cfg_path = os.path.join(self.path, "model_config.json")
        if os.path.exists(cfg_path):
            try:
                self.metadata["model_config"] = json.load(open(cfg_path))
            except Exception:
                self.metadata["model_config"] = None

        # Exported Scripted
        exp_scripted = os.path.join(self.path, "exported_scripted.pt")
        if os.path.exists(exp_scripted):
            self.exported_scripted = exp_scripted
            self.metadata["exported_scripted"] = exp_scripted

    def try_load(self, force_scripted: bool = False) -> bool:
        """Try to load model from available artifacts."""
        with self._lock:
            if self.model_obj or self.onnx_session:
                return True

            # 1️⃣ TorchScript
            if self.exported_scripted and os.path.exists(self.exported_scripted):
                try:
                    self.model_obj = torch.jit.load(self.exported_scripted, map_location="cpu")
                    self.model_obj.eval()
                    self.status = "loaded_exported_scripted"
                    print(f"[INFO] Loaded TorchScript for {self.run_name}")
                    return True
                except Exception as e:
                    self.load_error = f"torchscript_load_error: {e}"

            # 2️⃣ ONNX
            if HAS_ONNX and self.onnx_path:
                try:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
                    self.onnx_session = ort.InferenceSession(self.onnx_path, providers=providers)
                    self.status = "loaded_onnx"
                    print(f"[INFO] Loaded ONNX for {self.run_name}")
                    return True
                except Exception as e:
                    self.load_error = f"onnx_load_error: {e}"

            # 3️⃣ TorchScript candidate
            if self.best_model_path and _is_scripted_model(self.best_model_path):
                try:
                    self.model_obj = torch.jit.load(self.best_model_path, map_location=DEVICE)
                    self.model_obj.eval()
                    self.status = "loaded_scripted"
                    print(f"[INFO] Loaded scripted model for {self.run_name}")
                    return True
                except Exception as e:
                    self.load_error = f"scripted_load_error: {e}"

            # 4️⃣ Raw state_dict
            if not self.best_model_path:
                self.status = "no_artifact_found"
                self.load_error = "no_model_artifact_found"
                return False

            try:
                raw = torch.load(self.best_model_path, map_location="cpu")
            except Exception as e:
                self.status = "failed_load"
                self.load_error = str(e)
                return False

            try:
                if isinstance(raw, dict):
                    if "model_state" in raw:
                        state = raw["model_state"]
                    elif "model_state_dict" in raw:
                        state = raw["model_state_dict"]
                    elif all(isinstance(v, torch.Tensor) for v in raw.values()):
                        state = raw
                    else:
                        state = None

                    if state:
                        cfg = _sanitize_model_config(self.metadata.get("model_config", {}))
                        io = _infer_io_from_state_dict(state)
                        cfg.setdefault("input_dim", io["input_dim"])
                        cfg.setdefault("output_dim", io["output_dim"])
                        cfg.setdefault("hidden_sizes", (512, 256, 128))
                        model = MLPRegressor(
                            input_dim=cfg["input_dim"],
                            output_dim=cfg["output_dim"],
                            hidden_sizes=tuple(cfg["hidden_sizes"]),
                            activation=cfg.get("activation", "gelu"),
                            dropout=cfg.get("dropout", 0.0),
                        )
                        model.load_state_dict(state, strict=False)
                        model.eval()
                        self.model_obj = model.to(DEVICE)
                        self.status = "loaded_from_state_dict"
                        print(f"[INFO] Loaded model from state_dict for {self.run_name}")
                        return True

                self.status = "unknown_checkpoint_format"
                self.load_error = "unsupported_checkpoint"
                return False
            except Exception as e:
                self.status = "failed_load_state_dict"
                self.load_error = str(e)
                return False

    def ensure_loaded(self, force_scripted: bool = False) -> bool:
        return self.model_obj is not None or self.try_load(force_scripted)

    def predict(self, inputs: Any) -> Dict[str, Any]:
        with self._lock:
            try:
                x = np.array(inputs, dtype=np.float32)
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                if self.onnx_session is not None:
                    name = self.onnx_session.get_inputs()[0].name
                    out = self.onnx_session.run(None, {name: x})[0]
                    return {"ok": True, "result": out.tolist(), "meta": {"status": self.status}}
                if self.model_obj is not None:
                    tensor_in = torch.tensor(x, dtype=torch.float32, device=DEVICE)
                    with torch.no_grad():
                        out = self.model_obj(tensor_in).cpu().numpy()
                    return {"ok": True, "result": out.tolist(), "meta": {"status": self.status}}
                return _deterministic_demo_output(self.run_name, x)
            except Exception:
                return _deterministic_demo_output(self.run_name, inputs)

    def show_info(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "path": self.path,
            "status": self.status,
            "last_loaded": self.last_loaded,
            "load_error": self.load_error,
            "metadata": self.metadata,
            "has_model_obj": bool(self.model_obj),
            "has_onnx_session": bool(self.onnx_session),
        }


# ---------------------------------------------
# Global Model Manager
# ---------------------------------------------
class GlobalModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[str, ModelEntry] = {}
        self.base_dir = PIPELINE_OUT

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        runs.sort(reverse=True)
        results = []
        for r in runs:
            p = os.path.join(self.base_dir, r)
            entry = self._cache.get(r) or ModelEntry(r, p)
            status = "available" if entry.best_model_path or entry.exported_scripted else "incomplete"
            results.append({"run_name": r, "status": status, "path": p})
        return results

    def get(self, run_name: str):
        with self._lock:
            if run_name in self._cache:
                return self._cache[run_name]
            path = os.path.join(self.base_dir, run_name)
            if not os.path.exists(path):
                return None
            entry = ModelEntry(run_name, path)
            self._cache[run_name] = entry
            return entry

    def load_best_for(self, run_name: str):
        entry = self.get(run_name)
        if not entry:
            return {"ok": False, "status": "not_found"}
        ok = entry.ensure_loaded()
        return {"ok": ok, "status": entry.status, "error": entry.load_error}

    def predict(self, run_name: str, x):
        entry = self.get(run_name)
        if not entry:
            return {"ok": False, "error": f"run_not_found:{run_name}"}
        if not entry.ensure_loaded():
            return {"ok": False, "error": f"failed_to_load:{entry.load_error}"}
        return entry.predict(x)


# ---------------------------------------------
# Global Singleton
# ---------------------------------------------
GLOBAL_MODEL_MANAGER = GlobalModelManager()


# ---------------------------------------------
# CLI Testing
# ---------------------------------------------
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--load", type=str)
    parser.add_argument("--predict", nargs="+", type=float)
    parser.add_argument("--export", type=str, help="Export TorchScript model for given run")
    parser.add_argument("--show-model-info", type=str, help="Show full info for a given run")
    args = parser.parse_args()

    if args.list:
        import json
        print(json.dumps(GLOBAL_MODEL_MANAGER.list_runs(), indent=2))

    elif args.load:
        status = GLOBAL_MODEL_MANAGER.load_best_for(args.load)
        print(f"[LOAD] {args.load}: {status['status']} (ok={status['ok']})")
        if args.predict:
            preds = GLOBAL_MODEL_MANAGER.predict(args.load, np.array(args.predict, dtype=float))
            print(json.dumps(preds, indent=2))

    elif args.export:
        entry = GLOBAL_MODEL_MANAGER.get(args.export)
        if not entry:
            print(f"[ERROR] Run not found: {args.export}")
        else:
            # Ensure model is loaded before export
            load_status = GLOBAL_MODEL_MANAGER.load_best_for(args.export)
            if not load_status["ok"]:
                print(f"[ERROR] Failed to load model: {load_status.get('error')}")
            else:
                try:
                    scripted_path = os.path.join(entry.path, "exported_scripted.pt")
                    scripted_model = torch.jit.script(entry.model_obj)
                    scripted_model.save(scripted_path)
                    entry.exported_scripted = scripted_path
                    entry.status = "exported_scripted"
                    print(f"[OK] Exported TorchScript to {scripted_path}")
                except Exception as e:
                    print(f"[ERROR] Export failed: {e}")

    elif args.show_model_info:
        run = args.show_model_info
        entry = GLOBAL_MODEL_MANAGER.get(run)
        if not entry:
            print(f"[ERROR] Run not found: {run}")
        else:
            # ✅ ensure the model is actually loaded
            load_status = GLOBAL_MODEL_MANAGER.load_best_for(run)
            info = entry.show_info()
            info["load_status"] = load_status
            print(json.dumps(info, indent=2))