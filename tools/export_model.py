
"""
TenaciousLab Inference Engine
-----------------------------
Handles:
  • Smart model loading (TorchScript or state_dict)
  • Deterministic fallback outputs for demo
  • Auto-export integration with tools/export_model.py
"""

import torch
import json
from pathlib import Path
from typing import Any, Dict

class ModelManager:
    def __init__(self, run_dir: str = "pipeline_train_outputs/slm_best_run"):
        self.run_dir = Path(run_dir)
        self.model = None
        self.model_config = {}
        self.state_dict_only = False
        self.device = torch.device("cpu")

        self._load_model()

    def _load_model(self):
        """Attempts to load model in three prioritized ways."""
        print(f"[ModelManager] Searching for model in: {self.run_dir}")

        scripted_path = self.run_dir / "exported_scripted.pt"
        best_model_path = self.run_dir / "best_model.pt"
        config_path = self.run_dir / "model_config.json"

        # Load model config if exists
        if config_path.exists():
            with open(config_path, "r") as f:
                self.model_config = json.load(f)

        # 1️⃣ Prefer exported scripted model
        if scripted_path.exists():
            print("[ModelManager] Loading TorchScript model …")
            self.model = torch.jit.load(scripted_path, map_location=self.device)
            self.model.eval()
            return

        # 2️⃣ Try loading state_dict only
        if best_model_path.exists():
            print("[ModelManager] Found state_dict model (no class def).")
            self.state_dict_only = True
            return

        # 3️⃣ Try auto-export if possible
        exporter = Path("tools/export_model.py")
        if exporter.exists():
            print("[ModelManager] Attempting auto-export via tools/export_model.py …")
            import subprocess
            subprocess.run(["python", str(exporter)], check=False)
            if scripted_path.exists():
                print("[ModelManager] Export successful, reloading.")
                self.model = torch.jit.load(scripted_path, map_location=self.device)
                self.model.eval()
                return

        print("[ModelManager] ⚠️ No valid model found — using deterministic fallback.")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns model prediction or deterministic dummy output if state_dict_only or missing model.
        """

        if self.model and not self.state_dict_only:
            # Assume features is dict of numeric values
            x = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                y = self.model(x).cpu().numpy().tolist()
            return {"prediction": y, "mode": "inference"}

        # Fallback deterministic demo prediction
        demo_value = round(sum(map(float, features.values())) % 0.97, 4)
        return {"prediction": [demo_value], "mode": "deterministic-demo"}

    def info(self):
        return {
            "has_model": self.model is not None,
            "state_dict_only": self.state_dict_only,
            "config": self.model_config,
            "run_dir": str(self.run_dir),
        }
