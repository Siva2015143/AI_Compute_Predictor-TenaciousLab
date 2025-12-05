#!/usr/bin/env python3
import json, sys
from pathlib import Path

def generate_report(run_dir: Path):
    metrics_file = run_dir / "metrics_log.jsonl"
    if not metrics_file.exists():
        print(f"[WARN] No metrics_log.jsonl found in {run_dir}")
        return

    best = {"epoch": None, "val_r2": -1e9}
    mse_vals, mae_vals, r2_vals = [], [], []

    with open(metrics_file, "r") as f:
        for line in f:
            rec = json.loads(line)
            mse_vals.append(rec["val_mse"])
            mae_vals.append(rec["val_mae"])
            r2_vals.append(rec["val_r2"])
            if rec["val_r2"] > best["val_r2"]:
                best = rec

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": best["epoch"],
        "best_val_r2": best["val_r2"],
        "final_val_mse": mse_vals[-1],
        "final_val_mae": mae_vals[-1],
        "avg_val_mse": sum(mse_vals)/len(mse_vals),
        "avg_val_r2": sum(r2_vals)/len(r2_vals),
    }

    out_path = run_dir / "training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Report written to: {out_path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pipeline_train_outputs/slm_small_run")
    generate_report(run)
