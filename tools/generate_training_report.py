import json, os
from datetime import datetime
import pandas as pd

def generate_training_report():
    """
    Collects benchmark, config, and run history into one summary JSON + CSV.
    """
    base_dir = "pipeline_train_outputs/test_cpu"
    model_cfg = os.path.join(base_dir, "model_config.json")
    benchmark = "tools/benchmark_results.json"
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    report = {"timestamp": datetime.utcnow().isoformat()}

    # 1️⃣ Load model config
    if os.path.exists(model_cfg):
        with open(model_cfg) as f:
            report["model_config"] = json.load(f)

    # 2️⃣ Load benchmark
    if os.path.exists(benchmark):
        with open(benchmark) as f:
            report["benchmark"] = json.load(f)

    # 3️⃣ Load run logs
    db_path = "backend/runs.db"
    if os.path.exists(db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC", conn)
        conn.close()
        df.to_csv(os.path.join(report_dir, "run_history.csv"), index=False)
        report["run_count"] = len(df)

    # Save JSON
    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"[📄] Summary written to {report_dir}/summary.json")
    return report

if __name__ == "__main__":
    generate_training_report()
