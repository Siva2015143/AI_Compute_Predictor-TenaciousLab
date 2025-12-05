import json
from pathlib import Path

def preprocess_tokens(data_dir="data/compute_tokens", output="data/compute_tokens/normalized.jsonl"):
    """
    Preprocess and normalize all compute-token JSONL files.
    Ensures a fixed schema, safe numeric conversions, and robustness
    for large-scale training (1M+ records).
    """

    required_fields = [
        "total_tokens",
        "total_GFLOPs",
        "throughput_TFLOPs",
        "loss",
        "amplification",
        "b_hat",
    ]

    def safe_val(x, default=1e-6):
        """Convert safely to float, replacing 0, NaN, '', None, etc."""
        try:
            v = float(x)
            if v == 0.0 or v != v:  # zero or NaN
                return default
            return v
        except Exception:
            return default

    out = []
    bad = 0

    for f in Path(data_dir).glob("*.jsonl"):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                    clean = {}
                    # Force schema — never skip missing keys
                    for k in required_fields:
                        clean[k] = safe_val(rec.get(k, 0.0))

                    clean["timestamp"] = rec.get("timestamp", "")
                    clean["source_file"] = str(f)
                    out.append(clean)
                except Exception as e:
                    bad += 1

    # Smart fallbacks: never allow empty dataset
    if not out:
        raise RuntimeError(f"No valid records found in {data_dir} — check input files.")

    # Write normalized file
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    print(f"✅ Cleaned {len(out)} valid records, skipped {bad} bad lines → {output}")
    print(f"✅ Enforced {len(required_fields)} numeric features per record")

if __name__ == "__main__":
    preprocess_tokens()
