import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler

def normalize_and_save(input_path: str, output_path: str):
    """
    Reads a JSONL or JSON file of raw DataTokens,
    normalizes numeric fields (features only), and saves standardized version.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append(obj)

    # extract numeric features
    feats = []
    for d in data:
        feats.append([
            float(d.get("total_GFLOPs", 0.0)),
            float(d.get("total_tokens", 0)),
            float(d.get("loss", 0) or 0.0),
            float(d.get("latency_s", 0.0) or 0.0),
            float(d.get("amplification", 0.0) or 0.0),
        ])
    feats = np.array(feats, dtype=np.float32)

    # normalize
    scaler = StandardScaler()
    feats_norm = scaler.fit_transform(feats)

    # rebuild entries
    normalized = []
    for d, f in zip(data, feats_norm):
        normalized.append({
            "features": f.tolist(),
            "target": float(d.get("b_hat", 0.0) or 0.0),
            "meta": {
                "model": d.get("model_name", "unknown"),
                "timestamp": d.get("timestamp")
            }
        })

    # save as JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out:
        for obj in normalized:
            out.write(json.dumps(obj) + "\n")

    print(f"✅ Normalized {len(normalized)} tokens → {output_path}")
