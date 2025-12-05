import os
import json
import torch
from torch.utils.data import Dataset

class ComputeTokenDataset(Dataset):
    """
    Dataset loader for normalized compute token data.
    Expects JSONL files with 'features' and 'target' fields.
    """
    def __init__(self, data_dir):
        self.samples = []
        self.feature_dim = 0
        self.target_dim = 0

        for file in os.listdir(data_dir):
            if not file.endswith(".jsonl"):
                continue
            path = os.path.join(data_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        x = torch.tensor(item["features"], dtype=torch.float32)
                        y = torch.tensor([item["target"]], dtype=torch.float32)
                        self.samples.append((x, y))
                    except Exception:
                        continue

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")

        self.feature_dim = self.samples[0][0].shape[-1]
        self.target_dim = self.samples[0][1].shape[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def generate_data_config(dataset: ComputeTokenDataset, output_path="config/data_config.json", dataset_path="data/compute_tokens"):
    """
    Save dataset config for use in training and pipeline.
    Includes dataset path and dimensionality info.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cfg = {
        "dataset_path": os.path.abspath(dataset_path),
        "num_samples": len(dataset),
        "feature_dim": dataset.feature_dim,
        "target_dim": dataset.target_dim
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"💾 Saved data_config.json → {output_path}")
    return cfg
