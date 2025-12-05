import torch, json, os, time

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_training_flops_tokens(model, dataset_size: int = 10000):
    """Rough, order-of-magnitude compute estimate."""
    param_count = count_parameters(model)
    est_flops = param_count * dataset_size * 2  # forward + backward
    est_tokens = dataset_size * 32              # proxy
    return est_flops, est_tokens

def save_model_config(model, cfg, out_dir):
    """Save model config + telemetry next to checkpoints."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    from datetime import datetime
    model_info = {
        "timestamp": datetime.now().isoformat(),
        "model_name": cfg.model,
        "hidden_sizes": getattr(cfg, "hidden_sizes", None),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "activation": getattr(cfg, "activation", None),
        "dropout": getattr(cfg, "dropout", None),
    }

    param_count = count_parameters(model)
    est_flops, est_tokens = estimate_training_flops_tokens(model)
    model_info.update({
        "param_count": param_count,
        "est_flops": est_flops,
        "est_tokens": est_tokens
    })

    out_path = os.path.join(out_dir, "model_config.json")
    with open(out_path, "w") as f:
        json.dump(model_info, f, indent=2)

    # Also append summary to train.log
    with open(os.path.join(out_dir, "train.log"), "a") as f:
        f.write(f"\n[MODEL CONFIG]\n{json.dumps(model_info, indent=2)}\n")

    return out_path
