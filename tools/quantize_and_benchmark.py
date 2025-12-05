import torch
import time
import os
import json


def quantize_and_benchmark(model_path="pipeline_train_outputs/test_cpu/best_model.pt"):
    """
    Quantize the trained model dynamically (Linear layers only)
    and benchmark CPU inference latency.
    """
    if not os.path.exists(model_path):
        print(f"[❌] Model not found: {model_path}")
        return

    print(f"[ℹ️] Loading model from: {model_path}")
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Benchmark
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        start = time.time()
        _ = quantized_model(x)
        latency = (time.time() - start) * 1000  # in ms

    print(f"[✅] Inference latency (quantized): {latency:.2f} ms")

    result = {
        "model_path": model_path,
        "latency_ms": latency,
        "quantized": True,
    }

    os.makedirs("tools", exist_ok=True)
    out_path = "tools/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[💾] Results saved to {out_path}")


if __name__ == "__main__":
    quantize_and_benchmark()
