import math
import json
from config.model_params import (
    instantiate_configs,
    estimate_training_flops_tokens,
    humanize_params,
)

# 🧠 Available presets
PRESETS = [
    "slm_small",
    "slm_medium",
    "transformer_tiny",
    "mini_gpt_125M",
    "caif_mini_lab",
]


def plan_experiment(preset: str, input_dim=5, output_dim=1, dataset_tokens=9000):
    """
    Estimate model scale, FLOPs, and dataset match for a given preset.
    Returns a dictionary with all stats.
    """
    cfgs = instantiate_configs(preset, input_dim, output_dim)
    params = cfgs["param_count"]

    est = estimate_training_flops_tokens(params)
    needed_tokens = est["tokens"]
    flops = est["flops_TFLOPS"]
    dataset_match = min(100.0, (dataset_tokens / needed_tokens) * 100)

    result = {
        "preset": preset,
        "params": params,
        "params_human": humanize_params(params),
        "needed_tokens": needed_tokens,
        "flops_TFLOPS": flops,
        "dataset_tokens": dataset_tokens,
        "dataset_match": dataset_match,
    }

    return result


def analyze_and_recommend(results):
    """
    Analyze all presets and recommend the best-fitting one for current dataset size.
    """
    print("\n📊 MODEL ESTIMATION SUMMARY\n" + "=" * 70)
    for r in results:
        print(f"🧩 {r['preset']:<20} | Params: {r['params_human']:<8} | "
              f"Match: {r['dataset_match']:>6.2f}% | FLOPs: {r['flops_TFLOPS']:.3f} TFLOPs")

    print("=" * 70)

    # Rule-based scoring:
    # Prefer models with dataset_match between 10% and 80%
    # Penalize too large (>100%) or too small (<5%) matches
    def score(r):
        match = r["dataset_match"]
        if 10 <= match <= 80:
            return 1.0 - abs(45 - match) / 45  # closer to middle = better
        elif match > 80:
            return 0.3  # model too small
        else:
            return 0.1  # model too big for data

    scored = [(score(r), r) for r in results]
    best = max(scored, key=lambda x: x[0])[1]

    print(f"\n✅ Recommended Model Preset: **{best['preset']}**")
    print(f"   → Params: {best['params_human']}")
    print(f"   → Match:  {best['dataset_match']:.2f}%")
    print(f"   → FLOPs:  {best['flops_TFLOPS']:.3f} TFLOPs")

    # Give textual advice
    if best["dataset_match"] < 15:
        print("⚠️ Dataset is small; you may overfit. Try even smaller or regularized MLP.")
    elif best["dataset_match"] > 80:
        print("⚠️ Dataset is large relative to model — might underfit.")
    else:
        print("🚀 Excellent balance between capacity and data size.")

    return best


def save_results(results, path="reports/model_estimates.json"):
    """Save full estimation results as a JSON file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {path}")


if __name__ == "__main__":
    INPUT_DIM = 5
    OUTPUT_DIM = 1
    DATASET_TOKENS = 9000

    print("\n🧮 Running full model estimation and selection...\n")
    results = [plan_experiment(p, INPUT_DIM, OUTPUT_DIM, DATASET_TOKENS) for p in PRESETS]

    best = analyze_and_recommend(results)
    save_results(results)

    print("\n🎯 Final Recommended Preset:", best["preset"])
    print("✅ Estimation completed.\n")
