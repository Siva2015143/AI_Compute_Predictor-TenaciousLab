# AI Compute Predictor

**Predict neural network performance before training.**

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

</div>

---

## What is this?

A lightweight system that predicts how well a neural network will perform—before you train it.

Instead of training models for specific tasks, this trains a model to understand compute behavior patterns.

---

## Why it matters

Training neural networks is expensive and time-consuming. Most experiments don't work as expected.

This tool helps you:
- Estimate performance before training
- Compare different configurations
- Save compute resources

---

## Quick Start
```bash
# Clone and install
git clone https://github.com/Siva2015143/AI_Compute_Predictor-TenaciousLab.git
cd AI_Compute_Predictor-TenaciousLab
pip install -r requirements.txt

# Make a prediction
python engine/inference_engine.py --load test_cpu --predict 25000 400 0.45 1.1 0.95
```

**Output:**
```json
{
  "ok": true,
  "result": [[1915.32]]
}
```

---

## How it works

1. Collect data from past training runs
2. Train a small predictor model (~167K parameters)
3. Use it to predict new experiments

The model learns patterns between inputs (tokens, architecture) and outputs (performance metrics).

---

## Project Structure
```
├── engine/          # Inference and predictions
├── tokens/          # Data processing
├── training/        # Model training code
├── tools/           # Utilities
├── runs/            # Trained models
└── reports/         # Generated reports
```

---

## Usage

### List available models
```bash
python engine/inference_engine.py --list
```

### View model information
```bash
python engine/inference_engine.py --show-model-info test_cpu
```

### Make predictions
```bash
python engine/inference_engine.py --load test_cpu --predict [parameters]
```

---

## Key Features

- Fast predictions (<2ms on CPU)
- Small model size (~167K parameters)
- TorchScript export for deployment
- Simple JSON-based data format

---

## Technical Details

**Model:**
- Architecture: 3-layer MLP
- Parameters: ~167K
- Framework: PyTorch
- Export: TorchScript

**Training:**
- Time: ~2 minutes on CPU
- Optimizer: AdamW
- Loss: MSE

---

## Use Cases

- Estimate training costs before running experiments
- Compare different model configurations
- Prioritize which experiments to run

---

## What I learned

- Building meta-learning systems
- Production model deployment with TorchScript
- Working with structured experiment data
- Scaling law relationships in neural networks

---

## Contact

**Sivamani Battala**

📧 sivamani6104@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/sivamani-battala)  
💻 [GitHub](https://github.com/Siva2015143)

---

<div align="center">

*A practical tool for smarter ML experimentation*

</div>