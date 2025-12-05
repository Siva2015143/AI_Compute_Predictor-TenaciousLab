# backend/train_engine.py
import threading
import time
import torch
from torch import nn, optim
from neuralop.models import FNO
from .utils import log_run, update_run, save_model
import os

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "fno_model.pt")

def quick_synthetic_train(run_name="quick_run", epochs=3, batch_size=4, lr=1e-3, device="cpu"):
    run_id = log_run(run_name, status="queued", metrics={})
    def _train():
        update_run(run_id, status="running")
        device_local = torch.device(device)
        model = FNO(n_modes=(12,12), hidden_channels=32, in_channels=3, out_channels=1).to(device_local)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        try:
            for ep in range(1, epochs+1):
                model.train()
                # synthetic batch
                x = torch.randn(batch_size, 3, 64, 64, device=device_local)
                y = x.mean(dim=1, keepdim=True)  # toy target
                opt.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()
                # log progress (every epoch)
                update_run(run_id, status="running", metrics={"epoch": ep, "loss": float(loss.item())})
                print(f"[train] epoch {ep} loss {loss.item():.6f}")
                time.sleep(0.5)
            # save model
            save_model(model, MODEL_SAVE_PATH)
            update_run(run_id, status="completed", metrics={"final_loss": float(loss.item())})
            print("[train] training complete. saved to", MODEL_SAVE_PATH)
        except Exception as e:
            update_run(run_id, status="error", notes=str(e))
            print("[train] error:", e)

    t = threading.Thread(target=_train, daemon=True)
    t.start()
    return run_id
