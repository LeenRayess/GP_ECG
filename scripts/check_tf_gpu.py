"""
Quick GPU checks: (1) PyTorch — same logic as scripts/train_benchmark_cnns.py.
(2) TensorFlow — optional; native Windows often has no TF GPU even when PyTorch works.
"""
import os

# --- PyTorch: what train_benchmark_cnns.py uses --------------------------------
import torch

print("=== PyTorch (used by train_benchmark_cnns.py) ===")
print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(torch.version, "cuda", None))

cuda_ok = torch.cuda.is_available()
device = torch.device("cuda" if cuda_ok else "cpu")
if cuda_ok:
    print("GPU: CUDA available — {} (device {})".format(
        torch.cuda.get_device_name(0), device
    ))
    print("  cuda device count:", torch.cuda.device_count())
else:
    print("GPU: CUDA not available — train_benchmark_cnns would use CPU.")

# --- TensorFlow: separate stack; not used by train_benchmark_cnns --------------
print()
print("=== TensorFlow (not used by train_benchmark_cnns.py) ===")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
try:
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("TF GPU: YES —", len(gpus), "device(s)")
        for i, d in enumerate(gpus):
            print(f"  [{i}] {d.name}")
    else:
        print("TF GPU: NO — (on native Windows, pip TF 2.11+ is often CPU-only).")
except ModuleNotFoundError:
    print("TensorFlow not installed — skipped.")
