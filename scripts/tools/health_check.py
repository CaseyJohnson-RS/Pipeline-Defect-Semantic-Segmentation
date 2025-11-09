import importlib
import torch
import platform

REQUIRED_PACKAGES = [
    "dotenv",
    "PIL",
    "numpy",
    "matplotlib",
    "tqdm",
    "mlflow",
    "segmentation_models_pytorch",
    "cv2",
    "albumentations",
    "ipykernel",
    "rich"
]

print("=" * 60)
print(f"🔍 Python environment check — {platform.python_version()}")
print("=" * 60)

def check_package(name):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "built-in")
        print(f"[✔] {name} — version: {version}")
    except ImportError as e:
        print(f"[✘] {name} — NOT FOUND ({e.__class__.__name__})")
    except Exception as e:
        print(f"[⚠] {name} — import error: {e}")

for pkg in REQUIRED_PACKAGES:
    check_package(pkg)

print("\n" + "=" * 60)
print("🔍 PyTorch check")
print("=" * 60)

try:
    print(f"[✔] torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        x = torch.rand((2, 2), device='cuda')
        print(f"Tensor test on GPU OK → {x}")
    else:
        x = torch.rand((2, 2))
        print(f"Tensor test on CPU OK → {x}")
except Exception as e:
    print(f"[✘] PyTorch error: {e}")

print("\n" + "=" * 60)
print("✅ Environment check complete.")
print("=" * 60)
