import importlib
import torch
import platform
from typing import Dict, List, Optional, Tuple

# Mapping dictionary PyPI names -> import names
PACKAGE_NAME_MAP: Dict[str, str] = {
    "pillow": "PIL",
    "opencv-python": "cv2",
    "segmentation-models-pytorch": "segmentation_models_pytorch",
}

# List of dependencies with versions
REQUIREMENTS: List[str] = [
    "dotenv==0.9.9",
    "pillow==12.0.0",
    "numpy==2.2.6",
    "matplotlib==3.10.7",
    "tqdm==4.67.1",
    "mlflow==3.6.0",
    "segmentation-models-pytorch==0.5.0",
    "opencv-python==4.12.0.88",
    "albumentations==2.0.8",
    "ipykernel==7.1.0",
    "rich==14.2.0",
    "imagehash==4.3.2"
]

# Try to import packaging for version comparison
try:
    from packaging import version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False
    print("[⚠] 'packaging' not installed - version comparison will be limited")

print("=" * 60)
print(f"🔍 Python environment check — {platform.python_version()}")
print("=" * 60)

def parse_requirement(req: str) -> Tuple[str, Optional[str]]:
    """Parses dependency string like 'package==version'."""
    if "==" in req:
        pkg_name, req_version = req.split("==", 1)
        return pkg_name.strip(), req_version.strip()
    return req.strip(), None

def check_package(requirement: str):
    """Checks package and its version. Format: 'package==version'"""
    pkg_name, required_version = parse_requirement(requirement)
    import_name = PACKAGE_NAME_MAP.get(pkg_name, pkg_name)
    
    try:
        module = importlib.import_module(import_name)
        actual_version = getattr(module, "__version__", None)
        
        if actual_version:
            if required_version:
                if HAS_PACKAGING:
                    try:
                        req_ver = version.parse(required_version)
                        act_ver = version.parse(actual_version)
                        
                        if act_ver == req_ver:
                            status, msg = "[✔]", f"{actual_version} (matches)"
                        elif act_ver > req_ver:
                            status, msg = "[⚠]", f"{actual_version} (newer than {required_version})"
                        else:
                            status, msg = "[✘]", f"{actual_version} (older than {required_version})"
                    except Exception as _:
                        status, msg = "[✔]", f"{actual_version} (required: {required_version})"
                else:
                    status, msg = "[✔]", f"{actual_version} (required: {required_version})"
            else:
                status, msg = "[✔]", actual_version
        else:
            status = "[✔]"
            msg = "built-in (no version info)"
            
        print(f"{status} {pkg_name} — {msg}")
                
    except ImportError as e:
        print(f"[✘] {pkg_name} — NOT FOUND ({e.__class__.__name__})")
    except Exception as e:
        print(f"[⚠] {pkg_name} — import error: {e}")

for req in REQUIREMENTS:
    check_package(req)

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