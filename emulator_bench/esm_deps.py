"""Make the fair-esm package importable when the environment ships another `esm` package.

The mldb env holds the Biohub `esm` fork, which has no `esm.pretrained.load_model_and_alphabet`.
KcatNet needs fair-esm 2.0.0. It is unpacked under `<out_root>/pydeps` and put first on the path.
"""
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

FAIR_ESM_VERSION = "2.0.0"
FAIR_ESM_SHA256 = "7dcb58828a401155d1457d071a03800bd1a50331ddf24ac79ad1dd0f2ea682a2"
FAIR_ESM_INFO_URL = f"https://pypi.org/pypi/fair-esm/{FAIR_ESM_VERSION}/json"


def _has_fair_esm() -> bool:
    try:
        return hasattr(importlib.import_module("esm.pretrained"), "load_model_and_alphabet")
    except ImportError:
        return False


def _install_wheel(target: Path) -> None:
    with urllib.request.urlopen(FAIR_ESM_INFO_URL) as response:
        wheel_url = next(f["url"] for f in json.load(response)["urls"] if f["filename"].endswith(".whl"))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}_"))
    try:
        wheel_path = tmp_dir / "fair_esm.whl"
        urllib.request.urlretrieve(wheel_url, wheel_path)
        if hashlib.sha256(wheel_path.read_bytes()).hexdigest() != FAIR_ESM_SHA256:
            raise RuntimeError(f"fair-esm wheel from {wheel_url} has an unexpected sha256")
        with zipfile.ZipFile(wheel_path) as wheel:
            wheel.extractall(tmp_dir / "pkg")
        (tmp_dir / "pkg").rename(target)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def setup_esm(out_root: Path) -> None:
    """Call before any `import esm`. Children inherit the environment set here."""
    os.environ.setdefault("TORCH_HOME", str(Path(out_root) / "torch_home"))
    pydeps = Path(out_root) / "pydeps"
    if not _has_fair_esm():
        for name in [m for m in sys.modules if m == "esm" or m.startswith("esm.")]:
            del sys.modules[name]
        if not (pydeps / "esm" / "pretrained.py").exists():
            if pydeps.exists():
                shutil.rmtree(pydeps)
            _install_wheel(pydeps)
        sys.path.insert(0, str(pydeps))
    if str(pydeps) in sys.path:
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(pydeps), os.environ.get("PYTHONPATH")]))
