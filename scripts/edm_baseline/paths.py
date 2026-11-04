import os
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("EDM_WORKSPACE", str(CODE_ROOT))).expanduser().resolve()
CONFIG_ROOT = CODE_ROOT / "config/edm_baseline"
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT / "src"))
