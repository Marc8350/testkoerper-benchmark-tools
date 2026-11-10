import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
