"""Portable locations for the EV-charging rental-affordability replication code.

Set EVCS_DATA_ROOT to the original project data directory; otherwise data/ adjacent
 to this code/ directory is used. The variables below do not contain credentials.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent

def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()

DATA_ROOT = _env_path("EVCS_DATA_ROOT", PROJECT_ROOT / "data")
SHAPE_ROOT = _env_path("EVCS_SHAPE_ROOT", DATA_ROOT / "shapefiles")
BG_SHP = _env_path("EVCS_BG_SHP", SHAPE_ROOT / "BG_shapefile_cb" / "cb_2024_us_bg_500k.shp")
CBSA_SHP = _env_path("EVCS_CBSA_SHP", SHAPE_ROOT / "CBSA_shapefile" / "tl_2025_us_cbsa.shp")
HISTORY_DIR = _env_path("EVCS_HISTORY_DIR", DATA_ROOT / "intermediate" / "analysis_data_harmonized")
