"""
Shared helpers for the Streamlit app.

Each page under app/pages/ is now self-contained (it loads data, calibrates, and
renders its own controls inline), so this module only holds the small bits that
are still common across pages. At present that is page configuration.

Historical note: an earlier design centralised the sidebar controls, cached data
helpers, and a three-slot calibration panel here. Those were retired when the
pages were refactored to inline their own logic; the removed code lives in the
gitignored _graveyard.py at the repo root.

Imported by: app/pages/*.py (currently only 4_Risk_Dashboard.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so pages can `from shared import ...` and the
# service/model packages resolve when Streamlit runs a page as the entrypoint.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


def configure_page(title: str) -> None:
    """Set the Streamlit page title and wide layout. Call once at the top of a page."""
    st.set_page_config(page_title=title, layout="wide")
