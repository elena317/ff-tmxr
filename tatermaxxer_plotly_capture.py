"""Registration wrapper for the browser-side Plotly capture component."""

from pathlib import Path

import streamlit.components.v1 as components


plotly_capture = components.declare_component(
    "tatermaxxer_plotly_capture",
    path=str(Path(__file__).resolve().with_name("plotly_capture_component")),
)
