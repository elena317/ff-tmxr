"""Permanent Streamlit Community Cloud entrypoint for Tatermaxxer."""

from pathlib import Path


CURRENT_APP = Path(__file__).with_name("tatermaxxer_app_ui_v5.12.py")

if not CURRENT_APP.is_file():
    raise FileNotFoundError(f"Tatermaxxer UI entrypoint not found: {CURRENT_APP.name}")

exec(
    compile(CURRENT_APP.read_text(encoding="utf-8"), str(CURRENT_APP), "exec"),
    globals(),
    globals(),
)
