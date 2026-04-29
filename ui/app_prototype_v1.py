"""
app_prototype_v1.py — Streamlit prototype (legacy)

This was the original single-file demo interface for Sentinel.
It has been superseded by the React + FastAPI stack in war-damage-ui/.

See ui/app_v2_notes.md for the full migration history.

NOTE: This file is preserved for reference only. The Streamlit dependency
has been removed from requirements.txt. To run this you would need to
install streamlit separately: pip install streamlit
"""

# import streamlit as st
# from pathlib import Path
# import sys
# ROOT = Path(__file__).parent.parent
# sys.path.insert(0, str(ROOT / "src"))
# from api import analyze, AnalyzeRequest
#
# st.set_page_config(page_title="Sentinel — War Damage Detection", layout="wide")
# st.title("Sentinel — War Damage Detection")
#
# with st.sidebar:
#     location = st.text_input("Location (lat,lng)", "49.9935,36.2304")
#     start_date = st.date_input("Start date")
#     end_date = st.date_input("End date")
#     infra = st.multiselect("Infrastructure", ["Hospitals","Schools","Water","Power"])
#     run = st.button("Analyze")
#
# if run:
#     with st.spinner("Running pipeline... (this may take up to 15 minutes)"):
#         req = AnalyzeRequest(
#             location=location,
#             start_date=str(start_date),
#             end_date=str(end_date),
#             infrastructure_type=",".join(infra).lower() if infra else "all",
#         )
#         result = analyze(req)
#     st.success(f"Done — {result.metrics['zones_flagged']} zones flagged")
#     if result.post_image_b64:
#         import base64
#         from PIL import Image
#         import io
#         img = Image.open(io.BytesIO(base64.b64decode(result.post_image_b64)))
#         st.image(img, caption="Post-war satellite + damage mask", use_column_width=True)
#     st.json(result.metrics)

print("This is a preserved prototype. See ui/app_v2_notes.md for context.")
