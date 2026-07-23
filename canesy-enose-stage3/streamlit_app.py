import streamlit as st
from pathlib import Path

# Set full-width dark layout matching index.html
st.set_page_config(
    page_title="CaNeSy-eNose Monitoring Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to hide Streamlit margins, headers, and footers for a 100% immersive experience
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div.block-container {
            padding: 0rem !important;
            margin: 0rem !important;
            max-width: 100% !important;
        }
        iframe {
            border: none !important;
            width: 100vw !important;
            height: 100vh !important;
        }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
index_path = BASE_DIR / "static" / "index.html"

if index_path.exists():
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=1000, scrolling=True)
else:
    st.error("static/index.html not found.")
