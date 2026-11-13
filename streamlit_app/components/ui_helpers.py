from __future__ import annotations

import pandas as pd
import streamlit as st


def apply_theme() -> None:
    """Apply focused presentation polish on top of Streamlit's supported light theme."""
    st.markdown("""<style>
    :root {
        --km-forest: #1F5A45;
        --km-text: #24332D;
        --km-secondary: #58665F;
        --km-page: #F5F7F4;
        --km-card: #FFFFFF;
        --km-border: #DCE4DD;
        --km-pale: #EAF2EB;
    }
    .stApp { background: var(--km-page); color: var(--km-text); }
    .stApp p, .stApp li, .stApp [data-testid="stCaptionContainer"] { color: var(--km-secondary); }
    .stApp h1, .stApp h2, .stApp h3 { color: var(--km-forest); letter-spacing: -0.018em; }
    .stApp h1 { margin-bottom: 0.2rem; }
    .stApp h2, .stApp h3 { margin-top: 1.35rem; }

    [data-testid="stSidebar"] { background: #EEF4EF; border-right: 1px solid var(--km-border); }
    [data-testid="stSidebar"] * { color: var(--km-text); }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--km-secondary); }
    [data-testid="stSidebar"] [data-baseweb="radio"] { gap: 0.2rem; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background: transparent; border: 1px solid transparent; border-radius: 7px;
        padding: 0.4rem 0.5rem; margin: 0; width: 100%;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:has(input:checked) {
        background: var(--km-card); border-color: #B9CCBD; box-shadow: 0 1px 2px rgba(31, 90, 69, 0.08);
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:has(input:checked) div { color: var(--km-forest); font-weight: 600; }

    [data-testid="stMetric"] {
        min-height: 104px; background: var(--km-card); border: 1px solid var(--km-border);
        border-radius: 10px; padding: 0.85rem 0.95rem; box-shadow: 0 1px 2px rgba(36, 51, 45, 0.04);
    }
    [data-testid="stMetricLabel"] p { color: var(--km-secondary) !important; font-size: 0.83rem; font-weight: 500; }
    [data-testid="stMetricValue"] > div { color: var(--km-text) !important; font-size: clamp(1.1rem, 1.65vw, 1.65rem); font-weight: 650; line-height: 1.25; }
    [data-testid="stMetricDelta"] { color: var(--km-forest) !important; }

    .stApp [data-testid="stSelectbox"] label, .stApp [data-testid="stMultiSelect"] label,
    .stApp [data-testid="stSlider"] label, .stApp [data-testid="stRadio"] > label,
    .stApp [data-testid="stExpander"] summary { color: var(--km-text) !important; font-weight: 600; }
    .stApp [data-baseweb="select"] > div {
        background: var(--km-card); border-color: #B7C5BA; color: var(--km-text);
        min-height: 2.5rem;
    }
    .stApp [data-baseweb="select"] span, .stApp [data-baseweb="select"] input { color: var(--km-text) !important; }
    .stApp [data-baseweb="tag"] { background: var(--km-pale); color: var(--km-forest); }
    .stApp [data-baseweb="tag"] span { color: var(--km-forest) !important; }
    .stApp [data-baseweb="popover"] [role="option"] { color: var(--km-text); }
    .stApp [data-baseweb="slider"] [role="slider"] { background: var(--km-forest); }
    .stApp [data-baseweb="slider"] div { color: var(--km-text); }
    .stApp [data-baseweb="select"] > div:focus-within { border-color: var(--km-forest); box-shadow: 0 0 0 1px var(--km-forest); }

    [data-testid="stAlert"] { border-radius: 8px; border: 1px solid var(--km-border); }
    [data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--km-text) !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--km-border); border-radius: 8px; overflow: hidden; }
    [data-testid="stDataFrame"] * { color: var(--km-text); }
    [data-testid="stExpander"] { background: var(--km-card); border: 1px solid var(--km-border); border-radius: 8px; }
    [data-testid="stExpander"] details { background: var(--km-card); }
    .research-note { border-left: 4px solid var(--km-forest); background: var(--km-pale); color: var(--km-text); padding: .8rem 1rem; border-radius: 5px; }
    .workflow-step { border-left: 2px solid #86A38B; padding: .38rem 0 .38rem .85rem; margin: .1rem 0; color: var(--km-text); }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { border-color: var(--km-forest); color: var(--km-forest); border-radius: 7px; font-weight: 600; }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover { background: var(--km-pale); border-color: var(--km-forest); }

    @media (max-width: 900px) {
        [data-testid="stMetric"] { min-height: 88px; }
    }
    </style>""", unsafe_allow_html=True)


def metric(label: str, value: object, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def empty(message: str) -> None:
    st.info(message)


def dataframe(frame: pd.DataFrame, **kwargs: object) -> None:
    st.dataframe(frame, use_container_width=True, hide_index=True, **kwargs)


def caveat(text: str) -> None:
    st.markdown(f'<div class="research-note">{text}</div>', unsafe_allow_html=True)
