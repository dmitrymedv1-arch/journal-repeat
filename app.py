import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import re
import time
import json
import random
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set, Any
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ratelimit import limits, sleep_and_retry
import logging
import io
from pathlib import Path
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PDF IMPORTS - FIXED
# ============================================================================
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as reportlab_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("reportlab not installed. PDF export will be disabled. Install with: pip install reportlab")

# ============================================================================
# SCIENTIFIC COLOR PALETTES FOR PUBLICATIONS
# ============================================================================

# Colorblind-friendly palettes for scientific publications - DISCRETE
COLOR_PALETTES_SCIENTIFIC_DISCRETE = {
    'nature': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85'],
    'science': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'matplotlib': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'colorblind': ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000', '#999999', '#882255'],
    'vibrant': ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB', '#999933', '#882255', '#AA4499'],
    'ggplot': ['#F8766D', '#7CAE00', '#00BFC4', '#C77CFF', '#FF61CC', '#A3A500', '#00A9FF', '#E68613', '#B983FF', '#00CDCD'],
    'seaborn': ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#8C8C8C', '#A1A9B0', '#B0A1BA', '#F0B27A'],
    'tableau': ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'],
    'brewer_Set1': ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999', '#66C2A5'],
    'brewer_Set2': ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3', '#8DD3C7', '#FFFFB3'],
    'brewer_Set3': ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD'],
    'brewer_Paired': ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A23', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A'],
    'brewer_Dark2': ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666', '#1B9E77', '#D95F02'],
    'brewer_Accent': ['#7FC97F', '#BEAED4', '#FDC086', '#FFFF99', '#386CB0', '#F0027F', '#BF5B17', '#666666', '#1B9E77', '#D95F02'],
    'custom_bright': ['#FF0010', '#10FF00', '#0010FF', '#FFD700', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#008000', '#800000']
}

# Gradient palettes for continuous data - 15 VARIANTS
GRADIENT_PALETTES = {
    'viridis': plt.cm.viridis,
    'plasma': plt.cm.plasma,
    'inferno': plt.cm.inferno,
    'magma': plt.cm.magma,
    'cividis': plt.cm.cividis,
    'coolwarm': plt.cm.coolwarm,
    'RdYlBu': plt.cm.RdYlBu,
    'Spectral': plt.cm.Spectral,
    'Blues': plt.cm.Blues,
    'Reds': plt.cm.Reds,
    'Greens': plt.cm.Greens,
    'Purples': plt.cm.Purples,
    'Oranges': plt.cm.Oranges,
    'Greys': plt.cm.Greys,
    'YlOrRd': plt.cm.YlOrRd
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Publication Clustering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# EXTENDED COLOR PALETTES (10 VARIANTS) - FOR UI ONLY
# ============================================================================

UI_COLOR_PALETTES = [
    {
        'name': 'Ocean Blues',
        'primary': '#0066cc',
        'secondary': '#00a8cc',
        'gradient_start': '#0066cc',
        'gradient_end': '#00a8cc',
        'accent1': '#004d66',
        'accent2': '#66c2ff',
        'background': '#f0f8ff',
        'card_bg': '#ffffff',
        'text': '#003333',
        'border': '#b3e0ff'
    },
    {
        'name': 'Forest Green',
        'primary': '#2e7d32',
        'secondary': '#81c784',
        'gradient_start': '#1b5e20',
        'gradient_end': '#4caf50',
        'accent1': '#0d3d0d',
        'accent2': '#a5d6a7',
        'background': '#f1f8e9',
        'card_bg': '#ffffff',
        'text': '#1b3b1b',
        'border': '#c8e6c9'
    },
    {
        'name': 'Sunset Orange',
        'primary': '#e65100',
        'secondary': '#ffb74d',
        'gradient_start': '#bf360c',
        'gradient_end': '#ff9800',
        'accent1': '#8d2f00',
        'accent2': '#ffe082',
        'background': '#fff3e0',
        'card_bg': '#ffffff',
        'text': '#4a2c00',
        'border': '#ffe0b2'
    },
    {
        'name': 'Royal Purple',
        'primary': '#6a1b9a',
        'secondary': '#ba68c8',
        'gradient_start': '#4a148c',
        'gradient_end': '#9c27b0',
        'accent1': '#311b92',
        'accent2': '#ce93d8',
        'background': '#f3e5f5',
        'card_bg': '#ffffff',
        'text': '#2a0f3a',
        'border': '#e1bee7'
    },
    {
        'name': 'Ruby Red',
        'primary': '#b71c1c',
        'secondary': '#ef5350',
        'gradient_start': '#8b0000',
        'gradient_end': '#d32f2f',
        'accent1': '#5a0000',
        'accent2': '#ffcdd2',
        'background': '#ffebee',
        'card_bg': '#ffffff',
        'text': '#3b0000',
        'border': '#ffcdd2'
    },
    {
        'name': 'Amber Gold',
        'primary': '#ff8f00',
        'secondary': '#ffb300',
        'gradient_start': '#ff6f00',
        'gradient_end': '#ffa000',
        'accent1': '#b26500',
        'accent2': '#ffe082',
        'background': '#fff8e1',
        'card_bg': '#ffffff',
        'text': '#5c3f00',
        'border': '#ffecb3'
    },
    {
        'name': 'Teal Marine',
        'primary': '#00796b',
        'secondary': '#4db6ac',
        'gradient_start': '#004d40',
        'gradient_end': '#009688',
        'accent1': '#00332e',
        'accent2': '#b2dfdb',
        'background': '#e0f2f1',
        'card_bg': '#ffffff',
        'text': '#00332e',
        'border': '#b2dfdb'
    },
    {
        'name': 'Lavender Mist',
        'primary': '#7e57c2',
        'secondary': '#b085f5',
        'gradient_start': '#512da8',
        'gradient_end': '#9575cd',
        'accent1': '#311b92',
        'accent2': '#d1c4e9',
        'background': '#ede7f6',
        'card_bg': '#ffffff',
        'text': '#1e0f3a',
        'border': '#d1c4e9'
    },
    {
        'name': 'Crimson Rose',
        'primary': '#c2185b',
        'secondary': '#f06292',
        'gradient_start': '#880e4f',
        'gradient_end': '#e91e63',
        'accent1': '#560027',
        'accent2': '#f8bbd0',
        'background': '#fce4ec',
        'card_bg': '#ffffff',
        'text': '#33001a',
        'border': '#f8bbd0'
    },
    {
        'name': 'Slate Gray',
        'primary': '#546e7a',
        'secondary': '#90a4ae',
        'gradient_start': '#29434e',
        'gradient_end': '#607d8b',
        'accent1': '#1c313a',
        'accent2': '#cfd8dc',
        'background': '#eceff1',
        'card_bg': '#ffffff',
        'text': '#1c313a',
        'border': '#cfd8dc'
    }
]

# Select random UI palette at startup
if 'ui_color_palette' not in st.session_state:
    st.session_state['ui_color_palette'] = random.choice(UI_COLOR_PALETTES)

# Initialize color palette selections for plots
if 'discrete_palette_name' not in st.session_state:
    st.session_state['discrete_palette_name'] = 'nature'
if 'gradient_palette_name' not in st.session_state:
    st.session_state['gradient_palette_name'] = 'viridis'

ui_colors = st.session_state['ui_color_palette']

# ============================================================================
# SCIENTIFIC STYLE FOR PLOTS (INDEPENDENT FROM UI)
# ============================================================================

SCIENTIFIC_STYLE = {
    # Font sizes and weights
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    
    # Axes appearance
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    
    # Tick parameters
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    
    # Legend
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    
    # Figure
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    
    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
}

# Apply scientific style
plt.style.use('default')
plt.rcParams.update(SCIENTIFIC_STYLE)

# ============================================================================
# CUSTOM UI STYLES (ONLY FOR INTERFACE, NOT PLOTS)
# ============================================================================

st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {ui_colors['gradient_start']} 0%, {ui_colors['gradient_end']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }}
    
    .step-card {{
        background: linear-gradient(135deg, {ui_colors['gradient_start']}15 0%, {ui_colors['gradient_end']}15 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid {ui_colors['primary']};
        margin-bottom: 15px;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.04);
    }}
    
    .metric-card {{
        background: {ui_colors['card_bg']};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        border: 1px solid {ui_colors['border']};
        height: 100%;
        min-height: 90px;
    }}
    
    .metric-card h4 {{
        font-size: 0.85rem;
        margin: 0 0 8px 0;
        color: {ui_colors['accent1']};
    }}
    
    .metric-card .value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {ui_colors['text']};
        line-height: 1.2;
    }}
    
    .result-card {{
        background: {ui_colors['card_bg']};
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        border-left: 3px solid {ui_colors['primary']};
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }}
    
    .info-message {{
        background: linear-gradient(135deg, {ui_colors['primary']}15 0%, {ui_colors['secondary']}15 100%);
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid {ui_colors['primary']};
        font-size: 0.9rem;
        margin: 10px 0;
    }}
    
    .warning-message {{
        background: linear-gradient(135deg, #FF980015 0%, #EF6C0015 100%);
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #FF9800;
        font-size: 0.9rem;
        margin: 10px 0;
    }}
    
    .filter-section {{
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #dee2e6;
    }}
    
    .filter-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {ui_colors['accent1']};
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid {ui_colors['primary']};
    }}
    
    .filter-stats {{
        background: {ui_colors['card_bg']};
        border-radius: 8px;
        padding: 12px;
        border: 1px solid {ui_colors['border']};
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    
    .year-checkbox-container {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 15px;
    }}
    
    .year-checkbox-item {{
        background: white;
        border-radius: 6px;
        padding: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    .year-checkbox-item:hover {{
        border-color: {ui_colors['primary']};
        background-color: {ui_colors['background']};
    }}
    
    .year-checkbox-item.selected {{
        background: linear-gradient(135deg, {ui_colors['primary']}15 0%, {ui_colors['secondary']}15 100%);
        border-color: {ui_colors['primary']};
        color: {ui_colors['primary']};
        font-weight: 600;
    }}
    
    .scientific-plot {{
        background: white;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid {ui_colors['border']};
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }}
    
    .scientific-plot h4 {{
        color: {ui_colors['accent1']};
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid {ui_colors['primary']};
    }}
    
    .back-button {{
        background-color: {ui_colors['background']};
        color: {ui_colors['primary']};
        border: 2px solid {ui_colors['primary']};
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    
    .back-button:hover {{
        background-color: {ui_colors['primary']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# OPENALEX API CONFIGURATION
# ============================================================================

OPENALEX_BASE_URL = "https://api.openalex.org"
MAILTO = "your-email@example.com"
POLITE_POOL_HEADER = {'User-Agent': f'Publication-Clustering (mailto:{MAILTO})'}

# Rate limit settings
RATE_LIMIT_PER_SECOND = 8
CURSOR_PAGE_SIZE = 200
MAX_RETRIES = 3
INITIAL_DELAY = 1
MAX_DELAY = 60

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean text from HTML tags and extra characters"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def expand_wildcard(term: str) -> str:
    """
    Convert wildcard queries (e.g., "electroly*") to OR query for OpenAlex.
    electroly* -> (electrolyte OR electrolysis OR electrolyzer OR electrolytic OR electrolytical)
    """
    if '*' not in term:
        return term
    
    # Common expansions for typical roots
    expansions = {
        'electroly*': ['electrolyte', 'electrolysis', 'electrolyzer', 'electrolytic', 'electrolytical'],
        'cataly*': ['catalyst', 'catalysis', 'catalytic', 'catalyze', 'catalyser'],
        'polymer*': ['polymer', 'polymeric', 'polymerization', 'polymerisation', 'polymerase'],
        'nanomater*': ['nanomaterial', 'nanomaterials', 'nanostructured', 'nanoparticle'],
        'biomolec*': ['biomolecule', 'biomolecular', 'biomolecules'],
        'spectroscop*': ['spectroscopy', 'spectroscopic', 'spectroscope'],
        'chromatogra*': ['chromatography', 'chromatographic', 'chromatogram'],
        'thermodynam*': ['thermodynamics', 'thermodynamic', 'thermodynamical'],
        'quantum*': ['quantum', 'quantized', 'quantization'],
        'molecular*': ['molecular', 'molecule', 'molecules'],
    }
    
    # Check known patterns
    for pattern, expansions_list in expansions.items():
        if term.lower() == pattern.lower():
            return '(' + ' OR '.join(expansions_list) + ')'
    
    # For unknown patterns - general rule
    # Remove asterisk and look for common endings
    base = term.rstrip('*')
    common_endings = ['', 's', 'es', 'ing', 'ed', 'tion', 'tions', 'al', 'ic', 'ize', 'ise', 'izer', 'iser', 'lysis', 'lytic']
    expanded_terms = [base + ending for ending in common_endings if base + ending]
    
    if len(expanded_terms) > 1:
        return '(' + ' OR '.join(expanded_terms) + ')'
    
    return term

def parse_query_terms(term: str) -> str:
    """
    Parse search term for OpenAlex API.
    Enhanced version with proper phrase and wildcard handling.
    Supports:
    - Simple words
    - Phrases in quotes
    - Logical operators: AND, OR, NOT
    - Wildcard (*) queries
    """
    term = term.strip()
    
    # Check for wildcard
    if '*' in term and not (term.startswith('"') and term.endswith('"')):
        return expand_wildcard(term)
    
    # If it's a quoted phrase, leave as is
    if term.startswith('"') and term.endswith('"'):
        return term
    
    # If there's OR operator (case insensitive)
    if ' OR ' in term.upper():
        # Split by OR, process each part
        parts = re.split(r'\s+OR\s+', term, flags=re.IGNORECASE)
        processed_parts = []
        for part in parts:
            part = part.strip()
            if ' ' in part and not (part.startswith('"') and part.endswith('"')):
                # If part has spaces, wrap in quotes
                processed_parts.append(f'"{part}"')
            else:
                processed_parts.append(part)
        return ' OR '.join(processed_parts)
    
    # If there are spaces but no OR, it's a phrase - use quotes
    if ' ' in term:
        return f'"{term}"'
    
    return term

def create_metric_card(title: str, value, icon: str = "📊"):
    """Create compact metric card with formatted numbers"""
    # Format large numbers with commas
    if isinstance(value, (int, float)):
        formatted_value = f"{value:,}"
    else:
        formatted_value = str(value)
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>{icon} {title}</h4>
        <div class="value">{formatted_value}</div>
    </div>
    """, unsafe_allow_html=True)

def create_result_card(work: dict, index: int, topic: str):
    """Create result card"""
    citation_count = work.get('cited_by_count', 0)
    
    # Determine citation badge color
    if citation_count == 0:
        badge_color = "#4CAF50"
        badge_text = "0 citations"
    elif citation_count <= 3:
        badge_color = "#4CAF50"
        badge_text = f"{citation_count} citation{'s' if citation_count > 1 else ''}"
    elif citation_count <= 10:
        badge_color = "#FF9800"
        badge_text = f"{citation_count} citations"
    else:
        badge_color = "#f44336"
        badge_text = f"{citation_count} citations"
    
    oa_badge = '🔓' if work.get('is_oa') else '🔒'
    doi_url = work.get('doi_url', '')
    title = work.get('title', 'No title')
    authors = ', '.join(work.get('authors', [])[:2])
    if len(work.get('authors', [])) > 2:
        authors += ' et al.'
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <div>
                <span style="font-weight: 600; color: {ui_colors['primary']}; margin-right: 8px;">{topic} #{index}</span>
                <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                    {badge_text}
                </span>
                <span style="background: #e3f2fd; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-left: 5px;">
                    Score: {work.get('relevance_score', 0):.2f}
                </span>
            </div>
            <span style="color: #666; font-size: 0.8rem;">{work.get('publication_year', '')}</span>
        </div>
        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 5px; line-height: 1.3;">{title}</div>
        <div style="color: #555; font-size: 0.85rem; margin-bottom: 5px;">👤 {authors}</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
            <span>{oa_badge} {work.get('journal', '')[:30]}</span>
            <a href="{doi_url}" target="_blank" style="color: {ui_colors['primary']}; text-decoration: none; font-size: 0.85rem;">
                🔗 View Article
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def navigation_buttons(show_back: bool = True, show_new: bool = True):
    """Display navigation buttons"""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if show_back and st.session_state.step > 1:
            if st.button("← Back", key="back_btn", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()
    
    with col2:
        if show_new:
            if st.button("🔄 New Search", key="new_btn", use_container_width=True):
                # Clear session but keep terms for Step 1
                level1 = st.session_state.get('level1_input', '')
                level2 = st.session_state.get('level2_input', '')
                level3 = st.session_state.get('level3_input', [])
                years = st.session_state.get('years_input', [])
                
                for key in ['step', 'results', 'topic_counts', 'level1_count', 'level2_count', 'consistent_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.session_state.step = 1
                st.session_state['level1_input'] = level1
                st.session_state['level2_input'] = level2
                st.session_state['level3_input'] = level3
                st.session_state['years_input'] = years
                st.rerun()

# ============================================================================
# QUERY BUILDING FUNCTIONS
# ============================================================================

def build_search_filter(level1_term: str, level2_term: Optional[str] = None,
                       years: Optional[List[int]] = None) -> Dict[str, str]:
    """Build filters for OpenAlex API based on first two levels"""
    filters = {}
    
    # Build search query
    search_parts = []
    
    # Level 1 - main term
    if level1_term:
        parsed = parse_query_terms(level1_term)
        search_parts.append(parsed)
    
    # Level 2 - additional term (optional)
    if level2_term:
        parsed = parse_query_terms(level2_term)
        search_parts.append(parsed)
    
    # Combine all parts with AND
    if search_parts:
        # Use default.search instead of title_and_abstract.search for better results
        filters['default.search'] = ' AND '.join(search_parts)
    
    # Year filter
    if years:
        if len(years) == 1:
            filters['publication_year'] = str(years[0])
        else:
            # For range use format from:to
            filters['publication_year'] = f"{min(years)}-{max(years)}"
    
    return filters

def build_level3_filter(level3_term: str, base_filters: Dict[str, str]) -> str:
    """Build filter for level 3 term including all filters"""
    filter_parts = []
    
    if 'publication_year' in base_filters:
        filter_parts.append(f"publication_year:{base_filters['publication_year']}")
    
    # Collect search parts
    search_parts = []
    if 'default.search' in base_filters:
        search_parts.append(f"({base_filters['default.search']})")
    
    if level3_term:
        parsed = parse_query_terms(level3_term)
        search_parts.append(f"({parsed})")
    
    if search_parts:
        filter_parts.append(f"default.search:{' AND '.join(search_parts)}")
    
    return ','.join(filter_parts)

def build_count_filter(base_filters: Dict[str, str]) -> str:
    """Build filter only from first two levels"""
    filter_parts = []
    
    if 'publication_year' in base_filters:
        filter_parts.append(f"publication_year:{base_filters['publication_year']}")
    
    if 'default.search' in base_filters:
        filter_parts.append(f"default.search:{base_filters['default.search']}")
    
    return ','.join(filter_parts)

# ============================================================================
# OPENALEX API REQUEST FUNCTIONS
# ============================================================================

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_DELAY, max=MAX_DELAY),
    retry=retry_if_exception_type((requests.exceptions.RequestException,))
)
@sleep_and_retry
@limits(calls=RATE_LIMIT_PER_SECOND, period=1)
def make_openalex_request(url: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """Make request to OpenAlex API with rate limiting"""
    if params is None:
        params = {}
    
    params['mailto'] = MAILTO
    
    try:
        response = requests.get(
            url,
            params=params,
            headers=POLITE_POOL_HEADER,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 5))
            logger.warning(f"Rate limited. Waiting {retry_after} seconds")
            time.sleep(retry_after)
            raise requests.exceptions.RequestException("Rate limited")
        else:
            logger.error(f"Error {response.status_code}: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        logger.warning("Request timeout")
        raise
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        raise

def get_total_count(level1_term: str, level2_term: Optional[str] = None,
                   years: Optional[List[int]] = None) -> int:
    """Get total count of papers matching query"""
    filters = build_search_filter(level1_term, level2_term, years=years)
    filter_str = build_count_filter(filters)
    
    if not filter_str:
        return 0
    
    params = {
        'filter': filter_str,
        'per-page': 1
    }
    
    data = make_openalex_request(f"{OPENALEX_BASE_URL}/works", params)
    
    if data and 'meta' in data:
        return data['meta'].get('count', 0)
    
    return 0

def test_query(level1_term: str, level2_term: Optional[str] = None, years: Optional[List[int]] = None):
    """Test query and show how it will be sent to OpenAlex"""
    filters = build_search_filter(level1_term, level2_term, years)
    filter_str = build_count_filter(filters)
    
    st.write("**Debug Information:**")
    st.write(f"Original Level 1: {level1_term}")
    st.write(f"Parsed Level 1: {parse_query_terms(level1_term)}")
    if level2_term:
        st.write(f"Original Level 2: {level2_term}")
        st.write(f"Parsed Level 2: {parse_query_terms(level2_term)}")
    st.write(f"Filter string: {filter_str}")
    st.write(f"Full URL: https://api.openalex.org/works?filter={filter_str}&per-page=1")
    
    # Test request
    count = get_total_count(level1_term, level2_term, years)
    st.write(f"**Result count: {count:,}**")
    return count

def get_topic_counts(level1_term: str, level2_term: Optional[str],
                    level3_terms: List[str], years: Optional[List[int]],
                    progress_callback=None) -> Dict[str, int]:
    """Get paper counts for each level 3 term"""
    base_filters = build_search_filter(level1_term, level2_term, years=years)
    counts = {}
    
    for i, term in enumerate(level3_terms):
        if progress_callback:
            progress_callback(i / len(level3_terms), f"Analyzing: {term}")
        
        filter_str = build_level3_filter(term, base_filters)
        
        params = {
            'filter': filter_str,
            'per-page': 1
        }
        
        data = make_openalex_request(f"{OPENALEX_BASE_URL}/works", params)
        
        if data and 'meta' in data:
            counts[term] = data['meta'].get('count', 0)
        else:
            counts[term] = 0
        
        time.sleep(0.2)
    
    return counts

def fetch_top_works(level1_term: str, level2_term: Optional[str],
                   level3_term: str, years: Optional[List[int]],
                   limit: int = 100, progress_callback=None) -> List[Dict]:
    """Fetch top N most relevant works for a term"""
    base_filters = build_search_filter(level1_term, level2_term, years=years)
    filter_str = build_level3_filter(level3_term, base_filters)
    
    all_works = []
    cursor = "*"
    page = 0
    
    while len(all_works) < limit and cursor:
        page += 1
        if progress_callback:
            progress_callback(
                min(len(all_works) / limit, 0.99),
                f"Fetching {level3_term}: page {page}"
            )
        
        params = {
            'filter': filter_str,
            'per-page': min(CURSOR_PAGE_SIZE, limit - len(all_works)),
            'cursor': cursor,
            'sort': 'relevance_score:desc'
        }
        
        data = make_openalex_request(f"{OPENALEX_BASE_URL}/works", params)
        
        if not data or 'results' not in data:
            break
        
        works = data['results']
        if not works:
            break
        
        all_works.extend(works)
        cursor = data.get('meta', {}).get('next_cursor')
        time.sleep(0.1)
    
    return all_works[:limit]

def enrich_work_data(work: Dict) -> Dict:
    """Enrich work data with additional fields"""
    if not work:
        return {}
    
    doi_raw = work.get('doi')
    doi_clean = ''
    if doi_raw:
        doi_clean = str(doi_raw).replace('https://doi.org/', '')
    
    enriched = {
        'id': work.get('id', ''),
        'doi': doi_clean,
        'title': clean_text(work.get('title', '')),
        'publication_date': work.get('publication_date', ''),
        'publication_year': work.get('publication_year', 0),
        'cited_by_count': work.get('cited_by_count', 0),
        'type': work.get('type', ''),
        'doi_url': f"https://doi.org/{doi_clean}" if doi_clean else '',
        'relevance_score': work.get('relevance_score', 0),
        'fwci': work.get('fwci', None)  # Field-Weighted Citation Impact
    }
    
    # Authors
    authorships = work.get('authorships', [])
    authors = []
    for authorship in authorships[:5]:
        if authorship and 'author' in authorship:
            author_name = authorship['author'].get('display_name', '')
            if author_name:
                authors.append(author_name)
    enriched['authors'] = authors
    
    # Journal
    primary_location = work.get('primary_location')
    if primary_location and 'source' in primary_location:
        source = primary_location['source']
        enriched['journal'] = source.get('display_name', '') if source else ''
    else:
        enriched['journal'] = ''
    
    # Open Access
    open_access = work.get('open_access', {})
    enriched['is_oa'] = open_access.get('is_oa', False)
    
    return enriched

def get_yearly_distribution_group_by(level1_term: str, level2_term: Optional[str], 
                                    level3_term: str, years: List[int]) -> Dict[int, int]:
    """
    Get yearly distribution for a specific sub-topic using group_by (single request)
    This ensures perfect consistency between total count and yearly sum
    """
    base_filters = build_search_filter(level1_term, level2_term)
    filter_str = build_level3_filter(level3_term, base_filters)
    
    params = {
        'filter': filter_str,
        'group-by': 'publication_year',
        'per-page': 200
    }
    
    data = make_openalex_request(f"{OPENALEX_BASE_URL}/works", params)
    
    # Initialize all years with 0
    yearly_counts = {year: 0 for year in years}
    
    if data and 'group_by' in data:
        for group in data['group_by']:
            try:
                year = int(group['key'])
                if year in years:
                    yearly_counts[year] = group['count']
            except (ValueError, TypeError):
                continue
    
    return yearly_counts

# ============================================================================
# NEW FUNCTIONS FOR ENHANCED CITATION ANALYSIS
# ============================================================================

def get_citation_distribution_group_by(level1_term: str, level2_term: Optional[str],
                                      level3_term: str, years: List[int]) -> Dict[str, int]:
    """
    Get citation distribution using group_by with filters for citation ranges.
    Makes 7 separate API requests to get exact counts for each citation category.
    """
    base_filters = build_search_filter(level1_term, level2_term)
    
    # Define citation ranges - UPDATED TO 7 CATEGORIES
    citation_ranges = {
        '0': 'cited_by_count:0',
        '1-4': 'cited_by_count:1-4',
        '5-10': 'cited_by_count:5-10',
        '11-30': 'cited_by_count:11-30',
        '31-50': 'cited_by_count:31-50',
        '51-100': 'cited_by_count:51-100',
        '100+': 'cited_by_count:>100'
    }
    
    distribution = {}
    
    for range_name, range_filter in citation_ranges.items():
        # Combine base filter with citation range filter
        combined_filter = f"{range_filter}"
        if base_filters:
            filter_parts = []
            if 'publication_year' in base_filters:
                filter_parts.append(f"publication_year:{base_filters['publication_year']}")
            if 'default.search' in base_filters:
                filter_parts.append(f"default.search:{base_filters['default.search']}")
            if level3_term:
                parsed = parse_query_terms(level3_term)
                filter_parts.append(f"default.search:({parsed})")
            
            if filter_parts:
                combined_filter = f"{range_filter},{','.join(filter_parts)}"
        
        params = {
            'filter': combined_filter,
            'per-page': 1
        }
        
        data = make_openalex_request(f"{OPENALEX_BASE_URL}/works", params)
        
        if data and 'meta' in data:
            distribution[range_name] = data['meta'].get('count', 0)
        else:
            distribution[range_name] = 0
        
        time.sleep(0.1)  # Small delay to be polite to API
    
    return distribution

def get_yearly_citations(level1_term: str, level2_term: Optional[str],
                        level3_term: str, years: List[int],
                        top_works: List[Dict]) -> Dict[int, int]:
    """
    Get yearly citation counts for a sub-topic.
    Aggregates citations from top works by year.
    """
    yearly_citations = {year: 0 for year in years}
    
    for work in top_works:
        year = work.get('publication_year')
        citations = work.get('cited_by_count', 0)
        if year in yearly_citations:
            yearly_citations[year] += citations
    
    return yearly_citations

def get_journal_data(all_works_by_topic: Dict[str, List[Dict]]) -> Dict:
    """
    Collect comprehensive journal data for analysis.
    """
    journal_data = defaultdict(lambda: {
        'total_papers': 0,
        'topics': defaultdict(int),
        'citations': [],
        'papers_by_year': defaultdict(int)
    })
    
    for topic, works in all_works_by_topic.items():
        for work in works:
            enriched = enrich_work_data(work)
            journal = enriched.get('journal', 'Unknown Journal')
            if journal and journal != 'Unknown Journal':
                journal_data[journal]['total_papers'] += 1
                journal_data[journal]['topics'][topic] += 1
                journal_data[journal]['citations'].append(enriched.get('cited_by_count', 0))
                journal_data[journal]['papers_by_year'][enriched.get('publication_year', 0)] += 1
    
    return journal_data

def calculate_percentile_threshold(citations: List[int], percentile: float = 85) -> float:
    """
    Calculate citation threshold for given percentile.
    """
    if not citations:
        return 0
    return np.percentile(citations, percentile)

def get_highly_cited_papers(works_by_topic: Dict[str, List[Dict]], percentile: float = 85) -> List[Dict]:
    """
    Get papers above the given citation percentile.
    Returns ONLY UNIQUE papers (no duplicates) based on DOI or work ID.
    """
    all_papers = []
    seen_ids = set()  # Track unique papers by ID or DOI
    
    for topic, works in works_by_topic.items():
        for work in works:
            enriched = enrich_work_data(work)
            
            # Create unique identifier (use DOI if available, otherwise work ID)
            paper_id = enriched.get('doi', enriched.get('id', ''))
            if not paper_id:
                # If no DOI or ID, use title as fallback (less reliable)
                paper_id = enriched.get('title', '')
            
            # Only add if not seen before
            if paper_id and paper_id not in seen_ids:
                enriched['topic'] = topic
                all_papers.append(enriched)
                seen_ids.add(paper_id)
    
    if not all_papers:
        return []
    
    citations = [p.get('cited_by_count', 0) for p in all_papers]
    threshold = calculate_percentile_threshold(citations, percentile)
    
    highly_cited = [p for p in all_papers if p.get('cited_by_count', 0) >= threshold]
    
    # Sort by citations descending
    highly_cited_sorted = sorted(highly_cited, key=lambda x: x.get('cited_by_count', 0), reverse=True)
    
    return highly_cited_sorted

def get_unique_top_fwci_papers(works_by_topic: Dict[str, List[Dict]], top_n: int = 100) -> List[Dict]:
    """
    Get top N papers by Field-Weighted Citation Impact (FWCI).
    Returns ONLY UNIQUE papers (no duplicates) based on DOI or work ID.
    """
    all_papers = []
    seen_ids = set()  # Track unique papers by ID or DOI
    
    for topic, works in works_by_topic.items():
        for work in works:
            enriched = enrich_work_data(work)
            
            # Only include papers with valid FWCI
            if enriched.get('fwci') is not None and enriched.get('fwci') > 0:
                
                # Create unique identifier (use DOI if available, otherwise work ID)
                paper_id = enriched.get('doi', enriched.get('id', ''))
                if not paper_id:
                    # If no DOI or ID, use title as fallback (less reliable)
                    paper_id = enriched.get('title', '')
                
                # Only add if not seen before
                if paper_id and paper_id not in seen_ids:
                    enriched['topic'] = topic
                    all_papers.append(enriched)
                    seen_ids.add(paper_id)
    
    if not all_papers:
        return []
    
    # Sort by FWCI descending
    all_papers_sorted = sorted(all_papers, key=lambda x: x.get('fwci', 0), reverse=True)
    
    # Return top N unique papers
    return all_papers_sorted[:top_n]

def calculate_gini_coefficient(citations: List[int]) -> float:
    """
    Calculate Gini coefficient for citation distribution.
    Gini = 0 (perfect equality) to 1 (perfect inequality)
    """
    if citations is None or len(citations) == 0:
        return 0.0
    
    citations = np.array(citations)
    if len(citations) == 0 or np.sum(citations) == 0:
        return 0.0
    
    # Sort citations in ascending order
    citations = np.sort(citations)
    n = len(citations)
    
    # Calculate Gini coefficient using the formula:
    # G = (2 * Σ(i * x_i)) / (n * Σx) - (n + 1)/n
    indices = np.arange(1, n + 1)
    numerator = 2 * np.sum(indices * citations)
    denominator = n * np.sum(citations)
    
    if denominator == 0:
        return 0.0
    
    gini = (numerator / denominator) - (n + 1) / n
    return float(gini)

def calculate_cagr(start_count: int, end_count: int, num_years: int) -> float:
    """
    Calculate Compound Annual Growth Rate
    CAGR = (End/Start)^(1/num_years) - 1
    """
    if start_count == 0 or num_years == 0:
        return 0.0
    
    cagr = (end_count / start_count) ** (1 / num_years) - 1
    return float(cagr * 100)  # Return as percentage

def calculate_citation_velocity(citations: int, years_since_publication: int) -> float:
    """
    Calculate citation velocity (citations per year since publication)
    """
    if years_since_publication == 0:
        return 0.0
    return citations / years_since_publication

def get_consistent_topic_data(level1_term: str, level2_term: Optional[str],
                            level3_terms: List[str], years: List[int],
                            max_papers_to_fetch: int = 100,
                            progress_callback=None) -> Dict[str, Dict]:
    """
    Get consistent data for all topics using hybrid approach:
    - group_by for yearly distributions (single request per topic)
    - group_by for citation distributions (7 requests per topic)
    - topic_counts from the same group_by data (sum of yearly)
    - fetch top papers for detailed view (limited)
    
    This ensures all visualizations use the SAME source data
    """
    consistent_data = {}
    total_terms = len(level3_terms)
    
    for idx, term in enumerate(level3_terms):
        if progress_callback:
            progress_callback(
                idx / total_terms,
                f"Analyzing: {term}"
            )
        
        # Step 1: Get yearly distribution using group_by (1 request)
        yearly_dist = get_yearly_distribution_group_by(
            level1_term, level2_term, term, years
        )
        
        # Step 2: Get citation distribution using group_by (7 requests)
        citation_dist = get_citation_distribution_group_by(
            level1_term, level2_term, term, years
        )
        
        # Step 3: Calculate total from yearly data (ensures consistency)
        total_papers = sum(yearly_dist.values())
        
        # Step 4: Fetch top works for detailed view
        top_works = []
        if total_papers > 0:
            top_works = fetch_top_works(
                level1_term, level2_term, term, years,
                limit=max_papers_to_fetch
            )
        
        # Step 5: Calculate yearly citations from top works
        yearly_citations = get_yearly_citations(
            level1_term, level2_term, term, years, top_works
        )
        
        # Step 6: Calculate enhanced citation stats from top works
        citation_stats = {}
        if top_works:
            citations = [w.get('cited_by_count', 0) for w in top_works]
            
            # Calculate current year for citation velocity
            current_year = datetime.now().year
            
            # Calculate citation velocities
            velocities = []
            for work in top_works:
                pub_year = work.get('publication_year', current_year)
                years_since = max(1, current_year - pub_year)
                velocity = calculate_citation_velocity(
                    work.get('cited_by_count', 0), 
                    years_since
                )
                velocities.append(velocity)
            
            # Calculate Gini coefficient
            gini = calculate_gini_coefficient(citations)
            
            citation_stats = {
                'mean': float(np.mean(citations)),
                'median': float(np.median(citations)),
                'max': int(max(citations)),
                'gini': gini,
                'total_citations': int(sum(citations)),
                'mean_velocity': float(np.mean(velocities)),
                'median_velocity': float(np.median(velocities)),
                'distribution': {
                    '0': int(sum(1 for c in citations if c == 0)),
                    '1-4': int(sum(1 for c in citations if 1 <= c <= 4)),
                    '5-10': int(sum(1 for c in citations if 5 <= c <= 10)),
                    '11-30': int(sum(1 for c in citations if 11 <= c <= 30)),
                    '31-50': int(sum(1 for c in citations if 31 <= c <= 50)),
                    '51-100': int(sum(1 for c in citations if 51 <= c <= 100)),
                    '100+': int(sum(1 for c in citations if c > 100))
                },
                'highly_cited_50': int(sum(1 for c in citations if c > 50)),
                'highly_cited_100': int(sum(1 for c in citations if c > 100))
            }
        
        # Step 7: Calculate CAGR for last 5 years if enough data
        cagr_data = {}
        if len(years) >= 5:
            recent_years = sorted(years)[-5:]
            if len(recent_years) >= 2:
                start_year = recent_years[0]
                end_year = recent_years[-1]
                start_count = yearly_dist.get(start_year, 0)
                end_count = yearly_dist.get(end_year, 0)
                cagr_data = {
                    'period': f"{start_year}-{end_year}",
                    'cagr': calculate_cagr(start_count, end_count, len(recent_years) - 1),
                    'start_count': start_count,
                    'end_count': end_count
                }
        
        # Step 8: Find peak year
        peak_year = None
        peak_count = 0
        for year, count in yearly_dist.items():
            if count > peak_count:
                peak_count = count
                peak_year = year
        
        consistent_data[term] = {
            'total': total_papers,
            'yearly': yearly_dist,  # Exact data from group_by
            'yearly_citations': yearly_citations,  # Yearly citation sums
            'citation_distribution': citation_dist,  # Exact counts from group_by (7 categories)
            'top_works': top_works,
            'citation_stats': citation_stats,
            'cagr': cagr_data,
            'peak_year': peak_year,
            'peak_count': peak_count
        }
        
        # Small delay to be polite to API
        time.sleep(0.1)
    
    return consistent_data

# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS - ALL USING SCIENTIFIC STYLE
# ============================================================================

def get_current_palettes():
    """Get currently selected color palettes"""
    discrete_name = st.session_state.get('discrete_palette_name', 'nature')
    gradient_name = st.session_state.get('gradient_palette_name', 'viridis')
    
    discrete_palette = COLOR_PALETTES_SCIENTIFIC_DISCRETE.get(discrete_name, COLOR_PALETTES_SCIENTIFIC_DISCRETE['nature'])
    gradient_palette = GRADIENT_PALETTES.get(gradient_name, GRADIENT_PALETTES['viridis'])
    
    return discrete_palette, gradient_palette

def create_subtopic_distribution_absolute(topic_counts: Dict[str, int], level2_count: int):
    """
    4.1.1. Гистограмма с общим числом публикаций по запросам с учетом фильтров
    Внутри гистограммы должна быть легенда с общим числом статей по всем Sub-topic,
    а также отдельная легенда по числу статей по каждому Sub-topic
    """
    if not topic_counts:
        return None
    
    discrete_palette, _ = get_current_palettes()
    
    # Filter zero values
    non_zero = {k: v for k, v in topic_counts.items() if v > 0}
    if not non_zero:
        return None
    
    topics = list(non_zero.keys())
    counts = list(non_zero.values())
    total_papers = sum(counts)
    
    # Sort descending
    sorted_idx = np.argsort(counts)[::-1]
    topics = [topics[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(topics) * 0.4)))
    
    # Create horizontal bars
    y_pos = np.arange(len(topics))
    colors = discrete_palette[:len(topics)] if len(topics) <= len(discrete_palette) else plt.cm.tab20(np.linspace(0, 1, len(topics)))
    
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel('Number of Publications', fontsize=10, fontweight='bold')
    ax.set_title(f'Sub-topic Distribution - Absolute Counts\nTotal papers: {total_papers:,}', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=9, fontweight='bold')
    
    # Add total count annotation
    ax.text(0.98, 0.02, f'Total: {total_papers:,} papers', 
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    return fig

def create_subtopic_distribution_percentage(topic_counts: Dict[str, int], level2_count: int):
    """
    4.1.2. Аналогичная гистограмма, представленная в процентах
    Сумма процентов всех столбиков должна равна 100%
    """
    if not topic_counts:
        return None
    
    discrete_palette, _ = get_current_palettes()
    
    # Filter zero values
    non_zero = {k: v for k, v in topic_counts.items() if v > 0}
    if not non_zero:
        return None
    
    topics = list(non_zero.keys())
    counts = list(non_zero.values())
    total_papers = sum(counts)
    percentages = [(c / total_papers * 100) if total_papers > 0 else 0 for c in counts]
    
    # Sort descending
    sorted_idx = np.argsort(counts)[::-1]
    topics = [topics[i] for i in sorted_idx]
    percentages = [percentages[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(topics) * 0.4)))
    
    # Create horizontal bars
    y_pos = np.arange(len(topics))
    colors = discrete_palette[:len(topics)] if len(topics) <= len(discrete_palette) else plt.cm.tab20(np.linspace(0, 1, len(topics)))
    
    bars = ax.barh(y_pos, percentages, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel('Percentage of Publications (%)', fontsize=10, fontweight='bold')
    ax.set_title(f'Sub-topic Distribution - Percentage\nTotal papers: {total_papers:,} (100%)', 
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim(0, 100)
    
    # Add value labels on bars
    for i, (bar, pct, count) in enumerate(zip(bars, percentages, counts)):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% ({count:,})', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_stacked_yearly_chart(consistent_data: Dict[str, Dict], years: List[int], level2_term: Optional[str] = None):
    """
    4.2.1. Сводная гистограмма с накоплением числа публикаций от года
    """
    if not consistent_data:
        return None
    
    discrete_palette, _ = get_current_palettes()
    
    # Get topics with data
    topics = [t for t, data in consistent_data.items() if data['total'] > 0]
    if not topics:
        return None
    
    # Sort years
    years_sorted = sorted(years)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for stacking
    bottom = np.zeros(len(years_sorted))
    colors = discrete_palette[:len(topics)] if len(topics) <= len(discrete_palette) else plt.cm.tab20(np.linspace(0, 1, len(topics)))
    
    # Create stacked bars
    for idx, topic in enumerate(topics):
        yearly_data = consistent_data[topic]['yearly']
        counts = [yearly_data.get(year, 0) for year in years_sorted]
        
        ax.bar(years_sorted, counts, bottom=bottom, label=f"{topic} ({consistent_data[topic]['total']:,})",
               color=colors[idx], edgecolor='black', linewidth=0.5, width=0.8)
        bottom += np.array(counts)
    
    # Customize plot
    ax.set_xlabel('Publication Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Publications', fontsize=11, fontweight='bold')
    ax.set_title(f'Stacked Yearly Distribution' + (f' with {level2_term}' if level2_term else ''),
                fontsize=12, fontweight='bold', pad=10)
    
    ax.set_xticks(years_sorted)
    ax.set_xticklabels([str(y) for y in years_sorted], rotation=45, ha='right')
    
    # Add legend
    ax.legend(fontsize=9, frameon=True, edgecolor='black', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add total count
    total_papers = sum(bottom)
    ax.text(0.02, 0.98, f'Total: {total_papers:,} papers',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    return fig

def create_grouped_yearly_chart(consistent_data: Dict[str, Dict], years: List[int], level2_term: Optional[str] = None):
    """
    4.2.2. Сводная гистограмма с группировкой числа публикаций от года
    """
    if not consistent_data:
        return None
    
    discrete_palette, _ = get_current_palettes()
    
    # Get topics with data
    topics = [t for t, data in consistent_data.items() if data['total'] > 0]
    if not topics:
        return None
    
    # Sort years
    years_sorted = sorted(years)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data for grouped bars
    n_topics = len(topics)
    n_years = len(years_sorted)
    width = 0.8 / n_topics  # Width of each bar
    
    colors = discrete_palette[:n_topics] if n_topics <= len(discrete_palette) else plt.cm.tab20(np.linspace(0, 1, n_topics))
    
    # Create grouped bars
    for idx, topic in enumerate(topics):
        yearly_data = consistent_data[topic]['yearly']
        counts = [yearly_data.get(year, 0) for year in years_sorted]
        
        x_pos = np.arange(n_years) + idx * width - (n_topics * width / 2) + width/2
        bars = ax.bar(x_pos, counts, width, label=f"{topic} ({consistent_data[topic]['total']:,})",
                      color=colors[idx], edgecolor='black', linewidth=0.5)
        
        # Add small value labels for non-zero bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{count}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Customize plot
    ax.set_xlabel('Publication Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Publications', fontsize=11, fontweight='bold')
    ax.set_title(f'Grouped Yearly Distribution' + (f' with {level2_term}' if level2_term else ''),
                fontsize=12, fontweight='bold', pad=10)
    
    ax.set_xticks(np.arange(n_years))
    ax.set_xticklabels([str(y) for y in years_sorted], rotation=45, ha='right')
    
    # Add legend
    ax.legend(fontsize=9, frameon=True, edgecolor='black', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    return fig

def create_normalized_yearly_chart(consistent_data: Dict[str, Dict], years: List[int], level2_term: Optional[str] = None):
    """
    4.2.3. Сводная гистограмма с группировкой числа публикаций от года,
    нормированных на максимальное число публикации по какому-либо Sub-topic в какой-либо год
    """
    if not consistent_data:
        return None
    
    discrete_palette, _ = get_current_palettes()
    
    # Get topics with data
    topics = [t for t, data in consistent_data.items() if data['total'] > 0]
    if not topics:
        return None
    
    # Sort years
    years_sorted = sorted(years)
    
    # Find global maximum across all topics and years
    global_max = 0
    for topic in topics:
        yearly_data = consistent_data[topic]['yearly']
        global_max = max(global_max, max(yearly_data.values()) if yearly_data else 0)
    
    if global_max == 0:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data for grouped bars
    n_topics = len(topics)
    n_years = len(years_sorted)
    width = 0.8 / n_topics
    
    colors = discrete_palette[:n_topics] if n_topics <= len(discrete_palette) else plt.cm.tab20(np.linspace(0, 1, n_topics))
    
    # Create grouped bars with normalized values
    for idx, topic in enumerate(topics):
        yearly_data = consistent_data[topic]['yearly']
        normalized_counts = [yearly_data.get(year, 0) / global_max for year in years_sorted]
        actual_counts = [yearly_data.get(year, 0) for year in years_sorted]
        
        x_pos = np.arange(n_years) + idx * width - (n_topics * width / 2) + width/2
        bars = ax.bar(x_pos, normalized_counts, width, label=f"{topic} ({consistent_data[topic]['total']:,})",
                      color=colors[idx], edgecolor='black', linewidth=0.5)
        
        # Add small value labels for bars with significant height
        for bar, norm, actual in zip(bars, normalized_counts, actual_counts):
            if actual > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{actual}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Customize plot
    ax.set_xlabel('Publication Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Publication Count (max=1)', fontsize=11, fontweight='bold')
    ax.set_title(f'Normalized Yearly Distribution\n(Normalized to global maximum: {global_max:,} papers)',
                fontsize=12, fontweight='bold', pad=10)
    
    ax.set_xticks(np.arange(n_years))
    ax.set_xticklabels([str(y) for y in years_sorted], rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    
    # Add legend
    ax.legend(fontsize=9, frameon=True, edgecolor='black', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    return fig

def create_topic_yearly_publications(topic: str, data: Dict, years: List[int]):
    """
    4.3.1. Отдельные гистограммы публикаций каждого конкретного Sub-topic по годам с color-map
    """
    if data['total'] == 0:
        return None
    
    _, gradient_palette = get_current_palettes()
    
    years_sorted = sorted(years)
    counts = [data['yearly'].get(year, 0) for year in years_sorted]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color gradient based on counts
    norm = plt.Normalize(min(counts) if min(counts) > 0 else 0, max(counts))
    colors = gradient_palette(norm(counts))
    
    bars = ax.bar(years_sorted, counts, color=colors, edgecolor='black', linewidth=0.5, width=0.8)
    
    # Customize plot
    ax.set_xlabel('Publication Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Publications', fontsize=11, fontweight='bold')
    ax.set_title(f'{topic}: Publications by Year\nTotal: {data["total"]:,} papers', 
                fontsize=12, fontweight='bold', pad=10)
    
    ax.set_xticks(years_sorted)
    ax.set_xticklabels([str(y) for y in years_sorted], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=gradient_palette, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Publication Intensity', fontsize=9)
    
    # Verify total matches
    ax.text(0.98, 0.98, f'Sum: {sum(counts):,} papers\n(Matches topic total: ✓)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    return fig

def create_topic_yearly_citations(topic: str, data: Dict, years: List[int]):
    """
    4.3.2. Отдельные гистограммы числа всех цитирований всех публикаций Sub-topic по годам с color-map
    """
    if data['total'] == 0 or not data.get('yearly_citations'):
        return None
    
    _, gradient_palette = get_current_palettes()
    
    years_sorted = sorted(years)
    citations = [data['yearly_citations'].get(year, 0) for year in years_sorted]
    
    if sum(citations) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color gradient based on citation counts
    norm = plt.Normalize(min(citations) if min(citations) > 0 else 0, max(citations))
    colors = gradient_palette(norm(citations))
    
    bars = ax.bar(years_sorted, citations, color=colors, edgecolor='black', linewidth=0.5, width=0.8)
    
    # Customize plot
    ax.set_xlabel('Publication Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Citations', fontsize=11, fontweight='bold')
    ax.set_title(f'{topic}: Total Citations by Year\nTotal citations: {sum(citations):,}', 
                fontsize=12, fontweight='bold', pad=10)
    
    ax.set_xticks(years_sorted)
    ax.set_xticklabels([str(y) for y in years_sorted], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, cit in zip(bars, citations):
        if cit > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{cit:,}', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=gradient_palette, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Citation Intensity', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_topic_citation_distribution(topic: str, data: Dict):
    """
    4.3.3. Отдельные гистограммы распределения цитирований публикаций каждого конкретного Sub-topic
    по диапазонам с colormap: 0, 1-4, 5-10, 11-30, 31-50, 51-100, 100+
    """
    if data['total'] == 0 or not data.get('citation_distribution'):
        return None
    
    _, gradient_palette = get_current_palettes()
    
    # Get citation distribution (should already have the 7 categories)
    dist = data['citation_distribution']
    categories = ['0', '1-4', '5-10', '11-30', '31-50', '51-100', '100+']
    
    # Ensure all categories exist
    counts = [dist.get(cat, 0) for cat in categories]
    total = sum(counts)
    
    if total == 0:
        return None
    
    percentages = [c / total * 100 for c in counts]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color gradient based on citation impact (higher categories get warmer colors)
    norm = plt.Normalize(0, len(categories)-1)
    colors = gradient_palette(norm(range(len(categories))))
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Citation Range', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Papers', fontsize=11, fontweight='bold')
    ax.set_title(f'{topic}: Citation Distribution\nTotal papers: {total:,}', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Add value and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=gradient_palette, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Citation Impact →', fontsize=9)
    cbar.set_ticks([])
    
    plt.tight_layout()
    return fig

def create_top_journals_bar(journal_data: Dict, top_n: int = 15):
    """
    4.4.1. Линейчатые гистограммы топ-журналов по всем публикациям
    """
    if not journal_data:
        return None
    
    discrete_palette, _ = get_current_palettes()
    
    # Sort by total papers and get top N
    sorted_journals = sorted(journal_data.items(), key=lambda x: x[1]['total_papers'], reverse=True)[:top_n]
    
    journals = [j[0] for j in sorted_journals]
    papers = [j[1]['total_papers'] for j in sorted_journals]
    
    # Truncate long journal names
    journals = [j[:40] + '...' if len(j) > 40 else j for j in journals]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(journals) * 0.4)))
    
    # Create horizontal bars
    y_pos = np.arange(len(journals))
    colors = discrete_palette[:len(journals)] if len(journals) <= len(discrete_palette) else plt.cm.tab20(np.linspace(0, 1, len(journals)))
    
    bars = ax.barh(y_pos, papers, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(journals, fontsize=8)
    ax.set_xlabel('Number of Publications', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Journals by Publication Count\nTotal journals analyzed: {len(journal_data)}', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Add value labels
    for bar, paper in zip(bars, papers):
        ax.text(paper + max(papers)*0.01, bar.get_y() + bar.get_height()/2,
                f'{paper:,}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_top_journals_stacked(journal_data: Dict, topics: List[str], top_n: int = 15):
    """
    4.4.2. Линейчатые гистограммы с накоплением по Sub-topics
    """
    if not journal_data or not topics:
        return None
    
    discrete_palette, _ = get_current_palettes()
    
    # Sort by total papers and get top N
    sorted_journals = sorted(journal_data.items(), key=lambda x: x[1]['total_papers'], reverse=True)[:top_n]
    
    journals = [j[0] for j in sorted_journals]
    journals_short = [j[:30] + '...' if len(j) > 30 else j for j in journals]
    
    # Prepare stacked data
    topic_data = []
    for topic in topics:
        topic_counts = []
        for journal, data in sorted_journals:
            topic_counts.append(data['topics'].get(topic, 0))
        topic_data.append(topic_counts)
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(journals) * 0.4)))
    
    # Create stacked horizontal bars
    y_pos = np.arange(len(journals))
    left = np.zeros(len(journals))
    
    colors = discrete_palette[:len(topics)] if len(topics) <= len(discrete_palette) else plt.cm.tab20(np.linspace(0, 1, len(topics)))
    
    for idx, (topic, counts) in enumerate(zip(topics, topic_data)):
        bars = ax.barh(y_pos, counts, left=left, label=topic,
                       color=colors[idx], edgecolor='black', linewidth=0.5)
        left += np.array(counts)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(journals_short, fontsize=8)
    ax.set_xlabel('Number of Publications', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Journals - Distribution by Sub-topic', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Add legend
    ax.legend(fontsize=9, frameon=True, edgecolor='black', loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig

def create_top_journals_citations(journal_data: Dict, top_n: int = 15):
    """
    4.4.3. Среднее цитирование топ-журнала по найденным публикациям
    """
    if not journal_data:
        return None
    
    discrete_palette, _ = get_current_palettes()
    
    # Filter journals with at least one paper and calculate average citations
    journal_stats = []
    for journal, data in journal_data.items():
        if data['total_papers'] > 0 and data['citations']:
            avg_cit = np.mean(data['citations'])
            journal_stats.append((journal, avg_cit, data['total_papers']))
    
    # Sort by average citations and get top N
    journal_stats.sort(key=lambda x: x[1], reverse=True)
    journal_stats = journal_stats[:top_n]
    
    journals = [j[0] for j in journal_stats]
    avg_cits = [j[1] for j in journal_stats]
    papers = [j[2] for j in journal_stats]
    
    # Truncate long journal names
    journals = [j[:40] + '...' if len(j) > 40 else j for j in journals]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(journals) * 0.4)))
    
    # Create horizontal bars
    y_pos = np.arange(len(journals))
    colors = discrete_palette[:len(journals)] if len(journals) <= len(discrete_palette) else plt.cm.viridis(np.linspace(0.2, 0.9, len(journals)))
    
    bars = ax.barh(y_pos, avg_cits, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(journals, fontsize=8)
    ax.set_xlabel('Average Citations per Paper', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Journals by Average Citations\n(Number of papers in parentheses)', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Add value labels with paper count
    for bar, avg, paper in zip(bars, avg_cits, papers):
        ax.text(avg + max(avg_cits)*0.02, bar.get_y() + bar.get_height()/2,
                f'{avg:.1f} (n={paper})', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_citation_velocity_chart(consistent_data: Dict[str, Dict], current_year: int = None):
    """
    4.5. Citation Velocity (Citations per Year) для каждого Sub-topics с color-map и средними значениями
    """
    if current_year is None:
        current_year = datetime.now().year
    
    topics = []
    velocities = []
    total_citations = []
    
    for topic, data in consistent_data.items():
        if data['citation_stats'] and 'mean_velocity' in data['citation_stats']:
            topics.append(topic)
            velocities.append(data['citation_stats']['mean_velocity'])
            total_citations.append(data['citation_stats'].get('total_citations', 0))
    
    if not topics:
        return None
    
    # Sort by velocity
    sorted_data = sorted(zip(topics, velocities, total_citations), key=lambda x: x[1], reverse=True)
    topics = [x[0] for x in sorted_data]
    velocities = [x[1] for x in sorted_data]
    total_citations = [x[2] for x in sorted_data]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(topics) * 0.4)))
    
    # Create color gradient based on velocity
    norm = plt.Normalize(min(velocities), max(velocities))
    _, gradient_palette = get_current_palettes()
    colors = gradient_palette(norm(velocities))
    
    # Create horizontal bars
    y_pos = np.arange(len(topics))
    bars = ax.barh(y_pos, velocities, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add average line
    avg_velocity = np.mean(velocities)
    ax.axvline(avg_velocity, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_velocity:.2f} cit/year')
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel('Mean Citations per Year', fontsize=11, fontweight='bold')
    ax.set_title('Citation Velocity by Sub-topic\n(Higher values indicate "hotter" research areas)',
                fontsize=12, fontweight='bold', pad=10)
    
    # Add value labels
    for bar, vel, cit in zip(bars, velocities, total_citations):
        ax.text(vel + max(velocities)*0.02, bar.get_y() + bar.get_height()/2,
                f'{vel:.2f} (total: {cit:,})', va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    ax.legend(fontsize=9, frameon=True, edgecolor='black')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=gradient_palette, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Citation Velocity', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_matthew_effect_analysis(consistent_data: Dict[str, Dict]):
    """
    Дополнительный график 1: "Matthew Effect" Analysis
    Соотношение доли статей и доли цитирований
    """
    all_citations = []
    for topic, data in consistent_data.items():
        for work in data['top_works']:
            all_citations.append(work.get('cited_by_count', 0))
    
    if not all_citations:
        return None
    
    all_citations = np.sort(all_citations)[::-1]  # Sort descending
    n = len(all_citations)
    total_citations = sum(all_citations)
    
    # Calculate cumulative shares
    cum_papers = np.arange(1, n + 1) / n * 100
    cum_citations = np.cumsum(all_citations) / total_citations * 100
    
    # Find key percentiles
    percentiles = [1, 5, 10, 20, 50]
    percentile_indices = [int(p * n / 100) - 1 for p in percentiles]
    citation_shares = [cum_citations[i] for i in percentile_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Cumulative distribution
    ax1.plot(cum_papers, cum_citations, 'b-', linewidth=2, label='Actual distribution')
    ax1.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Perfect equality')
    ax1.fill_between(cum_papers, cum_papers, cum_citations, alpha=0.2, color='gray')
    
    # Mark key percentiles
    for p, share in zip(percentiles, citation_shares):
        ax1.plot(p, share, 'ro', markersize=6)
        ax1.annotate(f'{p}% papers\n{share:.1f}% citations',
                    (p, share), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Cumulative share of papers (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cumulative share of citations (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Matthew Effect Analysis\nConcentration of Citations', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Bar chart of top percentile shares
    ax2.bar([f'Top {p}%' for p in percentiles], citation_shares,
            color=plt.cm.Reds(np.linspace(0.4, 0.9, len(percentiles))),
            edgecolor='black', linewidth=0.5)
    
    for i, (p, share) in enumerate(zip(percentiles, citation_shares)):
        ax2.text(i, share + 2, f'{share:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Share of total citations (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Citation Concentration in Top Papers', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.suptitle(f'Analysis of Bibliometric Matthew Effect\nTotal papers: {n:,}, Total citations: {total_citations:,}',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def create_citation_half_life(consistent_data: Dict[str, Dict], current_year: int = None):
    """
    Дополнительный график 2: Citation Half-Life
    Медианный возраст цитируемых статей
    """
    if current_year is None:
        current_year = datetime.now().year
    
    topics = []
    half_lives = []
    
    for topic, data in consistent_data.items():
        if data['top_works']:
            ages = []
            for work in data['top_works']:
                pub_year = work.get('publication_year', current_year)
                age = current_year - pub_year
                citations = work.get('cited_by_count', 0)
                # Weight by citations for citation half-life
                ages.extend([age] * citations)
            
            if ages:
                half_life = np.median(ages)
                topics.append(topic)
                half_lives.append(half_life)
    
    if not topics:
        return None
    
    # Sort by half-life
    sorted_data = sorted(zip(topics, half_lives), key=lambda x: x[1])
    topics = [x[0] for x in sorted_data]
    half_lives = [x[1] for x in sorted_data]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(topics) * 0.4)))
    
    # Create horizontal bars
    y_pos = np.arange(len(topics))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(topics)))
    
    bars = ax.barh(y_pos, half_lives, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add average line
    avg_half_life = np.mean(half_lives)
    ax.axvline(avg_half_life, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_half_life:.1f} years')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel('Median Age of Cited Papers (years)', fontsize=11, fontweight='bold')
    ax.set_title('Citation Half-Life Analysis\n(Longer bars indicate slower knowledge obsolescence)',
                fontsize=12, fontweight='bold', pad=10)
    
    # Add value labels
    for bar, hl in zip(bars, half_lives):
        ax.text(hl + 0.5, bar.get_y() + bar.get_height()/2,
                f'{hl:.1f} years', va='center', fontsize=9, fontweight='bold')
    
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig

def create_collaboration_intensity(consistent_data: Dict[str, Dict], years: List[int]):
    """
    Дополнительный график 3: Collaboration Network Intensity
    Среднее число авторов на статью по годам
    """
    if not consistent_data:
        return None
    
    years_sorted = sorted(years)
    
    # Collect average authors per year across all topics
    yearly_authors = defaultdict(list)
    yearly_papers = defaultdict(int)
    
    for topic, data in consistent_data.items():
        for work in data['top_works']:
            year = work.get('publication_year')
            if year in years_sorted:
                enriched = enrich_work_data(work)
                n_authors = len(enriched.get('authors', []))
                if n_authors > 0:
                    yearly_authors[year].append(n_authors)
                yearly_papers[year] += 1
    
    if not yearly_authors:
        return None
    
    # Calculate averages
    years_with_data = sorted([y for y in years_sorted if y in yearly_authors])
    avg_authors = [np.mean(yearly_authors[y]) for y in years_with_data]
    paper_counts = [yearly_papers[y] for y in years_with_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Average authors over time
    ax1.plot(years_with_data, avg_authors, 'b-o', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgecolor='blue', markeredgewidth=1.5)
    
    # Add trend line
    if len(years_with_data) > 1:
        z = np.polyfit(years_with_data, avg_authors, 1)
        p = np.poly1d(z)
        ax1.plot(years_with_data, p(years_with_data), "r--", alpha=0.8, linewidth=1,
                label=f'Trend (slope: {z[0]:.3f})')
    
    ax1.set_xlabel('Publication Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Number of Authors', fontsize=11, fontweight='bold')
    ax1.set_title('Collaboration Intensity Over Time', fontsize=12, fontweight='bold')
    ax1.set_xticks(years_with_data)
    ax1.set_xticklabels([str(y) for y in years_with_data], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)
    
    # Plot 2: Distribution by topic
    topic_authors = []
    topics = []
    for topic, data in consistent_data.items():
        authors_list = []
        for work in data['top_works']:
            enriched = enrich_work_data(work)
            authors_list.append(len(enriched.get('authors', [])))
        if authors_list:
            topic_authors.append(np.mean(authors_list))
            topics.append(topic)
    
    if topics:
        y_pos = np.arange(len(topics))
        colors = plt.cm.Paired(np.linspace(0, 1, len(topics)))
        
        bars = ax2.barh(y_pos, topic_authors, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(topics, fontsize=8)
        ax2.set_xlabel('Average Number of Authors', fontsize=11, fontweight='bold')
        ax2.set_title('Collaboration Intensity by Topic', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, topic_authors):
            ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Collaboration Network Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def create_journal_if_vs_citations(consistent_data: Dict[str, Dict]):
    """
    Дополнительный график 4: Journal Impact Factor vs Citation Performance
    (Используем среднее цитирование как прокси, так как IF не доступен напрямую)
    """
    journal_stats = defaultdict(lambda: {'citations': [], 'papers': 0, 'topics': set()})
    
    for topic, data in consistent_data.items():
        for work in data['top_works']:
            enriched = enrich_work_data(work)
            journal = enriched.get('journal', 'Unknown')
            if journal != 'Unknown':
                journal_stats[journal]['citations'].append(enriched.get('cited_by_count', 0))
                journal_stats[journal]['papers'] += 1
                journal_stats[journal]['topics'].add(topic)
    
    # Filter journals with at least 3 papers
    journals = []
    avg_cits = []
    papers = []
    topic_diversity = []
    
    for journal, stats in journal_stats.items():
        if stats['papers'] >= 3:
            journals.append(journal[:30] + '...' if len(journal) > 30 else journal)
            avg_cits.append(np.mean(stats['citations']))
            papers.append(stats['papers'])
            topic_diversity.append(len(stats['topics']))
    
    if len(journals) < 5:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(papers, avg_cits, c=topic_diversity, s=np.array(papers)*20,
                        cmap='viridis', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add journal labels for top ones
    for i, (journal, paper, cit) in enumerate(zip(journals, papers, avg_cits)):
        if paper > 5 or cit > 50:  # Label prominent journals
            ax.annotate(journal, (paper, cit), xytext=(5, 5), textcoords='offset points',
                       fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Number of Papers in Sample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Citations per Paper', fontsize=11, fontweight='bold')
    ax.set_title('Journal Impact Analysis\nSize → paper count, Color → topic diversity',
                fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Topics Covered', fontsize=9)
    
    # Add trend line
    if len(papers) > 1:
        z = np.polyfit(papers, avg_cits, 1)
        p = np.poly1d(z)
        ax.plot(sorted(papers), p(sorted(papers)), "r--", alpha=0.5, linewidth=1,
               label=f'Trend line')
    
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_research_front_velocity(consistent_data: Dict[str, Dict], current_year: int = None):
    """
    Дополнительный график 5: Research Front Velocity
    Соотношение "горячих" статей (последние 2 года) к общему числу публикаций
    """
    if current_year is None:
        current_year = datetime.now().year
    
    recent_years = [current_year - 1, current_year - 2]
    
    topics = []
    hot_paper_shares = []
    total_papers_list = []
    
    for topic, data in consistent_data.items():
        if data['total'] > 0:
            total = data['total']
            # Count papers from last 2 years
            hot_papers = sum(data['yearly'].get(year, 0) for year in recent_years if year in data['yearly'])
            hot_share = (hot_papers / total) * 100 if total > 0 else 0
            
            topics.append(topic)
            hot_paper_shares.append(hot_share)
            total_papers_list.append(total)
    
    if not topics:
        return None
    
    # Sort by hot paper share
    sorted_data = sorted(zip(topics, hot_paper_shares, total_papers_list), key=lambda x: x[1], reverse=True)
    topics = [x[0] for x in sorted_data]
    hot_shares = [x[1] for x in sorted_data]
    totals = [x[2] for x in sorted_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Hot paper shares
    y_pos = np.arange(len(topics))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(topics)))
    
    bars = ax1.barh(y_pos, hot_shares, color=colors, edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(topics, fontsize=9)
    ax1.set_xlabel('Share of papers from last 2 years (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Research Front Velocity\n(Share of "hot" papers: {recent_years[0]}-{recent_years[1]})',
                 fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar, share, total in zip(bars, hot_shares, totals):
        ax1.text(share + 1, bar.get_y() + bar.get_height()/2,
                f'{share:.1f}% (n={total})', va='center', fontsize=8, fontweight='bold')
    
    # Plot 2: Time trend of hot paper percentage
    # Calculate yearly hot paper percentage
    years_sorted = sorted(set().union(*[set(data['yearly'].keys()) for data in consistent_data.values()]))
    years_sorted = [y for y in years_sorted if y <= current_year]
    
    if len(years_sorted) >= 3:
        yearly_hot_pct = []
        for year in years_sorted:
            # For each year, calculate what percentage of papers from that year are "hot" (i.e., recent)
            # This is a simplification - we're looking at the age distribution
            total_papers_year = sum(data['yearly'].get(year, 0) for data in consistent_data.values())
            if total_papers_year > 0:
                yearly_hot_pct.append(100)  # All papers were hot when published
            else:
                yearly_hot_pct.append(0)
        
        ax2.plot(years_sorted, yearly_hot_pct, 'b-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Publication Year', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Papers from this year (% of total)', fontsize=11, fontweight='bold')
        ax2.set_title('Publication Activity Over Time', fontsize=12, fontweight='bold')
        ax2.set_xticks(years_sorted)
        ax2.set_xticklabels([str(y) for y in years_sorted], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Research Front Velocity Analysis\nIdentifying Emerging Research Areas',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def create_lorenz_curve(all_works_by_topic: Dict[str, List[Dict]], title: str = "Citation Inequality (Lorenz Curve)"):
    """Create Lorenz curve showing citation inequality across all topics combined"""
    # Collect all citations from all topics
    all_citations = []
    for topic, works in all_works_by_topic.items():
        for work in works:
            all_citations.append(work.get('cited_by_count', 0))
    
    if not all_citations:
        return None
    
    # Sort citations
    all_citations = np.sort(all_citations)
    n = len(all_citations)
    
    # Calculate cumulative shares
    cum_citations = np.cumsum(all_citations)
    total_citations = cum_citations[-1] if cum_citations[-1] > 0 else 1
    
    # Population shares (x-axis)
    population_share = np.arange(1, n + 1) / n
    
    # Citation shares (y-axis)
    citation_share = cum_citations / total_citations
    
    # Calculate Gini coefficient for all papers combined
    gini_all = calculate_gini_coefficient(all_citations)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Apply scientific style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Plot Lorenz curve
    ax.plot(population_share, citation_share, 'b-', linewidth=2, label=f'Lorenz curve (Gini = {gini_all:.3f})')
    
    # Plot equality line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect equality')
    
    # Fill area between curves
    ax.fill_between(population_share, population_share, citation_share, alpha=0.2, color='gray')
    
    ax.set_xlabel('Cumulative share of papers (from least to most cited)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative share of citations', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text box with interpretation
    if gini_all > 0.6:
        interpretation = "Very high inequality (Matthew effect)"
    elif gini_all > 0.4:
        interpretation = "High inequality"
    elif gini_all > 0.2:
        interpretation = "Moderate inequality"
    else:
        interpretation = "Low inequality"
    
    ax.text(0.05, 0.95, f"Interpretation: {interpretation}\nTotal papers: {n:,}\nTotal citations: {total_citations:,}", 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    return fig

def create_scientific_bar_chart(data: Dict[str, int], level2_count: int, title: str):
    """Create scientific bar chart with publication-quality colors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Apply scientific style
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Filter zero values
    non_zero = {k: v for k, v in data.items() if v > 0}
    if not non_zero:
        return None
    
    topics = list(non_zero.keys())
    counts = list(non_zero.values())
    percentages = [(c / level2_count * 100) if level2_count > 0 else 0 for c in counts]
    
    # Sort descending
    sorted_idx = np.argsort(counts)[::-1]
    topics = [topics[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]
    percentages = [percentages[i] for i in sorted_idx]
    
    # Use vibrant color palette
    colors_count = plt.cm.viridis(np.linspace(0.2, 0.9, len(topics)))
    colors_pct = plt.cm.plasma(np.linspace(0.2, 0.9, len(topics)))
    
    # Count plot
    bars1 = ax1.barh(range(len(topics)), counts, color=colors_count, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(topics)))
    ax1.set_yticklabels(topics, fontsize=9)
    ax1.set_xlabel('Number of Publications', fontsize=10, fontweight='bold')
    ax1.set_title('A) Publication Counts', fontsize=11, fontweight='bold', pad=10)
    
    # Add values to bars
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        ax1.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{count:,}', va='center', fontsize=8)
    
    # Percentage plot
    bars2 = ax2.barh(range(len(topics)), percentages, color=colors_pct, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(topics)))
    ax2.set_yticklabels([])  # Remove labels as they're on first plot
    ax2.set_xlabel('Percentage of Total (%)', fontsize=10, fontweight='bold')
    ax2.set_title('B) Percentage Distribution', fontsize=11, fontweight='bold', pad=10)
    
    # Add percentages to bars
    for i, (bar, pct) in enumerate(zip(bars2, percentages)):
        ax2.text(pct + max(percentages)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=8)
    
    plt.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def create_yearly_distribution_chart(yearly_data: Dict[int, int], title: str):
    """Create yearly distribution chart with gradient colors"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Apply scientific style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Ensure keys are integers and sort
    years_sorted = sorted([int(y) for y in yearly_data.keys()])
    counts = [yearly_data[y] for y in years_sorted]
    
    # Create color gradient based on count values
    norm = plt.Normalize(min(counts) if min(counts) > 0 else 0, max(counts))
    colors = plt.cm.viridis(norm(counts))
    
    # Create bar chart with gradient colors
    bars = ax.bar(years_sorted, counts, color=colors, edgecolor='black', linewidth=0.5, width=0.8)
    
    # Set integer labels on X axis
    ax.set_xticks(years_sorted)
    ax.set_xticklabels([str(y) for y in years_sorted], rotation=45, ha='right')
    
    ax.set_xlabel('Publication Year', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Publications', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Add values to bars if not too many
    if len(years_sorted) <= 15:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show positive values
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=8, 
                       color='black', fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Publication Intensity', fontsize=9)
    
    # Add total count
    total = sum(counts)
    ax.text(0.98, 0.98, f'Total: {total:,}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    return fig

def create_citation_distribution_chart(works_or_stats, title: str, is_stats: bool = False, is_full_distribution: bool = False):
    """Create citation distribution chart with publication-quality colors"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Apply scientific style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    if is_full_distribution:
        # For full citation distribution from group_by
        dist = works_or_stats
        categories = list(dist.keys())
        counts = list(dist.values())
        total = sum(counts)
        
        # Calculate percentages
        percentages = [(c / total * 100) if total > 0 else 0 for c in counts]
        
        # Use color palette based on citation impact
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(categories)))
        
        # Create bar chart
        bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Citation Categories', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Papers', fontsize=10, fontweight='bold')
        
        # Add values and percentages on bars
        for bar, count, pct in zip(bars, counts, percentages):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', 
                       fontsize=7, fontweight='bold')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, max(counts)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Citation Impact', fontsize=9)
        
        # Add total
        ax.text(0.98, 0.98, f'Total: {total:,} papers', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
    elif is_stats:
        # For citation stats from top_works
        dist = works_or_stats.get('distribution', {})
        categories = list(dist.keys())
        counts = list(dist.values())
        
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(categories)))
        
        bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Citation Categories', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Papers', fontsize=10, fontweight='bold')
        
        # Add values on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{count:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add note about sample
        total = sum(counts)
        ax.text(0.98, 0.98, f'Sample: {total} papers', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, fontstyle='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
    else:
        # For raw works list
        citations = [w.get('cited_by_count', 0) for w in works_or_stats]
        if not citations:
            return None
        
        # Create histogram with gradient
        n, bins, patches = ax.hist(citations, bins=20, edgecolor='black', 
                                   linewidth=0.5, alpha=0.8)
        
        # Color bars by height
        cmap = plt.cm.viridis
        for i, (patch, height) in enumerate(zip(patches, n)):
            patch.set_facecolor(cmap(height / max(n)))
        
        ax.set_xlabel('Number of Citations', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Papers', fontsize=10, fontweight='bold')
        
        # Add statistics with colored lines
        mean_cit = np.mean(citations)
        median_cit = np.median(citations)
        ax.axvline(mean_cit, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_cit:.1f}')
        ax.axvline(median_cit, color='blue', linestyle=':', linewidth=2, 
                  label=f'Median: {median_cit:.1f}')
        ax.legend(fontsize=8, frameon=True, edgecolor='black')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, max(n)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Frequency', fontsize=9)
        
        # Add sample size
        ax.text(0.98, 0.98, f'Sample: {len(citations)} papers', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, fontstyle='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig

def create_combined_yearly_charts(consistent_data: Dict[str, Dict], 
                                 years_input: List[int], 
                                 level2_term: Optional[str] = None):
    """Create combined chart with yearly distributions using publication colors"""
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Apply scientific style
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Determine all available years - sort and convert to integers
    years = sorted([int(y) for y in set(years_input)])
    topics = [t for t, data in consistent_data.items() if data['total'] > 0]
    
    if not topics or not years:
        plt.close(fig)
        return None
    
    # Use vibrant color palette for topics
    topic_colors = plt.cm.tab10(np.linspace(0, 1, len(topics)))
    if len(topics) > 10:
        topic_colors = plt.cm.viridis(np.linspace(0, 1, len(topics)))
    
    # Subplot 1: Stacked
    ax = axes[0]
    bottom = np.zeros(len(years))
    
    # Create list to store yearly totals for validation
    yearly_totals = np.zeros(len(years))
    
    for idx, topic in enumerate(topics):
        # Get REAL data from consistent_data
        topic_yearly = consistent_data[topic]['yearly']
        counts = [topic_yearly.get(year, 0) for year in years]
        
        ax.bar(years, counts, bottom=bottom, label=topic, 
               color=topic_colors[idx], edgecolor='black', linewidth=0.5, width=0.8,
               alpha=0.8)
        bottom += counts
        yearly_totals += counts
    
    # Set integer labels on X axis
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right')
    
    ax.set_xlabel('Publication Year', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Publications', fontsize=10, fontweight='bold')
    ax.set_title('A) Stacked Yearly Distribution', fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=8, frameon=True, edgecolor='black', loc='upper left')
    
    # Add total count above plot
    total_papers = int(sum(yearly_totals))
    ax.text(0.5, 0.98, f'Total: {total_papers:,} papers', 
            transform=ax.transAxes, ha='center', va='top', 
            fontsize=9, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Subplot 2: Normalized (by maximum of each topic)
    ax = axes[1]
    
    for idx, topic in enumerate(topics):
        topic_yearly = consistent_data[topic]['yearly']
        counts = np.array([topic_yearly.get(year, 0) for year in years])
        if counts.max() > 0:
            normalized = counts / counts.max()
            ax.plot(years, normalized, marker='o', linewidth=2, markersize=6, 
                   label=topic, color=topic_colors[idx], markerfacecolor='white',
                   markeredgewidth=1.5, markeredgecolor=topic_colors[idx])
    
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right')
    
    ax.set_xlabel('Publication Year', fontsize=10, fontweight='bold')
    ax.set_ylabel('Normalized Intensity (max=1)', fontsize=10, fontweight='bold')
    ax.set_title('B) Normalized by Maximum', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, frameon=True, edgecolor='black', loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    # Subplot 3: Logarithmic scale (absolute values)
    ax = axes[2]
    
    # Find global maximum for Y axis tuning
    all_counts = []
    for topic in topics:
        topic_yearly = consistent_data[topic]['yearly']
        counts = [topic_yearly.get(year, 0) for year in years]
        all_counts.extend(counts)
    max_count = max(all_counts) if all_counts else 1
    
    for idx, topic in enumerate(topics):
        topic_yearly = consistent_data[topic]['yearly']
        counts = [topic_yearly.get(year, 0) for year in years]
        if max(counts) > 0:
            # For log scale, replace 0 with 0.1 (below minimum)
            counts_log = [c if c > 0 else 0.1 for c in counts]
            ax.semilogy(years, counts_log, marker='s', linewidth=2, markersize=6,
                       label=topic, color=topic_colors[idx], markerfacecolor='white',
                       markeredgewidth=1.5, markeredgecolor=topic_colors[idx])
    
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right')
    
    ax.set_xlabel('Publication Year', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Publications (log scale)', fontsize=10, fontweight='bold')
    ax.set_title('C) Logarithmic Scale', fontsize=11, fontweight='bold', pad=10)
    
    # Configure logarithmic Y scale
    y_min = 0.5
    y_max = max_count * 2
    ax.set_ylim(y_min, y_max)
    
    # Add grid lines for log scale
    ax.grid(True, alpha=0.3, linestyle='--', which='both', color='gray')
    ax.legend(fontsize=8, frameon=True, edgecolor='black', loc='best')
    
    plt.suptitle(f'Comparative Yearly Distribution Analysis' + (f' (with {level2_term})' if level2_term else ''), 
                 fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    return fig

def create_scientific_tree_visualization(topic_counts: Dict[str, int], level1_term: str, level2_term: Optional[str] = None):
    """
    Create scientific tree visualization
    """
    topics = [t for t, count in topic_counts.items() if count > 0]
    if not topics:
        return None
    
    # Sort topics descending
    topics_sorted = sorted(topics, key=lambda x: topic_counts[x], reverse=True)
    counts = [topic_counts[t] for t in topics_sorted]
    max_count = max(counts) if counts else 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Apply scientific style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                  labelbottom=False, labelleft=False)
    
    # Node positions
    n_topics = len(topics_sorted)
    
    # Root node
    root_x, root_y = 0.5, 0.9
    
    # Leaf positions (distribute along an arc)
    leaf_x = []
    leaf_y = []
    
    if n_topics == 1:
        leaf_x = [0.5]
        leaf_y = [0.3]
    else:
        # Distribute along arc
        angles = np.linspace(np.pi/4, 3*np.pi/4, n_topics)
        leaf_x = 0.5 + 0.35 * np.cos(angles)
        leaf_y = 0.3 + 0.15 * np.sin(angles)
    
    # Draw connections (branches)
    for i in range(n_topics):
        # Line thickness proportional to publication count
        line_width = 1 + 3 * (counts[i] / max_count)
        
        # Draw branch with slight curve
        x_vals = [root_x, root_x - 0.1 + 0.2 * i / n_topics, leaf_x[i]]
        y_vals = [root_y, root_y - 0.3, leaf_y[i]]
        
        ax.plot(x_vals, y_vals, 'k-', linewidth=line_width, alpha=0.7, solid_capstyle='round')
    
    # Draw root node
    root_size = 300 + 100 * (sum(counts) / max_count) if max_count > 0 else 300
    ax.scatter([root_x], [root_y], s=root_size, c='white', edgecolor='black', 
               linewidth=1.5, zorder=10)
    
    # Add root node text
    root_label = f"{level1_term}"
    if level2_term:
        root_label += f"\n+ {level2_term}"
    ax.annotate(root_label, (root_x, root_y), ha='center', va='center', 
                fontsize=10, fontweight='bold', zorder=11)
    
    # Draw leaf nodes
    for i in range(n_topics):
        # Node size proportional to publication count
        node_size = 200 + 300 * (counts[i] / max_count)
        
        ax.scatter([leaf_x[i]], [leaf_y[i]], s=node_size, c='white', edgecolor='black', 
                   linewidth=1.0, zorder=10)
        
        # Add label
        ax.annotate(f"{topics_sorted[i]}\n({counts[i]:,})", 
                   (leaf_x[i], leaf_y[i]), ha='center', va='center', 
                   fontsize=8, zorder=11)
    
    # Add title
    ax.set_title('Hierarchical Topic Structure\nBranch thickness proportional to publication count', 
                fontsize=12, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_csv(works_by_topic: Dict[str, List[Dict]]) -> bytes:
    """Export results to CSV"""
    all_rows = []
    for topic, works in works_by_topic.items():
        for work in works:
            enriched = enrich_work_data(work)
            enriched['sub_topic'] = topic
            all_rows.append(enriched)
    
    df = pd.DataFrame(all_rows)
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

def export_to_excel(works_by_topic: Dict[str, List[Dict]], highly_cited_papers: List[Dict] = None) -> bytes:
    """Export results to Excel with papers sorted from newest to oldest and additional sheets"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main sheet - собираем все статьи
        all_rows = []
        seen_main = set()  # Track unique papers for main sheet
        
        for topic, works in works_by_topic.items():
            for work in works:
                enriched = enrich_work_data(work)
                
                # Check for duplicates in main sheet
                paper_id = enriched.get('doi', enriched.get('id', enriched.get('title', '')))
                
                if paper_id not in seen_main:
                    enriched['sub_topic'] = topic
                    all_rows.append(enriched)
                    seen_main.add(paper_id)
        
        if all_rows:
            df_all = pd.DataFrame(all_rows)
            
            # СОРТИРОВКА: от новых к старым
            df_all = df_all.sort_values(
                by=['publication_year', 'publication_date'], 
                ascending=[False, False]
            )
            
            df_all.to_excel(writer, sheet_name='All Papers', index=False)
        
        # Separate sheets for each sub-topic (within a topic, duplicates are fine as they're the same paper)
        for topic, works in works_by_topic.items():
            if works:
                topic_rows = []
                seen_topic = set()  # Track unique papers within this topic sheet
                
                for work in works:
                    enriched = enrich_work_data(work)
                    
                    # Check for duplicates within this topic
                    paper_id = enriched.get('doi', enriched.get('id', enriched.get('title', '')))
                    
                    if paper_id not in seen_topic:
                        topic_rows.append(enriched)
                        seen_topic.add(paper_id)
                
                df_topic = pd.DataFrame(topic_rows)
                df_topic = df_topic.sort_values(
                    by=['publication_year', 'publication_date'], 
                    ascending=[False, False]
                )
                
                sheet_name = re.sub(r'[^\w\s-]', '', topic)[:31]
                df_topic.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 4.6. Highly cited papers sheet (85th percentile) - USING UNIQUE PAPERS
        if highly_cited_papers:
            # highly_cited_papers should already be unique from get_highly_cited_papers
            df_highly_cited = pd.DataFrame(highly_cited_papers)
            if not df_highly_cited.empty:
                df_highly_cited = df_highly_cited.sort_values(
                    by=['cited_by_count'], 
                    ascending=[False]
                )
                df_highly_cited.to_excel(writer, sheet_name=f'Highly Cited (85th pct)', index=False)
        
        # 4.6. Top 100 papers by Field-Weighted Citation Impact - USING UNIQUE PAPERS
        top_fwci_papers = get_unique_top_fwci_papers(works_by_topic, top_n=100)
        
        if top_fwci_papers:
            df_fwci = pd.DataFrame(top_fwci_papers)
            df_fwci = df_fwci.sort_values(by=['fwci'], ascending=[False])
            df_fwci.to_excel(writer, sheet_name='Top 100 by FWCI', index=False)
        
        # Formatting
        workbook = writer.book
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': st.session_state['ui_color_palette']['primary'],
            'font_color': 'white',
            'border': 1
        })
        
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            if sheet_name == 'All Papers':
                df = df_all
            elif sheet_name == 'Highly Cited (85th pct)' and highly_cited_papers:
                df = pd.DataFrame(highly_cited_papers)
            elif sheet_name == 'Top 100 by FWCI' and top_fwci_papers:
                df = pd.DataFrame(top_fwci_papers)
            else:
                df = None
                for t, works in works_by_topic.items():
                    if re.sub(r'[^\w\s-]', '', t)[:31] == sheet_name:
                        temp_rows = []
                        seen_temp = set()
                        for work in works:
                            enriched = enrich_work_data(work)
                            paper_id = enriched.get('doi', enriched.get('id', enriched.get('title', '')))
                            if paper_id not in seen_temp:
                                temp_rows.append(enriched)
                                seen_temp.add(paper_id)
                        df = pd.DataFrame(temp_rows)
                        df = df.sort_values(
                            by=['publication_year', 'publication_date'], 
                            ascending=[False, False]
                        )
                        break
            
            if df is not None and not df.empty:
                for col_num, col_name in enumerate(df.columns):
                    worksheet.write(0, col_num, col_name, header_format)
                    max_len = max(
                        df[col_name].astype(str).map(len).max() if not df[col_name].empty else 0,
                        len(str(col_name))
                    ) + 2
                    worksheet.set_column(col_num, col_num, min(max_len, 50))
    
    return output.getvalue()

def generate_pdf_report(works_by_topic: Dict[str, List[Dict]], level1_term: str, level2_term: Optional[str] = None, years: Optional[List[int]] = None, highly_cited_papers: List[Dict] = None) -> Optional[bytes]:
    """Generate PDF report with analysis results including highly cited papers and FWCI analysis"""
    if not PDF_AVAILABLE:
        return None
    
    buffer = io.BytesIO()
    
    # Document setup
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm
    )
    
    styles = getSampleStyleSheet()
    
    # ========== CREATE CUSTOM STYLES ==========
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=reportlab_colors.HexColor('#2C3E50'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=reportlab_colors.HexColor('#34495E'),
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    topic_style = ParagraphStyle(
        'CustomTopic',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=reportlab_colors.HexColor('#16A085'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    meta_style = ParagraphStyle(
        'CustomMeta',
        parent=styles['Normal'],
        fontSize=10,
        textColor=reportlab_colors.HexColor('#7F8C8D'),
        spaceAfter=3,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    paper_title_style = ParagraphStyle(
        'CustomPaperTitle',
        parent=styles['Heading4'],
        fontSize=11,
        textColor=reportlab_colors.HexColor('#2980B9'),
        spaceAfter=4,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    authors_style = ParagraphStyle(
        'CustomAuthors',
        parent=styles['Normal'],
        fontSize=9,
        textColor=reportlab_colors.HexColor('#2C3E50'),
        spaceAfter=2,
        alignment=TA_LEFT,
        fontName='Helvetica'
    )
    
    details_style = ParagraphStyle(
        'CustomDetails',
        parent=styles['Normal'],
        fontSize=8,
        textColor=reportlab_colors.HexColor('#7F8C8D'),
        spaceAfter=2,
        alignment=TA_LEFT,
        fontName='Helvetica'
    )
    
    metrics_style = ParagraphStyle(
        'CustomMetrics',
        parent=styles['Normal'],
        fontSize=9,
        textColor=reportlab_colors.HexColor('#27AE60'),
        spaceAfter=0,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    footer_style = ParagraphStyle(
        'CustomFooter',
        parent=styles['Normal'],
        fontSize=8,
        textColor=reportlab_colors.HexColor('#95A5A6'),
        spaceBefore=15,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    separator_style = ParagraphStyle(
        'CustomSeparator',
        parent=styles['Normal'],
        fontSize=8,
        textColor=reportlab_colors.HexColor('#BDC3C7'),
        spaceAfter=10,
        spaceBefore=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    story = []
    
    # ========== TITLE PAGE ==========
    
    story.append(Spacer(1, 1*cm))
    
    # Title
    story.append(Paragraph("Publication Clustering Report", title_style))
    story.append(Paragraph("Multi-level Literature Analysis", subtitle_style))
    story.append(Spacer(1, 0.8*cm))
    
    # Query information
    topic_name = level1_term
    if level2_term:
        topic_name += f" + {level2_term}"
    story.append(Paragraph(f"RESEARCH TOPIC:", topic_style))
    story.append(Paragraph(topic_name, subtitle_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Meta information
    current_date = datetime.now().strftime('%B %d, %Y at %H:%M')
    story.append(Paragraph(f"Generated on {current_date}", meta_style))
    
    total_papers = sum(len(works) for works in works_by_topic.values())
    story.append(Paragraph(f"Total papers analyzed: {total_papers:,}", meta_style))
    
    if years:
        year_range = f"{min(years)}-{max(years)}"
        story.append(Paragraph(f"Publication years: {year_range}", meta_style))
    
    story.append(Spacer(1, 1.5*cm))
    
    # Copyright
    story.append(Paragraph("© Publication Clustering Tool", footer_style))
    story.append(Paragraph("Powered by OpenAlex API", footer_style))
    
    # Page break
    story.append(PageBreak())
    
    # ========== TABLE OF CONTENTS ==========
    
    story.append(Paragraph("TABLE OF CONTENTS", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    toc_items = [
        "1. Topic Distribution Summary",
        "2. Highly Cited Papers (>85th percentile)",
        "3. Field-Weighted Citation Impact Analysis (Top 100)",
        "4. Detailed Paper Analysis by Topic"
    ]
    
    for item in toc_items:
        story.append(Paragraph(item, details_style))
    
    story.append(PageBreak())
    
    # ========== TOPIC DISTRIBUTION ==========
    
    story.append(Paragraph("1. TOPIC DISTRIBUTION SUMMARY", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Topic distribution table
    topic_data = [["Topic", "Papers Found", "Percentage"]]
    total_all = sum(len(works) for works in works_by_topic.values())
    
    for topic, works in works_by_topic.items():
        if works:
            count = len(works)
            percentage = (count / total_all * 100) if total_all > 0 else 0
            topic_data.append([topic, f"{count:,}", f"{percentage:.1f}%"])
    
    if len(topic_data) > 1:
        topic_table = Table(topic_data, colWidths=[doc.width/2, doc.width/4, doc.width/4])
        topic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.HexColor('#D5DBDB')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(topic_table)
    
    story.append(PageBreak())
    
    # ========== 4.6. HIGHLY CITED PAPERS (>85th PERCENTILE) ==========
    
    story.append(Paragraph("2. HIGHLY CITED PAPERS (>85th PERCENTILE)", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    if highly_cited_papers and len(highly_cited_papers) > 0:
        # Calculate statistics
        citation_values = [p.get('cited_by_count', 0) for p in highly_cited_papers]
        mean_citations = np.mean(citation_values) if citation_values else 0
        median_citations = np.median(citation_values) if citation_values else 0
        max_citations = max(citation_values) if citation_values else 0
        
        story.append(Paragraph(f"Total highly cited papers: {len(highly_cited_papers)}", subtitle_style))
        story.append(Spacer(1, 0.3*cm))
        
        # Summary statistics table
        stats_data = [
            ["Metric", "Value"],
            ["Total Papers", f"{len(highly_cited_papers):,}"],
            ["Mean Citations", f"{mean_citations:.1f}"],
            ["Median Citations", f"{median_citations:.1f}"],
            ["Maximum Citations", f"{max_citations:,}"],
            ["Percentage of All Papers", f"{(len(highly_cited_papers)/total_all*100):.1f}%" if total_all > 0 else "0%"]
        ]
        
        stats_table = Table(stats_data, colWidths=[doc.width/2, doc.width/3])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.HexColor('#E74C3C')),
            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.HexColor('#D5DBDB')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(stats_table)
        
        story.append(Spacer(1, 0.5*cm))
        
        # Distribution by topic
        topic_distribution = defaultdict(int)
        for paper in highly_cited_papers:
            topic_distribution[paper.get('topic', 'Unknown')] += 1
        
        story.append(Paragraph("Distribution by Topic:", topic_style))
        story.append(Spacer(1, 0.2*cm))
        
        topic_dist_data = [["Topic", "Highly Cited Papers", "Percentage of Topic"]]
        for topic, count in sorted(topic_distribution.items(), key=lambda x: x[1], reverse=True):
            topic_total = len(works_by_topic.get(topic, []))
            topic_pct = (count / topic_total * 100) if topic_total > 0 else 0
            topic_dist_data.append([topic, str(count), f"{topic_pct:.1f}%"])
        
        if len(topic_dist_data) > 1:
            dist_table = Table(topic_dist_data, colWidths=[doc.width/2, doc.width/4, doc.width/4])
            dist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.HexColor('#D5DBDB')),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            story.append(dist_table)
        
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("List of Highly Cited Papers:", topic_style))
        story.append(Spacer(1, 0.3*cm))
        
        for i, paper in enumerate(highly_cited_papers[:50], 1):  # Show top 50 to save space
            # Title
            title = clean_text(paper.get('title', 'No title'))
            story.append(Paragraph(f"{i}. {title}", paper_title_style))
            
            # Authors
            authors = paper.get('authors', [])
            if authors:
                authors_text = ', '.join(authors[:3])
                if len(authors) > 3:
                    authors_text += f' et al. ({len(authors)} authors)'
                story.append(Paragraph(f"Authors: {authors_text}", authors_style))
            
            # Metrics
            metrics = f"Citations: {paper.get('cited_by_count', 0):,} | Year: {paper.get('publication_year', 'N/A')} | Topic: {paper.get('topic', 'N/A')}"
            story.append(Paragraph(metrics, metrics_style))
            
            # DOI
            doi = paper.get('doi', '')
            if doi:
                if doi.startswith('10.'):
                    doi_url = f"https://doi.org/{doi}"
                elif doi.startswith('https://doi.org/'):
                    doi_url = doi
                else:
                    doi_url = f"https://doi.org/{doi}"
                
                doi_link = f'<link href="{doi_url}"><font color="blue"><u>{doi}</u></font></link>'
                story.append(Paragraph(f"DOI: {doi_link}", details_style))
            
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("─" * 30, separator_style))
            story.append(Spacer(1, 0.2*cm))
    else:
        story.append(Paragraph("No highly cited papers found in this dataset.", details_style))
    
    story.append(PageBreak())
    
    # ========== 4.6. FIELD-WEIGHTED CITATION IMPACT ANALYSIS (TOP 100) ==========
    
    story.append(Paragraph("3. FIELD-WEIGHTED CITATION IMPACT ANALYSIS (TOP 100)", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Collect and sort by FWCI
    fwci_papers = []
    seen_fwci = set()
    
    for topic, works in works_by_topic.items():
        for work in works:
            enriched = enrich_work_data(work)
            if enriched.get('fwci') is not None and enriched.get('fwci') > 0:
                # Create unique identifier
                paper_id = enriched.get('doi', enriched.get('id', enriched.get('title', '')))
                
                if paper_id not in seen_fwci:
                    enriched['topic'] = topic
                    fwci_papers.append(enriched)
                    seen_fwci.add(paper_id)
    
    if fwci_papers:
        fwci_papers.sort(key=lambda x: x.get('fwci', 0), reverse=True)
        top_100_fwci = fwci_papers[:100]
        
        # Calculate statistics
        fwci_values = [p.get('fwci', 0) for p in top_100_fwci]
        mean_fwci = np.mean(fwci_values) if fwci_values else 0
        median_fwci = np.median(fwci_values) if fwci_values else 0
        max_fwci = max(fwci_values) if fwci_values else 0
        
        story.append(Paragraph(f"Total papers with FWCI data: {len(fwci_papers)}", subtitle_style))
        story.append(Paragraph(f"Showing Top 100 by FWCI", subtitle_style))
        story.append(Spacer(1, 0.3*cm))
        
        # Summary statistics table
        fwci_stats_data = [
            ["Metric", "Value"],
            ["Papers with FWCI", f"{len(fwci_papers):,}"],
            ["Mean FWCI (Top 100)", f"{mean_fwci:.2f}"],
            ["Median FWCI (Top 100)", f"{median_fwci:.2f}"],
            ["Maximum FWCI", f"{max_fwci:.2f}"],
            ["Papers with FWCI > 3", f"{sum(1 for p in top_100_fwci if p.get('fwci', 0) > 3)}"],
            ["Papers with FWCI > 5", f"{sum(1 for p in top_100_fwci if p.get('fwci', 0) > 5)}"]
        ]
        
        fwci_stats_table = Table(fwci_stats_data, colWidths=[doc.width/2, doc.width/3])
        fwci_stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.HexColor('#F39C12')),
            ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.HexColor('#D5DBDB')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(fwci_stats_table)
        
        story.append(Spacer(1, 0.5*cm))
        
        # Distribution by FWCI ranges
        story.append(Paragraph("FWCI Distribution:", topic_style))
        story.append(Spacer(1, 0.2*cm))
        
        fwci_ranges = {
            '< 1': sum(1 for p in top_100_fwci if p.get('fwci', 0) < 1),
            '1-2': sum(1 for p in top_100_fwci if 1 <= p.get('fwci', 0) < 2),
            '2-3': sum(1 for p in top_100_fwci if 2 <= p.get('fwci', 0) < 3),
            '3-5': sum(1 for p in top_100_fwci if 3 <= p.get('fwci', 0) < 5),
            '> 5': sum(1 for p in top_100_fwci if p.get('fwci', 0) >= 5)
        }
        
        fwci_range_data = [["FWCI Range", "Number of Papers", "Percentage"]]
        for range_name, count in fwci_ranges.items():
            percentage = (count / len(top_100_fwci) * 100) if top_100_fwci else 0
            fwci_range_data.append([range_name, str(count), f"{percentage:.1f}%"])
        
        if len(fwci_range_data) > 1:
            range_table = Table(fwci_range_data, colWidths=[doc.width/3, doc.width/3, doc.width/3])
            range_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.HexColor('#27AE60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.HexColor('#D5DBDB')),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            story.append(range_table)
        
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Top 100 Papers by Field-Weighted Citation Impact:", topic_style))
        story.append(Spacer(1, 0.3*cm))
        
        for i, paper in enumerate(top_100_fwci, 1):
            # Title with FWCI
            title = clean_text(paper.get('title', 'No title'))
            story.append(Paragraph(f"{i}. {title} (FWCI: {paper.get('fwci', 0):.2f})", paper_title_style))
            
            # Authors
            authors = paper.get('authors', [])
            if authors:
                authors_text = ', '.join(authors[:2])
                if len(authors) > 2:
                    authors_text += f' et al.'
                story.append(Paragraph(f"Authors: {authors_text}", authors_style))
            
            # Metrics
            metrics = f"Citations: {paper.get('cited_by_count', 0):,} | Year: {paper.get('publication_year', 'N/A')} | Topic: {paper.get('topic', 'N/A')}"
            story.append(Paragraph(metrics, details_style))
            
            # DOI
            doi = paper.get('doi', '')
            if doi:
                if doi.startswith('10.'):
                    doi_url = f"https://doi.org/{doi}"
                elif doi.startswith('https://doi.org/'):
                    doi_url = doi
                else:
                    doi_url = f"https://doi.org/{doi}"
                
                doi_link = f'<link href="{doi_url}"><font color="blue"><u>{doi}</u></font></link>'
                story.append(Paragraph(f"DOI: {doi_link}", details_style))
            
            story.append(Spacer(1, 0.1*cm))
            
            if i < len(top_100_fwci):
                story.append(Paragraph("─" * 20, separator_style))
                story.append(Spacer(1, 0.1*cm))
    else:
        story.append(Paragraph("No Field-Weighted Citation Impact data available for these papers.", details_style))
    
    story.append(PageBreak())
    
    # ========== DETAILED ANALYSIS ==========
    
    story.append(Paragraph("4. DETAILED PAPER ANALYSIS BY TOPIC", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    for topic, works in works_by_topic.items():
        if works:
            story.append(Paragraph(f"Topic: {topic}", topic_style))
            story.append(Spacer(1, 0.3*cm))
            
            # Count highly cited in this topic
            topic_highly_cited = [p for p in (highly_cited_papers or []) if p.get('topic') == topic]
            if topic_highly_cited:
                story.append(Paragraph(f"★ {len(topic_highly_cited)} highly cited papers in this topic", metrics_style))
                story.append(Spacer(1, 0.2*cm))
            
            for i, work in enumerate(works[:20], 1):  # Show top 20 per topic to save space
                enriched = enrich_work_data(work)
                
                # Check if highly cited
                is_highly_cited = any(p.get('doi') == enriched.get('doi') for p in (highly_cited_papers or []))
                star_prefix = "★ " if is_highly_cited else ""
                
                # Title
                title = clean_text(enriched.get('title', 'No title'))
                story.append(Paragraph(f"{i}. {star_prefix}{title}", paper_title_style))
                
                # Authors
                authors = enriched.get('authors', [])
                if authors:
                    authors_text = ', '.join(authors[:3])
                    if len(authors) > 3:
                        authors_text += f' et al. ({len(authors)} authors)'
                    story.append(Paragraph(f"Authors: {authors_text}", authors_style))
                
                # Metrics
                fwci_text = f" | FWCI: {enriched.get('fwci', 0):.2f}" if enriched.get('fwci') is not None else ""
                metrics = f"Citations: {enriched.get('cited_by_count', 0):,} | Year: {enriched.get('publication_year', 'N/A')} | OA: {'Yes' if enriched.get('is_oa') else 'No'}{fwci_text}"
                story.append(Paragraph(metrics, metrics_style))
                
                # DOI
                doi = enriched.get('doi', '')
                if doi:
                    if doi.startswith('10.'):
                        doi_url = f"https://doi.org/{doi}"
                    elif doi.startswith('https://doi.org/'):
                        doi_url = doi
                    else:
                        doi_url = f"https://doi.org/{doi}"
                    
                    doi_link = f'<link href="{doi_url}"><font color="blue"><u>{doi}</u></font></link>'
                    story.append(Paragraph(f"DOI: {doi_link}", details_style))
                
                story.append(Spacer(1, 0.1*cm))
            
            story.append(PageBreak())
    
    # ========== CONCLUSION ==========
    
    story.append(Paragraph("CONCLUSION", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Collect final statistics
    n_highly_cited = len(highly_cited_papers) if highly_cited_papers else 0
    n_with_fwci = len([p for topic, works in works_by_topic.items() for p in works if enrich_work_data(p).get('fwci') is not None])
    
    conclusions = [
        f"This report analyzed {total_papers:,} papers across {len([t for t, w in works_by_topic.items() if w])} topics.",
        f"Identified {n_highly_cited} highly cited papers (>85th percentile), representing the most impactful research in this field.",
        f"Field-Weighted Citation Impact data available for {n_with_fwci} papers, with top performers highlighted.",
        "The analysis provides comprehensive insights into the distribution, impact, and trends of research in these areas.",
        "For the most current data, please visit the original sources via the provided DOIs."
    ]
    
    for conclusion in conclusions:
        story.append(Paragraph(f"• {conclusion}", details_style))
    
    # Footer
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("© Publication Clustering Tool - Generated Automatically", footer_style))
    story.append(Paragraph(f"Report ID: {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}", footer_style))
    
    # ========== GENERATE PDF ==========
    
    doc.build(story)
    return buffer.getvalue()

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main application function"""
    
    # Header
    st.markdown(f'<h1 class="main-header">Publication Clustering</h1>', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-size: 1rem; color: {ui_colors['text']}; margin-bottom: 1.5rem;">
    Multi-level literature search with topic clustering and advanced bibliometric analysis
    </p>
    """, unsafe_allow_html=True)
    
    # Display current UI theme info
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.8rem; color: {ui_colors['primary']}; margin-bottom: 0.5rem;">
        UI Theme: {ui_colors['name']}
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # COLOR PALETTE SELECTORS (NEW)
    # ========================================================================
    
    with st.sidebar:
        st.markdown("### 🎨 Plot Color Settings")
        st.markdown("These settings affect only the scientific plots, not the UI")
        
        # Discrete palette selector
        discrete_options = list(COLOR_PALETTES_SCIENTIFIC_DISCRETE.keys())
        selected_discrete = st.selectbox(
            "Discrete Color Palette",
            options=discrete_options,
            index=discrete_options.index(st.session_state.get('discrete_palette_name', 'nature')),
            key='discrete_palette_selector'
        )
        st.session_state['discrete_palette_name'] = selected_discrete
        
        # Show preview of discrete palette
        preview_colors = COLOR_PALETTES_SCIENTIFIC_DISCRETE[selected_discrete][:7]
        preview_html = '<div style="display: flex; gap: 5px; margin: 10px 0;">'
        for color in preview_colors:
            preview_html += f'<div style="width: 30px; height: 20px; background-color: {color}; border: 1px solid black;"></div>'
        preview_html += '</div>'
        st.markdown(preview_html, unsafe_allow_html=True)
        
        # Gradient palette selector
        gradient_options = list(GRADIENT_PALETTES.keys())
        selected_gradient = st.selectbox(
            "Gradient Color Map",
            options=gradient_options,
            index=gradient_options.index(st.session_state.get('gradient_palette_name', 'viridis')),
            key='gradient_palette_selector'
        )
        st.session_state['gradient_palette_name'] = selected_gradient
        
        # Show preview of gradient
        st.markdown("Preview:")
        gradient_preview = np.linspace(0, 1, 100).reshape(1, -1)
        fig_preview, ax_preview = plt.subplots(figsize=(5, 0.5))
        ax_preview.imshow(gradient_preview, aspect='auto', cmap=GRADIENT_PALETTES[selected_gradient])
        ax_preview.set_xticks([])
        ax_preview.set_yticks([])
        for spine in ax_preview.spines.values():
            spine.set_visible(False)
        st.pyplot(fig_preview)
        plt.close(fig_preview)
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state['step'] = 1
    if 'results' not in st.session_state:
        st.session_state['results'] = {}
    if 'topic_counts' not in st.session_state:
        st.session_state['topic_counts'] = {}
    if 'level1_count' not in st.session_state:
        st.session_state['level1_count'] = 0
    if 'level2_count' not in st.session_state:
        st.session_state['level2_count'] = 0
    if 'level1_input' not in st.session_state:
        st.session_state['level1_input'] = '"metal-organic frameworks" OR MOF'
    if 'level2_input' not in st.session_state:
        st.session_state['level2_input'] = ''
    if 'level3_input' not in st.session_state:
        st.session_state['level3_input'] = ['MIL', 'ZIF', 'IRMOF', 'UiO', 'HKUST']
    if 'years_input' not in st.session_state:
        st.session_state['years_input'] = list(range(2000, 2026))
    if 'consistent_data' not in st.session_state:
        st.session_state['consistent_data'] = {}
    
    # ========================================================================
    # STEP 1: TERM INPUT
    # ========================================================================
    
    if st.session_state.step == 1:
        st.markdown(f"""
        <div class="step-card">
            <h3 style="margin: 0; font-size: 1.3rem;">📥 Step 1: Enter Search Terms</h3>
            <p style="margin: 5px 0; font-size: 0.9rem;">Define your multi-level search query</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced instructions
        st.markdown(f"""
        <div class="info-message">
            <strong>📚 How the multi-level search works:</strong><br><br>
            • <b>Level 1 (Main domain)</b> - broad research area that defines the overall scope<br>
            • <b>Level 2 (Optional refinement)</b> - narrows down the search within Level 1<br>
            • <b>Level 3 terms (Classification topics)</b> - these are the specific sub-topics that will be used for <b>clustering and classification</b>. 
            The system will count papers matching each Level 3 term within the context of Level 1+2, and then fetch the most relevant papers for detailed analysis.
            <br><br>
            <i>Example: If Level 1 is "metal-organic frameworks" and Level 3 terms are "MIL", "ZIF", "UiO" - 
            the system will create separate clusters for each MOF family and analyze their publication patterns.</i>
        </div>
        
        <div class="info-message" style="margin-top: 10px;">
            <strong>💡 Search Syntax Tips:</strong><br>
            • Use <b>OR</b> for logical OR (e.g., "MOF OR COF")<br>
            • Use quotes for exact phrases (e.g., "metal-organic frameworks")<br>
            • Use <b>*</b> for wildcard (e.g., "electroly*" matches electrolyte, electrolysis, electrolyzer)<br>
            • Multiple words without OR are treated as a phrase automatically
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Level 1 (required):**")
            level1 = st.text_input(
                "Main domain (broad research area)",
                value=st.session_state['level1_input'],
                key="level1_input_widget",
                label_visibility="collapsed",
                placeholder="e.g., \"machine learning\" OR \"artificial intelligence\""
            )
            
            st.markdown("**Level 2 (optional):**")
            level2 = st.text_input(
                "Refinement term (narrows down Level 1)",
                value=st.session_state['level2_input'] if st.session_state['level2_input'] else '',
                key="level2_input_widget",
                label_visibility="collapsed",
                placeholder="e.g., \"neural networks\" OR deep learning"
            )
        
        with col2:
            st.markdown("**Level 3 terms (one per line - these will become your clusters):**")
            level3_text = st.text_area(
                "Sub-topics for classification",
                value='\n'.join(st.session_state['level3_input']),
                height=120,
                key="level3_input_widget",
                label_visibility="collapsed",
                placeholder="Enter each sub-topic on a new line"
            )
        
        # Year filter
        st.markdown("---")
        st.markdown("**📅 Publication Years:**")
        
        current_year = datetime.now().year
        
        year_option = st.radio(
            "Year filter type",
            ["Range", "Single year", "Multiple years"],
            horizontal=True,
            key="year_type",
            index=0
        )
        
        if year_option == "Single year":
            years = [st.slider("Select year", 2000, current_year, current_year)]
        elif year_option == "Range":
            default_range = (2000, current_year)
            year_range = st.slider("Select range", 2000, current_year, default_range)
            years = list(range(year_range[0], year_range[1] + 1))
        else:  # Multiple years
            default_years = [current_year-2, current_year-1, current_year]
            years = st.multiselect(
                "Select years",
                list(range(current_year, 2000-1, -1)),
                default=default_years
            )
        
        # Test query
        with st.expander("🔧 Test Query Before Full Analysis"):
            if st.button("Test Current Query"):
                with st.spinner("Testing query..."):
                    level2_for_test = level2.strip() if level2 and level2.strip() else None
                    temp_count = get_total_count(level1.strip(), level2_for_test, years)
                    
                    if temp_count > 0:
                        st.success(f"✅ Found {temp_count:,} papers matching your Level 1+2 criteria")
                        
                        parsed_level2 = parse_query_terms(level2.strip()) if level2 and level2.strip() else '(not specified)'
                        st.markdown(f"""
                        <div class="info-message">
                            <strong>Query Analysis:</strong><br>
                            • Parsed Level 1: {parse_query_terms(level1.strip())}<br>
                            • Parsed Level 2: {parsed_level2}<br>
                            • Years: {min(years)}-{max(years)}<br>
                            • Total papers: {temp_count:,}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("""
                        ❌ No results found! Try:
                        - Using fewer or more general terms
                        - Checking your spelling
                        - Expanding the year range
                        - Using quotes for exact phrases
                        - Using wildcard (*) for word variations
                        """)
        
        # Start button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
                if not level1 or not level1.strip():
                    st.error("❌ Please enter Level 1 term")
                elif not level3_text or not level3_text.strip():
                    st.error("❌ Please enter at least one Level 3 term")
                else:
                    # Save to session with proper None handling
                    st.session_state['level1_input'] = level1.strip()
                    st.session_state['level2_input'] = level2.strip() if level2 and level2.strip() else None
                    st.session_state['level3_input'] = [t.strip() for t in level3_text.split('\n') if t.strip()]
                    st.session_state['years_input'] = years
                    st.session_state['step'] = 2
                    st.rerun()
    
    # ========================================================================
    # STEP 2: ANALYSIS
    # ========================================================================
    
    elif st.session_state.step == 2:
        st.markdown(f"""
        <div class="step-card">
            <h3 style="margin: 0; font-size: 1.3rem;">🔍 Step 2: Analysis in Progress</h3>
            <p style="margin: 5px 0; font-size: 0.9rem;">Fetching data from OpenAlex...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show query parameters
        st.markdown(f"""
        <div class="filter-stats">
            <strong>Query Parameters:</strong><br>
            Level 1: {st.session_state.level1_input}<br>
            Level 2: {st.session_state.level2_input or '(not specified)'}<br>
            Level 3: {', '.join(st.session_state.level3_input)}<br>
            Years: {min(st.session_state.years_input)}-{max(st.session_state.years_input)}
        </div>
        """, unsafe_allow_html=True)
        
        # Back button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("← Back to Step 1", key="back_from_step2"):
                st.session_state.step = 1
                st.rerun()
        
        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(value, message):
            progress_bar.progress(value)
            status_text.text(message)
        
        try:
            # Step 1: Level 1 count
            update_progress(0.1, "Getting Level 1 count...")
            st.session_state['level1_count'] = get_total_count(
                st.session_state['level1_input'], None, st.session_state['years_input']
            )
            
            # Step 2: Level 2 count (if applicable)
            if st.session_state['level2_input']:
                update_progress(0.2, "Getting Level 2 count...")
                st.session_state['level2_count'] = get_total_count(
                    st.session_state['level1_input'], st.session_state['level2_input'], st.session_state['years_input']
                )
            else:
                st.session_state['level2_count'] = st.session_state['level1_count']
            
            # Step 3: Get CONSISTENT data using hybrid approach with enhanced citation distribution
            update_progress(0.3, "Analyzing Level 3 terms with group_by...")
            st.session_state['consistent_data'] = get_consistent_topic_data(
                st.session_state['level1_input'],
                st.session_state['level2_input'],
                st.session_state['level3_input'],
                st.session_state['years_input'],
                max_papers_to_fetch=100,
                progress_callback=lambda p, m: update_progress(0.3 + p*0.6, m)
            )
            
            # Step 4: Extract topic_counts and results from consistent_data for backward compatibility
            st.session_state['topic_counts'] = {
                term: data['total'] 
                for term, data in st.session_state['consistent_data'].items()
            }
            
            st.session_state['results'] = {
                term: data['top_works'] 
                for term, data in st.session_state['consistent_data'].items()
            }
            
            update_progress(1.0, "✅ Analysis complete!")
            time.sleep(0.5)
            
            st.session_state['step'] = 3
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            if st.button("← Back to Step 1"):
                st.session_state['step'] = 1
                st.rerun()
    
    # ========================================================================
    # STEP 3: RESULTS - UPDATED WITH NEW ANALYTICS
    # ========================================================================
    
    elif st.session_state.step == 3:
        st.markdown(f"""
        <div class="step-card">
            <h3 style="margin: 0; font-size: 1.3rem;">📊 Step 3: Results</h3>
            <p style="margin: 5px 0; font-size: 0.9rem;">Analysis complete - review the findings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use consistent_data for all visualizations
        consistent_data = st.session_state.get('consistent_data', {})
        
        # Navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
        
        with nav_col1:
            if st.button("← Back to Step 1", key="back_from_step3"):
                st.session_state.step = 1
                st.rerun()
        
        with nav_col2:
            if st.button("🔄 New Search", key="new_from_step3"):
                # Clear session but keep terms for Step 1
                level1 = st.session_state.get('level1_input', '')
                level2 = st.session_state.get('level2_input', '')
                level3 = st.session_state.get('level3_input', [])
                years = st.session_state.get('years_input', [])
                
                for key in ['step', 'results', 'topic_counts', 'level1_count', 'level2_count', 'consistent_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.session_state.step = 1
                st.session_state['level1_input'] = level1
                st.session_state['level2_input'] = level2
                st.session_state['level3_input'] = level3
                st.session_state['years_input'] = years
                st.rerun()
        
        # Show data consistency info
        st.markdown(f"""
        <div class="info-message">
            <strong>✅ Data Consistency Note:</strong><br>
            All charts use the SAME source data from group_by queries.<br>
            • Topic totals are calculated from yearly distributions<br>
            • Yearly distributions sum exactly to topic totals<br>
            • Citation distributions are now calculated for ALL papers using group_by (7 requests per topic)<br>
            • All visualizations use scientific publication style independent of UI theme
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Level 1 Papers", st.session_state.level1_count, "📄")
        
        with col2:
            create_metric_card("After Level 2", st.session_state.level2_count, "🔍")
        
        with col3:
            total_found = sum(len(works) for works in st.session_state.results.values())
            create_metric_card("Top Papers Found", total_found, "🎯")
        
        with col4:
            topics_with_results = sum(1 for data in consistent_data.values() if data['total'] > 0)
            create_metric_card("Topics with results", topics_with_results, "✅")
        
        st.markdown("---")
        
        # Topic distribution info
        st.markdown(f"""
        <div class="info-message">
            <strong>📊 Topic Distribution Analysis:</strong><br>
            Total papers matching Level 1+2 filters: {st.session_state.level2_count:,}<br>
            Sum of papers in all sub-topics: {sum(st.session_state.topic_counts.values()):,}<br>
            <i>Note: Papers containing multiple sub-topic keywords are counted in each category, 
            so the sum may exceed the total.</i>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for different views - UPDATED: REMOVED CLUSTER GRAPH, ADDED ADVANCED
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Topic Distribution", 
            "📋 Papers by Topic", 
            "📥 Export",
            "📊 Advanced Bibliometrics"
        ])
        
        with tab1:
            # 4.1. Sub-topic Distribution
            st.markdown('<div class="scientific-plot">', unsafe_allow_html=True)
            st.markdown("<h4>4.1. Sub-topic Distribution Analysis</h4>", unsafe_allow_html=True)
            
            col_abs, col_pct = st.columns(2)
            
            with col_abs:
                # 4.1.1. Absolute counts
                fig_abs = create_subtopic_distribution_absolute(
                    st.session_state.topic_counts,
                    st.session_state.level2_count
                )
                if fig_abs:
                    st.pyplot(fig_abs)
                    plt.close(fig_abs)
                else:
                    st.info("No data for absolute distribution")
            
            with col_pct:
                # 4.1.2. Percentage distribution
                fig_pct = create_subtopic_distribution_percentage(
                    st.session_state.topic_counts,
                    st.session_state.level2_count
                )
                if fig_pct:
                    st.pyplot(fig_pct)
                    plt.close(fig_pct)
                else:
                    st.info("No data for percentage distribution")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 4.2. Comparative Yearly Distribution
            st.markdown('<div class="scientific-plot">', unsafe_allow_html=True)
            st.markdown("<h4>4.2. Comparative Yearly Distribution Analysis</h4>", unsafe_allow_html=True)
            
            if consistent_data:
                col_stack, col_group = st.columns(2)
                
                with col_stack:
                    # 4.2.1. Stacked
                    fig_stack = create_stacked_yearly_chart(
                        consistent_data,
                        st.session_state.years_input,
                        st.session_state.level2_input
                    )
                    if fig_stack:
                        st.pyplot(fig_stack)
                        plt.close(fig_stack)
                    else:
                        st.info("No data for stacked chart")
                
                with col_group:
                    # 4.2.2. Grouped
                    fig_group = create_grouped_yearly_chart(
                        consistent_data,
                        st.session_state.years_input,
                        st.session_state.level2_input
                    )
                    if fig_group:
                        st.pyplot(fig_group)
                        plt.close(fig_group)
                    else:
                        st.info("No data for grouped chart")
                
                # 4.2.3. Normalized
                fig_norm = create_normalized_yearly_chart(
                    consistent_data,
                    st.session_state.years_input,
                    st.session_state.level2_input
                )
                if fig_norm:
                    st.pyplot(fig_norm)
                    plt.close(fig_norm)
                else:
                    st.info("No data for normalized chart")
            else:
                st.info("No consistent data available")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 4.3. Analysis for each Sub-topic
            st.markdown('<div class="scientific-plot">', unsafe_allow_html=True)
            st.markdown("<h4>4.3. Detailed Sub-topic Analysis</h4>", unsafe_allow_html=True)
            
            for term, data in consistent_data.items():
                if data['total'] > 0:
                    st.markdown(f"#### {term}")
                    
                    col_pub, col_cit = st.columns(2)
                    
                    with col_pub:
                        # 4.3.1. Yearly publications
                        fig_pub = create_topic_yearly_publications(
                            term, data, st.session_state.years_input
                        )
                        if fig_pub:
                            st.pyplot(fig_pub)
                            plt.close(fig_pub)
                    
                    with col_cit:
                        # 4.3.2. Yearly citations
                        fig_cit = create_topic_yearly_citations(
                            term, data, st.session_state.years_input
                        )
                        if fig_cit:
                            st.pyplot(fig_cit)
                            plt.close(fig_cit)
                    
                    # 4.3.3. Citation distribution
                    fig_dist = create_topic_citation_distribution(term, data)
                    if fig_dist:
                        st.pyplot(fig_dist)
                        plt.close(fig_dist)
                    
                    st.markdown("---")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 4.4. Top Journals Analysis
            st.markdown('<div class="scientific-plot">', unsafe_allow_html=True)
            st.markdown("<h4>4.4. Top Journals Analysis</h4>", unsafe_allow_html=True)
            
            journal_data = get_journal_data(st.session_state.results)
            topics = [t for t, d in consistent_data.items() if d['total'] > 0]
            
            if journal_data:
                col_j1, col_j2 = st.columns(2)
                
                with col_j1:
                    # 4.4.1. Top journals bar
                    fig_j1 = create_top_journals_bar(journal_data, top_n=15)
                    if fig_j1:
                        st.pyplot(fig_j1)
                        plt.close(fig_j1)
                
                with col_j2:
                    # 4.4.2. Stacked by topics
                    fig_j2 = create_top_journals_stacked(journal_data, topics, top_n=15)
                    if fig_j2:
                        st.pyplot(fig_j2)
                        plt.close(fig_j2)
                
                # 4.4.3. Average citations
                fig_j3 = create_top_journals_citations(journal_data, top_n=15)
                if fig_j3:
                    st.pyplot(fig_j3)
                    plt.close(fig_j3)
            else:
                st.info("No journal data available")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 4.5. Citation Velocity
            st.markdown('<div class="scientific-plot">', unsafe_allow_html=True)
            st.markdown("<h4>4.5. Citation Velocity Analysis</h4>", unsafe_allow_html=True)
            
            fig_vel = create_citation_velocity_chart(consistent_data)
            if fig_vel:
                st.pyplot(fig_vel)
                plt.close(fig_vel)
            else:
                st.info("No data for citation velocity analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            # 4.6. Papers by Topic (includes highly cited indicator)
            highly_cited_papers = get_highly_cited_papers(st.session_state.results, percentile=85)
            
            st.markdown(f"""
            <div class="info-message">
                <strong>📋 Highly Cited Papers (>85th percentile):</strong> {len(highly_cited_papers)} papers identified.
                These will be exported to a separate sheet in Excel and included in PDF report.
            </div>
            """, unsafe_allow_html=True)
            
            for term, works in st.session_state.results.items():
                if works:
                    # Mark highly cited papers in this topic
                    topic_highly_cited = [p for p in highly_cited_papers if p.get('topic') == term]
                    
                    with st.expander(f"📚 {term} - {len(works)} papers ({len(topic_highly_cited)} highly cited)"):
                        for i, work in enumerate(works[:20], 1):
                            enriched = enrich_work_data(work)
                            # Check if highly cited
                            is_highly_cited = any(p.get('doi') == enriched.get('doi') for p in highly_cited_papers)
                            if is_highly_cited:
                                enriched['title'] = "⭐ " + enriched.get('title', '')
                            create_result_card(enriched, i, term)
                            
                            if i < len(works[:20]):
                                st.markdown("---")
        
        with tab3:
            # Export (was tab4) - updated with highly cited papers
            st.markdown("### 📥 Export Results")
            
            highly_cited_papers = get_highly_cited_papers(st.session_state.results, percentile=85)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export
                csv_data = export_to_csv(st.session_state.results)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"publication_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel export with additional sheets
                excel_data = export_to_excel(st.session_state.results, highly_cited_papers)
                st.download_button(
                    label="📈 Download Excel",
                    data=excel_data,
                    file_name=f"publication_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                # PDF export
                if PDF_AVAILABLE:
                    pdf_data = generate_pdf_report(
                        st.session_state.results,
                        st.session_state.level1_input,
                        st.session_state.level2_input,
                        st.session_state.years_input,
                        highly_cited_papers
                    )
                    if pdf_data:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_data,
                            file_name=f"publication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.button("📄 PDF Report", disabled=True, use_container_width=True)
                else:
                    st.warning("PDF export requires reportlab. Install with: pip install reportlab")
                    st.button("📄 PDF Report", disabled=True, use_container_width=True)
        
        with tab4:
            # New tab: Advanced Bibliometrics (5 additional graphs)
            st.markdown('<div class="scientific-plot">', unsafe_allow_html=True)
            st.markdown("<h4>📊 Advanced Bibliometric Analysis</h4>", unsafe_allow_html=True)
            
            # Create 2x3 grid for 6 graphs
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                # Graph 1: Matthew Effect Analysis
                st.markdown("**1. Matthew Effect Analysis**")
                fig_matthew = create_matthew_effect_analysis(consistent_data)
                if fig_matthew:
                    st.pyplot(fig_matthew)
                    plt.close(fig_matthew)
                else:
                    st.info("Insufficient data for Matthew effect analysis")
            
            with col_a2:
                # Graph 2: Citation Half-Life
                st.markdown("**2. Citation Half-Life Analysis**")
                fig_half_life = create_citation_half_life(consistent_data)
                if fig_half_life:
                    st.pyplot(fig_half_life)
                    plt.close(fig_half_life)
                else:
                    st.info("Insufficient data for half-life analysis")
            
            # Second row
            col_b1, col_b2 = st.columns(2)
            
            with col_b1:
                # Graph 3: Collaboration Intensity
                st.markdown("**3. Collaboration Network Intensity**")
                fig_collab = create_collaboration_intensity(consistent_data, st.session_state.years_input)
                if fig_collab:
                    st.pyplot(fig_collab)
                    plt.close(fig_collab)
                else:
                    st.info("Insufficient data for collaboration analysis")
            
            with col_b2:
                # Graph 4: Journal Impact Analysis
                st.markdown("**4. Journal Impact vs Citation Performance**")
                fig_journal_impact = create_journal_if_vs_citations(consistent_data)
                if fig_journal_impact:
                    st.pyplot(fig_journal_impact)
                    plt.close(fig_journal_impact)
                else:
                    st.info("Insufficient journal data for analysis")
            
            # Third row
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                # Graph 5: Research Front Velocity
                st.markdown("**5. Research Front Velocity**")
                fig_front = create_research_front_velocity(consistent_data)
                if fig_front:
                    st.pyplot(fig_front)
                    plt.close(fig_front)
                else:
                    st.info("Insufficient data for research front analysis")
            
            with col_c2:
                # Graph 6: Lorenz Curve (overall citation inequality)
                st.markdown("**6. Overall Citation Inequality (Lorenz Curve)**")
                fig_lorenz = create_lorenz_curve(st.session_state.results, 
                                                 "Citation Distribution - All Topics Combined")
                if fig_lorenz:
                    st.pyplot(fig_lorenz)
                    plt.close(fig_lorenz)
                else:
                    st.info("Insufficient data for Lorenz curve analysis")
            
            # Add interpretation guide
            st.markdown("""
            <div class="info-message" style="margin-top: 20px;">
                <strong>📈 Interpretation Guide:</strong><br>
                • <b>Matthew Effect:</b> Shows concentration of citations - higher concentration indicates "rich get richer" phenomenon<br>
                • <b>Citation Half-Life:</b> Median age of cited papers - longer half-life indicates slower knowledge obsolescence<br>
                • <b>Collaboration Intensity:</b> Average authors per paper over time - increasing trend suggests growing team science<br>
                • <b>Journal Impact:</b> Relationship between publication volume and citation impact<br>
                • <b>Research Front Velocity:</b> Share of recent papers - higher values indicate "hotter" research areas<br>
                • <b>Lorenz Curve:</b> Visual representation of citation inequality - greater deviation from diagonal = higher inequality
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()








