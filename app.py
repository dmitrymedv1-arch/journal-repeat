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
# NEW IMPORTS FOR DOCX PROCESSING AND NAME PARSING
# ============================================================================
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("python-docx not installed. DOCX upload will be disabled. Install with: pip install python-docx")

try:
    from nameparser import HumanName
    NAMEPARSER_AVAILABLE = True
except ImportError:
    NAMEPARSER_AVAILABLE = False
    st.warning("nameparser not installed. Author name parsing will be basic. Install with: pip install nameparser")

# ============================================================================
# PDF IMPORTS
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
# SCIENTIFIC COLOR PALETTES
# ============================================================================

COLOR_PALETTES_SCIENTIFIC_DISCRETE = {
    'nature': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85'],
    'science': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'colorblind': ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000', '#999999', '#882255'],
}

GRADIENT_PALETTES = {
    'viridis': plt.cm.viridis,
    'plasma': plt.cm.plasma,
    'inferno': plt.cm.inferno,
    'magma': plt.cm.magma,
    'cividis': plt.cm.cividis,
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Author Journal Publication Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# OPENALEX API CONFIGURATION
# ============================================================================

OPENALEX_BASE_URL = "https://api.openalex.org"
MAILTO = "your-email@example.com"  # Change to your email
POLITE_POOL_HEADER = {'User-Agent': f'Author-Journal-Analyzer (mailto:{MAILTO})'}

RATE_LIMIT_PER_SECOND = 8
MAX_RETRIES = 3
INITIAL_DELAY = 1
MAX_DELAY = 60

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_issn(issn: str) -> str:
    """Clean ISSN input (remove spaces, hyphens, etc.)"""
    if not issn:
        return ""
    # Remove all non-alphanumeric characters
    cleaned = re.sub(r'[^0-9Xx]', '', issn.strip().upper())
    return cleaned

def validate_issn(issn: str) -> bool:
    """Basic ISSN validation (length and format)"""
    cleaned = clean_issn(issn)
    # ISSN can be 8 characters (with possible X at end)
    return len(cleaned) == 8 and (cleaned[:-1].isdigit() and (cleaned[-1].isdigit() or cleaned[-1] == 'X'))

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_DELAY, max=MAX_DELAY)
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

# ============================================================================
# AUTHOR NAME PARSING FUNCTIONS
# ============================================================================

def parse_author_name(author_string: str) -> Dict[str, str]:
    """
    Parse author name string to extract last name and first initial.
    Handles various formats: "Smith J.", "Smith, J.", "Smith, John", "J. Smith", etc.
    """
    author_string = author_string.strip()
    if not author_string:
        return {"last_name": "", "first_initial": ""}
    
    # Try using nameparser if available
    if NAMEPARSER_AVAILABLE:
        try:
            name = HumanName(author_string)
            if name.last and name.first:
                first_initial = name.first[0].upper() if name.first else ""
                return {
                    "last_name": name.last,
                    "first_initial": first_initial
                }
        except:
            pass
    
    # Fallback to regex patterns
    
    # Pattern 1: "Last, First" or "Last, F." or "Last, F"
    pattern1 = r'^([^,]+),\s*([A-Za-z])\.?\s*(?:[A-Za-z]*)?$'
    match = re.match(pattern1, author_string)
    if match:
        return {
            "last_name": match.group(1).strip(),
            "first_initial": match.group(2).upper()
        }
    
    # Pattern 2: "First Last" - extract first letter of first word as initial
    pattern2 = r'^([A-Za-z])\.?\s+([A-Za-z]+)$'
    match = re.match(pattern2, author_string)
    if match:
        return {
            "last_name": match.group(2),
            "first_initial": match.group(1).upper()
        }
    
    # Pattern 3: "F. Last"
    pattern3 = r'^([A-Za-z])\.\s+([A-Za-z]+)$'
    match = re.match(pattern3, author_string)
    if match:
        return {
            "last_name": match.group(2),
            "first_initial": match.group(1).upper()
        }
    
    # Pattern 4: "Last F" (no comma, no dot)
    pattern4 = r'^([A-Za-z]+)\s+([A-Za-z])$'
    match = re.match(pattern4, author_string)
    if match:
        return {
            "last_name": match.group(1),
            "first_initial": match.group(2).upper()
        }
    
    # If all else fails, assume the last word is the last name and first character is initial
    parts = author_string.split()
    if len(parts) >= 2:
        return {
            "last_name": parts[-1],
            "first_initial": parts[0][0].upper()
        }
    elif len(parts) == 1:
        return {
            "last_name": parts[0],
            "first_initial": ""
        }
    
    return {"last_name": "", "first_initial": ""}

def extract_authors_from_docx(docx_file) -> List[Dict[str, str]]:
    """
    Extract author names from first page of DOCX file.
    Returns list of dicts with 'last_name' and 'first_initial'.
    Also attempts to extract affiliations.
    """
    if not DOCX_AVAILABLE:
        st.error("python-docx is required for DOCX processing")
        return []
    
    doc = Document(docx_file)
    
    # Get text from first page (first few paragraphs)
    first_page_text = []
    for para in doc.paragraphs[:20]:  # First 20 paragraphs should cover first page
        if para.text.strip():
            first_page_text.append(para.text.strip())
    
    full_text = ' '.join(first_page_text)
    
    # Common patterns for author sections
    author_patterns = [
        r'Authors?:\s*(.*?)(?:\n|\.\.\.|$)',
        r'By\s+(.*?)(?:\n|\.\.\.|$)',
        r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]+)+)',
    ]
    
    author_string = ""
    for pattern in author_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
        if match:
            author_string = match.group(1)
            break
    
    if not author_string:
        # Try to find lines that look like author names (multiple capitalized words with commas/and)
        lines = full_text.split('\n')
        for line in lines:
            if re.search(r'[A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]+)+', line):
                if len(line.split()) <= 10:  # Reasonable for author list
                    author_string = line
                    break
    
    if not author_string:
        return []
    
    # Split authors by common delimiters
    author_string = re.sub(r'\s+and\s+', ', ', author_string)
    author_string = re.sub(r'\s*,\s*', ',', author_string)
    author_string = re.sub(r'[;]', ',', author_string)
    
    author_list = [a.strip() for a in author_string.split(',') if a.strip()]
    
    # Parse each author
    parsed_authors = []
    for author in author_list:
        parsed = parse_author_name(author)
        if parsed['last_name']:
            parsed_authors.append(parsed)
    
    return parsed_authors

# ============================================================================
# OPENALEX SEARCH FUNCTIONS
# ============================================================================

def find_journal_by_issn(issn: str) -> Optional[Dict]:
    """
    Find journal in OpenAlex by ISSN.
    Returns journal info including OpenAlex ID.
    """
    cleaned_issn = clean_issn(issn)
    if not validate_issn(cleaned_issn):
        return None
    
    # Search by ISSN (OpenAlex stores both ISSN and ISSN-L)
    params = {
        'filter': f'issn:{cleaned_issn}',
        'per-page': 1
    }
    
    data = make_openalex_request(f"{OPENALEX_BASE_URL}/sources", params)
    
    if data and 'results' in data and data['results']:
        journal = data['results'][0]
        return {
            'id': journal.get('id'),
            'display_name': journal.get('display_name'),
            'issn': journal.get('issn', []),
            'issn_l': journal.get('issn_l'),
            'publisher': journal.get('publisher'),
            'works_count': journal.get('works_count'),
            'cited_by_count': journal.get('cited_by_count')
        }
    
    return None

def find_author_candidates(last_name: str, first_initial: str) -> List[Dict]:
    """
    Find author candidates in OpenAlex by last name and first initial.
    Returns list of possible authors with their details.
    """
    if not last_name:
        return []
    
    # Build search query
    search_query = f"{last_name}"
    if first_initial:
        search_query += f" {first_initial}"
    
    params = {
        'search': search_query,
        'per-page': 10,  # Get top 10 candidates
        'sort': 'relevance_score:desc'
    }
    
    data = make_openalex_request(f"{OPENALEX_BASE_URL}/authors", params)
    
    candidates = []
    if data and 'results' in data:
        for author in data['results']:
            # Get last known institution
            institution = None
            last_known_institution = author.get('last_known_institution')
            if last_known_institution:
                institution = last_known_institution.get('display_name')
            
            # Get recent works for identification
            recent_works = []
            if author.get('works_count', 0) > 0:
                works_params = {
                    'filter': f'author.id:{author["id"]}',
                    'sort': 'publication_date:desc',
                    'per-page': 3
                }
                works_data = make_openalex_request(f"{OPENALEX_BASE_URL}/works", works_params)
                if works_data and 'results' in works_data:
                    for work in works_data['results']:
                        recent_works.append({
                            'title': work.get('title', ''),
                            'year': work.get('publication_year'),
                            'cited_by_count': work.get('cited_by_count', 0)
                        })
            
            candidate = {
                'id': author.get('id'),
                'display_name': author.get('display_name'),
                'orcid': author.get('orcid'),
                'institution': institution,
                'works_count': author.get('works_count', 0),
                'cited_by_count': author.get('cited_by_count', 0),
                'relevance_score': author.get('relevance_score', 0),
                'recent_works': recent_works
            }
            candidates.append(candidate)
    
    return candidates

def get_author_publications_in_journal(author_id: str, journal_id: str, years: Optional[List[int]] = None) -> List[Dict]:
    """
    Get all publications by author in specific journal.
    """
    filters = [f'author.id:{author_id}', f'primary_location.source.id:{journal_id}']
    
    if years:
        if len(years) == 1:
            filters.append(f'publication_year:{years[0]}')
        else:
            filters.append(f'publication_year:{min(years)}-{max(years)}')
    
    filter_str = ','.join(filters)
    
    params = {
        'filter': filter_str,
        'per-page': 200,  # Get as many as possible
        'sort': 'publication_date:desc'
    }
    
    data = make_openalex_request(f"{OPENALEX_BASE_URL}/works", params)
    
    publications = []
    if data and 'results' in data:
        for work in data['results']:
            pub = {
                'title': work.get('title'),
                'publication_year': work.get('publication_year'),
                'publication_date': work.get('publication_date'),
                'cited_by_count': work.get('cited_by_count', 0),
                'doi': work.get('doi'),
                'doi_url': f"https://doi.org/{work.get('doi', '').replace('https://doi.org/', '')}" if work.get('doi') else None,
                'type': work.get('type'),
                'open_access': work.get('open_access', {}).get('is_oa', False)
            }
            publications.append(pub)
    
    return publications

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main application function"""
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0066cc 0%, #00a8cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }
    .step-card {
        background: linear-gradient(135deg, #0066cc15 0%, #00a8cc15 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid #0066cc;
        margin-bottom: 15px;
    }
    .author-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .candidate-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #0066cc;
    }
    .affiliation-badge {
        background: #e3f2fd;
        color: #0066cc;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 2px;
    }
    .publication-item {
        background: white;
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Author Journal Publication Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 1rem; color: #333; margin-bottom: 1.5rem;">
    Upload an article (DOCX) and enter journal ISSN to find previous publications by the same authors in that journal.
    </p>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'extracted_authors' not in st.session_state:
        st.session_state.extracted_authors = []
    if 'author_affiliations' not in st.session_state:
        st.session_state.author_affiliations = []
    if 'selected_authors' not in st.session_state:
        st.session_state.selected_authors = {}
    if 'journal_info' not in st.session_state:
        st.session_state.journal_info = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # ========================================================================
    # STEP 1: Upload DOCX and Enter ISSN
    # ========================================================================
    
    if st.session_state.step == 1:
        st.markdown("""
        <div class="step-card">
            <h3 style="margin: 0; font-size: 1.3rem;">📄 Step 1: Upload Article and Enter Journal ISSN</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📁 Upload Article (DOCX)**")
            
            if not DOCX_AVAILABLE:
                st.error("python-docx is not installed. Please install it to use this feature.")
                st.code("pip install python-docx nameparser")
            else:
                uploaded_file = st.file_uploader(
                    "Choose a DOCX file",
                    type=['docx'],
                    help="Upload the first page of the article with author names",
                    label_visibility="collapsed"
                )
                
                if uploaded_file is not None:
                    with st.spinner("Extracting authors from DOCX..."):
                        st.session_state.extracted_authors = extract_authors_from_docx(uploaded_file)
                    
                    if st.session_state.extracted_authors:
                        st.success(f"✅ Found {len(st.session_state.extracted_authors)} authors")
                        
                        # Show extracted authors with editable fields
                        st.markdown("**Extracted Authors (you can edit):**")
                        st.session_state.author_affiliations = []
                        
                        for i, author in enumerate(st.session_state.extracted_authors):
                            col_a, col_b, col_c = st.columns([2, 1, 2])
                            with col_a:
                                last_name = st.text_input(
                                    f"Last Name {i+1}",
                                    value=author['last_name'],
                                    key=f"last_name_{i}"
                                )
                            with col_b:
                                first_init = st.text_input(
                                    f"Initial {i+1}",
                                    value=author['first_initial'],
                                    key=f"first_init_{i}",
                                    max_chars=1
                                )
                            with col_c:
                                affiliation = st.text_input(
                                    f"Affiliation (if known) {i+1}",
                                    value="",
                                    key=f"affiliation_{i}",
                                    placeholder="e.g., MIT"
                                )
                            
                            # Update author data
                            st.session_state.extracted_authors[i] = {
                                'last_name': last_name,
                                'first_initial': first_init.upper() if first_init else ""
                            }
                            st.session_state.author_affiliations.append(affiliation)
                    else:
                        st.warning("No authors found. You can enter them manually below.")
                        
                        # Manual entry option
                        st.markdown("**✏️ Manual Author Entry**")
                        num_authors = st.number_input("Number of authors", min_value=1, max_value=20, value=1)
                        
                        st.session_state.extracted_authors = []
                        st.session_state.author_affiliations = []
                        
                        for i in range(num_authors):
                            col_a, col_b, col_c = st.columns([2, 1, 2])
                            with col_a:
                                last_name = st.text_input(f"Last Name {i+1}", key=f"manual_last_{i}")
                            with col_b:
                                first_init = st.text_input(f"Initial {i+1}", key=f"manual_init_{i}", max_chars=1)
                            with col_c:
                                affiliation = st.text_input(f"Affiliation {i+1}", key=f"manual_aff_{i}")
                            
                            if last_name:
                                st.session_state.extracted_authors.append({
                                    'last_name': last_name,
                                    'first_initial': first_init.upper() if first_init else ""
                                })
                                st.session_state.author_affiliations.append(affiliation)
        
        with col2:
            st.markdown("**🔖 Journal ISSN**")
            
            issn_input = st.text_input(
                "ISSN",
                placeholder="e.g., 0028-0836 or 00280836",
                help="Enter ISSN with or without hyphen",
                label_visibility="collapsed"
            )
            
            if issn_input:
                cleaned_issn = clean_issn(issn_input)
                if validate_issn(cleaned_issn):
                    st.success(f"✓ Valid ISSN format: {cleaned_issn}")
                    
                    # Optional: Test journal lookup
                    if st.button("🔍 Test Journal Lookup"):
                        with st.spinner("Searching for journal..."):
                            journal = find_journal_by_issn(cleaned_issn)
                            if journal:
                                st.markdown(f"""
                                <div style="background: #e8f5e9; padding: 10px; border-radius: 8px;">
                                    <strong>✅ Found:</strong> {journal['display_name']}<br>
                                    <strong>Publisher:</strong> {journal.get('publisher', 'N/A')}<br>
                                    <strong>Total works:</strong> {journal['works_count']:,}<br>
                                    <strong>Total citations:</strong> {journal['cited_by_count']:,}
                                </div>
                                """, unsafe_allow_html=True)
                                st.session_state.journal_info = journal
                            else:
                                st.error("Journal not found in OpenAlex")
                else:
                    st.error("Invalid ISSN format. Should be 8 digits (with optional X at end)")
            
            # Year range filter
            st.markdown("**📅 Publication Years (optional filter)**")
            current_year = datetime.now().year
            year_option = st.radio(
                "Year filter",
                ["All years", "Range", "Last 5 years", "Last 10 years"],
                horizontal=True,
                key="year_filter_step1"
            )
            
            if year_option == "Range":
                year_range = st.slider(
                    "Select range",
                    1950, current_year,
                    (current_year-10, current_year)
                )
                years_filter = list(range(year_range[0], year_range[1] + 1))
            elif year_option == "Last 5 years":
                years_filter = list(range(current_year-4, current_year+1))
            elif year_option == "Last 10 years":
                years_filter = list(range(current_year-9, current_year+1))
            else:
                years_filter = None
            
            st.session_state.years_filter = years_filter
        
        # Continue button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("➡️ Continue to Author Confirmation", type="primary", use_container_width=True):
                if not st.session_state.extracted_authors:
                    st.error("❌ Please extract or enter at least one author")
                elif not issn_input or not validate_issn(clean_issn(issn_input)):
                    st.error("❌ Please enter a valid ISSN")
                else:
                    # Find journal info
                    with st.spinner("Looking up journal..."):
                        st.session_state.journal_info = find_journal_by_issn(clean_issn(issn_input))
                        if not st.session_state.journal_info:
                            st.error("❌ Journal not found in OpenAlex. Please check the ISSN.")
                            st.stop()
                    
                    st.session_state.issn = clean_issn(issn_input)
                    st.session_state.step = 2
                    st.rerun()
    
    # ========================================================================
    # STEP 2: Author Confirmation
    # ========================================================================
    
    elif st.session_state.step == 2:
        st.markdown("""
        <div class="step-card">
            <h3 style="margin: 0; font-size: 1.3rem;">👥 Step 2: Confirm Authors</h3>
            <p style="margin: 5px 0;">Select the correct OpenAlex profile for each author</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show journal info
        if st.session_state.journal_info:
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <strong>📚 Journal:</strong> {st.session_state.journal_info['display_name']}<br>
                <strong>ISSN:</strong> {st.session_state.issn}<br>
                <strong>Publisher:</strong> {st.session_state.journal_info.get('publisher', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
        
        # Progress
        total_authors = len(st.session_state.extracted_authors)
        confirmed = len([a for a in st.session_state.selected_authors.values() if a])
        st.progress(confirmed / total_authors if total_authors > 0 else 0)
        st.markdown(f"**Progress:** {confirmed}/{total_authors} authors confirmed")
        
        # Author confirmation cards
        st.markdown("### Select Author Profiles")
        
        selected_authors = {}
        
        for idx, author in enumerate(st.session_state.extracted_authors):
            last_name = author['last_name']
            first_init = author['first_initial']
            affiliation_from_article = st.session_state.author_affiliations[idx] if idx < len(st.session_state.author_affiliations) else ""
            
            st.markdown(f"""
            <div class="author-card">
                <h4>Author {idx+1}: {last_name}, {first_init}.</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if affiliation_from_article:
                st.markdown(f"<span class='affiliation-badge'>📌 From article: {affiliation_from_article}</span>", unsafe_allow_html=True)
            
            # Search for candidates
            with st.spinner(f"Searching for {last_name} {first_init}..."):
                candidates = find_author_candidates(last_name, first_init)
            
            if candidates:
                # Create options for selectbox
                options = {}
                for cand in candidates:
                    # Format display string
                    inst = cand['institution'] or "No affiliation"
                    inst_short = inst[:40] + "..." if len(inst) > 40 else inst
                    display = f"{cand['display_name']} | {inst_short} | Works: {cand['works_count']}"
                    options[display] = cand
                
                # Add "None of the above" option
                options["❌ None of the above / Skip"] = None
                
                # Default to previously selected if any
                default_idx = 0
                prev_selected = st.session_state.selected_authors.get(idx)
                if prev_selected:
                    for i, (display, cand) in enumerate(options.items()):
                        if cand and cand['id'] == prev_selected['id']:
                            default_idx = i
                            break
                
                selected_display = st.selectbox(
                    f"Select profile for {last_name}, {first_init}.",
                    options=list(options.keys()),
                    index=default_idx,
                    key=f"author_select_{idx}"
                )
                
                selected_candidate = options[selected_display]
                
                if selected_candidate:
                    # Show candidate details
                    with st.expander(f"📊 Show details for {selected_candidate['display_name']}"):
                        st.markdown(f"""
                        **ORCID:** {selected_candidate['orcid'] or 'Not available'}<br>
                        **Institution:** {selected_candidate['institution'] or 'Unknown'}<br>
                        **Total publications:** {selected_candidate['works_count']:,}<br>
                        **Total citations:** {selected_candidate['cited_by_count']:,}<br>
                        **Relevance score:** {selected_candidate['relevance_score']:.2f}
                        """, unsafe_allow_html=True)
                        
                        if selected_candidate['recent_works']:
                            st.markdown("**Recent publications:**")
                            for work in selected_candidate['recent_works']:
                                st.markdown(f"• {work['title']} ({work['year']}) - {work['cited_by_count']} citations")
                    
                    # Save selection
                    selected_authors[idx] = {
                        'id': selected_candidate['id'],
                        'display_name': selected_candidate['display_name'],
                        'institution': selected_candidate['institution'],
                        'orcid': selected_candidate['orcid']
                    }
                else:
                    selected_authors[idx] = None
            else:
                st.warning(f"No candidates found for {last_name}, {first_init}.")
                selected_authors[idx] = None
            
            st.markdown("---")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("← Back to Step 1", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        
        with col2:
            confirmed_count = len([v for v in selected_authors.values() if v])
            if st.button(f"➡️ Continue with {confirmed_count} confirmed authors", 
                        type="primary", use_container_width=True,
                        disabled=confirmed_count == 0):
                st.session_state.selected_authors = selected_authors
                st.session_state.step = 3
                st.rerun()
    
    # ========================================================================
    # STEP 3: Analysis Results
    # ========================================================================
    
    elif st.session_state.step == 3:
        st.markdown("""
        <div class="step-card">
            <h3 style="margin: 0; font-size: 1.3rem;">📊 Step 3: Analysis Results</h3>
            <p style="margin: 5px 0;">Previous publications in the selected journal</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show journal info
        if st.session_state.journal_info:
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <strong>📚 Journal:</strong> {st.session_state.journal_info['display_name']}<br>
                <strong>ISSN:</strong> {st.session_state.issn}
            </div>
            """, unsafe_allow_html=True)
        
        # Run analysis if not already done
        if not st.session_state.analysis_results:
            with st.spinner("Analyzing authors' publications in this journal..."):
                journal_id = st.session_state.journal_info['id']
                
                for idx, author_info in st.session_state.selected_authors.items():
                    if author_info:  # Skip if None
                        author_id = author_info['id']
                        author_name = author_info['display_name']
                        
                        publications = get_author_publications_in_journal(
                            author_id, 
                            journal_id,
                            st.session_state.get('years_filter')
                        )
                        
                        if publications:
                            st.session_state.analysis_results[author_name] = publications
        
        # Display results
        if st.session_state.analysis_results:
            total_pubs = sum(len(pubs) for pubs in st.session_state.analysis_results.values())
            
            st.markdown(f"""
            <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <strong>✅ Found {total_pubs} previous publications</strong> by {len(st.session_state.analysis_results)} authors
            </div>
            """, unsafe_allow_html=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Authors with publications", len(st.session_state.analysis_results))
            with col2:
                st.metric("Total publications", total_pubs)
            with col3:
                total_citations = sum(
                    sum(pub['cited_by_count'] for pub in pubs) 
                    for pubs in st.session_state.analysis_results.values()
                )
                st.metric("Total citations", f"{total_citations:,}")
            with col4:
                avg_citations = total_citations / total_pubs if total_pubs > 0 else 0
                st.metric("Avg citations per paper", f"{avg_citations:.1f}")
            
            # Detailed results by author
            for author_name, publications in st.session_state.analysis_results.items():
                with st.expander(f"📚 {author_name} - {len(publications)} publications"):
                    # Sort by year (newest first)
                    publications.sort(key=lambda x: x.get('publication_year', 0), reverse=True)
                    
                    for pub in publications:
                        year = pub.get('publication_year', 'Unknown')
                        title = pub.get('title', 'No title')
                        citations = pub.get('cited_by_count', 0)
                        doi_url = pub.get('doi_url')
                        oa_status = "🔓" if pub.get('open_access') else "🔒"
                        
                        st.markdown(f"""
                        <div class="publication-item">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: 600;">{year}</span>
                                <span>{oa_status} {citations} citations</span>
                            </div>
                            <div style="margin: 5px 0;">{title}</div>
                            {f'<a href="{doi_url}" target="_blank">🔗 DOI</a>' if doi_url else ''}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Export options
            st.markdown("### 📥 Export Results")
            
            # Prepare data for export
            export_data = []
            for author_name, publications in st.session_state.analysis_results.items():
                for pub in publications:
                    export_data.append({
                        'Author': author_name,
                        'Title': pub.get('title'),
                        'Year': pub.get('publication_year'),
                        'Date': pub.get('publication_date'),
                        'Citations': pub.get('cited_by_count'),
                        'DOI': pub.get('doi'),
                        'Type': pub.get('type'),
                        'Open Access': pub.get('open_access')
                    })
            
            if export_data:
                df = pd.DataFrame(export_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV export
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 Download CSV",
                        csv,
                        f"author_journal_pubs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Excel export
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Publications', index=False)
                        
                        # Summary sheet
                        summary = df.groupby('Author').agg({
                            'Title': 'count',
                            'Citations': 'sum'
                        }).rename(columns={'Title': 'Publications'})
                        summary.to_excel(writer, sheet_name='Summary')
                        
                        # Formatting
                        workbook = writer.book
                        header_format = workbook.add_format({'bold': True, 'bg_color': '#0066cc', 'font_color': 'white'})
                        
                        for sheet_name in writer.sheets:
                            worksheet = writer.sheets[sheet_name]
                            for col_num, value in enumerate(df.columns if sheet_name == 'Publications' else summary.columns):
                                worksheet.write(0, col_num, value, header_format)
                    
                    st.download_button(
                        "📈 Download Excel",
                        output.getvalue(),
                        f"author_journal_pubs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        
        else:
            st.info("No previous publications found in this journal for the selected authors.")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("← Back to Author Selection", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        
        with col2:
            if st.button("🔄 New Analysis", use_container_width=True):
                # Clear session but keep settings for step 1
                for key in ['step', 'extracted_authors', 'author_affiliations', 
                           'selected_authors', 'journal_info', 'analysis_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.step = 1
                st.rerun()

if __name__ == "__main__":
    main()
