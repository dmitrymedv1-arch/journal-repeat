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
import re
import time
import json
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set, Any
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ratelimit import limits, sleep_and_retry
import logging
import io
from nameparser import HumanName
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Author Publication History Analyzer",
    page_icon="👨‍🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# OPENALEX API CONFIGURATION
# ============================================================================

OPENALEX_BASE_URL = "https://api.openalex.org"
MAILTO = "your-email@example.com"  # Replace with your email
POLITE_POOL_HEADER = {'User-Agent': f'Author-Analyzer (mailto:{MAILTO})'}

# Rate limit settings
RATE_LIMIT_PER_SECOND = 10
MAX_RETRIES = 3
INITIAL_DELAY = 1
MAX_DELAY = 60

# ============================================================================
# UI STYLES
# ============================================================================

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
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.04);
    }
    
    .author-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .author-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0066cc;
        margin-bottom: 10px;
    }
    
    .candidate-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 3px solid #0066cc;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .candidate-card:hover {
        background: #e8f4ff;
        transform: translateX(5px);
    }
    
    .candidate-card.selected {
        background: #e3f2fd;
        border-left: 3px solid #00a8cc;
        border: 2px solid #0066cc;
    }
    
    .affiliation-badge {
        background: #e3f2fd;
        color: #0066cc;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px 0;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin: 10px 0;
    }
    
    .metric-box {
        background: white;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0066cc;
    }
    
    .paper-item {
        background: white;
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #e9ecef;
        font-size: 0.9rem;
    }
    
    .paper-title {
        font-weight: 500;
        color: #212529;
    }
    
    .paper-meta {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 3px;
    }
    
    .progress-header {
        font-size: 1rem;
        font-weight: 600;
        color: #0066cc;
        margin: 10px 0;
    }
    
    .info-message {
        background: linear-gradient(135deg, #0066cc15 0%, #00a8cc15 100%);
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #0066cc;
        font-size: 0.9rem;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# RATE LIMITED API REQUESTS
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

# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_issn(issn_input: str) -> str:
    """
    Parse ISSN from various formats: 12345678, 1234-5678, 1234 5678, etc.
    Returns clean ISSN without hyphen for API queries
    """
    if not issn_input:
        return ""
    
    # Remove all non-digit characters
    clean = re.sub(r'[^\dXx]', '', issn_input)
    
    # ISSN should be 8 characters (digits or X)
    if len(clean) >= 8:
        clean = clean[:8].upper()
        return clean
    
    return ""

def parse_author_string(author_str: str) -> Dict[str, str]:
    """
    Parse author string to extract last name and first initial.
    Handles various formats: "Ivanov, I.I.", "Petrov P.", "Smith, John", "von Neumann J."
    """
    if not author_str:
        return {"last_name": "", "first_initial": ""}
    
    # Clean the string
    author_str = author_str.strip()
    
    try:
        # Use nameparser library for robust parsing
        name = HumanName(author_str)
        
        last_name = name.last
        first_name = name.first
        
        # If no last name found, try to guess from format
        if not last_name:
            # Try comma-separated format: "Last, First"
            if ',' in author_str:
                parts = author_str.split(',', 1)
                last_name = parts[0].strip()
                first_part = parts[1].strip() if len(parts) > 1 else ""
                # Get first initial from first part
                if first_part:
                    first_initial = first_part[0].upper() if first_part[0].isalpha() else ""
                else:
                    first_initial = ""
            else:
                # Try space-separated: "First Last" or "F. Last"
                parts = author_str.split()
                if len(parts) >= 2:
                    # Assume last word is last name
                    last_name = parts[-1]
                    # First part might be initial or name
                    first_part = parts[0]
                    if first_part and first_part[0].isalpha():
                        first_initial = first_part[0].upper()
                    else:
                        first_initial = ""
                else:
                    last_name = author_str
                    first_initial = ""
        else:
            # Get first initial from first name
            if first_name:
                first_initial = first_name[0].upper() if first_name[0].isalpha() else ""
            else:
                first_initial = ""
        
        return {
            "last_name": last_name,
            "first_initial": first_initial
        }
        
    except Exception as e:
        logger.warning(f"Error parsing name '{author_str}': {str(e)}")
        
        # Fallback: simple parsing
        if ',' in author_str:
            parts = author_str.split(',', 1)
            last_name = parts[0].strip()
            first_part = parts[1].strip() if len(parts) > 1 else ""
            first_initial = first_part[0].upper() if first_part and first_part[0].isalpha() else ""
        else:
            parts = author_str.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_part = parts[0]
                first_initial = first_part[0].upper() if first_part and first_part[0].isalpha() else ""
            else:
                last_name = author_str
                first_initial = ""
        
        return {
            "last_name": last_name,
            "first_initial": first_initial
        }

def extract_authors_from_text(text: str) -> List[str]:
    """
    Extract author names from article text.
    This is a simplified version - in production, you'd need more sophisticated parsing
    based on the article format.
    """
    # Look for common author sections
    author_patterns = [
        r'Authors?:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'By\s+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Author\(s\):?\s*(.+?)(?:\n\n|\n[A-Z]|$)'
    ]
    
    all_authors = []
    
    for pattern in author_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            # Split by common separators
            authors = re.split(r'[;,]\s*|\s+and\s+|\s+&\s+', match)
            for author in authors:
                author = author.strip()
                if author and len(author) > 3 and not author.isdigit():
                    all_authors.append(author)
    
    # If no authors found, try to split by lines and look for name patterns
    if not all_authors:
        lines = text.split('\n')
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            # Simple heuristic: contains at least one space and no digits
            if line and ' ' in line and not any(c.isdigit() for c in line):
                # Check if it looks like a name (has uppercase letters)
                if any(c.isupper() for c in line):
                    # Split by common separators
                    candidates = re.split(r'[;,]\s*|\s+and\s+', line)
                    for candidate in candidates:
                        candidate = candidate.strip()
                        if candidate and len(candidate) > 3:
                            all_authors.append(candidate)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_authors = []
    for author in all_authors:
        if author not in seen:
            seen.add(author)
            unique_authors.append(author)
    
    return unique_authors

# ============================================================================
# OPENALEX SEARCH FUNCTIONS
# ============================================================================

def find_journal_by_issn(issn: str) -> Optional[Dict]:
    """
    Find journal/source in OpenAlex by ISSN
    """
    if not issn:
        return None
    
    # Clean ISSN
    clean_issn = parse_issn(issn)
    if not clean_issn:
        return None
    
    # Search by ISSN
    params = {
        'filter': f'issn:{clean_issn}',
        'per-page': 1
    }
    
    data = make_openalex_request(f"{OPENALEX_BASE_URL}/sources", params)
    
    if data and 'results' in data and data['results']:
        source = data['results'][0]
        return {
            'id': source.get('id', ''),
            'display_name': source.get('display_name', ''),
            'issn': source.get('issn', []),
            'works_count': source.get('works_count', 0),
            'cited_by_count': source.get('cited_by_count', 0)
        }
    
    return None

def search_author_candidates(last_name: str, first_initial: str, 
                            affiliation_hint: Optional[str] = None) -> List[Dict]:
    """
    Search for author candidates in OpenAlex by name
    """
    if not last_name:
        return []
    
    # Build search query
    if first_initial:
        # Search with first initial
        search_query = f"{last_name} {first_initial}"
    else:
        search_query = last_name
    
    params = {
        'search': search_query,
        'per-page': 10,
        'sort': 'relevance_score:desc'
    }
    
    data = make_openalex_request(f"{OPENALEX_BASE_URL}/authors", params)
    
    candidates = []
    if data and 'results' in data:
        for author in data['results']:
            # Get affiliation
            affiliation = None
            last_known_institution = author.get('last_known_institution')
            if last_known_institution:
                affiliation = last_known_institution.get('display_name')
            
            # Get recent works for context
            recent_works = []
            works_url = author.get('works_api_url')
            if works_url:
                works_params = {'per-page': 5, 'sort': 'publication_date:desc'}
                works_data = make_openalex_request(works_url, works_params)
                if works_data and 'results' in works_data:
                    for work in works_data['results'][:3]:
                        recent_works.append({
                            'title': work.get('title', ''),
                            'year': work.get('publication_year', ''),
                            'cited_by_count': work.get('cited_by_count', 0)
                        })
            
            # Calculate match score (simple heuristic)
            display_name = author.get('display_name', '').lower()
            last_name_lower = last_name.lower()
            score = 1.0 if last_name_lower in display_name else 0.5
            
            # Boost score if affiliation matches hint
            if affiliation_hint and affiliation and affiliation_hint.lower() in affiliation.lower():
                score *= 1.2
            
            candidate = {
                'id': author.get('id', ''),
                'display_name': author.get('display_name', ''),
                'orcid': author.get('orcid', ''),
                'works_count': author.get('works_count', 0),
                'cited_by_count': author.get('cited_by_count', 0),
                'affiliation': affiliation,
                'recent_works': recent_works,
                'relevance_score': min(score, 1.0)  # Cap at 1.0
            }
            candidates.append(candidate)
    
    # Sort by relevance score
    candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return candidates

def get_author_works_in_journal(author_id: str, journal_id: str, 
                               years: Optional[List[int]] = None,
                               limit: int = 100) -> List[Dict]:
    """
    Get all works by an author in a specific journal
    """
    if not author_id or not journal_id:
        return []
    
    # Extract ID from full URL if needed
    if '/' in author_id:
        author_id = author_id.split('/')[-1]
    if '/' in journal_id:
        journal_id = journal_id.split('/')[-1]
    
    # Build filter
    filter_parts = [
        f"author.id:{author_id}",
        f"primary_location.source.id:{journal_id}"
    ]
    
    if years:
        year_str = f"{min(years)}-{max(years)}"
        filter_parts.append(f"publication_year:{year_str}")
    
    filter_str = ','.join(filter_parts)
    
    all_works = []
    cursor = "*"
    
    while len(all_works) < limit and cursor:
        params = {
            'filter': filter_str,
            'per-page': 25,
            'cursor': cursor,
            'sort': 'publication_date:desc'
        }
        
        data = make_openalex_request(f"{OPENALEX_BASE_URL}/works", params)
        
        if not data or 'results' not in data:
            break
        
        works = data['results']
        if not works:
            break
        
        for work in works:
            enriched = enrich_work_data(work)
            all_works.append(enriched)
        
        cursor = data.get('meta', {}).get('next_cursor')
        time.sleep(0.1)
    
    return all_works

def enrich_work_data(work: Dict) -> Dict:
    """Enrich work data with additional fields"""
    if not work:
        return {}
    
    doi_raw = work.get('doi')
    doi_clean = ''
    if doi_raw:
        doi_clean = str(doi_raw).replace('https://doi.org/', '')
    
    # Get authors
    authors = []
    authorships = work.get('authorships', [])
    for authorship in authorships[:10]:
        if authorship and 'author' in authorship:
            author_name = authorship['author'].get('display_name', '')
            if author_name:
                authors.append(author_name)
    
    # Get journal
    journal = ''
    primary_location = work.get('primary_location')
    if primary_location and 'source' in primary_location:
        source = primary_location['source']
        if source:
            journal = source.get('display_name', '')
    
    enriched = {
        'id': work.get('id', ''),
        'doi': doi_clean,
        'title': clean_text(work.get('title', '')),
        'publication_date': work.get('publication_date', ''),
        'publication_year': work.get('publication_year', 0),
        'cited_by_count': work.get('cited_by_count', 0),
        'type': work.get('type', ''),
        'doi_url': f"https://doi.org/{doi_clean}" if doi_clean else '',
        'authors': authors,
        'journal': journal,
        'is_oa': work.get('open_access', {}).get('is_oa', False)
    }
    
    return enriched

def clean_text(text: str) -> str:
    """Clean text from HTML tags and extra characters"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_candidate_card(candidate: Dict, is_selected: bool = False):
    """Display a candidate author card"""
    selected_class = "selected" if is_selected else ""
    
    # Format affiliation
    affiliation_html = ""
    if candidate.get('affiliation'):
        affiliation_html = f'<span class="affiliation-badge">🏛️ {candidate["affiliation"]}</span>'
    
    # Format ORCID
    orcid_html = ""
    if candidate.get('orcid'):
        orcid_html = f'<span class="affiliation-badge" style="background: #e8f5e9; color: #2e7d32;">🆔 {candidate["orcid"]}</span>'
    
    # Recent works
    recent_works_html = ""
    if candidate.get('recent_works'):
        recent_works_html = "<div style='margin-top: 10px;'><strong>Recent works:</strong>"
        for work in candidate['recent_works']:
            recent_works_html += f"""
            <div class="paper-item">
                <div class="paper-title">{work['title'][:80]}...</div>
                <div class="paper-meta">{work['year']} · {work['cited_by_count']} citations</div>
            </div>
            """
        recent_works_html += "</div>"
    
    # Metrics
    metrics_html = f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-label">Publications</div>
            <div class="metric-value">{candidate.get('works_count', 0):,}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Total citations</div>
            <div class="metric-value">{candidate.get('cited_by_count', 0):,}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Relevance</div>
            <div class="metric-value">{candidate.get('relevance_score', 0):.1%}</div>
        </div>
    </div>
    """
    
    st.markdown(f"""
    <div class="candidate-card {selected_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="font-size: 1.1rem;">{candidate['display_name']}</strong>
            </div>
            <div>
                {affiliation_html}
                {orcid_html}
            </div>
        </div>
        {metrics_html}
        {recent_works_html}
    </div>
    """, unsafe_allow_html=True)

def display_results_summary(author_results: Dict[str, List[Dict]]):
    """Display summary of results for all authors"""
    if not author_results:
        return
    
    total_papers = sum(len(papers) for papers in author_results.values())
    authors_with_papers = sum(1 for papers in author_results.values() if papers)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total authors analyzed", len(author_results))
    with col2:
        st.metric("Authors with papers in journal", authors_with_papers)
    with col3:
        st.metric("Total papers found", total_papers)
    
    # Distribution
    st.markdown("### 📊 Publication Distribution")
    
    data = []
    for author_name, papers in author_results.items():
        data.append({
            'Author': author_name,
            'Papers': len(papers),
            'Citations': sum(p.get('cited_by_count', 0) for p in papers)
        })
    
    if data:
        df = pd.DataFrame(data)
        df = df.sort_values('Papers', ascending=False)
        
        fig = px.bar(df, x='Author', y='Papers', 
                    title='Publications per Author',
                    color='Papers',
                    color_continuous_scale='Blues',
                    hover_data=['Citations'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def display_author_papers(author_name: str, papers: List[Dict]):
    """Display papers for a specific author"""
    if not papers:
        st.info(f"No papers found for {author_name}")
        return
    
    st.markdown(f"#### 📚 Papers by {author_name} ({len(papers)} found)")
    
    # Sort by year descending
    papers_sorted = sorted(papers, key=lambda x: x.get('publication_year', 0), reverse=True)
    
    for i, paper in enumerate(papers_sorted, 1):
        with st.expander(f"{i}. {paper.get('title', 'No title')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Year:** {paper.get('publication_year', 'N/A')}")
                st.markdown(f"**Journal:** {paper.get('journal', 'N/A')}")
                st.markdown(f"**Authors:** {', '.join(paper.get('authors', []))}")
                
                if paper.get('doi'):
                    st.markdown(f"**DOI:** [{paper['doi']}](https://doi.org/{paper['doi']})")
            
            with col2:
                st.metric("Citations", paper.get('cited_by_count', 0))
                if paper.get('is_oa'):
                    st.success("🔓 Open Access")
                else:
                    st.info("🔒 Closed Access")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Author Publication History Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-message">
        <strong>🔍 How it works:</strong><br>
        1. Upload an article (paste text) - authors will be extracted automatically<br>
        2. Enter the ISSN of the journal you want to analyze<br>
        3. For each extracted author, review candidate profiles from OpenAlex and select the correct one<br>
        4. The system will find all previous publications by these authors in the specified journal
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'extracted_authors' not in st.session_state:
        st.session_state.extracted_authors = []
    if 'parsed_authors' not in st.session_state:
        st.session_state.parsed_authors = {}  # Dict mapping original string to parsed {last, first}
    if 'author_candidates' not in st.session_state:
        st.session_state.author_candidates = {}  # Dict mapping author to list of candidates
    if 'selected_authors' not in st.session_state:
        st.session_state.selected_authors = {}  # Dict mapping author to selected candidate ID
    if 'journal_info' not in st.session_state:
        st.session_state.journal_info = None
    if 'results' not in st.session_state:
        st.session_state.results = {}  # Dict mapping author name to list of papers
    if 'current_author_index' not in st.session_state:
        st.session_state.current_author_index = 0
    
    # ========================================================================
    # STEP 1: INPUT
    # ========================================================================
    
    if st.session_state.step == 1:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("### 📥 Step 1: Input Data")
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📄 Article Text**")
            article_text = st.text_area(
                "Paste the article text here (authors will be extracted automatically)",
                height=300,
                placeholder="Paste the full article text including author names...",
                key="article_input"
            )
        
        with col2:
            st.markdown("**📰 Journal ISSN**")
            issn_input = st.text_input(
                "Enter ISSN (formats: 12345678, 1234-5678, 1234 5678)",
                placeholder="e.g., 0028-0836",
                key="issn_input"
            )
            
            # Show ISSN preview
            if issn_input:
                clean_issn = parse_issn(issn_input)
                if clean_issn:
                    st.info(f"✓ Parsed ISSN: {clean_issn[:4]}-{clean_issn[4:]}")
                else:
                    st.warning("Invalid ISSN format")
            
            st.markdown("---")
            st.markdown("**🔍 Optional Filters**")
            
            years_filter = st.multiselect(
                "Filter by publication years (optional)",
                options=list(range(2000, datetime.now().year + 1)),
                default=[],
                help="Select years to limit the search"
            )
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Extract Authors & Start", type="primary", use_container_width=True):
                if not article_text:
                    st.error("Please paste the article text")
                elif not issn_input:
                    st.error("Please enter the journal ISSN")
                else:
                    with st.spinner("Processing..."):
                        # Parse ISSN and find journal
                        clean_issn = parse_issn(issn_input)
                        if not clean_issn:
                            st.error("Invalid ISSN format")
                            st.stop()
                        
                        journal_info = find_journal_by_issn(clean_issn)
                        
                        if not journal_info:
                            st.error(f"Journal with ISSN {clean_issn[:4]}-{clean_issn[4:]} not found in OpenAlex")
                            st.stop()
                        
                        st.session_state.journal_info = journal_info
                        
                        # Extract authors from text
                        author_strings = extract_authors_from_text(article_text)
                        
                        if not author_strings:
                            st.warning("No authors found in text. Please check the format or enter manually.")
                            # Allow manual entry
                            manual_authors = st.text_area(
                                "Enter authors manually (one per line)",
                                height=100,
                                placeholder="Smith, J.\nJohnson, M.\nWilliams, R."
                            )
                            if manual_authors:
                                author_strings = [a.strip() for a in manual_authors.split('\n') if a.strip()]
                        
                        if author_strings:
                            st.session_state.extracted_authors = author_strings
                            
                            # Parse each author
                            parsed = {}
                            for author_str in author_strings:
                                parsed[author_str] = parse_author_string(author_str)
                            st.session_state.parsed_authors = parsed
                            
                            st.session_state.step = 2
                            st.rerun()
                        else:
                            st.error("No authors found. Please enter authors manually.")
        
        # Manual entry option
        with st.expander("✏️ Or enter authors manually"):
            manual_authors = st.text_area(
                "Enter authors (one per line, e.g., 'Smith, J.' or 'John Smith')",
                height=150
            )
            if st.button("Use Manual Entry"):
                if manual_authors:
                    author_strings = [a.strip() for a in manual_authors.split('\n') if a.strip()]
                    st.session_state.extracted_authors = author_strings
                    
                    # Parse each author
                    parsed = {}
                    for author_str in author_strings:
                        parsed[author_str] = parse_author_string(author_str)
                    st.session_state.parsed_authors = parsed
                    
                    st.session_state.step = 2
                    st.rerun()
    
    # ========================================================================
    # STEP 2: AUTHOR CONFIRMATION
    # ========================================================================
    
    elif st.session_state.step == 2:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("### 👤 Step 2: Confirm Authors")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show journal info
        if st.session_state.journal_info:
            st.markdown(f"""
            <div class="info-message">
                <strong>📰 Target Journal:</strong> {st.session_state.journal_info['display_name']}<br>
                <strong>ISSN:</strong> {', '.join(st.session_state.journal_info.get('issn', []))}<br>
                <strong>Total publications in OpenAlex:</strong> {st.session_state.journal_info.get('works_count', 0):,}
            </div>
            """, unsafe_allow_html=True)
        
        # Progress
        total_authors = len(st.session_state.extracted_authors)
        confirmed = len([k for k in st.session_state.selected_authors.keys() if st.session_state.selected_authors.get(k)])
        
        st.markdown(f"""
        <div class="progress-header">
            Progress: {confirmed} of {total_authors} authors confirmed
        </div>
        """, unsafe_allow_html=True)
        
        progress = confirmed / total_authors if total_authors > 0 else 0
        st.progress(progress)
        
        # Process each author
        for idx, author_str in enumerate(st.session_state.extracted_authors):
            parsed = st.session_state.parsed_authors.get(author_str, {})
            
            st.markdown(f"<div class='author-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='author-name'>Author {idx + 1}: {author_str}</div>", unsafe_allow_html=True)
            
            # Show parsed name
            if parsed.get('last_name'):
                st.markdown(f"**Parsed as:** Last name: `{parsed['last_name']}`, Initial: `{parsed['first_initial'] or '?'}`")
            
            # Check if already confirmed
            if author_str in st.session_state.selected_authors:
                st.success(f"✅ Confirmed: {st.session_state.selected_authors[author_str]}")
                st.markdown("</div>", unsafe_allow_html=True)
                continue
            
            # Search for candidates if not already searched
            if author_str not in st.session_state.author_candidates:
                with st.spinner(f"Searching for {author_str}..."):
                    candidates = search_author_candidates(
                        parsed.get('last_name', ''),
                        parsed.get('first_initial', '')
                    )
                    st.session_state.author_candidates[author_str] = candidates
            
            candidates = st.session_state.author_candidates.get(author_str, [])
            
            if not candidates:
                st.warning("No candidates found in OpenAlex")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"❌ Skip {author_str}", key=f"skip_{idx}"):
                        st.session_state.selected_authors[author_str] = "SKIPPED"
                        st.rerun()
                with col2:
                    if st.button(f"➕ Manual ID", key=f"manual_{idx}"):
                        st.session_state[f"show_manual_{idx}"] = True
                
                # Manual ID input
                if st.session_state.get(f"show_manual_{idx}", False):
                    manual_id = st.text_input(f"Enter OpenAlex ID for {author_str}", key=f"manual_id_{idx}")
                    if st.button(f"Confirm Manual ID", key=f"confirm_manual_{idx}"):
                        if manual_id:
                            st.session_state.selected_authors[author_str] = manual_id
                            st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
                continue
            
            # Display candidates
            st.markdown("**Select the correct profile:**")
            
            # Create radio options
            options = {}
            for i, cand in enumerate(candidates):
                label = f"{cand['display_name']}"
                if cand.get('affiliation'):
                    label += f" - {cand['affiliation']}"
                if cand.get('orcid'):
                    label += f" (ORCID: {cand['orcid']})"
                options[label] = cand['id']
            
            # Add "None of these" option
            options["❌ None of these / Skip"] = "SKIP"
            
            selected_label = st.radio(
                "Choose profile:",
                options=list(options.keys()),
                key=f"candidate_select_{idx}",
                index=None
            )
            
            # Show details of selected candidate
            if selected_label and options[selected_label] != "SKIP":
                selected_id = options[selected_label]
                # Find the candidate
                selected_cand = next((c for c in candidates if c['id'] == selected_id), None)
                if selected_cand:
                    with st.expander("Show details"):
                        display_candidate_card(selected_cand, is_selected=True)
            
            # Confirm button
            if selected_label:
                if st.button(f"✅ Confirm Selection", key=f"confirm_{idx}"):
                    selected_id = options[selected_label]
                    st.session_state.selected_authors[author_str] = selected_id
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Back to Input", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        
        with col2:
            confirmed = len([k for k in st.session_state.selected_authors.keys() if st.session_state.selected_authors.get(k) and st.session_state.selected_authors[k] != "SKIP"])
            if confirmed > 0 and st.button("🚀 Analyze Publications", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
    
    # ========================================================================
    # STEP 3: ANALYSIS
    # ========================================================================
    
    elif st.session_state.step == 3:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Step 3: Analyzing Publications")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get journal ID
        journal_id = st.session_state.journal_info.get('id', '')
        
        # Get selected authors (exclude skipped)
        selected = {}
        for author_str, selection in st.session_state.selected_authors.items():
            if selection and selection != "SKIP":
                selected[author_str] = selection
        
        if not selected:
            st.warning("No authors selected for analysis")
            if st.button("← Back to Author Confirmation"):
                st.session_state.step = 2
                st.rerun()
            st.stop()
        
        results = {}
        total = len(selected)
        
        for i, (author_str, author_id) in enumerate(selected.items()):
            # Update progress
            progress = i / total
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {author_str}... ({i+1}/{total})")
            
            # Get papers
            papers = get_author_works_in_journal(
                author_id,
                journal_id,
                years=None  # Add year filtering if needed
            )
            
            results[author_str] = papers
            
            # Small delay to be polite
            time.sleep(0.2)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        st.session_state.results = results
        st.session_state.step = 4
        time.sleep(1)
        st.rerun()
    
    # ========================================================================
    # STEP 4: RESULTS
    # ========================================================================
    
    elif st.session_state.step == 4:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Step 4: Results")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show journal info
        if st.session_state.journal_info:
            st.markdown(f"""
            <div class="info-message">
                <strong>📰 Journal:</strong> {st.session_state.journal_info['display_name']}
            </div>
            """, unsafe_allow_html=True)
        
        # Display summary
        display_results_summary(st.session_state.results)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📑 Papers by Author", "📈 Analysis", "📥 Export"])
        
        with tab1:
            for author_name, papers in st.session_state.results.items():
                display_author_papers(author_name, papers)
                st.markdown("---")
        
        with tab2:
            st.markdown("### 📈 Publication Timeline")
            
            # Prepare timeline data
            timeline_data = []
            for author_name, papers in st.session_state.results.items():
                for paper in papers:
                    timeline_data.append({
                        'Author': author_name,
                        'Year': paper.get('publication_year', 0),
                        'Title': paper.get('title', ''),
                        'Citations': paper.get('cited_by_count', 0)
                    })
            
            if timeline_data:
                df = pd.DataFrame(timeline_data)
                
                # Filter out years with no data
                df = df[df['Year'] > 0]
                
                if not df.empty:
                    # Publications over time
                    fig = px.histogram(df, x='Year', color='Author',
                                      title='Publications Over Time',
                                      barmode='group',
                                      color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Citations over time
                    fig2 = px.scatter(df, x='Year', y='Citations', color='Author',
                                     size='Citations', hover_data=['Title'],
                                     title='Citations by Year',
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Statistics
                    st.markdown("### 📊 Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Papers", len(df))
                    with col2:
                        st.metric("Total Citations", df['Citations'].sum())
                    with col3:
                        st.metric("Average Citations", f"{df['Citations'].mean():.1f}")
                    with col4:
                        st.metric("Year Range", f"{df['Year'].min()} - {df['Year'].max()}")
                    
                    # Author collaboration network (simplified)
                    st.markdown("### 🤝 Co-authorship Network")
                    
                    # Count co-authored papers
                    coauth_matrix = defaultdict(int)
                    for author_name, papers in st.session_state.results.items():
                        for paper in papers:
                            authors = paper.get('authors', [])
                            # Check which selected authors are in this paper
                            selected_in_paper = [a for a in st.session_state.results.keys() 
                                                if any(name in a for name in authors)]
                            for i, a1 in enumerate(selected_in_paper):
                                for a2 in selected_in_paper[i+1:]:
                                    coauth_matrix[(a1, a2)] += 1
                    
                    if coauth_matrix:
                        # Create network data
                        nodes = list(st.session_state.results.keys())
                        links = []
                        for (a1, a2), count in coauth_matrix.items():
                            links.append({'source': a1, 'target': a2, 'value': count})
                        
                        fig3 = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=nodes,
                                color="blue"
                            ),
                            link=dict(
                                source=[nodes.index(l['source']) for l in links],
                                target=[nodes.index(l['target']) for l in links],
                                value=[l['value'] for l in links]
                            )
                        )])
                        
                        fig3.update_layout(title_text="Co-authorship Network", font_size=10)
                        st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No timeline data available")
        
        with tab3:
            st.markdown("### 📥 Export Results")
            
            # Prepare data for export
            export_data = []
            for author_name, papers in st.session_state.results.items():
                for paper in papers:
                    export_data.append({
                        'Author': author_name,
                        'Author ID': st.session_state.selected_authors.get(author_name, ''),
                        'Title': paper.get('title', ''),
                        'Year': paper.get('publication_year', ''),
                        'Journal': paper.get('journal', ''),
                        'Citations': paper.get('cited_by_count', 0),
                        'DOI': paper.get('doi', ''),
                        'DOI URL': paper.get('doi_url', ''),
                        'Type': paper.get('type', ''),
                        'Open Access': paper.get('is_oa', False)
                    })
            
            if export_data:
                df = pd.DataFrame(export_data)
                
                # CSV export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"author_publications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Excel export
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Publications', index=False)
                    
                    # Add summary sheet
                    summary = df.groupby('Author').agg({
                        'Title': 'count',
                        'Citations': 'sum'
                    }).rename(columns={'Title': 'Papers'})
                    summary['Average Citations'] = summary['Citations'] / summary['Papers']
                    summary.to_excel(writer, sheet_name='Summary')
                    
                    workbook = writer.book
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#0066cc',
                        'font_color': 'white',
                        'border': 1
                    })
                    
                    for sheet_name in writer.sheets:
                        worksheet = writer.sheets[sheet_name]
                        for col_num, value in enumerate(df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                
                st.download_button(
                    label="📈 Download Excel",
                    data=output.getvalue(),
                    file_name=f"author_publications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # New search button
        if st.button("🔄 New Analysis", use_container_width=True):
            # Clear session
            for key in ['step', 'extracted_authors', 'parsed_authors', 'author_candidates',
                       'selected_authors', 'journal_info', 'results', 'current_author_index']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
