import streamlit as st
import pandas as pd
import plotly.express as px
import fitz
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="KrishiEvidence AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM UI DESIGN
# =========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F5F7FA;
    color: #17263C;
}

.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111D32;
    border-right: 1px solid #263750;
}

section[data-testid="stSidebar"] * {
    color: #E7EDF5;
}

section[data-testid="stSidebar"] hr {
    border-color: #34445A;
}

section[data-testid="stSidebar"] label {
    color: #D3DCE8 !important;
}

/* Sidebar radio navigation */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 8px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: #1C2C43;
    padding: 12px 14px;
    border-radius: 9px;
    border: 1px solid #293D57;
    transition: 0.2s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #263D58;
    border-color: #16A6A0;
}

/* Main headings */
h1, h2, h3 {
    color: #17263C !important;
    font-weight: 700;
}

/* Brand */
.brand-title {
    font-size: 27px;
    font-weight: 750;
    color: #FFFFFF;
    letter-spacing: -0.7px;
}

.brand-subtitle {
    font-size: 10px;
    letter-spacing: 1.5px;
    color: #9EAFBF;
    margin-top: 3px;
}

/* Main header */
.page-eyebrow {
    color: #159A96;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.hero-title {
    font-size: 35px;
    font-weight: 750;
    color: #17263C;
    letter-spacing: -1px;
    margin-top: 5px;
}

.hero-description {
    color: #64748B;
    font-size: 15px;
    margin-top: 5px;
    margin-bottom: 25px;
}

/* Section cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 13px;
    border-color: #DFE6ED;
    background-color: #FFFFFF;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E7EF;
    border-radius: 12px;
    padding: 17px;
}

/* Buttons */
.stButton > button {
    background-color: #159D99;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 20px;
    font-weight: 650;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #117E7B;
    color: white;
    border: none;
}

/* Text input */
.stTextInput input,
.stTextArea textarea {
    background-color: #FFFFFF;
    border: 1px solid #D8E1EA;
    border-radius: 9px;
    color: #17263C;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #6D879D;
    border-radius: 10px;
    padding: 8px;
}

/* Evidence */
.evidence-label {
    color: #159D99;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.evidence-source {
    color: #526477;
    font-size: 13px;
}

/* Info boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #E0E7EF;
}

/* Hide Streamlit's default decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR BRAND AND NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
        <div style="display:flex;align-items:center;gap:11px;">
            <div style="
                background:#159D99;
                padding:10px;
                border-radius:10px;
                font-size:23px;">
                📚
            </div>
            <div>
                <div class="brand-title">KrishiEvidence AI</div>
                <div class="brand-subtitle">
                    RESEARCH INTELLIGENCE
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        "<p style='font-size:11px;color:#91A5B9;"
        "letter-spacing:1.5px;font-weight:700;'>WORKSPACE</p>",
        unsafe_allow_html=True
    )

    selected_module = st.radio(
        "Navigation",
        [
            "Research Assistant",
            "Evidence Verification",
            "Field Intelligence",
            "About Platform"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "<p style='font-size:11px;color:#91A5B9;"
        "letter-spacing:1.5px;font-weight:700;'>KNOWLEDGE BASE</p>",
        unsafe_allow_html=True
    )

    pdfs = st.file_uploader(
        "Upload research publications",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.caption(
        "Upload ICAR publications, agricultural research "
        "papers and technical reports."
    )

    if pdfs:
        st.success(f"{len(pdfs)} PDF file(s) selected.")

    st.divider()

    st.markdown(
        "<p style='font-size:11px;color:#91A5B9;"
        "letter-spacing:1.5px;font-weight:700;'>FIELD DATA</p>",
        unsafe_allow_html=True
    )

    feedback_file = st.file_uploader(
        "Upload field feedback CSV",
        type=["csv"]
    )

    st.divider()

    st.caption("KrishiEvidence AI · Prototype v0.1")

# =========================================================
# PDF EXTRACTION — ORIGINAL LOGIC
# =========================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner=False)
def extract_pdf(file_bytes, filename):

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    passages = []

    for page_number, page in enumerate(document, start=1):

        text = clean_text(page.get_text())
        words = text.split()

        chunk_size = 180
        overlap = 35

        for start in range(
            0,
            len(words),
            chunk_size - overlap
        ):

            chunk = " ".join(
                words[start:start + chunk_size]
            )

            if len(chunk) > 50:

                passages.append({
                    "source": filename,
                    "page": page_number,
                    "text": chunk
                })

    return passages

# =========================================================
# BUILD KNOWLEDGE BASE — ORIGINAL LOGIC
# =========================================================

documents = []

if pdfs:

    with st.spinner("Processing research publications..."):

        for pdf in pdfs:

            try:
                documents.extend(
                    extract_pdf(pdf.getvalue(), pdf.name)
                )

            except Exception as error:
                st.error(f"Error reading {pdf.name}: {error}")

# =========================================================
# RETRIEVAL ENGINE — ORIGINAL LOGIC
# =========================================================

def retrieve_evidence(question, top_k=4):

    corpus = [
        document["text"]
        for document in documents
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000
    )

    matrix = vectorizer.fit_transform(
        corpus + [question]
    )

    scores = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).flatten()

    indexes = scores.argsort()[::-1][:top_k]

    return [
        (int(index), float(scores[index]))
        for index in indexes
    ]

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="page-eyebrow">AGRICULTURAL RESEARCH WORKSPACE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">KrishiEvidence AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-description">'
    'Discover, retrieve and examine evidence from agricultural research.'
    '</div>',
    unsafe_allow_html=True
)

if documents:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Research Documents",
        len(pdfs)
    )

    col2.metric(
        "Searchable Passages",
        len(documents)
    )

    col3.metric(
        "Knowledge Base",
        "Active"
    )

else:

    st.info(
        "Your research workspace is ready. "
        "Upload agricultural research PDFs from the left sidebar "
        "to activate the knowledge base."
    )

st.divider()

# =========================================================
# MODULE 1 — RESEARCH ASSISTANT
# =========================================================

if selected_module == "Research Assistant":

    st.markdown("### Research Assistant")

    st.caption(
        "Ask questions and retrieve relevant passages "
        "from your uploaded agricultural publications."
    )

    with st.container(border=True):

        st.markdown(
            '<div class="evidence-label">RESEARCH QUERY</div>',
            unsafe_allow_html=True
        )

        question = st.text_input(
            "Ask your research question",
            placeholder=(
                "Example: What are the major climate-related "
                "challenges affecting soybean production?"
            ),
            label_visibility="collapsed"
        )

        top_k = st.slider(
            "Number of evidence passages",
            1, 8, 4
        )

        search_button = st.button(
            "Search Research Evidence",
            type="primary",
            disabled=not question.strip() or not documents
        )

    if not documents:

        st.warning(
            "Upload research PDFs from the sidebar to begin."
        )

    if search_button:

        with st.spinner("Searching the research knowledge base..."):

            results = retrieve_evidence(
                question,
                top_k
            )

        if not results or results[0][1] < 0.025:

            st.warning(
                "No sufficiently relevant passage was found. "
                "Try different keywords or another document."
            )

        else:

            st.markdown("### Retrieved Research Evidence")

            st.caption(
                "Original source passages retrieved from your documents. "
                "These are not AI-generated conclusions."
            )

            for number, (index, score) in enumerate(
                results,
                start=1
            ):

                item = documents[index]

                with st.container(border=True):

                    st.markdown(
                        f'<div class="evidence-label">'
                        f'EVIDENCE {number}</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"**{item['source']}** · Page {item['page']}"
                    )

                    st.caption(
                        f"Text similarity: {score:.3f}"
                    )

                    st.write(item["text"])

            st.info(
                "Check publication date, crop, region, season "
                "and experimental conditions before applying "
                "any research finding."
            )

# =========================================================
# MODULE 2 — EVIDENCE VERIFICATION
# =========================================================

elif selected_module == "Evidence Verification":

    st.markdown("### Evidence Verification")

    st.caption(
        "Examine whether an agricultural statement "
        "has related supporting passages in your documents."
    )

    with st.container(border=True):

        claim = st.text_area(
            "Statement to examine",
            placeholder=(
                "Example: Drought during flowering "
                "reduces soybean productivity."
            )
        )

        verify_button = st.button(
            "Find Supporting Evidence",
            type="primary",
            disabled=not claim.strip()
        )

    if verify_button:

        if not documents:

            st.warning("Upload PDFs first.")

        else:

            with st.spinner("Examining available evidence..."):

                results = retrieve_evidence(claim, 3)

            if not results or results[0][1] < 0.025:

                st.error(
                    "No closely related passage found "
                    "in the current knowledge base."
                )

                st.write(
                    "Status: Not verified from uploaded documents. "
                    "This does not prove the statement is false."
                )

            else:

                st.warning(
                    "Related evidence found. "
                    "Human expert review is still required."
                )

                for index, score in results:

                    item = documents[index]

                    with st.container(border=True):

                        st.markdown(
                            f"**{item['source']}** · "
                            f"Page {item['page']}"
                        )

                        st.caption(
                            f"Text similarity: {score:.3f}"
                        )

                        st.write(item["text"])

# =========================================================
# MODULE 3 — FIELD INTELLIGENCE
# =========================================================

elif selected_module == "Field Intelligence":

    st.markdown("### Field Intelligence")

    st.caption(
        "Explore recurring agricultural problems "
        "reported through field observations."
    )

    sample_path = (
        Path(__file__).parent
        / "sample_data"
        / "synthetic_field_feedback.csv"
    )

    if feedback_file:

        try:

            feedback = pd.read_csv(feedback_file)
            synthetic = False

        except Exception as error:

            st.error(str(error))
            feedback = pd.DataFrame()
            synthetic = False

    else:

        feedback = pd.read_csv(sample_path)
        synthetic = True

    required_columns = [
        "issue",
        "location",
        "crop",
        "observation"
    ]

    if not feedback.empty and all(
        column in feedback.columns
        for column in required_columns
    ):

        if synthetic:

            st.warning(
                "SYNTHETIC DEMONSTRATION DATA — "
                "NOT REAL FARMER REPORTS"
            )

        else:

            st.info(
                "Use authorized and anonymized records only."
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            len(feedback)
        )

        col2.metric(
            "Locations",
            feedback["location"].nunique()
        )

        col3.metric(
            "Issue Categories",
            feedback["issue"].nunique()
        )

        st.markdown("#### Agricultural Issue Distribution")

        issue_counts = (
            feedback["issue"]
            .value_counts()
            .rename_axis("Issue")
            .reset_index(name="Records")
        )

        chart = px.bar(
            issue_counts,
            x="Issue",
            y="Records",
            color="Records",
            color_continuous_scale=[
                "#B8DCD8",
                "#159D99"
            ]
        )

        chart.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#26384C"
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        st.markdown("#### Location-wise Issues")

        location_data = (
            feedback.groupby(
                ["location", "issue"]
            )
            .size()
            .reset_index(name="records")
        )

        chart2 = px.bar(
            location_data,
            x="location",
            y="records",
            color="issue",
            barmode="stack"
        )

        chart2.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#26384C"
        )

        st.plotly_chart(
            chart2,
            use_container_width=True
        )

        st.markdown("#### Field Observation Records")

        st.dataframe(
            feedback,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Issue Summary",
            issue_counts.to_csv(index=False).encode(),
            "feedback_summary.csv",
            "text/csv"
        )

    else:

        st.warning(
            "CSV must contain: issue, location, crop, observation."
        )

# =========================================================
# MODULE 4 — ABOUT PLATFORM
# =========================================================

elif selected_module == "About Platform":

    st.markdown("### About KrishiEvidence AI")

    st.write(
        "KrishiEvidence AI is an independent student research "
        "prototype designed to help agricultural researchers "
        "search publications, inspect supporting passages "
        "and explore field-feedback themes."
    )

    st.markdown("#### Current Capabilities")

    capabilities = [
        "Agricultural PDF document extraction",
        "Page-level source identification",
        "Searchable document passages",
        "Evidence retrieval using TF-IDF",
        "Related-passage verification",
        "Field feedback analysis"
    ]

    for capability in capabilities:

        st.markdown(f"- {capability}")

    st.divider()

    st.markdown("#### Current Limitations")

    limitations = [
        "Not yet a complete generative RAG system",
        "Similarity does not prove scientific correctness",
        "Cannot establish causation",
        "Cannot certify agricultural recommendations",
        "Scanned PDFs may require OCR"
    ]

    for limitation in limitations:

        st.markdown(f"- {limitation}")

    st.divider()

    st.markdown("#### Future Development")

    roadmap = [
        "Semantic search using embeddings",
        "Vector database integration",
        "LLM-based answers with source citations",
        "Advanced evidence verification",
        "Research-gap analytics",
        "Expert-reviewed evaluation"
    ]

    for item in roadmap:

        st.markdown(f"- {item}")

    st.info(
        "Independent student prototype. "
        "No ICAR affiliation or endorsement is implied."
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#8090A1;
        font-size:12px;
        padding:10px;">
        KrishiEvidence AI · Research Intelligence Platform · v0.1
        <br>
        Research support, not a substitute for expert judgement.
    </div>
    """,
    unsafe_allow_html=True
)