import streamlit as st
import pandas as pd
import plotly.express as px
import fitz
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="KrishiEvidence AI",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}
h1, h2, h3 {
    color: #1E563E;
}
div[data-testid="stMetric"] {
    background: #F0F6F1;
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 KrishiEvidence AI")
st.caption("Agricultural Research-to-Field Intelligence System")

st.info(
    "Independent research prototype. Answers are based on documents "
    "uploaded in this session. Expert review is required."
)

# SIDEBAR
with st.sidebar:
    st.header("Research Workspace")

    pdfs = st.file_uploader(
        "Upload agricultural research PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.caption(
        "Upload agricultural research papers, ICAR publications, "
        "crop advisories and technical reports."
    )

    st.divider()

    feedback_file = st.file_uploader(
        "Upload field feedback CSV (optional)",
        type=["csv"]
    )


# PDF EXTRACTION
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


# BUILD KNOWLEDGE BASE
documents = []

if pdfs:

    with st.spinner("Reading agricultural documents..."):

        for pdf in pdfs:

            try:
                documents.extend(
                    extract_pdf(pdf.getvalue(), pdf.name)
                )

            except Exception as error:
                st.error(f"Error reading {pdf.name}: {error}")


# SEARCH FUNCTION
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


# APPLICATION TABS
research_tab, verify_tab, feedback_tab, about_tab = st.tabs([
    "🔎 Research Assistant",
    "🛡️ Evidence Verification",
    "📊 Field Feedback",
    "ℹ️ About"
])


# TAB 1: RESEARCH ASSISTANT
with research_tab:

    st.subheader("Agricultural Research Assistant")

    if documents:

        st.success(
            f"{len(pdfs)} PDF(s) loaded | "
            f"{len(documents)} searchable passages"
        )

        question = st.text_input(
            "Ask an agricultural research question",
            placeholder="What does the literature say about soybean seed treatment?"
        )

        top_k = st.slider(
            "Number of evidence passages",
            1, 8, 4
        )

        if st.button(
            "Search Evidence",
            type="primary",
            disabled=not question.strip()
        ):

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

                st.markdown("### Retrieved Evidence")

                st.caption(
                    "These are retrieved source passages, "
                    "not AI-generated conclusions."
                )

                for number, (index, score) in enumerate(
                    results, start=1
                ):

                    item = documents[index]

                    with st.container(border=True):

                        st.markdown(
                            f"**Evidence {number}**"
                        )

                        st.write(
                            f"Source: {item['source']}"
                        )

                        st.write(
                            f"Page: {item['page']}"
                        )

                        st.caption(
                            f"Text similarity: {score:.3f}"
                        )

                        st.write(item["text"])

                st.info(
                    "Check the publication date, crop, region, "
                    "season and experimental conditions before "
                    "applying any research finding."
                )

    else:

        st.write(
            "Upload agricultural research PDFs using "
            "the left sidebar to create your knowledge base."
        )


# TAB 2: EVIDENCE VERIFICATION
with verify_tab:

    st.subheader("Evidence Verification")

    st.write(
        "Enter a proposed agricultural statement. "
        "The system will search for related passages."
    )

    claim = st.text_area(
        "Statement to verify",
        placeholder="Enter a proposed agricultural recommendation..."
    )

    if st.button(
        "Find Supporting Evidence",
        disabled=not claim.strip()
    ):

        if not documents:

            st.warning("Upload PDFs first.")

        else:

            results = retrieve_evidence(claim, 3)

            if results[0][1] < 0.025:

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
                            f"**{item['source']} | Page {item['page']}**"
                        )

                        st.caption(
                            f"Similarity: {score:.3f}"
                        )

                        st.write(item["text"])


# TAB 3: FIELD FEEDBACK
with feedback_tab:

    st.subheader("Field Feedback Intelligence")

    st.write(
        "Analyze recurring agricultural issues "
        "reported in field observations."
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
                "#A8C9AE",
                "#245B43"
            ]
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        st.subheader("Location-wise Issues")

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

        st.plotly_chart(
            chart2,
            use_container_width=True
        )

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


# TAB 4: ABOUT
with about_tab:

    st.subheader("About KrishiEvidence AI")

    st.write(
        "KrishiEvidence AI is an independent student research "
        "prototype designed to help agricultural researchers "
        "search publications, inspect supporting passages "
        "and explore field-feedback themes."
    )

    st.markdown("### Current Capabilities")

    st.markdown("""
    - Agricultural PDF document extraction.
    - Page-level source identification.
    - Searchable document passages.
    - Evidence retrieval using TF-IDF.
    - Related-passage verification.
    - Field feedback analysis.
    """)

    st.markdown("### Current Limitations")

    st.markdown("""
    - Not yet a complete generative RAG system.
    - Similarity does not prove scientific correctness.
    - Cannot establish causation.
    - Cannot certify agricultural recommendations.
    - Scanned PDFs may require OCR.
    """)

    st.markdown("### Future Development")

    st.markdown("""
    1. Semantic search using embeddings.
    2. Vector database integration.
    3. LLM-based answers with source citations.
    4. Advanced evidence verification.
    5. Research-gap analytics.
    6. Expert-reviewed evaluation.
    """)

    st.caption(
        "Independent student prototype. "
        "No ICAR affiliation or endorsement is implied."
    )


st.divider()

st.caption(
    "KrishiEvidence AI | Prototype v0.1 | "
    "Research support, not a substitute for expert judgement."
)