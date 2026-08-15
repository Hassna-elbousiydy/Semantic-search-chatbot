import sys
from pathlib import Path

import streamlit as st


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.preprocessing import load_jsonl
from src.rag import RAGSystem


CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "scientific"
    / "processed"
    / "scientific_chunks.jsonl"
)

RELEVANCE_THRESHOLD = 0.30
RERANKER_CONFIDENCE_THRESHOLD = 0.0
CANDIDATE_POOL = 5


# ------------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Scientific Battery RAG",
    page_icon="🔬",
    layout="wide",
)


# ------------------------------------------------------------------
# Application header
# ------------------------------------------------------------------

st.title(
    "🔬 Scientific Battery RAG"
)

st.write(
    """
    Ask questions about the scientific battery literature
    indexed in this demo.

    The system retrieves evidence from research papers,
    reranks candidate passages, generates a grounded answer,
    and displays the scientific source used for the answer.
    """
)


# ------------------------------------------------------------------
# System information
# ------------------------------------------------------------------

with st.expander(
    "How the system works",
    expanded=False,
):

    st.markdown(
        """
        **Pipeline**

        1. Scientific PDF extraction with PyMuPDF
        2. Text cleaning and token-aware chunking
        3. Semantic embeddings with MiniLM
        4. FAISS dense retrieval
        5. Cross-Encoder reranking
        6. Grounded answer generation with FLAN-T5-base
        7. Out-of-domain refusal using a relevance threshold

        **Current experimental configuration**

        - Candidate pool: `5`
        - FAISS relevance threshold: `0.30`
        - Generator: `google/flan-t5-base`
        - Retrieval benchmark: 25 questions
        """
    )


# ------------------------------------------------------------------
# Load RAG only once
# ------------------------------------------------------------------

@st.cache_resource
def load_rag_system():

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            "Scientific corpus not found at: "
            f"{CORPUS_PATH}"
        )

    corpus = load_jsonl(
        CORPUS_PATH
    )

    rag = RAGSystem(
        corpus,
        relevance_threshold=(
            RELEVANCE_THRESHOLD
        ),
        use_reranker=True,
        reranker_confidence_threshold=(
            RERANKER_CONFIDENCE_THRESHOLD
        ),
        candidate_pool=(
            CANDIDATE_POOL
        ),
    )

    return rag, len(corpus)


try:

    rag, corpus_size = (
        load_rag_system()
    )

except Exception as error:

    st.error(
        "The Scientific RAG system "
        "could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# ------------------------------------------------------------------
# Corpus information
# ------------------------------------------------------------------

st.caption(
    f"Scientific knowledge base: "
    f"{corpus_size} indexed passages"
)


# ------------------------------------------------------------------
# Example questions
# ------------------------------------------------------------------

st.subheader(
    "Example questions"
)

example_questions = [
    (
        "What degradation mechanisms are considered "
        "in the fractional-order battery model?"
    ),
    (
        "What problem is the AC + DC heating strategy "
        "designed to prevent?"
    ),
    (
        "Which three states were investigated in detail "
        "for photovoltaic size, battery capacity, "
        "performance, and cost?"
    ),
    (
        "What optimization method is used to evaluate "
        "all possible combinations of components and "
        "control strategies?"
    ),
    (
        "What battery degradation phenomenon does "
        "the reduced-order model predict?"
    ),
]


selected_example = st.selectbox(
    "Choose an example or write your own question below:",
    options=[
        "Write my own question"
    ]
    + example_questions,
)


default_question = ""

if (
    selected_example
    != "Write my own question"
):

    default_question = (
        selected_example
    )


# ------------------------------------------------------------------
# Question input
# ------------------------------------------------------------------

question = st.text_area(
    "Scientific question",
    value=default_question,
    height=100,
    placeholder=(
        "Example: What problem is the "
        "AC + DC heating strategy "
        "designed to prevent?"
    ),
)


ask_button = st.button(
    "Ask the Scientific RAG",
    type="primary",
    use_container_width=True,
)


# ------------------------------------------------------------------
# Answer generation
# ------------------------------------------------------------------

if ask_button:

    clean_question = (
        question.strip()
    )

    if not clean_question:

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Retrieving scientific evidence "
            "and generating the answer..."
        ):

            try:

                result = rag.answer(
                    clean_question,
                    k=3,
                )

            except Exception as error:

                st.error(
                    "An error occurred while "
                    "processing the question."
                )

                st.exception(
                    error
                )

                st.stop()


        # ----------------------------------------------------------
        # Generated answer
        # ----------------------------------------------------------

        st.divider()

        st.subheader(
            "Answer"
        )

        answer = result.get(
            "answer",
            (
                "I don't know based on "
                "the provided context."
            ),
        )

        st.success(
            answer
        )


        # ----------------------------------------------------------
        # Selection method
        # ----------------------------------------------------------

        selection_method = result.get(
            "selection_method",
            "unknown",
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.metric(
                "Selection method",
                selection_method,
            )

        with col2:

            st.metric(
                "Candidate pool",
                CANDIDATE_POOL,
            )


        # ----------------------------------------------------------
        # Scientific sources
        # ----------------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        if not sources:

            st.info(
                "No supporting scientific "
                "source was returned."
            )

        else:

            st.subheader(
                "Scientific evidence"
            )

            for index, source in enumerate(
                sources,
                start=1,
            ):

                source_file = source.get(
                    "source_file",
                    "Unknown source",
                )

                page = source.get(
                    "page",
                    "Unknown",
                )

                chunk_id = source.get(
                    "chunk_id",
                    "Unknown",
                )

                text = source.get(
                    "text",
                    "",
                )

                faiss_score = source.get(
                    "retrieval_score",
                    source.get("score"),
                )

                rerank_score = source.get(
                    "rerank_score"
                )


                with st.expander(
                    (
                        f"Evidence {index} — "
                        f"page {page}"
                    ),
                    expanded=(
                        index == 1
                    ),
                ):

                    st.markdown(
                        "**Article**"
                    )

                    st.code(
                        source_file,
                        language=None,
                    )

                    metadata_col1, metadata_col2 = (
                        st.columns(2)
                    )

                    with metadata_col1:

                        st.write(
                            f"**Page:** "
                            f"{page}"
                        )

                        st.write(
                            f"**Chunk:** "
                            f"{chunk_id}"
                        )

                    with metadata_col2:

                        if (
                            faiss_score
                            is not None
                        ):

                            st.write(
                                "**FAISS score:** "
                                f"{faiss_score:.4f}"
                            )

                        if (
                            rerank_score
                            is not None
                        ):

                            st.write(
                                "**Reranker score:** "
                                f"{rerank_score:.4f}"
                            )


                    st.markdown(
                        "**Retrieved passage**"
                    )

                    if text:

                        st.write(
                            text
                        )

                    else:

                        st.caption(
                            "Passage text unavailable."
                        )


        # ----------------------------------------------------------
        # Safety / grounding note
        # ----------------------------------------------------------

        st.divider()

        st.caption(
            "This demo answers from the indexed "
            "scientific corpus only. "
            "When semantic relevance is below the "
            "configured threshold, the system "
            "refuses the question instead of "
            "attempting an unsupported answer."
        )


# ------------------------------------------------------------------
# Portfolio benchmark
# ------------------------------------------------------------------

st.divider()

st.subheader(
    "Evaluation snapshot"
)

metric_col1, metric_col2, metric_col3, metric_col4 = (
    st.columns(4)
)

with metric_col1:

    st.metric(
        "Hit@5",
        "77.3%",
    )

with metric_col2:

    st.metric(
        "MRR@5",
        "0.455",
    )

with metric_col3:

    st.metric(
        "Source accuracy",
        "68.2%",
    )

with metric_col4:

    st.metric(
        "OOD refusal",
        "100%",
    )


st.caption(
    """
    Results are measured on the current 25-question
    manually annotated scientific evaluation set.
    The benchmark is intentionally small and should
    not be interpreted as general-domain performance.
    """
)