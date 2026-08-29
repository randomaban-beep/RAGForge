import sys
from pathlib import Path

import streamlit as st


# =========================================================
# PATH SETUP
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =========================================================
# IMPORT PIPELINE
# =========================================================

from pipeline import RAGPipeline


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAGForge",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("🧠 RAGForge")

st.caption(
    "Adaptive Hybrid Retrieval-Augmented Generation"
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# =========================================================
# INITIALIZE RAG PIPELINE
# =========================================================

@st.cache_resource
def load_rag():
    return RAGPipeline()


try:

    rag = load_rag()

    st.session_state.rag_initialized = True

except Exception as error:

    st.session_state.rag_initialized = False

    st.error(
        f"Failed to initialize RAGForge: {error}"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.write(
        "Upload documents to add them "
        "to the knowledge base."
    )


    # =====================================================
    # FILE UPLOADER
    # =====================================================

    uploaded_files = st.file_uploader(

        "Upload documents",

        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],

        accept_multiple_files=True,
    )


    # =====================================================
    # UPLOAD / INDEX
    # =====================================================

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} "
            f"file(s) uploaded"
        )


        if st.button(
            "🔍 Index Documents",
            use_container_width=True,
        ):

            upload_dir = (
                ROOT_DIR
                / "data"
                / "uploads"
            )

            upload_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            file_paths = []


            # -------------------------------------------------
            # SAVE FILES
            # -------------------------------------------------

            with st.spinner(
                "Saving documents..."
            ):

                for uploaded_file in uploaded_files:

                    file_path = (
                        upload_dir
                        / uploaded_file.name
                    )


                    with open(
                        file_path,
                        "wb",
                    ) as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )


                    file_paths.append(
                        str(file_path)
                    )


            # -------------------------------------------------
            # INDEX DOCUMENTS
            # -------------------------------------------------

            try:

                with st.spinner(
                    "Indexing documents..."
                ):

                    count = rag.index_documents(
                        file_paths
                    )


                st.success(
                    f"Successfully indexed "
                    f"{count} chunks!"
                )


                st.session_state.indexed = True


            except Exception as error:

                st.error(
                    f"Indexing failed: {error}"
                )


    else:

        st.info(
            "Upload one or more documents "
            "to build your knowledge base."
        )


    # =====================================================
    # SYSTEM STATUS
    # =====================================================

    st.divider()

    st.header("⚙️ System")


    if st.session_state.rag_initialized:

        st.success(
            "RAGForge is ready"
        )

    else:

        st.error(
            "RAGForge initialization failed"
        )


    st.caption(
        "Router → Hybrid Retrieval → "
        "Reranker → LLM"
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        # -------------------------------------------------
        # DISPLAY SOURCES
        # -------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown(
                "### Sources"
            )


            for source in message["sources"]:

                source_id = source.get(
                    "id",
                    "?",
                )


                score = source.get(
                    "score"
                )


                content = source.get(
                    "content",
                    "",
                )


                metadata = source.get(
                    "metadata",
                    {},
                )


                # -------------------------------------------------
                # Source label
                # -------------------------------------------------

                if score is not None:

                    label = (
                        f"Source {source_id} "
                        f"— Score: {score:.4f}"
                    )

                else:

                    label = (
                        f"Source {source_id}"
                    )


                with st.expander(
                    label
                ):

                    st.write(
                        content
                    )


                    if metadata:

                        st.caption(
                            f"Metadata: {metadata}"
                        )


# =========================================================
# CHAT INPUT
# =========================================================

query = st.chat_input(
    "Ask something about your documents..."
)


# =========================================================
# HANDLE USER QUERY
# =========================================================

if query:

    # =====================================================
    # IMPORTANT
    #
    # Capture the OLD conversation BEFORE adding
    # the current user message.
    #
    # This is what allows:
    #
    # User: Tell me an interesting fact
    # Assistant: ...
    # User: Another one
    #
    # to be understood correctly.
    # =====================================================

    conversation_history = (
        st.session_state.messages.copy()
    )


    # =====================================================
    # DISPLAY USER MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            query
        )


    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            try:

                # -------------------------------------------------
                # PASS CONVERSATION HISTORY
                # -------------------------------------------------

                result = rag.query(

                    user_query=query,

                    conversation_history=(
                        conversation_history
                    ),
                )


                # =================================================
                # RESULT
                # =================================================

                answer = result.get(
                    "answer",
                    "I couldn't generate an answer.",
                )


                sources = result.get(
                    "sources",
                    [],
                )


                route = result.get(
                    "route",
                    "unknown",
                )


                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.markdown(
                    answer
                )


                # =================================================
                # PIPELINE DETAILS
                # =================================================

                with st.expander(
                    "🔀 Pipeline details"
                ):

                    st.write(
                        f"**Route:** `{route}`"
                    )


                    if route == "greeting":

                        st.write(
                            "Greeting detected — "
                            "retrieval skipped."
                        )


                    elif route == "casual":

                        st.write(
                            "Casual query detected — "
                            "retrieval skipped."
                        )


                    elif route == "document":

                        st.write(
                            "Document query detected — "
                            "hybrid retrieval + "
                            "reranking used."
                        )


                # =================================================
                # SOURCES
                # =================================================

                if sources:

                    st.markdown(
                        "### Sources"
                    )


                    for source in sources:

                        source_id = source.get(
                            "id",
                            "?",
                        )


                        score = source.get(
                            "score"
                        )


                        content = source.get(
                            "content",
                            "",
                        )


                        metadata = source.get(
                            "metadata",
                            {},
                        )


                        # -------------------------------------------------
                        # Source label
                        # -------------------------------------------------

                        if score is not None:

                            label = (
                                f"Source {source_id} "
                                f"— Score: {score:.4f}"
                            )

                        else:

                            label = (
                                f"Source {source_id}"
                            )


                        with st.expander(
                            label
                        ):

                            st.write(
                                content
                            )


                            if metadata:

                                st.caption(
                                    f"Metadata: {metadata}"
                                )


                # =================================================
                # SAVE ASSISTANT MESSAGE
                # =================================================

                st.session_state.messages.append(

                    {
                        "role": "assistant",

                        "content": answer,

                        "sources": sources,
                    }
                )


            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )


                st.error(
                    error_message
                )


                st.session_state.messages.append(

                    {
                        "role": "assistant",

                        "content": error_message,

                        "sources": [],
                    }
                )