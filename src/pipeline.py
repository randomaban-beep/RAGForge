import sys
from pathlib import Path


# =========================================================
# PATH SETUP
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =========================================================
# IMPORTS
# =========================================================

from retrieval.router import QueryRouter

from retrieval.hybrid import (
    load_documents,
    create_chunks,
    create_embeddings,
    create_vector_store,
    create_bm25,
    hybrid_search,
)

from retrieval.reranker import Reranker
from generation.generator import RAGGenerator


# =========================================================
# RAG PIPELINE
# =========================================================

class RAGPipeline:

    def __init__(self):

        print("Initializing RAGForge...")


        # -------------------------------------------------
        # 1. Router
        # -------------------------------------------------

        print("Loading semantic router...")

        self.router = QueryRouter()


        # -------------------------------------------------
        # 2. Embeddings
        # -------------------------------------------------

        print("Loading embedding model...")

        self.embeddings = create_embeddings()


        # -------------------------------------------------
        # 3. Empty knowledge base
        # -------------------------------------------------

        print(
            "Starting with an empty knowledge base..."
        )

        self.chunks = []

        self.vector_store = None

        self.bm25 = None


        # -------------------------------------------------
        # 4. Reranker
        # -------------------------------------------------

        print("Loading reranker...")

        self.reranker = Reranker()


        # -------------------------------------------------
        # 5. Generator
        # -------------------------------------------------

        print("Loading generator...")

        self.generator = RAGGenerator()


        print(
            "\nRAGForge initialized successfully."
        )


    # =====================================================
    # BUILD INDEXES
    # =====================================================

    def _build_indexes(
        self,
        documents
    ):

        if not documents:

            raise ValueError(
                "No documents were loaded."
            )


        # -------------------------------------------------
        # Chunk documents
        # -------------------------------------------------

        print(
            "Creating chunks..."
        )

        self.chunks = create_chunks(
            documents
        )

        print(
            f"Created {len(self.chunks)} chunks."
        )


        if not self.chunks:

            raise ValueError(
                "No chunks were created from "
                "the documents."
            )


        # -------------------------------------------------
        # Vector store
        # -------------------------------------------------

        print(
            "Creating vector store..."
        )

        self.vector_store = create_vector_store(

            self.chunks,

            self.embeddings
        )


        # -------------------------------------------------
        # BM25
        # -------------------------------------------------

        print(
            "Creating BM25 index..."
        )

        self.bm25 = create_bm25(
            self.chunks
        )


    # =====================================================
    # INDEX UPLOADED DOCUMENTS
    # =====================================================

    def index_documents(
        self,
        file_paths
    ):

        if not file_paths:

            raise ValueError(
                "No documents were provided."
            )


        print(
            "\nIndexing uploaded documents..."
        )


        # -------------------------------------------------
        # Load documents
        # -------------------------------------------------

        documents = load_documents(
            file_paths
        )

        print(
            f"Loaded {len(documents)} "
            f"document(s)."
        )


        if not documents:

            raise ValueError(
                "No content could be extracted "
                "from the uploaded documents."
            )


        # -------------------------------------------------
        # Build indexes
        # -------------------------------------------------

        self._build_indexes(
            documents
        )


        print(
            "Documents indexed successfully."
        )


        return len(
            self.chunks
        )


    # =====================================================
    # QUERY
    # =====================================================

    def query(
        self,
        user_query: str,
        conversation_history=None
    ):

        # -------------------------------------------------
        # IMPORTANT:
        # Make sure history always exists.
        # -------------------------------------------------

        if conversation_history is None:

            conversation_history = []


        print(
            f"\nUSER QUERY: {user_query}"
        )


        # =================================================
        # STEP 1 — ROUTER
        # =================================================

        decision = self.router.route(

            user_query,

            conversation_history
        )


        route = decision.route.value


        print(
            f"ROUTE: {route}"
        )

        print(
            f"REASON: {decision.reason}"
        )


        # =================================================
        # GREETING
        # =================================================

        if route == "greeting":

            return {

                "answer": (
                    "Hey! 👋 I'm RAGForge. "
                    "Upload a document and ask me "
                    "something about it."
                ),

                "sources": [],

                "route": route,
            }


        # =================================================
        # CASUAL
        # =================================================

        if route == "casual":

            answer = (

                self.generator.generate_casual(

                    query=user_query,

                    conversation_history=(
                        conversation_history
                    )
                )
            )


            return {

                "answer": answer,

                "sources": [],

                "route": route,
            }


        # =================================================
        # DOCUMENT QUERY
        # =================================================

        # -------------------------------------------------
        # Make sure documents exist
        # -------------------------------------------------

        if not self.chunks:

            return {

                "answer": (
                    "I don't have any documents "
                    "in the knowledge base yet. "
                    "Please upload and index a document "
                    "first."
                ),

                "sources": [],

                "route": "document",
            }


        # -------------------------------------------------
        # Hybrid retrieval
        # -------------------------------------------------

        print(
            "\nRetrieving..."
        )


        candidates = hybrid_search(

            query=user_query,

            chunks=self.chunks,

            vector_store=self.vector_store,

            bm25=self.bm25,

            k=4,
        )


        print(
            f"Retrieved "
            f"{len(candidates)} candidates."
        )


        # -------------------------------------------------
        # Reranking
        # -------------------------------------------------

        reranked = self.reranker.rerank(

            query=user_query,

            documents=candidates,

            top_k=3,
        )


        print(
            f"Reranked to "
            f"{len(reranked)} documents."
        )


        # -------------------------------------------------
        # Extract documents
        # -------------------------------------------------

        documents = [

            document

            for document, score

            in reranked
        ]


        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        answer = self.generator.generate(

            query=user_query,

            documents=documents,

            conversation_history=(
                conversation_history
            )
        )


        # =================================================
        # SOURCES
        # =================================================

        sources = []


        for i, (
            document,
            score,
        ) in enumerate(

            reranked,

            start=1
        ):

            sources.append({

                "id": i,

                "content":
                    document.page_content,

                "score":
                    float(score),

                "metadata":
                    document.metadata,
            })


        # =================================================
        # RETURN
        # =================================================

        return {

            "answer": answer,

            "sources": sources,

            "route": route,
        }


    # =====================================================
    # ASK
    #
    # Streamlit can use either:
    #
    # rag.query(...)
    # rag.ask(...)
    #
    # =====================================================

    def ask(
        self,
        user_query: str,
        conversation_history=None
    ):

        return self.query(

            user_query=user_query,

            conversation_history=(
                conversation_history
            )
        )


# =========================================================
# CLI
# =========================================================

def main():

    pipeline = RAGPipeline()


    print(
        "\n" + "=" * 60
    )


    print(
        "RAGForge ready 🚀"
    )


    print(
        "Type 'exit' to quit."
    )


    print(
        "=" * 60
    )


    # -----------------------------------------------------
    # CLI conversation history
    # -----------------------------------------------------

    conversation_history = []


    while True:

        query = input(
            "\nYou: "
        ).strip()


        if query.lower() == "exit":

            print(
                "\nGoodbye! 👋"
            )

            break


        if not query:

            continue


        try:

            result = pipeline.query(

                query,

                conversation_history
            )


            print(
                "\nRAGForge:"
            )


            print(
                result["answer"]
            )


            # -------------------------------------------------
            # Save conversation
            # -------------------------------------------------

            conversation_history.append({

                "role": "user",

                "content": query,
            })


            conversation_history.append({

                "role": "assistant",

                "content": result["answer"],
            })


            # -------------------------------------------------
            # Sources
            # -------------------------------------------------

            if result["sources"]:

                print(
                    "\nSources:"
                )


                for source in (
                    result["sources"]
                ):

                    print(

                        f"\n[{source['id']}] "
                        f"Score: "
                        f"{source['score']:.4f}"
                    )


                    print(
                        source["content"][:500]
                    )


            print(
                "\n" + "-" * 60
            )


        except Exception as error:

            print(
                f"\nERROR: {error}"
            )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()