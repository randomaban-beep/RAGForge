# Repository review and cleanup

Reviewed every original tracked source file, README, requirements file, and .gitignore.

| Original issue | Change |
|---|---|
| Cached mutable pipeline shared across Streamlit users | Pipeline and knowledge base live in each session; isolation test |
| Uploaded filenames used as disk paths | Parse bytes in memory; sanitize display names; cap upload sizes |
| Ingestion implemented twice; active pipeline bypassed ingestion package | Single loader and chunker path with stable IDs and source metadata |
| RRF keyed by text, merging different sources with identical content | Fuse by stable document ID; regression test |
| Manual scripts named test_*.py required API keys or missing sample.txt | Replace with pytest tests, offline CLI smoke gate, and UI test |
| API routing call and general chat obscured the document-focused use case | Remove router/casual path; self-contained questions only |
| Follow-up claims exceeded retrieval behavior | Remove unsupported conversation-rewriting claims; UI states scope |
| In-memory Qdrant and LangChain wrappers added dependencies without persistence | Local exact dense index plus BM25; document scale/persistence limitations |
| CrossEncoder score transformation suggested probability semantics | Return native model ranking score; never label it confidence |
| Large environment freeze omitted app/document dependencies | Separate direct runtime, development, and neural requirements; tested environment snapshot |
| README repeated feature lists and listed nonexistent evaluation directory | Runnable quickstart, actual benchmark, protocol, research, and resume evidence |

Neural retrieval, hybrid RRF, cross-encoder reranking, optional OpenAI generation, PDF/DOCX/TXT/MD
ingestion, and source inspection remain available. The default is a lightweight BM25/excerpt demo.
No production readiness or LLM quality is inferred from offline retrieval measurements.
