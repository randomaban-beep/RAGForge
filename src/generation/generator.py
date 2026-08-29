import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class RAGGenerator:

    def __init__(
        self,
        model="gpt-5-mini"
    ):

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "OPENAI_API_KEY is missing from .env"
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = model


    # ======================================================
    # FORMAT CONVERSATION
    # ======================================================

    def _format_history(
        self,
        conversation_history
    ):

        if not conversation_history:

            return "No previous conversation."


        recent_messages = (
            conversation_history[-10:]
        )

        history = []

        for message in recent_messages:

            role = message.get(
                "role",
                "user"
            )

            content = message.get(
                "content",
                ""
            )

            history.append(
                f"{role.upper()}: {content}"
            )

        return "\n".join(history)


    # ======================================================
    # DOCUMENT GENERATION
    # ======================================================

    def generate(
        self,
        query,
        documents,
        conversation_history=None
    ):

        if conversation_history is None:

            conversation_history = []


        # --------------------------------------------------
        # Build retrieved context
        # --------------------------------------------------

        context_parts = []

        for i, document in enumerate(
            documents,
            start=1
        ):

            context_parts.append(
                f"[Source {i}]\n"
                f"{document.page_content}"
            )

        context = "\n\n".join(
            context_parts
        )


        # --------------------------------------------------
        # Conversation history
        # --------------------------------------------------

        history = self._format_history(
            conversation_history
        )


        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------

        prompt = f"""
You are RAGForge, a Retrieval-Augmented
Generation assistant.

Answer the CURRENT USER QUESTION using
the retrieved document context.

You may use the conversation history to
understand references and follow-up questions.

IMPORTANT RULES:

- Use the retrieved document context as the
  factual source.
- Do not invent information.
- If the retrieved context does not contain
  enough information, say that you cannot find
  the answer in the provided documents.
- Understand follow-up questions using the
  conversation history.
- If the user asks "another one", "tell me more",
  "what about that?", etc., resolve the reference
  using the conversation history.
- Cite the sources you actually use.
- Keep the answer clear and reasonably concise.

CONVERSATION HISTORY:
{history}

CURRENT USER QUESTION:
{query}

RETRIEVED DOCUMENT CONTEXT:
{context}
"""


        # --------------------------------------------------
        # OpenAI
        # --------------------------------------------------

        response = self.client.responses.create(

            model=self.model,

            input=prompt
        )

        return response.output_text


    # ======================================================
    # CASUAL GENERATION
    # ======================================================

    def generate_casual(
        self,
        query,
        conversation_history=None
    ):

        if conversation_history is None:

            conversation_history = []


        history = self._format_history(
            conversation_history
        )


        prompt = f"""
You are RAGForge, a helpful AI assistant.

The user's current message is casual
conversation and does not require information
from the document knowledge base.

Respond naturally.

Use the conversation history when useful.

Do not retrieve documents.
Do not cite sources.
Do not invent information about the
user's documents.

CONVERSATION HISTORY:
{history}

CURRENT USER MESSAGE:
{query}
"""


        response = self.client.responses.create(

            model=self.model,

            input=prompt
        )

        return response.output_text