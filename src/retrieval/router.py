import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()


# ==========================================================
# ROUTE TYPES
# ==========================================================

class Route(str, Enum):

    GREETING = "greeting"
    CASUAL = "casual"
    DOCUMENT = "document"


# ==========================================================
# ROUTER DECISION
# ==========================================================

class RouterDecision(BaseModel):

    route: Route
    reason: str


# ==========================================================
# QUERY ROUTER
# ==========================================================

class QueryRouter:

    def __init__(
        self,
        model="gpt-5-mini"
    ):

        print("Loading semantic router...")

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
    # ROUTE QUERY
    # ======================================================

    def route(
        self,
        query,
        conversation_history=None
    ):

        if conversation_history is None:

            conversation_history = []


        # --------------------------------------------------
        # Convert history into readable text
        # --------------------------------------------------

        history_text = ""

        if conversation_history:

            recent_messages = (
                conversation_history[-10:]
            )

            for message in recent_messages:

                role = message.get(
                    "role",
                    "user"
                )

                content = message.get(
                    "content",
                    ""
                )

                history_text += (
                    f"{role.upper()}: "
                    f"{content}\n"
                )

        else:

            history_text = (
                "No previous conversation."
            )


        # --------------------------------------------------
        # Router prompt
        # --------------------------------------------------

        prompt = f"""
You are the semantic router for RAGForge.

RAGForge has access to a user-provided document
knowledge base.

Classify the CURRENT USER QUERY into exactly
one of these routes:

1. greeting
   The user is simply greeting the assistant.

2. casual
   The user is having normal conversation and
   does not need information from the documents.

3. document
   The user is asking about information that could
   come from the uploaded documents.

IMPORTANT:

Use the conversation history to understand
follow-up queries and references.

For example:

Previous:
User: Tell me an interesting fact from this paper.
Assistant: The paper discusses...

Current:
User: Another one.

The current query should be classified as
DOCUMENT because "another one" refers to the
previous document-related request.

Other examples:

"tell me more"
"another one"
"what about that?"
"explain that"
"what else?"
"give me another example"

These should use the previous conversation
to determine what the user is referring to.

Do NOT hard-code specific phrases.

CURRENT USER QUERY:
{query}

CONVERSATION HISTORY:
{history_text}
"""


        # --------------------------------------------------
        # Structured response
        # --------------------------------------------------

        response = self.client.responses.parse(

            model=self.model,

            input=prompt,

            text_format=RouterDecision
        )


        decision = response.output_parsed

        if decision is None:

            raise ValueError(
                "Router failed to produce "
                "a valid decision."
            )

        return decision