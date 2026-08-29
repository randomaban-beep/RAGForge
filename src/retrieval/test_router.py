from router import QueryRouter


def main():

    router = QueryRouter()

    queries = [
        "hi",
        "yo what's up?",
        "how are you?",
        "what am I good at?",
        "walk me through my background",
        "what technologies have I used?",
        "tell me about something I built",
        "what did I do during my internship?",
        "can you explain my experience?",
        "thanks",
        "what can you do?",
    ]

    for query in queries:

        result = router.route(query)

        print("\nQUERY:", query)
        print("ROUTE:", result.route.value)
        print("REASON:", result.reason)


if __name__ == "__main__":
    main()