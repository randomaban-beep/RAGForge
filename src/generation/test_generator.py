from generator import RAGGenerator


def main():

    generator = RAGGenerator()

    fake_document = type(
        "Document",
        (),
        {
            "page_content":
            "Hybrid retrieval combines dense "
            "vector retrieval with BM25 keyword "
            "retrieval."
        }
    )()

    answer = generator.generate(
        "What is hybrid retrieval?",
        [fake_document]
    )

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    main()