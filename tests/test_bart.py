
from app.summarizer import get_summarizer

print("Loading BART model...")

try:
    summarizer = get_summarizer()
    print("Model loaded successfully!")

    text = """
    Artificial intelligence allows computers to perform tasks
    that usually require human intelligence. Machine learning
    helps computers learn patterns from data. Natural language
    processing enables computers to understand and generate text.
    """

    result = summarizer(
        text,
        max_length=60,
        min_length=5,
        do_sample=False
    )

    print("\nGenerated summary:")
    print(result[0]["summary_text"])
    print("\nBART test completed successfully!")

except Exception as exc:
    print("\nBART test failed.")
    print(f"Error: {type(exc).__name__}: {exc}")
    raise
