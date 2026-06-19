from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

while True:
    text = input("Enter a sentence (or 'exit' to quit): ").strip()
    if text.lower() == 'exit':
        break
    result = sentiment_analyzer(text)[0]
    print(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
