from gensim.downloader import load
from transformers import pipeline
import nltk
import string
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

print("Loading word vectors...")
word_vectors = load("glove-wiki-gigaword-100")

print("Loading GPT-2...")
generator = pipeline("text-generation", model="gpt2")

original_prompt = "write an essay on natural disaster"
keyword = "disaster"

words = word_tokenize(original_prompt)
enriched_words = []
for word in words:
    cleaned = word.lower().strip(string.punctuation)
    if cleaned == keyword:
        similar = word_vectors.most_similar(cleaned, topn=1)
        replacement = similar[0][0]
        print(f"Replacing {word} -> {replacement}")
        enriched_words.append(replacement)
    else:
        enriched_words.append(word)

enriched_prompt = " ".join(enriched_words)
print("Enriched prompt:", enriched_prompt)

original_response = generator(original_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
enriched_response = generator(enriched_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

print("\nOriginal response:\n", original_response)
print("\nEnriched response:\n", enriched_response)

print("\nOriginal length:", len(original_response), "| sentences:", original_response.count("."))
print("Enriched length:", len(enriched_response), "| sentences:", enriched_response.count("."))
