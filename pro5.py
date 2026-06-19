import random
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")

seed_word = input("Enter a seed word: ").strip().lower()
similar_words = [word for word, _ in model.most_similar(seed_word, topn=5)]

story = (
    f"Once upon a time, a {seed_word} embarked on a journey. Along the way, it encountered "
    f"a {random.choice(similar_words)}, which led it to a hidden {random.choice(similar_words)}. "
    f"Despite the challenges, it found {random.choice(similar_words)} and embraced the "
    f"adventure with {random.choice(similar_words)}. In the end, the journey was a tale of "
    f"{random.choice(similar_words)} and discovery."
)

print(story)
