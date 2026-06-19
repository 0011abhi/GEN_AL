from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

corpus = [
    "The patient was diagnosed with diabetes and hypertension.",
    "MRI scans reveal abnormalities in the brain tissue.",
    "The treatment involves antibiotics and regular monitoring.",
    "Symptoms include fever, fatigue, and muscle pain.",
    "The vaccine is effective against several viral infections.",
    "Doctors recommend physical therapy for recovery.",
    "The clinical trial results were published in the journal.",
    "The surgeon performed a minimally invasive procedure.",
    "The prescription includes pain relievers and anti-inflammatory drugs.",
    "The diagnosis confirmed a rare genetic disorder."
]

token_corp = [sentence.lower().split() for sentence in corpus]
model = Word2Vec(sentences=token_corp, vector_size=5, window=2, min_count=1, epochs=1000)

w = input("enter a word:").lower()
if w in model.wv:
    similar = model.wv.most_similar(w, topn=5)
    print(f"word similar to {w}")
    for i, (wo, score) in enumerate(similar, 1):
        print(f"{i}.{wo} similarity:{score}")
else:
    print("word not found in the vocabulary")

words = list(model.wv.index_to_key)
word_vectors = model.wv[words]
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)

plt.figure(figsize=(10, 8))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("word embeddings visualization")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
