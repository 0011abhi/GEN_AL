# Generative AI Lab — BAIL657C

A collection of 10 lab programs covering word embeddings, transformers, LangChain, and Retrieval-based chatbots, with sample outputs and a viva question bank.

## 📑 Table of Contents

| # | Program | Topic |
|---|---------|-------|
| 1 | [Pre-trained Word Vectors](#program-1--explore-pre-trained-word-vectors) | GloVe vector arithmetic |
| 2 | [Dimensionality Reduction](#program-2--visualize-word-embeddings) | PCA visualization of embeddings |
| 3 | [Custom Word2Vec](#program-3--train-a-custom-word2vec-model) | Domain-specific embeddings |
| 4 | [Prompt Enrichment](#program-4--enrich-genai-prompts-with-embeddings) | Embeddings + GPT-2 |
| 5 | [Creative Text Generation](#program-5--generate-creative-text-with-embeddings) | Story generation from seed words |
| 6 | [Sentiment Analysis](#program-6--sentiment-analysis-with-hugging-face) | Hugging Face pipeline |
| 7 | [Text Summarization](#program-7--summarize-text-with-bart) | BART summarization |
| 8 | [LangChain + Cohere](#program-8--langchain--cohere-document-qa) | Prompt templates & document analysis |
| 9 | [Structured Extraction](#program-9--structured-wikipedia-extraction-with-pydantic) | Pydantic schema + Wikipedia |
| 10 | [IPC Chatbot](#program-10--indian-penal-code-chatbot) | PDF search chatbot |

[Viva Questions](#-viva-questions) are listed at the end of this document.

---

## Program 1 — Explore Pre-trained Word Vectors

**Objective:** Explore pre-trained word vectors. Perform vector arithmetic (e.g. `king - man + woman`) and analyze word relationships.

```python
from gensim.downloader import load

# Load the pre-trained GloVe model (50 dimensions)
print("Loading pre-trained GloVe model (50 dimensions)...")
model = load("glove-wiki-gigaword-50")

# Function to perform vector arithmetic and analyze relationships
def ewr():
    result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    print("\nking - man + woman = ?", result[0][0])
    print("similarity:", result[0][1])

    result = model.most_similar(positive=['paris', 'italy'], negative=['france'], topn=1)
    print("\nparis - france + italy = ?", result[0][0])
    print("similarity:", result[0][1])

    # Example 3: Find analogies for programming
    result = model.most_similar(positive=['programming'], topn=5)
    print("\nTop 5 words similar to 'programming':")
    for word, similarity in result:
        print(word, similarity)

ewr()
```

**Output:**
```
Loading pre-trained Glove model (50 dimensions)
king - man + woman = ? queen
similarity: 0.8523604273796082

paris - france + italy = ? rome
similarity: 0.8465589284896851

Top 5 words similar to 'programming':
network 0.7707955241203308
interactive 0.7613597512245178
format 0.7584694623947144
channels 0.753067672252655
networks 0.752894937992096
```

---

## Program 2 — Visualize Word Embeddings

**Objective:** Use PCA to reduce embedding dimensions to 2D, visualize a domain-specific word set (sports), and generate semantically similar words.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.downloader import load

# Dimensionality reduction using PCA
def rd(ems):
    pca = PCA(n_components=2)
    r = pca.fit_transform(ems)
    return r

# Visualize word embeddings
def visualize(words, ems):
    plt.figure(figsize=(10, 6))
    for i, word in enumerate(words):
        x, y = ems[i]
        plt.scatter(x, y, marker='o', color='blue')
        plt.text(x + 0.02, y + 0.02, word, fontsize=12)
    plt.show()

# Generate semantically similar words
def gsm(word):
    sw = model.most_similar(word, topn=5)
    for word, s in sw:
        print(word, s)

# Load pre-trained GloVe model from Gensim API
print("Loading pre-trained GloVe model (50 dimensions)...")
model = load("glove-wiki-gigaword-50")

words = ['football', 'basketball', 'soccer', 'tennis', 'cricket']
ems = [model[word] for word in words]
e = rd(ems)
visualize(words, e)
gsm("programming")
```

**Output:**
```
Loading pre-trained GloVe model (50 dimensions)...

network 0.7707955241203308
interactive 0.7613597512245178
format 0.7584694623947144
channels 0.753067672252655
networks 0.752894937992096
```
*(Scatter plot shows football, soccer, tennis, basketball, cricket clustered together in 2D PCA space.)*

---

## Program 3 — Train a Custom Word2Vec Model

**Objective:** Train Word2Vec embeddings on a small medical-domain corpus and analyze how the model captures domain-specific semantics.

```python
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
```

**Output:** A 2D scatter plot of all corpus words (e.g. *diagnosis*, *symptoms*, *vaccine*, *prescription*) showing how medical terms cluster based on co-occurrence in the training sentences.

---

## Program 4 — Enrich GenAI Prompts with Embeddings

**Objective:** Use GloVe embeddings to replace a keyword in a prompt with a semantically similar word, then compare GPT-2 outputs for the original vs. enriched prompt.

```bash
pip install gensim
pip install nltk
pip install transformers
```

```python
from gensim.downloader import load
from transformers import pipeline
import nltk
import string
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

print("loading pre trained word vectors")
word_vectors = load("glove-wiki-gigaword-100")

def replace_keyword_in_prompt(prompt, keyword, word_vectors, topn=1):
    words = word_tokenize(prompt)
    enriched_words = []
    for word in words:
        cleaned_word = word.lower().strip(string.punctuation)
        if cleaned_word == keyword.lower():
            try:
                similar_words = word_vectors.most_similar(cleaned_word, topn=topn)
                if similar_words:
                    replacement_word = similar_words[0][0]
                    print(f"Replacing {word} -> {replacement_word}")
                    enriched_words.append(replacement_word)
                    continue
            except KeyError:
                print(f"{keyword} not found in vocabulary, using original word")
        enriched_words.append(word)
    enriched_prompt = " ".join(enriched_words)
    print(f"\nEnriched Prompt: {enriched_prompt}")
    return enriched_prompt

print("\nLoading GPT-2 model")
generator = pipeline("text-generation", model="gpt2")

def generate_response(prompt, max_length=100):
    try:
        response = generator(prompt, max_length=max_length, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        print(f"error generating response {e}")
        return None

original_prompt = "write an essay on natural disaster"
print(f"Original prompt: {original_prompt}")

k_term = "disaster"
enriched_prompt = replace_keyword_in_prompt(original_prompt, k_term, word_vectors)

print("\ngenerating response for original prompt")
original_response = generate_response(original_prompt)
print(original_response)

print("\ngenerating response for enriched prompt")
enriched_response = generate_response(enriched_prompt)
print(enriched_response)

print("\ncomparison of responses")
print("original prompt response length", len(original_response))
print("enriched prompt response length", len(enriched_response))
print("original prompt response detail", original_response.count("."))
print("enriched prompt response detail", enriched_response.count("."))
```

> ⚠️ Note: the original transcript referenced "GPT-4" in a comment but the code actually loads `gpt2` via `pipeline("text-generation", model="gpt2")` — corrected above for clarity.

---

## Program 5 — Generate Creative Text with Embeddings

**Objective:** Take a seed word, retrieve semantically similar words, and weave them into a short generated story/paragraph.

```python
import random
import gensim.downloader as api

# Load a pre-trained word embedding model
model = api.load("glove-wiki-gigaword-50")  # 50D GloVe embeddings

def get_similar_words(seed_word, top_n=5):
    try:
        similar_words = [word for word, _ in model.most_similar(seed_word, topn=top_n)]
        return similar_words
    except KeyError:
        return []

def create_paragraph(seed_word):
    similar_words = get_similar_words(seed_word)
    if not similar_words:
        return f"Could not find similar words for '{seed_word}'. Try another word!"

    paragraph = (
        f"Once upon a time, a {seed_word} embarked on a journey. Along the way, it encountered "
        f"a {random.choice(similar_words)}, which led it to a hidden {random.choice(similar_words)}. "
        f"Despite the challenges, it found {random.choice(similar_words)} and embraced the "
        f"adventure with {random.choice(similar_words)}. In the end, the journey was a tale of "
        f"{random.choice(similar_words)} and discovery."
    )
    return paragraph

# Example usage
seed_word = input("Enter a seed word: ").strip().lower()
print("\nGenerated Story:\n")
print(create_paragraph(seed_word))
```

**Sample Run:**
```
Enter a seed word: adventure

Generated Story:

Once upon a time, a adventure embarked on a journey. Along the way, it encountered
a adventures, which led it to a hidden romance. Despite the challenges, it found
mystery and embraced the adventure with adventures. In the end, the journey was a
tale of adventures and discovery.
```

---

## Program 6 — Sentiment Analysis with Hugging Face

**Objective:** Load a pre-trained Hugging Face sentiment-analysis pipeline and classify user-entered sentences interactively.

```python
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label} (Confidence: {score:.2f})"

while True:
    user_input = input("Enter a sentence for sentiment analysis (or 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break
    print(analyze_sentiment(user_input))
```

---

## Program 7 — Summarize Text with BART

**Objective:** Use Facebook's `bart-large-cnn` model to generate an abstractive summary of a long passage (air pollution article).

```bash
!pip install transformers torch
```

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """Air pollution is the presence of substances in the air that are
harmful to humans, other living beings or the environment. Pollutants can be
gases, like ozone or nitrogen oxides, or small particles like soot and dust.
Both outdoor and indoor air can be polluted. Outdoor air pollution comes
from burning fossil fuels for electricity and transport, wildfires, some
industrial processes, waste management, demolition and agriculture. Indoor
air pollution is often from burning firewood or agricultural waste for
cooking and heating.

Other sources of air pollution include dust storms and volcanic eruptions.
Many sources of local air pollution, especially burning fossil fuels, also
release greenhouse gases that cause global warming. However, air pollution
may limit warming locally. Air pollution kills 7 or 8 million people each
year. It is a significant risk factor for a number of diseases, including
stroke, heart disease, chronic obstructive pulmonary disease (COPD), asthma,
coronavirus and lung cancer. Particulate matter is the most deadly, both for
indoor and outdoor pollution. Ozone affects crops, and forests are damaged
by the pollution that causes acid rain. Overall, the World Bank has
estimated that welfare losses (premature deaths) and productivity losses
(lost labor) caused by air pollution cost the world economy over $8 trillion
per year. Various technologies and strategies reduce air pollution. Key
approaches include clean cookers, fire protection, improved waste
management, dust control, industrial scrubbers, electric vehicles and
renewable energy. National air quality laws have often been effective,
notably the 1956 Clean Air Act in Britain and the 1963 US Clean Air Act.
International efforts have had mixed results: the Montreal Protocol almost
eliminated harmful ozone-depleting chemicals, while international action on
climate change has been less successful."""

# Tokenize the input text
inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

# Generate the summary
summary_ids = model.generate(
    inputs,
    max_length=50,
    min_length=25,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
```

> 🛠 Fixed typos from the original transcript: `truncation-True` → `truncation=True`, `Inputs=` → `inputs =`.

---

## Program 8 — LangChain + Cohere Document QA

**Objective:** Load a text document from Google Drive, build a `PromptTemplate`, and use `ChatCohere` to summarize, extract key takeaways, and assess sentiment.

```bash
!pip install langchain cohere langchain-community google-colab
!pip install langchain-cohere
```

```python
import cohere
import getpass
from langchain import PromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from google.colab import auth
from google.colab import drive

auth.authenticate_user()
drive.mount('/content/drive')

file_path = "/content/drive/MyDrive/GenerativeAI.txt"
try:
    with open(file_path, "r", encoding="utf-8") as file:
        text_content = file.read()
    print("File loaded successfully!")
except Exception as e:
    print("Error loading file:", str(e))

COHERE_API_KEY = getpass.getpass("Enter your Cohere API Key: ")

cohere_llm = ChatCohere(cohere_api_key=COHERE_API_KEY, model="command-a-03-2025")

template = """
You are an AI assistant helping to summarize and analyze a text document.
Here is the document content:
{text}

* Summary:
- Provide a concise summary of the document.

* Key Takeaways:
- List 3 important points from the text.

* Sentiment Analysis:
- Determine if the sentiment of the document is Positive, Negative, or Neutral.
"""

prompt_template = PromptTemplate(input_variables=["text"], template=template)
formatted_prompt = prompt_template.format(text=text_content)
print("formatted_prompt := ", formatted_prompt)

response = cohere_llm.invoke([HumanMessage(content=formatted_prompt)]).content
print("\n**Formatted Output**")
print(response)
```

> 🔒 **Security note:** never commit a real API key to GitHub. Use `getpass` (as shown) or an environment variable / `.env` file excluded via `.gitignore`.

**Sample Output:**
```
**Formatted Output**

### Summary:
The document highlights key trends in AI for 2026, including the shift
from chatbots to autonomous "agentic AI" capable of independent task
execution, the rise of multimodal AI as the industry standard, and the
move toward post-training optimization and domain-specific models.

### Key Takeaways:
1. Agentic AI is Transforming Automation
2. Multimodal AI is the New Standard
3. Post-Training and Specialization Dominate

### Sentiment Analysis:
Positive — the document presents a forward-looking, optimistic view of
AI advancements with no negative tone or criticism.
```

---

## Program 9 — Structured Wikipedia Extraction with Pydantic

**Objective:** Take an institution name as input, define an output schema with Pydantic, and extract founder, founding year, branches, employee count, and a summary from Wikipedia.

```bash
!pip install wikipedia-api pydantic
```

```python
import re
import wikipediaapi
from pydantic import BaseModel, Field
from typing import List, Optional

class InstitutionDetails(BaseModel):
    name: str
    founder: Optional[str] = None
    founded: Optional[str] = None
    branches: List[str] = Field(default_factory=list)
    number_of_employees: Optional[int] = None
    summary: Optional[str] = None

def fetch_institution_details(institution_name: str) -> InstitutionDetails:
    user_agent = "InstitutionScraper/1.0 (contact: myemail@example.com)"
    wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language='en')
    page = wiki.page(institution_name)

    if not page.exists():
        raise ValueError(f"The page for '{institution_name}' does not exist.")

    full_text = page.text

    def extract_pattern(pattern, text, is_list=False):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if is_list:
                return [item.strip() for item in content.split(',')]
            return content
        return [] if is_list else None

    founder_pattern = r"(?:founded|established|started)\s+by\s+([^.\n,]+)"
    founder_match = re.search(founder_pattern, full_text, re.IGNORECASE)
    founder = founder_match.group(1).strip() if founder_match else "Unknown"

    year_pattern = r"(?:founded|established|started|incorporated)(?:\s+in)?(?:\s+the\s+year)?\s+(\d{4})"
    year_match = re.search(year_pattern, full_text, re.IGNORECASE)
    founded = year_match.group(1) if year_match else "Unknown"

    branches = extract_pattern(r"Branches\s*[:\-]?\s*(.*)", full_text, is_list=True)

    raw_employees = extract_pattern(r"Number of employees\s*[:\-]?\s*([\d,]+)", full_text)
    emp_count = None
    if raw_employees:
        try:
            emp_count = int(raw_employees.replace(',', ''))
        except ValueError:
            emp_count = None

    return InstitutionDetails(
        name=page.title,
        founder=founder,
        founded=founded,
        branches=branches,
        number_of_employees=emp_count,
        summary=page.summary[:500] + "..."
    )

try:
    val = input("Enter the Institution name: ")
    data = fetch_institution_details(val)
    print(data.model_dump_json(indent=2))
except Exception as e:
    print(f"Error: {e}")
```

> 🛠 Fixed a syntax error in the original transcript: the `extract_pattern` helper's `return [] if is_list else` had no value after `else` — corrected to `return [] if is_list else None`.

---

## Program 10 — Indian Penal Code Chatbot

**Objective:** Download the IPC PDF, extract its text with PyMuPDF, and build a simple keyword-search chatbot for IPC sections.

```bash
!pip install pymupdf
```

```python
import fitz

def extract(file):
    text = ""
    with fitz.open(file) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def search(query, ipc):
    query = query.lower()
    lines = ipc.split("\n")
    results = [line for line in lines if query in line.lower()]
    if results:
        return results[:15]
    else:
        return ["No relevant section found."]

def chatbot():
    print("Loading IPC document...")
    ipc = extract(r"/Users/ananthas/Desktop/ipc.pdf")
    while True:
        query = input("Ask a question about the IPC: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        results = search(query, ipc)
        print("\n".join(results))
        print("-" * 50)

chatbot()
```

**Sample Run:**
```
Loading IPC document...
Ask a question about the IPC: murder
300. Murder.
When culpable homicide is not murder.
302. Punishment for murder.
303. Punishment for murder by life-convict.
304. Punishment for culpable homicide not amounting to murder.
307. Attempt to murder.
364. Kidnapping or abducting in order to murder.
396. Dacoity with murder.
--------------------------------------------------
Ask a question about the IPC: exit
Goodbye!
```

---

## ❓ Viva Questions

<details>
<summary><strong>Word Embeddings & GloVe (Q1–21)</strong></summary>

1. **What is GloVe?** A pre-trained word embedding model (Global Vectors for Word Representation) that converts words into numerical vectors.
2. **What is word embedding?** A technique to represent words as numerical vectors so machines can understand their meanings.
3. **Why use pre-trained models?** To save time and reuse knowledge already learned from large datasets.
4. **What does `most_similar()` do?** Finds words whose vectors are closest to a given word vector.
5. **What is vector arithmetic in NLP?** Mathematical operations on word vectors to discover relationships between words.
6. **Dimension of `glove-wiki-gigaword-50`?** 50 dimensions.
7. **What is cosine similarity?** A measure of similarity between two vectors based on the angle between them.
8. **Why is the similarity value between 0 and 1?** Because cosine similarity measures closeness/angle between vectors.
9. **What are `positive` and `negative` in `most_similar()`?** `positive` = words to add; `negative` = words to subtract.
10. **What dataset trains `glove-wiki-gigaword-50`?** Wikipedia and the Gigaword corpus.
11. **Why are word embeddings useful?** They capture semantic meaning as vectors, enabling similarity comparisons.
14. **What happens if a word isn't in vocabulary?** The model raises a `KeyError` / cannot compute similarity.
15. **Why does vector arithmetic work?** Because embeddings capture semantic relationships geometrically.
16. **Why reduce dimensions with PCA?** To visualize high-dimensional data in 2D while preserving variance.
17. **Why choose 2D for visualization?** It's easy to plot and interpret clusters of similar words.
18. **Cosine similarity (again)?** Measures the angle between vectors — smaller angle = more similar.
19. **PCA vs t-SNE?** PCA is linear and preserves global structure; t-SNE is non-linear and better preserves local neighborhoods/clusters.
20. **How are similar words generated in code?** Via `most_similar()`, which computes cosine similarity against all vocabulary words.
21. **Why do football and soccer cluster together?** They're near-synonyms used in similar contexts, so embeddings place them close together.

</details>

<details>
<summary><strong>Word2Vec & Domain Embeddings (Q22–28)</strong></summary>

22. **What is Word2Vec?** A neural-network-based model that learns embeddings by predicting words from context (or vice versa).
23. **Why use a domain-specific corpus?** To capture domain-specific semantics (e.g., medical terms cluster together).
24. **CBOW vs Skip-gram?** CBOW predicts a word from its context; Skip-gram predicts context from a word. Skip-gram tends to perform better on small datasets.
25. **Why `vector_size=50`?** A standard, efficient setting that balances training speed and embedding quality for small datasets.
26. **What does `most_similar` do (again)?** Finds semantically closest words via cosine similarity.
27. **How do embeddings capture meaning?** Words used in similar contexts get similar vectors.
28. **What if the corpus is very small?** Embeddings become noisy due to insufficient context.

</details>

<details>
<summary><strong>Prompt Enrichment & Creative Generation (Q29–41)</strong></summary>

29. **Why enrich prompts with embeddings?** Semantically similar words make prompts richer, producing more detailed responses.
30. **How are similar words retrieved?** Via `most_similar()` on GloVe vectors.
31. **Why does GPT-2 respond better to enriched prompts?** Extra contextual words help the model understand scope and nuance.
32. **What is `no_repeat_ngram_size`?** Prevents repeating the same n-gram, improving fluency.
33. **Risks of over-enrichment?** Too many unrelated words can confuse the model and produce off-topic text.
34. **Can this apply to other models?** Yes — any generative model (GPT-3, LLaMA, BLOOM, etc.) benefits from enriched prompts.
35. **Embedding-based enrichment vs keyword insertion?** Embeddings capture semantic similarity, not just literal synonyms — more context-aware.
36. **Why use embeddings for creative writing?** They supply contextually related words for coherent, meaningful content.
37. **Why import `nltk`?** For pre-built NLP tools (tokenization, etc.) without writing the logic from scratch.
38. **Why import `transformers`?** To easily access and deploy state-of-the-art pre-trained models (BERT, GPT, T5...).
39. **What does `pipeline()` do?** Provides a high-level API that wraps preprocessing, inference, and postprocessing into one call.
40. **What is Gensim?** "Generate Similar" — an open-source NLP library for topic modeling, building word/document vectors, and semantic comparison.
41. **How does `topn` affect results?** More words → richer but less focused output; fewer words → concise but less varied.

</details>

<details>
<summary><strong>Sentiment Analysis (Q42–52)</strong></summary>

42. **Role of `pipeline()` in sentiment analysis?** Loads a pre-trained sentiment model with a simple high-level API.
43. **Default model for `pipeline("sentiment-analysis")`?** Typically a DistilBERT model fine-tuned on SST-2.
44. **What does `analyze_sentiment()` do?** Passes text to the analyzer and returns a formatted label + confidence string.
45. **What does the `result` variable contain?** A list of dicts with `label` and `score`.
46. **Why use `result[0]`?** The pipeline always returns a list, even for one input.
47. **What does `score` represent?** The model's confidence (0 to 1) in its prediction.
48. **Why `.strip()` the user input?** Removes leading/trailing whitespace for clean processing.
49. **Can it handle multiple sentences at once?** Not by default — but the pipeline accepts a list if modified.
50. **How do transformers understand sentiment?** Via self-attention, analyzing the whole sentence at once to capture context (e.g., negation).
51. **Limitations of an un-fine-tuned pipeline?** Poor performance on domain-specific text, slang, or sarcasm.
52. **How to get more sentiment classes (e.g., neutral)?** Use/fine-tune a multi-class sentiment model and update the pipeline accordingly.

</details>

<details>
<summary><strong>Summarization with BART (Q53–61)</strong></summary>

53. **Why is BART suitable for summarization?** Its encoder-decoder design understands context (like BERT) and generates coherent text (like GPT).
54. **Extractive vs abstractive summarization?** Extractive selects original sentences; abstractive (BART) generates new sentences preserving meaning.
55. **Why `AutoTokenizer` / `AutoModelForSeq2SeqLM`?** Generic classes that auto-select the correct tokenizer/architecture for a given model name.
56. **Significance of `max_length=512`?** BART's maximum input token limit; longer text gets truncated.
57. **Purpose of `num_beams=4`?** Beam search tracks 4 candidate sequences, improving summary quality at extra computational cost.
58. **What does `length_penalty` do?** Discourages overly short summaries, encouraging more informative output.
59. **Why `no_repeat_ngram_size=3`?** Prevents repeating 3-word sequences, improving readability.
60. **Role of `repetition_penalty`?** Discourages repeated words/phrases for more natural output.
61. **Effect of `early_stopping=True`?** Stops beam search once all candidates finish, saving computation.

</details>

<details>
<summary><strong>LangChain, Cohere & Pydantic (Q62–79)</strong></summary>

62. **Role of Cohere in Program 8?** Provides the LLM (`command-a-03-2025`) for summarization, key takeaways, and sentiment analysis.
63. **Why use LangChain?** Structures prompts, manages LLM calls, and simplifies chaining tasks.
64. **Purpose of `PromptTemplate`?** Standardizes and dynamically inserts text into a consistent prompt structure.
65. **`ChatCohere` vs a standard API call?** An abstraction that handles message formatting/response parsing for conversational use.
66. **Why `getpass.getpass()` for the API key?** Hides the key from being displayed/logged.
67. **Purpose of Colab auth + Drive mount?** Authenticates Google services and enables reading files from Drive.
68. **Structure of the prompt used?** Document content + instructions (summary, key takeaways, sentiment) for organized output.
69. **Why `HumanMessage` in `invoke()`?** Represents user input in chat-format, compatible with chat-based LLMs.
70. **Role of Pydantic?** Defines a structured schema (`InstitutionDetails`) with type validation and clean JSON output.
71. **Why use `BaseModel` for the schema?** Enforces a fixed, validated structure instead of unstructured dicts.
72. **Purpose of `wikipedia-api`?** Programmatically fetches Wikipedia page content (full text + summary).
73. **Why is a `User-Agent` required?** Wikipedia requires it to identify the client and prevent misuse.
74. **How is the founder extracted?** Via regex matching phrases like "founded by ...".
75. **Limitations of regex extraction?** Fails on format changes/complex text; lacks true contextual understanding.
76. **Why are some fields `Optional`?** Because data like founder or employee count isn't always available.
77. **How are missing employee numbers handled?** Set to `None` via exception handling if conversion fails.
78. **Why `Field(default_factory=list)` for `branches`?** Avoids the mutable-default-argument pitfall; each instance gets its own list.
79. **Why `page.summary[:500] + "..."`?** Keeps the summary concise while indicating truncation.

</details>

<details>
<summary><strong>IPC Chatbot / PyMuPDF (Q80–82)</strong></summary>

80. **Role of PyMuPDF (`fitz`)?** Opens and reads the PDF, extracting text for searching.
81. **How does `extract()` work?** Opens the PDF, iterates through pages, calls `page.get_text()`, and concatenates results.
82. **Limitations of `page.get_text()` for legal documents?** May not preserve formatting, tables, or section hierarchy — complex layouts can yield disordered text.

</details>

---

## 📂 Suggested Repository Structure

```
generative-ai-lab/
├── README.md                  ← this file
├── program1_word_vectors.py
├── program2_pca_visualization.py
├── program3_custom_word2vec.py
├── program4_prompt_enrichment.py
├── program5_creative_generation.py
├── program6_sentiment_analysis.py
├── program7_summarization.py
├── program8_langchain_cohere.py
├── program9_pydantic_wikipedia.py
├── program10_ipc_chatbot.py
├── requirements.txt
└── .gitignore                 ← exclude API keys, .env, __pycache__/
```

**Suggested `requirements.txt`:**
```
gensim
matplotlib
scikit-learn
transformers
torch
nltk
langchain
langchain-community
langchain-cohere
cohere
pydantic
wikipedia-api
pymupdf
```
