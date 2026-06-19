from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """Air pollution is the presence of substances in the air that are
harmful to humans, other living beings or the environment. Pollutants can be
gases, like ozone or nitrogen oxides, or small particles like soot and dust.
Air pollution kills 7 or 8 million people each year and is a major risk
factor for stroke, heart disease, COPD, asthma and lung cancer."""

inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs, max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
