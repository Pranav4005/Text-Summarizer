from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
model_path = "artifacts/model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Example dialogue
text = """
Amanda: Hi John, are we still meeting today?
John: Yes, at 5 PM.
Amanda: Great, see you then!
"""

# T5 requires prefix
input_text = "summarize: " + text

inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

outputs = model.generate(
    inputs["input_ids"],
    max_length=128,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== INPUT ===")
print(text)

print("\n=== SUMMARY ===")
print(summary)