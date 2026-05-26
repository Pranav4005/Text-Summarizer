from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from textSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load once (NOT inside predict)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        self.model.to(self.device)

    def predict(self, text: str) -> str:
        # ✅ Add T5 prefix
        input_text = "summarize: " + text

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=8,
            length_penalty=0.8,
            early_stopping=True
        )

        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return output