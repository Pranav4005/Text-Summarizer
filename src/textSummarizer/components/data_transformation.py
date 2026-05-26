import os
from transformers import T5Tokenizer
from textSummarizer.logging import logger
from datasets import load_from_disk
from textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):

        # ✅ Add T5 prefix
        inputs = ["summarize: " + dialogue for dialogue in example_batch["dialogue"]]

        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        # Tokenize targets
        labels = self.tokenizer(
            example_batch["summary"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        # ✅ Replace padding token id's in labels with -100
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(token if token != self.tokenizer.pad_token_id else -100) for token in label]
            for label in labels_ids
        ]

        model_inputs["labels"] = labels_ids

        return model_inputs

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)

        logger.info("Starting dataset transformation for T5...")

        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features,
            batched=True
        )

        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )

        logger.info("Dataset transformation completed and saved.")