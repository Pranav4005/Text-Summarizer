import os
from textSummarizer.entity import DataValidationConfig
from textSummarizer.logging import logger
from pathlib import Path

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files(self) -> bool:
        try:
            status_path = Path(self.config.STATUS_FILE)

            # ✅ create directory
            status_path.parent.mkdir(parents=True, exist_ok=True)

            validation_status = True

            # (your validation logic here)

            with open(status_path, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e