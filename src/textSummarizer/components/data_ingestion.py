import os
import urllib.request as request
import zipfile
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size
from pathlib import Path
from textSummarizer.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_file(self):
        # ✅ create parent directory first
        Path(self.config.local_data_file).parent.mkdir(parents=True, exist_ok=True)

        # ✅ keep this INSIDE the method
        if not Path(self.config.local_data_file).exists():
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
    