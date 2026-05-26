from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="artifacts/model",
    repo_id="Pranav4005/text-summarizer-t5",
    repo_type="model"
)

print("Model uploaded successfully")
