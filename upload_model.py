import sys
import subprocess
import getpass

# 1. Ensure huggingface_hub is installed
try:
    from huggingface_hub import HfApi
except ImportError:
    print("huggingface_hub library not found. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import HfApi

def main():
    repo_id = "Siddharthv06/BrainTumorAI"
    model_path = "model.keras"
    
    print("\n" + "="*50)
    print("🧠 Hugging Face Model Uploader")
    print("="*50)
    print(f"Target Repo: https://huggingface.co/{repo_id}")
    print(f"File to Upload: {model_path}\n")
    
    # Get write token
    print("To get a Write Token, visit: https://huggingface.co/settings/tokens")
    token = getpass.getpass("Enter your Hugging Face Write Token (input will be hidden): ").strip()
    
    if not token:
        print("❌ Error: Token cannot be empty.")
        return

    api = HfApi()
    
    try:
        print("\nUploading model.keras to Hugging Face (this may take a few minutes)...")
        # Upload file with built-in progress bar
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("\n✅ Success! Your model has been uploaded successfully.")
        print(f"Check it out here: https://huggingface.co/{repo_id}/blob/main/{model_path}")
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")

if __name__ == "__main__":
    main()
