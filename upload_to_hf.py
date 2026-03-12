#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Download from ModelScope, convert, and upload to Hugging Face")
    parser.add_argument("--model", choices=["fireredasr2", "sensevoice", "both"], required=True,
                        help="Which model to process")
    parser.add_argument("--token", required=True, 
                        help="HuggingFace token with write access to mlx-community")
    args = parser.parse_args()

    # Install required packages if missing
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("[*] Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi, login

    try:
        import modelscope
    except ImportError:
        print("[*] Installing modelscope...")
        subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], check=True)
        
    try:
        import kaldiio
        import sentencepiece
    except ImportError:
        print("[*] Installing kaldiio and sentencepiece...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaldiio", "sentencepiece"], check=True)

    from modelscope.hub.snapshot_download import snapshot_download

    models_to_process = ["fireredasr2", "sensevoice"] if args.model == "both" else [args.model]

    print(f"[*] Logging into HuggingFace...")
    login(token=args.token)
    api = HfApi()

    for model in models_to_process:
        print(f"\n{'='*50}\nProcessing {model}...\n{'='*50}")
        if model == "fireredasr2":
            ms_repo = "FireRedTeam/FireRedASR2-AED"
            repo_id = "mlx-community/FireRedASR2-AED-mlx"
            convert_module = "mlx_audio.stt.models.fireredasr2.convert"
        else:
            ms_repo = "iic/SenseVoiceSmall"
            repo_id = "mlx-community/SenseVoiceSmall-mlx"
            convert_module = "mlx_audio.stt.models.sensevoice.convert"

        print(f"[*] Ensuring HF repo '{repo_id}' exists...")
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

        # Use a temporary directory to avoid leaving massive files behind
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            print(f"[*] Downloading {ms_repo} from ModelScope...")
            snapshot_download(ms_repo, local_dir=input_dir)

            print(f"[*] Converting weights...")
            # We need to set PYTHONPATH so it can find mlx_audio
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
            
            subprocess.run([
                sys.executable, "-m", convert_module, input_dir, output_dir
            ], env=env, check=True)

            print(f"[*] Uploading converted weights to HF repo {repo_id}...")
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add converted MLX weights for {model}"
            )
            print(f"[*] Success! {model} uploaded to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
