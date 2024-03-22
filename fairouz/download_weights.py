from huggingface_hub import snapshot_download
import sys

if __name__ == "__main__":
    if sys.argv[1] == "mistral":
        snapshot_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", local_dir="./Mistral-7B-Instruct-v0.2-AWQ", local_dir_use_symlinks=False)
    if sys.argv[1] == "mxbai":
        snapshot_download(repo_id="mixedbread-ai/mxbai-embed-large-v1", local_dir="./mxbai-embed-large-v1", local_dir_use_symlinks=False)

        