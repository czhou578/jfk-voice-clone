# Installation and Setup Instructions

Follow these steps to set up the environment, install required dependencies, and configure Git LFS for downloading large models.

## 1. Install System Dependencies

You need to install system-level dependencies for audio processing packages (like `av` and `torchaudio`) to build successfully.
Run the following commands in your terminal:

```bash
apt-get update && apt-get install -y pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

## 2. Install Python Dependencies

Once the system dependencies are installed, install the Python packages specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 3. Set Up Git LFS

The models used in the pipeline may require Git Large File Storage (LFS) to download correctly. 

Install Git LFS using the package manager:

```bash
sudo apt-get install -y git-lfs
```

Initialize Git LFS in your system (you only need to run this once per user account):

```bash
git lfs install
```

## 4. Pull from Git LFS

If you have cloned a repository containing large files (like Hugging Face model repositories) and the large files were not downloaded as actual files (often showing up as tiny text pointers instead), navigate to the repository directory and pull the actual large files:

```bash
# Navigate to the cloned repository
# cd /path/to/cloned/repository

# Pull the large files
git lfs pull
```

This will download the actual model weights, replacing the LFS pointer files.
