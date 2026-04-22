# Running CAID in WSL (Windows Subsystem for Linux)

WSL allows you to run a Linux environment directly on Windows, which is often preferred for deep learning due to better package compatibility and performance.

## 1. Accessing Your Files
In WSL, your Windows drive is mounted under `/mnt/`. Navigate to the project folder:
```bash
cd /mnt/c/Users/Granbell/CAID-Coronary-AI-Ischemia-Detector
```

## 2. Setting Up the Environment
It is highly recommended to use a **native Linux virtual environment** instead of sharing the Windows `.venv`.

```bash
# Update and install python venv
sudo apt update
sudo apt install python3-venv python3-pip

# Create a Linux venv
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate
```

## 3. GPU Support (NVIDIA)
To use your GPU in WSL, you must have the **NVIDIA Windows Display Driver** installed on your host Windows machine. WSL will automatically share the GPU.

Install PyTorch for Linux with CUDA 12.4:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 5. Running the Application

### Training
```bash
python3 scripts/train.py
```

### Web UI
```bash
python3 app.py
```
> [!TIP]
> Even though Flask is running inside WSL, you can still access it from your Windows browser at `http://localhost:5000`.

## Troubleshooting
- **Permission Errors**: If you encounter issues writing to the `/mnt/c/` drive, ensure your WSL user has the necessary permissions or consider moving the project files into the WSL filesystem (e.g., `~/projects/`) for better performance.
- **CUDA not found**: Run `nvidia-smi` in WSL. If it doesn't work, you likely need to update your Windows NVIDIA drivers or install the [WSL2 GPU Support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
