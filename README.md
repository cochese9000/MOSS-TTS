# MOSS-TTS Windows GGUF Optimized

> **Important**: This is a modified version of the original [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) project. This repository includes massive GGUF performance improvements (using full GPU offloading and PyTorch LM heads) and minor UI/QoL improvements in the `tts3.py` app (like segregation of generated history audio vs reference audio, and aggressive VRAM garbage collection).

This guide walks you through the exact steps to install and compile the MOSS-TTS GGUF backend on a **Windows PC** to achieve maximum performance and ~10x generation speedups compared to the default setup.

## 1. Prerequisites
- **NVIDIA GPU** with CUDA installed (e.g., CUDA 12.6).
- **Visual Studio** (C++ Desktop Development workload) & **Visual Studio Developer Terminal**.
- **Ninja** (A fast build system that bypasses Windows MSBuild bugs).
- **Conda**

## 2. Environment Setup
Open your **Visual Studio Developer PowerShell** (or Command Prompt) and run:

```powershell
conda create -n moss python=3.12 -y
conda activate moss
```

## 3. Install Python Dependencies
Install PyTorch and the required inference libraries. We explicitly install `onnxruntime-gpu` and `flash_attn` to ensure the audio tokenizer and PyTorch LM heads utilize your GPU efficiently.

```powershell
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
pip install -e ".[llama-cpp-onnx,llama-cpp-torch]"
pip install onnxruntime-gpu flash_attn --no-build-isolation
```

## 4. Download GGUF & ONNX Weights
Download the quantized GGUF models and the ONNX audio tokenizer.

```powershell
huggingface-cli download OpenMOSS-Team/MOSS-TTS-GGUF --local-dir weights/MOSS-TTS-GGUF
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX --local-dir weights/MOSS-Audio-Tokenizer-ONNX
```

## 5. Compile `llama.cpp` with CUDA (Crucial for Speed)
To unlock full GPU performance, you **must** compile `llama.cpp` with the `-DGGML_CUDA=ON` flag using the **Ninja** generator. Using the default Microsoft MSBuild often silently drops the CUDA flag resulting in the program executing exclusively on your CPU.

In your **Visual Studio Developer Terminal**:
```powershell
# Go to your llama.cpp directory (assuming it's checked out alongside this repo)
# Adjust the path below if yours is located elsewhere
cd c:\tts\llama.cpp  

# 1. Clear any broken build files
rm -r build

# 2. Configure with Ninja and CUDA support
cmake -B build -G Ninja -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON

# 3. Build (Ninja automatically uses all available CPU cores)
cmake --build build --config Release
```

## 6. Build the Python C-Bridge
Return to the `moss` project directory and compile the C-bridge so Python can talk to your newly compiled `llama.cpp` `.dll`.

```powershell
# Navigate back to your MOSS-TTS bridge directory
cd c:\tts\moss\moss_tts_delay\llama_cpp

# Run the Windows build script
.\build_bridge.cmd
```
*Note: Make sure the `LLAMA_CPP_DIR` variable inside `build_bridge.cmd` is pointing to your `llama.cpp` directory before running.*

## 7. Run the WebUI
Once compilation finishes successfully, you can launch the enhanced Gradio UI!

```powershell
cd c:\tts\moss
python clis/tts3.py
```

### Improvements Included in this Repo:
- **Automatic GGUF Pre-loading**: `tts3.py` skips loading the 24GB FP16 model into VRAM during startup, immediately saving 20+ GB of memory.
- **Aggressive VRAM Purging**: Seamlessly switches models with `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()` separating memory contexts to prevent hidden caching leaks from `lru_cache`.
- **Reference Audio Segregation**: Reference `.wav` files are loaded exclusively from `./reference_audio`, separating them from the app generating outputs natively to `./output/iterations`.
- **Full GPU Offloading**: Re-configured the LlamaCppPipeline to correctly map `100%` of transformer layers (`n_gpu_layers=-1`) and use PyTorch accelerated matrices (`heads_backend="torch"`) instead of CPU NumPy math.
- **Console Generation Timer**: See precise logging metrics on how long each sequence took to synthesize.
