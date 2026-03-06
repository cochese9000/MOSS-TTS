import os
import argparse
import functools
import importlib.util
from pathlib import Path
import re
import time
import orjson
import datetime
import shutil
import scipy.io.wavfile as wavfile
import gc

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, login, whoami
import sys
import random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoModel, AutoProcessor

from moss_tts_delay.llama_cpp.pipeline import PipelineConfig, LlamaCppPipeline

# New storage directory for your session iterations
ITERATION_DIR = Path("iterations")
ITERATION_DIR.mkdir(exist_ok=True)

try:
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")
except:
    print("Not logged in")

# Download the model files first
model_id = "OpenMOSS-Team/MOSS-TTS"
print(f"Downloading {model_id}...")
MODEL_PATH = snapshot_download(repo_id=model_id, local_files_only=False)


DEFAULT_ATTN_IMPLEMENTATION = "auto"
DEFAULT_MAX_NEW_TOKENS = 4096
CONTINUATION_NOTICE = (
    "Continuation mode is active. The system will automatically prepend your entire Line History text to provide context."
)

MODE_CLONE = "Clone"
MODE_CONTINUE = "Continuation"
MODE_CONTINUE_CLONE = "Continuation + Clone"
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219
REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio"
EXAMPLE_TEXTS_JSONL_PATH = Path(__file__).resolve().parent.parent / "assets" / "text" / "moss_tts_example_texts.jsonl"


# --- BACKEND MANAGEMENT (FP16 & GGUF) ---

MODEL_TYPE_FP16 = "FP16 (PyTorch)"
MODEL_TYPE_GGUF = "GGUF (llama.cpp)"

_ACTIVE_BACKEND_TYPE = None
_ACTIVE_BACKEND_OBJ = None
_ACTIVE_PROCESSOR_OBJ = None
_ACTIVE_DEVICE = None
_ACTIVE_SAMPLE_RATE = None

_ACTIVE_GGUF_FILE = None

def get_available_gguf_models():
    """Returns a list of available .gguf files in the weights directory."""
    gguf_dir = Path("weights/MOSS-TTS-GGUF")
    if not gguf_dir.exists():
        return []
    return [f.name for f in gguf_dir.glob("*.gguf")]

def load_active_backend(model_type: str, gguf_file: str, args: argparse.Namespace):
    """Dynamically loads and caches the selected backend model, freeing VRAM when switching."""
    global _ACTIVE_BACKEND_TYPE, _ACTIVE_BACKEND_OBJ, _ACTIVE_PROCESSOR_OBJ, _ACTIVE_DEVICE, _ACTIVE_SAMPLE_RATE, _ACTIVE_GGUF_FILE
    
    # If same backend and (not GGUF or same GGUF file), do nothing
    if _ACTIVE_BACKEND_TYPE == model_type:
        if model_type == MODEL_TYPE_FP16 or (model_type == MODEL_TYPE_GGUF and _ACTIVE_GGUF_FILE == gguf_file):
            return _ACTIVE_BACKEND_OBJ, _ACTIVE_PROCESSOR_OBJ, _ACTIVE_DEVICE, _ACTIVE_SAMPLE_RATE

    print(f"\n[INFO] Switching backend to: {model_type}", flush=True)
    
    # 1. Unload existing
    if _ACTIVE_BACKEND_OBJ is not None:
        if hasattr(_ACTIVE_BACKEND_OBJ, "close"):
            _ACTIVE_BACKEND_OBJ.close()
            
        del _ACTIVE_BACKEND_OBJ
        del _ACTIVE_PROCESSOR_OBJ
        
        _ACTIVE_BACKEND_OBJ = None
        _ACTIVE_PROCESSOR_OBJ = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Force a secondary collect pass
            gc.collect()
            torch.cuda.empty_cache()
            
    # 2. Load new
    if model_type == MODEL_TYPE_FP16:
        model, processor, device, sample_rate = load_backend(
            model_path=args.model_path, 
            device_str=args.device, 
            attn_implementation=args.attn_implementation
        )
        _ACTIVE_BACKEND_OBJ = model
        _ACTIVE_PROCESSOR_OBJ = processor
        _ACTIVE_DEVICE = device
        _ACTIVE_SAMPLE_RATE = sample_rate
        _ACTIVE_BACKEND_TYPE = MODEL_TYPE_FP16
        _ACTIVE_GGUF_FILE = None
        
    elif model_type == MODEL_TYPE_GGUF:
        print(f"[INFO] Loading GGUF Backbone ({gguf_file}) via LlamaCppPipeline...", flush=True)
        config = PipelineConfig(
            backbone_gguf=f"weights/MOSS-TTS-GGUF/{gguf_file}",
            embedding_dir="weights/MOSS-TTS-GGUF/embeddings",
            lm_head_dir="weights/MOSS-TTS-GGUF/lm_heads",
            tokenizer_dir="weights/MOSS-TTS-GGUF/tokenizer",
            audio_backend="onnx", # Using ONNX if installed
            audio_encoder_onnx="weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx",
            audio_decoder_onnx="weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx",
            heads_backend="auto",
            n_ctx=4096,
            max_new_tokens=2000,
            use_gpu_audio=True
        )
        
        # Verify paths are present (will throw if missing)
        try:
            config.validate()
        except FileNotFoundError as e:
            # Fallback to pure PyTorch for audio tokenizer (downloads it automatically) if ONNX is missing
            print(f"[WARN] ONNX backend files missing: {e}. Falling back to PyTorch audio tokenizer.", flush=True)
            config.audio_backend = "torch"
            config.audio_model_name_or_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
        
        pipeline = LlamaCppPipeline(config)
        _ACTIVE_BACKEND_OBJ = pipeline
        _ACTIVE_PROCESSOR_OBJ = None # GGUF pipeline handles this internally
        _ACTIVE_DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
        _ACTIVE_SAMPLE_RATE = 24000
        _ACTIVE_BACKEND_TYPE = MODEL_TYPE_GGUF
        _ACTIVE_GGUF_FILE = gguf_file
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return _ACTIVE_BACKEND_OBJ, _ACTIVE_PROCESSOR_OBJ, _ACTIVE_DEVICE, _ACTIVE_SAMPLE_RATE


# --- STATE & STORAGE HELPERS ---

def load_startup_gallery():
    """Loads previously saved *reference* wav files from the iterations directory on startup.
    Only includes files explicitly saved as references or uploads, skipping auto-generated lines."""
    gallery = {}
    if ITERATION_DIR.exists():
        for filepath in sorted(ITERATION_DIR.glob("*.wav"), key=os.path.getmtime):
            # Only include files that start with 'Ref_' or 'UPLOAD_' (our manual saves)
            if filepath.name.startswith("Ref_") or filepath.name.startswith("UPLOAD_"):
                # Use the filename as the display label
                gallery[filepath.stem] = str(filepath)
    return dict(sorted(gallery.items()))

def save_to_gallery(audio_data, name, text_snippet, current_gallery):
    """Saves audio and optionally updates the session gallery if passed explicitly."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_text = re.sub(r'[^\w\s-]', '', text_snippet)[:30].strip().replace(" ", "_")
    filename = f"{name}_{clean_text}_{timestamp}"
    filepath = ITERATION_DIR / f"{filename}.wav"
    
    wavfile.write(filepath, audio_data[0], audio_data[1])
    
    # We purposefully do not add generated lines to the reference gallery dictionary here!
    return current_gallery, str(filepath)

def add_custom_ref(file_path, custom_name, current_gallery):
    """Handles manual file uploads to the gallery."""
    if not file_path or not custom_name:
        return current_gallery, gr.update()
    
    safe_name = re.sub(r'[^\w\s-]', '', custom_name).strip().replace(" ", "_")
    dest_path = ITERATION_DIR / f"Ref_{safe_name}.wav"
    shutil.copy(file_path, dest_path)
    
    current_gallery[f"UPLOAD_{custom_name}"] = str(dest_path)
    sorted_gallery = dict(sorted(current_gallery.items()))
    return sorted_gallery, gr.update(choices=list(sorted_gallery.keys()), value=f"UPLOAD_{custom_name}")

# --- INFERENCE WRAPPERS ---

def generate_audio_logic(
    target_text, history, ref_label, gallery, mode_with_reference, 
    duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens,
    args, model_type, gguf_file, use_random_seed, seed_val
):
    """Core logic for synthesizing a line, handling history prepending for continuation modes."""
    ref_path = gallery.get(ref_label) if ref_label else None
    base_ref = str(ref_label).split("_")[0] if ref_label else "DirectGen"
    
    # Auto-prepend history text if in continuation mode
    if mode_with_reference in [MODE_CONTINUE, MODE_CONTINUE_CLONE]:
        history_context = " ".join([item["text"] for item in history if item["text"].strip()])
        synthesis_text = f"{history_context} {target_text}".strip()
    else:
        synthesis_text = target_text

    audio_np_tuple, status_msg, active_seed = run_inference(
        synthesis_text, ref_path, mode_with_reference, 
        duration_enabled, duration_tokens,
        temp, top_p, top_k, rep_penalty,
        max_new_tokens, args, model_type, gguf_file, use_random_seed, seed_val
    )
    
    new_gallery, saved_path = save_to_gallery(audio_np_tuple, base_ref, target_text, gallery)
    
    # Do not auto-select the newly generated line in the reference dropdown
    # We just keep the existing gallery keys to return to the dropdown (which shouldn't include this auto-generated file)
    return saved_path, status_msg, gallery, active_seed

def generate_new_line(
    new_text, history, ref_label, gallery, mode_with_reference, 
    duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, model_type, gguf_file, use_random_seed, seed_val, args
):
    if not new_text.strip():
        return history, gallery, gr.update(), "Error: Text is empty", seed_val
        
    saved_path, status_msg, new_gallery, active_seed = generate_audio_logic(
        new_text, history, ref_label, gallery, mode_with_reference, 
        duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, args, model_type, gguf_file, use_random_seed, seed_val
    )
    
    # Push to history
    history.append({
        "text": new_text,
        "audio_path": saved_path
    })
    
    # keep the same choices
    return history, gallery, gr.update(choices=list(gallery.keys())), status_msg, active_seed

def redo_specific_line(
    index, updated_text, history, ref_label, gallery, mode_with_reference, 
    duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, model_type, gguf_file, use_random_seed, seed_val, args
):
    """Regenerates a specific line in the history without appending a new one."""
    # Temporarily slice history so continuation only uses context *prior* to this line
    temp_history_context = history[:index]
    
    saved_path, status_msg, new_gallery, active_seed = generate_audio_logic(
        updated_text, temp_history_context, ref_label, gallery, mode_with_reference, 
        duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, args, model_type, gguf_file, use_random_seed, seed_val
    )
    
    history[index]["text"] = updated_text
    history[index]["audio_path"] = saved_path
    
    return history, gallery, gr.update(choices=list(gallery.keys())), status_msg, active_seed


# --- ORIGINAL BACKEND FUNCTIONS ---

def load_backend(model_path: str, device_str: str, attn_implementation: str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    resolved_attn_implementation = resolve_attn_implementation(requested=attn_implementation, device=device, dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if resolved_attn_implementation:
        model_kwargs["attn_implementation"] = resolved_attn_implementation
    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()
    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    return model, processor, device, sample_rate

def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> str | None:
    requested_norm = (requested or "").strip().lower()
    if requested_norm in {"none"}: return None
    if requested_norm not in {"", "auto"}: return requested
    if (device.type == "cuda" and importlib.util.find_spec("flash_attn") is not None and dtype in {torch.float16, torch.bfloat16}):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8: return "flash_attention_2"
    if device.type == "cuda": return "sdpa"
    return "eager"

def detect_text_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0 and en_chars == 0: return "en"
    return "zh" if zh_chars >= en_chars else "en"

def supports_duration_control(mode_with_reference: str) -> bool:
    return mode_with_reference not in {MODE_CONTINUE, MODE_CONTINUE_CLONE}

def build_conversation(text: str, reference_audio: str | None, mode_with_reference: str, expected_tokens: int | None, processor):
    text = (text or "").strip()
    if not text: raise ValueError("Please enter text to synthesize.")
    user_kwargs = {"text": text}
    if expected_tokens is not None: user_kwargs["tokens"] = int(expected_tokens)
    if not reference_audio:
        return [[processor.build_user_message(**user_kwargs)]], "generation", "Direct Generation"
    if mode_with_reference == MODE_CLONE:
        clone_kwargs = dict(user_kwargs)
        clone_kwargs["reference"] = [reference_audio]
        return [[processor.build_user_message(**clone_kwargs)]], "generation", MODE_CLONE
    if mode_with_reference == MODE_CONTINUE:
        return [[processor.build_user_message(**user_kwargs), processor.build_assistant_message(audio_codes_list=[reference_audio])]], "continuation", MODE_CONTINUE
    continue_clone_kwargs = dict(user_kwargs)
    continue_clone_kwargs["reference"] = [reference_audio]
    return [[processor.build_user_message(**continue_clone_kwargs), processor.build_assistant_message(audio_codes_list=[reference_audio])]], "continuation", MODE_CONTINUE_CLONE

def run_inference(text, reference_audio, mode_with_reference, duration_control_enabled, duration_tokens, temperature, top_p, top_k, repetition_penalty, max_new_tokens, args, model_type, gguf_file, use_random_seed, seed_val):
    started_at = time.monotonic()
    
    if use_random_seed:
        active_seed = random.randint(0, 2**31 - 1)
    else:
        active_seed = int(seed_val)
        
    torch.manual_seed(active_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(active_seed)
    np.random.seed(active_seed)
    random.seed(active_seed)
    
    # 1. Obtain active backend
    backend_obj, processor, torch_device, sample_rate = load_active_backend(model_type, gguf_file, args)

    if model_type == MODEL_TYPE_FP16:
        model = backend_obj
        duration_enabled = bool(duration_control_enabled and supports_duration_control(mode_with_reference))
        expected_tokens = int(duration_tokens) if duration_enabled else None
        conversations, mode, mode_name = build_conversation(text=text, reference_audio=reference_audio, mode_with_reference=mode_with_reference, expected_tokens=expected_tokens, processor=processor)
        batch = processor(conversations, mode=mode)
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        print(f"generating (FP16)... text length: {len(text)}", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=int(max_new_tokens),
                audio_temperature=float(temperature), audio_top_p=float(top_p), audio_top_k=int(top_k), audio_repetition_penalty=float(repetition_penalty),
            )
        print("finished generating.", flush=True)
        messages = processor.decode(outputs)
        if not messages or messages[0] is None: raise RuntimeError("The model did not return a decodable audio result.")
        audio = messages[0].audio_codes_list[0]
        if isinstance(audio, torch.Tensor): audio_np = audio.detach().float().cpu().numpy()
        else: audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1: audio_np = audio_np.reshape(-1)
        audio_np = audio_np.astype(np.float32, copy=False)
        
    elif model_type == MODEL_TYPE_GGUF:
        pipeline = backend_obj
        mode_name = mode_with_reference if reference_audio else "Direct Generation"
        
        # Duration Control
        expected_tokens = int(duration_tokens) if (duration_control_enabled and supports_duration_control(mode_with_reference)) else None
        
        print(f"generating (GGUF)... text length: {len(text)}", flush=True)
        pipeline.config.audio_temperature = float(temperature)
        pipeline.config.audio_top_p = float(top_p)
        pipeline.config.audio_top_k = int(top_k)
        pipeline.config.audio_repetition_penalty = float(repetition_penalty)
        pipeline.config.max_new_tokens = int(max_new_tokens)
        
        audio_np = pipeline.generate(
            text=text,
            reference_audio=reference_audio,
            tokens=expected_tokens,
            max_new_tokens=int(max_new_tokens)
        )
        print("finished generating.", flush=True)

    elapsed = time.monotonic() - started_at
    status = f"Done | backend: {model_type} | mode: {mode_name} | elapsed: {elapsed:.2f}s | seed: {active_seed} | max_new_tokens={int(max_new_tokens)}"
    
    # Force garbage collection after generation to keep VRAM tight
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    return (sample_rate, audio_np), status, active_seed


# --- GRADIO UI ---

def build_demo(args: argparse.Namespace):
    custom_css = """
    :root { --bg: #f6f7f8; --panel: #ffffff; --ink: #111418; --muted: #4d5562; --line: #e5e7eb; --accent: #0f766e; }
    .gradio-container { background: linear-gradient(180deg, #f7f8fa 0%, #f3f5f7 100%); color: var(--ink); }
    .app-card { border: 1px solid var(--line); border-radius: 16px; background: var(--panel); padding: 14px; margin-bottom: 12px; }
    .history-row { border: 1px solid var(--line); padding: 8px; border-radius: 8px; background: #fafafa; margin-bottom: 8px; align-items: center; }
    #run-btn { background: var(--accent); border: none; color: white; }
    """

    with gr.Blocks(title="MOSS-TTS Line Iteration Workbench", css=custom_css) as demo:
        
        # State Variables
        gallery_state = gr.State(load_startup_gallery()) 
        history_state = gr.State([]) # list of dicts: {"text": str, "audio_path": str}

        gr.Markdown('<div class="app-card"><h2>MOSS-TTS Line-by-Line Workbench (Multi-Backend)</h2><p>Generate sequentially. Switch between PyTorch FP16 and GGUF models. History is auto-prepended in Continuation modes.</p></div>')

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 1. Generation Settings")
                
                # Backend Choice
                available_gguf_models = get_available_gguf_models()
                default_gguf = "MOSS_TTS_backbone_q4k_m.gguf" if "MOSS_TTS_backbone_q4k_m.gguf" in available_gguf_models else (available_gguf_models[0] if available_gguf_models else None)
                
                model_type_dropdown = gr.Dropdown(choices=[MODEL_TYPE_FP16, MODEL_TYPE_GGUF], value=MODEL_TYPE_GGUF, label="Model Backend")
                gguf_file_dropdown = gr.Dropdown(choices=available_gguf_models, value=default_gguf, label="GGUF Model File", visible=True)
                
                def toggle_gguf_dropdown(m_type):
                    return gr.update(visible=(m_type == MODEL_TYPE_GGUF))
                    
                model_type_dropdown.change(fn=toggle_gguf_dropdown, inputs=[model_type_dropdown], outputs=[gguf_file_dropdown])
                
                with gr.Group():
                    init_choices = list(gallery_state.value.keys())
                    ref_dropdown = gr.Dropdown(label="Select Reference Audio", choices=init_choices, value=init_choices[-1] if init_choices else None, interactive=True)
                    ref_preview = gr.Audio(label="Preview Selected Reference", interactive=False)
                    mode = gr.Radio(choices=[MODE_CLONE, MODE_CONTINUE, MODE_CONTINUE_CLONE], value=MODE_CLONE, label="Synthesis Mode")

                with gr.Accordion("Sampling Controls", open=False):
                    with gr.Row():
                        use_random_seed = gr.Checkbox(label="Use Random Seed", value=True)
                        seed_val = gr.Number(label="Seed", value=42, precision=0, interactive=True)
                    temp = gr.Slider(0.1, 3.0, 1.7, step=0.05, label="Temperature")
                    p = gr.Slider(0.1, 1.0, 0.8, step=0.01, label="Top P")
                    k = gr.Slider(1, 200, 25, step=1, label="Top K")
                    rep = gr.Slider(0.8, 2.0, 1.0, step=0.05, label="Repetition Penalty")
                    max_tokens = gr.Slider(256, 8192, 4096, step=128, label="Max Tokens")

                with gr.Accordion("Import New Reference File", open=False):
                    new_ref_file = gr.Audio(label="Upload Audio File", type="filepath")
                    new_ref_name = gr.Textbox(label="Name this Voice")
                    add_btn = gr.Button("Add to Library")

            with gr.Column(scale=5):
                gr.Markdown("### 2. Sequence History")
                
                # Dynamic Render Block for History Lines
                @gr.render(inputs=[history_state])
                def render_history(history):
                    if not history:
                        gr.Markdown("*History is empty. Generate your first line below.*")
                        return
                    
                    for i, item in enumerate(history):
                        with gr.Row(elem_classes="history-row"):
                            line_text = gr.Textbox(value=item["text"], show_label=False, lines=2, scale=4)
                            
                            if item["audio_path"] and os.path.exists(item["audio_path"]):
                                line_audio = gr.Audio(value=item["audio_path"], show_label=False, interactive=False, scale=3)
                            else:
                                line_audio = gr.HTML("<div style='color:red;'>Audio deleted/missing</div>", scale=3)

                            with gr.Column(scale=1):
                                redo_btn = gr.Button("🔄 Redo")
                                del_aud_btn = gr.Button("🗑️ Audio")
                                del_line_btn = gr.Button("❌ Line")

                            # Per-line events
                            redo_btn.click(
                                fn=lambda idx=i, t=line_text, *args_list: redo_specific_line(idx, t, *(list(args_list) + [args])),
                                inputs=[history_state, ref_dropdown, gallery_state, mode, gr.State(False), gr.State(1), temp, p, k, rep, max_tokens, model_type_dropdown, gguf_file_dropdown, use_random_seed, seed_val],
                                outputs=[history_state, gallery_state, ref_dropdown, status, seed_val]
                            )
                            
                            def del_audio(hist, idx=i):
                                hist[idx]["audio_path"] = None
                                return hist
                            
                            del_aud_btn.click(fn=del_audio, inputs=[history_state], outputs=[history_state])
                            
                            def del_line(hist, idx=i):
                                hist.pop(idx)
                                return hist
                                
                            del_line_btn.click(fn=del_line, inputs=[history_state], outputs=[history_state])

                gr.Markdown("### 3. Next Line Input")
                new_text_input = gr.Textbox(label="Text for New Line", lines=3, placeholder="Type the next segment here...")
                run_btn = gr.Button("Generate & Push to History", variant="primary", elem_id="run-btn")
                status = gr.Textbox(label="Console Status", lines=2, interactive=False)

        # --- Base UI Events ---
        add_btn.click(fn=add_custom_ref, inputs=[new_ref_file, new_ref_name, gallery_state], outputs=[gallery_state, ref_dropdown])
        ref_dropdown.change(fn=lambda label, gal: gal.get(label) if label else None, inputs=[ref_dropdown, gallery_state], outputs=[ref_preview])

        run_btn.click(
            fn=lambda *args_list: generate_new_line(*(list(args_list) + [args])),
            inputs=[
                new_text_input, history_state, ref_dropdown, gallery_state, mode,
                gr.State(False), gr.State(1), temp, p, k, rep, max_tokens, model_type_dropdown, gguf_file_dropdown, use_random_seed, seed_val
            ],
            outputs=[history_state, gallery_state, ref_dropdown, status, seed_val]
        ).then(
            # Clear the input box after successful generation
            fn=lambda: "", inputs=None, outputs=[new_text_input]
        )

    return demo

def main():
    parser = argparse.ArgumentParser(description="MossTTS Gradio Demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    runtime_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
    args.attn_implementation = resolve_attn_implementation(
        requested=args.attn_implementation, device=runtime_device, dtype=runtime_dtype,
    ) or "none"
    print(f"[INFO] Using attn_implementation={args.attn_implementation}", flush=True)

    preload_started_at = time.monotonic()
    
    # PRELOAD GGUF BY DEFAULT (MATCHING UI DEFAULT)
    available_gguf_models = get_available_gguf_models()
    default_gguf = "MOSS_TTS_backbone_q4k_m.gguf" if "MOSS_TTS_backbone_q4k_m.gguf" in available_gguf_models else (available_gguf_models[0] if available_gguf_models else None)
    load_active_backend(MODEL_TYPE_GGUF, default_gguf, args)
    
    print(f"[Startup] Backend preload finished in {time.monotonic() - preload_started_at:.2f}s", flush=True)

    demo = build_demo(args)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
    )

if __name__ == "__main__":
    main()
