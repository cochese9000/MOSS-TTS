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

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, login, whoami
from transformers import AutoModel, AutoProcessor

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

MODEL_PATH = snapshot_download(
    repo_id=model_id,
    local_files_only=False
)

print(f"Downloaded to: {MODEL_PATH}")

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


# --- STATE & STORAGE HELPERS ---

def load_startup_gallery():
    """Loads all previously saved wav files from the iterations directory on startup."""
    gallery = {}
    if ITERATION_DIR.exists():
        for filepath in sorted(ITERATION_DIR.glob("*.wav"), key=os.path.getmtime):
            # Use the filename as the display label
            gallery[filepath.stem] = str(filepath)
    return dict(sorted(gallery.items()))

def save_to_gallery(audio_data, name, text_snippet, current_gallery):
    """Saves audio and updates the session gallery."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_text = re.sub(r'[^\w\s-]', '', text_snippet)[:30].strip().replace(" ", "_")
    filename = f"{name}_{clean_text}_{timestamp}"
    filepath = ITERATION_DIR / f"{filename}.wav"
    
    wavfile.write(filepath, audio_data[0], audio_data[1])
    
    current_gallery[filename] = str(filepath)
    return dict(sorted(current_gallery.items())), str(filepath)

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
    duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, args
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

    audio_np_tuple, status_msg = run_inference(
        synthesis_text, ref_path, mode_with_reference, 
        duration_enabled, duration_tokens,
        temp, top_p, top_k, rep_penalty,
        args.model_path, args.device, args.attn_implementation, max_new_tokens
    )
    
    new_gallery, saved_path = save_to_gallery(audio_np_tuple, base_ref, target_text, gallery)
    return saved_path, status_msg, new_gallery

def generate_new_line(
    new_text, history, ref_label, gallery, mode_with_reference, 
    duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, args
):
    if not new_text.strip():
        return history, gallery, gr.update(), "Error: Text is empty"
        
    saved_path, status_msg, new_gallery = generate_audio_logic(
        new_text, history, ref_label, gallery, mode_with_reference, 
        duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, args
    )
    
    # Push to history
    history.append({
        "text": new_text,
        "audio_path": saved_path
    })
    
    return history, new_gallery, gr.update(choices=list(new_gallery.keys()), value=list(new_gallery.keys())[-1]), status_msg

def redo_specific_line(
    index, updated_text, history, ref_label, gallery, mode_with_reference, 
    duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, args
):
    """Regenerates a specific line in the history without appending a new one."""
    # Temporarily slice history so continuation only uses context *prior* to this line
    temp_history_context = history[:index]
    
    saved_path, status_msg, new_gallery = generate_audio_logic(
        updated_text, temp_history_context, ref_label, gallery, mode_with_reference, 
        duration_enabled, duration_tokens, temp, top_p, top_k, rep_penalty, max_new_tokens, args
    )
    
    history[index]["text"] = updated_text
    history[index]["audio_path"] = saved_path
    
    return history, new_gallery, gr.update(choices=list(new_gallery.keys()), value=list(new_gallery.keys())[-1]), status_msg


# --- ORIGINAL BACKEND FUNCTIONS ---

def _parse_example_id(example_id: str) -> tuple[str, int] | None:
    matched = re.fullmatch(r"(zh|en)/(\d+)", (example_id or "").strip())
    if matched is None:
        return None
    return matched.group(1), int(matched.group(2))

def _resolve_reference_audio_path(language: str, index: int) -> Path | None:
    stem_candidates = [f"reference_{language}_{index}"]
    for stem in stem_candidates:
        for ext in (".wav", ".mp3"):
            audio_path = REFERENCE_AUDIO_DIR / f"{stem}{ext}"
            if audio_path.exists():
                return audio_path
    return None

def build_example_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with open(EXAMPLE_TEXTS_JSONL_PATH, "rb") as f:
        for line in f:
            if not line.strip(): continue
            sample = orjson.loads(line)
            parsed = _parse_example_id(sample.get("id", ""))
            if parsed is None: continue
            language, index = parsed
            text = str(sample.get("text", "")).strip()
            audio_path = _resolve_reference_audio_path(language, index)
            if audio_path is None: continue
            rows.append((sample['role'], str(audio_path), text))
    return rows

EXAMPLE_ROWS = build_example_rows()

@functools.lru_cache(maxsize=1)
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

def estimate_duration_tokens(text: str) -> tuple[str, int, int, int]:
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    language = detect_text_language(normalized)
    factor = ZH_TOKENS_PER_CHAR if language == "zh" else EN_TOKENS_PER_CHAR
    default_tokens = max(1, int(effective_len * factor))
    min_tokens = max(1, int(default_tokens * 0.5))
    max_tokens = max(min_tokens, int(default_tokens * 1.5))
    return language, default_tokens, min_tokens, max_tokens

def update_duration_controls(enabled: bool, text: str, current_tokens: float | int | None, mode_with_reference: str):
    if not supports_duration_control(mode_with_reference):
        return (gr.update(visible=False), "Duration control disabled for Continuation.", gr.update(value=False, interactive=False))
    checkbox_update = gr.update(interactive=True)
    if not enabled:
        return gr.update(visible=False), "Duration control is disabled.", checkbox_update
    language, default_tokens, min_tokens, max_tokens = estimate_duration_tokens(text)
    if current_tokens is None or int(current_tokens) == 1: slider_value = default_tokens
    else: slider_value = max(min_tokens, min(max_tokens, int(current_tokens)))
    language_label = "Chinese" if language == "zh" else "English"
    hint = f"Duration control enabled | detected: {language_label} | default={default_tokens}, range=[{min_tokens}, {max_tokens}]"
    return (gr.update(visible=True, minimum=min_tokens, maximum=max_tokens, value=slider_value, step=1), hint, checkbox_update)

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

def render_mode_hint(reference_audio: str | None, mode_with_reference: str):
    if not reference_audio: return "Current mode: **Direct Generation** (no ref audio uploaded)"
    if mode_with_reference == MODE_CLONE: return "Current mode: **Clone** (timbre will be cloned)"
    return f"Current mode: **{mode_with_reference}** \n> {CONTINUATION_NOTICE}"

def run_inference(text, reference_audio, mode_with_reference, duration_control_enabled, duration_tokens, temperature, top_p, top_k, repetition_penalty, model_path, device, attn_implementation, max_new_tokens):
    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = load_backend(model_path=model_path, device_str=device, attn_implementation=attn_implementation)
    duration_enabled = bool(duration_control_enabled and supports_duration_control(mode_with_reference))
    expected_tokens = int(duration_tokens) if duration_enabled else None
    conversations, mode, mode_name = build_conversation(text=text, reference_audio=reference_audio, mode_with_reference=mode_with_reference, expected_tokens=expected_tokens, processor=processor)
    batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(torch_device)
    attention_mask = batch["attention_mask"].to(torch_device)
    print("generating...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=int(max_new_tokens),
            audio_temperature=float(temperature), audio_top_p=float(top_p), audio_top_k=int(top_k), audio_repetition_penalty=float(repetition_penalty),
        )
    print("finished generating.")
    messages = processor.decode(outputs)
    if not messages or messages[0] is None: raise RuntimeError("The model did not return a decodable audio result.")
    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor): audio_np = audio.detach().float().cpu().numpy()
    else: audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1: audio_np = audio_np.reshape(-1)
    audio_np = audio_np.astype(np.float32, copy=False)
    elapsed = time.monotonic() - started_at
    status = f"Done | mode: {mode_name} | elapsed: {elapsed:.2f}s | max_new_tokens={int(max_new_tokens)}, expected_tokens={expected_tokens if expected_tokens is not None else 'off'}"
    return (sample_rate, audio_np), status


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

        gr.Markdown('<div class="app-card"><h2>MOSS-TTS Line-by-Line Workbench</h2><p>Generate sequentially. History is auto-prepended in Continuation modes.</p></div>')

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 1. Generation Settings")
                with gr.Group():
                    init_choices = list(gallery_state.value.keys())
                    ref_dropdown = gr.Dropdown(label="Select Reference Audio", choices=init_choices, value=init_choices[-1] if init_choices else None, interactive=True)
                    ref_preview = gr.Audio(label="Preview Selected Reference", interactive=False)
                    mode = gr.Radio(choices=[MODE_CLONE, MODE_CONTINUE, MODE_CONTINUE_CLONE], value=MODE_CLONE, label="Synthesis Mode")
                    mode_hint = gr.Markdown(render_mode_hint(None, MODE_CLONE))

                with gr.Accordion("Sampling Controls", open=False):
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
                                inputs=[history_state, ref_dropdown, gallery_state, mode, gr.State(False), gr.State(1), temp, p, k, rep, max_tokens],
                                outputs=[history_state, gallery_state, ref_dropdown, status]
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
        mode.change(fn=render_mode_hint, inputs=[ref_dropdown, mode], outputs=[mode_hint])

        run_btn.click(
            fn=lambda *args_list: generate_new_line(*(list(args_list) + [args])),
            inputs=[
                new_text_input, history_state, ref_dropdown, gallery_state, mode,
                gr.State(False), gr.State(1), temp, p, k, rep, max_tokens
            ],
            outputs=[history_state, gallery_state, ref_dropdown, status]
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
    load_backend(model_path=args.model_path, device_str=args.device, attn_implementation=args.attn_implementation)
    print(f"[Startup] Backend preload finished in {time.monotonic() - preload_started_at:.2f}s", flush=True)

    demo = build_demo(args)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
    )

if __name__ == "__main__":
    main()