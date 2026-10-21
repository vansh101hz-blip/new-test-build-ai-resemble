import gc
import os
import subprocess
import tempfile

import gradio as gr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
from scipy.signal import butter, sosfilt

from resemble_enhance.enhancer.inference import load_enhancer
from resemble_enhance.inference import inference

device = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SECONDS = 30.0
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

_enhancer = None


def get_enhancer():
    """Load the model once and reuse it."""
    global _enhancer
    if _enhancer is None:
        _enhancer = load_enhancer(None, device)
    return _enhancer


def to_mono_wav(src, tmp):
    """Convert any audio/video file to a 44.1 kHz mono wav with ffmpeg."""
    out = os.path.join(tmp, "input.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "44100", out],
        check=True, capture_output=True,
    )
    return out


def highpass(wav, sr, cutoff):
    if cutoff <= 0:
        return wav
    sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, wav).astype(np.float32)


def normalize_loudness(wav, sr, target_lufs):
    if target_lufs is None:
        return wav
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    if not np.isfinite(loudness):
        return wav
    wav = pyln.normalize.loudness(wav, loudness, target_lufs)
    peak = np.max(np.abs(wav))
    if peak > 0.97:  # peak guard so normalization never clips
        wav = wav * (0.97 / peak)
    return wav.astype(np.float32)


def _fn(
    path, mode, solver, nfe, tau, denoise_strength, overlap_seconds,
    dry_wet, highpass_hz, normalize, target_lufs, out_format,
):
    if path is None:
        return None, None

    out_dir = tempfile.mkdtemp()
    enhancer = get_enhancer()
    enhancer.configurate_(
        nfe=int(nfe), solver=solver.lower(), lambd=float(denoise_strength), tau=tau
    )

    wav_path = to_mono_wav(path, out_dir)
    dwav, sr = torchaudio.load(wav_path)
    dwav = dwav.mean(dim=0)

    model = enhancer.denoiser if mode == "Denoise only" else enhancer
    wet, new_sr = inference(
        model=model, dwav=dwav, sr=sr, device=device,
        chunk_seconds=CHUNK_SECONDS, overlap_seconds=overlap_seconds,
    )
    wet = wet.cpu().numpy().astype(np.float32)

    # dry/wet: blend some of the original back in (1.0 = fully processed)
    if dry_wet < 1.0:
        dry = torchaudio.functional.resample(dwav, sr, new_sr).numpy()
        n = min(len(dry), len(wet))
        wet = dry_wet * wet[:n] + (1 - dry_wet) * dry[:n]

    wet = highpass(wet, new_sr, highpass_hz)
    if normalize:
        wet = normalize_loudness(wet, new_sr, target_lufs)

    base = os.path.join(out_dir, "result")
    wav_out = base + ".wav"
    sf.write(wav_out, wet, new_sr)

    is_video = os.path.splitext(path)[1].lower() in VIDEO_EXTS
    if is_video:
        final = base + ".mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-i", wav_out, "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", final],
            check=True, capture_output=True,
        )
    elif out_format == "wav":
        final = wav_out
    else:
        final = f"{base}.{out_format}"
        subprocess.run(["ffmpeg", "-y", "-i", wav_out, final], check=True, capture_output=True)

    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return wav_out, final


def main():
    inputs = [
        gr.File(label="Input (audio or video)", type="filepath", file_types=["audio", "video"]),
        gr.Radio(["Enhance", "Denoise only"], value="Enhance", label="Mode"),
        gr.Dropdown(["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver"),
        gr.Slider(1, 128, value=64, step=1, label="CFM Number of Function Evaluations (quality vs speed)"),
        gr.Slider(0, 1, value=0.5, step=0.01, label="CFM Prior Temperature"),
        gr.Slider(0, 1, value=0.5, step=0.05, label="Denoise strength (0 = keep background, 1 = remove max)"),
        gr.Slider(1, 10, value=4, step=0.5, label="Chunk overlap (seconds)"),
        gr.Slider(0, 1, value=1.0, step=0.05, label="Dry/wet (1 = fully processed, lower = more original)"),
        gr.Slider(0, 200, value=80, step=10, label="High-pass filter Hz (0 = off, removes rumble)"),
        gr.Checkbox(value=True, label="Normalize loudness"),
        gr.Slider(-24, -10, value=-16, step=1, label="Target loudness (LUFS)"),
        gr.Dropdown(["wav", "flac", "mp3"], value="wav", label="Download format (audio input)"),
    ]
    outputs = [
        gr.Audio(label="Preview", type="filepath"),
        gr.File(label="Download"),
    ]
    gr.Interface(
        fn=_fn,
        title="Resemble Enhance",
        description="Remove background noise and clean up voice recordings.",
        inputs=inputs,
        outputs=outputs,
    ).launch(share=True)


if __name__ == "__main__":
    main()
