import os
import shutil

from app import _fn

IN_DIR = "/content/drive/MyDrive/voice_in"
OUT_DIR = "/content/drive/MyDrive/voice_out"
EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
        ".mp4", ".mov", ".mkv", ".webm"}

os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

files = [f for f in sorted(os.listdir(IN_DIR)) if os.path.splitext(f)[1].lower() in EXTS]
print(f"Found {len(files)} file(s) in {IN_DIR}")

for i, name in enumerate(files, 1):
    stem = os.path.splitext(name)[0]
    # skip anything already processed
    if any(o.startswith(stem + "_clean") for o in os.listdir(OUT_DIR)):
        print(f"[{i}/{len(files)}] skip {name} (already done)")
        continue
    try:
        print(f"[{i}/{len(files)}] processing {name} ...")
        # path, mode, solver, nfe, tau, denoise_strength, overlap,
        # dry_wet, highpass_hz, normalize, target_lufs, out_format
        _, final = _fn(
            os.path.join(IN_DIR, name), "Enhance", "Midpoint", 64, 0.5, 0.8, 4,
            0.85, 80, True, -16, "wav",
        )
        dest = os.path.join(OUT_DIR, stem + "_clean" + os.path.splitext(final)[1])
        shutil.copy(final, dest)
    except Exception as e:
        print(f"  failed on {name}: {e}")

print("Done. Cleaned files are in", OUT_DIR)
