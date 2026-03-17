import sys
import os
import torch
import numpy as np

# Add src to sys.path
project_root = os.getcwd()
sys.path.append(os.path.join(project_root, "src"))

from chatterbox.tts_turbo import ChatterboxTurboTTS

def find_double_tensors(model, name):
    print(f"\n--- Checking {name} ---")
    found = False
    for n, p in model.named_parameters():
        if p.dtype == torch.float64:
            print(f"PARAM: {n} is {p.dtype}")
            found = True
    for n, b in model.named_buffers():
        if b.dtype == torch.float64:
            print(f"BUFFER: {n} is {b.dtype}")
            found = True
    
    if not found:
        print("No Double tensors found.")

try:
    print("Loading Turbo model (this may take a moment)...")
    # Load to CPU to avoid CUDA dependency issues in script
    model = ChatterboxTurboTTS.from_pretrained('cpu')
    
    print("\n[V] Manual .float() and check...")
    model.ve.float()
    model.t3.float()
    model.s3gen.float()
    
    find_double_tensors(model.ve, "Voice Encoder")
    find_double_tensors(model.t3, "T3 (Turbo)")
    find_double_tensors(model.s3gen, "S3Gen")
    
    if hasattr(model.s3gen, 'matcha'):
        find_double_tensors(model.s3gen.matcha, "Matcha (in S3Gen)")
    if hasattr(model.s3gen, 'mel2wav'):
        find_double_tensors(model.s3gen.mel2wav, "Vocoder (in S3Gen)")

    # Check if any submodules in T3 are still double
    print("\n--- Checking T3 Submodules ---")
    for n, m in model.t3.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if p.dtype == torch.float64:
                print(f"Module {n} / Param {pn} is {p.dtype}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
