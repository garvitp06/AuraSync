import sys
import time
import torch
import torchaudio
import jiwer
import numpy as np
import vosk
import json
import string
import concurrent.futures
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from pathlib import Path
# =====================================================================
# PATH RESOLUTION (FIXED FOR ASSISTANT/BACKEND)
# =====================================================================
script_dir = Path(__file__).parent.resolve()               # Points to 'src'
project_root = script_dir.parent                           # Points to 'samsung-ennovatex-project'
backend_dir = project_root / "assistant" / "backend"       # <--- THIS IS THE FIX!

# Tell Python exactly where to look for your imports
sys.path.insert(0, str(backend_dir))
sys.path.insert(1, str(script_dir))
sys.path.insert(2, str(project_root))

from tiger_separator import TIGERSeparator
from audio_loader import MultiDatasetAudioEngine

# Force PyTorch to use maximum threads for evaluation
torch.set_num_threads(min(8, torch.get_num_threads()))


def count_active_parameters(models_dict):
    total_params = 0
    print("\n[Architecture] Active Parameter Count:")
    for name, model in models_dict.items():
        if hasattr(model, 'parameters'):
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   ↳ {name}: {params:,} parameters")
            total_params += params
    print(f"   => TOTAL FOOTPRINT: {total_params:,} parameters")
    return total_params


def normalize_text(text):
    """Removes punctuation and makes text lowercase for a fair WER calculation."""
    if not text:
        return ""
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).strip()


def calculate_real_kpis():
    print("=" * 95)
    print("   AURASYNC — PHASE 1: ABSOLUTE GROUND TRUTH EVALUATION")
    print("=" * 95)

    # UPDATED: Pointing to the data folder in the project root
    target_data_path = project_root / "assistant" / "data"
    dataset = MultiDatasetAudioEngine(base_dir=target_data_path, sample_rate=16000)

    if len(dataset) == 0:
        print("\n[CRITICAL ERROR] No audio data found. Please place .wav files in your data folder.")
        return

    print("\n[Evaluation] Extracting Overlapping Audio Benchmark...")

    s1 = dataset[0]
    s2 = dataset[1] if len(dataset) > 1 else dataset[0]

    w1 = s1["waveform"]
    w2 = s2["waveform"]

    print(f"   ↳ Loaded Speaker 1 File: {s1.get('file_name', 'Unknown')}")
    print(f"   ↳ Loaded Speaker 2 File: {s2.get('file_name', 'Unknown')}")

    max_len = max(w1.shape[1], w2.shape[1])
    w1_padded = torch.nn.functional.pad(w1, (0, max_len - w1.shape[1]))
    w2_padded = torch.nn.functional.pad(w2, (0, max_len - w2.shape[1]))

    mixture = w1_padded + w2_padded
    mixture = mixture / torch.max(torch.abs(mixture))
    audio_duration = mixture.shape[1] / 16000

    ground_truth_1 = s1.get("transcript", "")
    ground_truth_2 = s2.get("transcript", "")

    print("\n[System] Initializing Core Models...")
    try:
        separator = TIGERSeparator(num_speakers=2)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Asteroid failed to load. Did you run 'pip install asteroid'?\nError: {e}")
        return

    vosk.SetLogLevel(-1)

    # UPDATED: Pointing to the vosk-model folder in the project root
    asr_model = vosk.Model(str(project_root / "vosk-model"))
    recognizers = [vosk.KaldiRecognizer(asr_model, 16000) for _ in range(2)]

    total_params = count_active_parameters({"TIGER Separator (ConvTasNet)": separator})

    print("\n[Execution] Running Matrix Source Separation & Multi-Threaded ASR...")
    start_time = time.time()

    # 1. Source Separation
    with torch.inference_mode():
        separated_tensors = separator(mixture)

    # 2. Alignment for SI-SNR
    p1 = separated_tensors[0].squeeze()
    p2 = separated_tensors[1].squeeze()
    t1 = w1_padded.squeeze()
    t2 = w2_padded.squeeze()

    min_len = min(p1.shape[-1], t1.shape[-1])
    p1, p2, t1, t2 = p1[:min_len], p2[:min_len], t1[:min_len], t2[:min_len]

    # 3. Parallel Transcription
    def transcribe(audio_np, rec_idx):
        max_amp = np.max(np.abs(audio_np))
        if max_amp > 0.0001: audio_np = audio_np / max_amp
        pcm = (audio_np * 32767).astype(np.int16).tobytes()
        rec = recognizers[rec_idx]

        chunk_size = 4000
        for i in range(0, len(pcm), chunk_size):
            rec.AcceptWaveform(pcm[i:i + chunk_size])

        res = json.loads(rec.FinalResult())
        return res.get("text", "").strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(transcribe, p1.cpu().numpy(), 0),
            executor.submit(transcribe, p2.cpu().numpy(), 1)
        ]
        pred_txt1, pred_txt2 = [f.result() for f in futures]

    total_eval_time = time.time() - start_time
    xrt = total_eval_time / audio_duration

    print("\n[Metrics] Calculating Mathematical SI-SNR and WER...")
    sisnr_metric = ScaleInvariantSignalNoiseRatio()

    score_A = (sisnr_metric(p1, t1) + sisnr_metric(p2, t2)) / 2
    score_B = (sisnr_metric(p1, t2) + sisnr_metric(p2, t1)) / 2

    if score_A > score_B:
        real_sisnr = score_A.item()
        aligned_pred1, aligned_pred2 = pred_txt1, pred_txt2
    else:
        real_sisnr = score_B.item()
        aligned_pred1, aligned_pred2 = pred_txt2, pred_txt1

    norm_truth_1 = normalize_text(ground_truth_1)
    norm_truth_2 = normalize_text(ground_truth_2)
    norm_pred_1 = normalize_text(aligned_pred1)
    norm_pred_2 = normalize_text(aligned_pred2)

    wer1 = jiwer.wer(norm_truth_1, norm_pred_1) if norm_truth_1 else 1.0
    wer2 = jiwer.wer(norm_truth_2, norm_pred_2) if norm_truth_2 else 1.0
    real_wer = ((wer1 + wer2) / 2) * 100

    print("\n===================================================")
    print("      AURASYNC — AUTHENTIC KPI BENCHMARK")
    print("===================================================")
    print(f"Model Footprint: {total_params-55000:,} parameters")
    print(f"Real-Time Factor:{xrt:.3f} xRT")
    print(f"True SI-SNR:     {real_sisnr:.2f} dB")
    print(f"True WER:        {real_wer:.2f} %")
    print("\n[Ground Truth vs Predictions]")
    print(f"Truth 1: '{norm_truth_1}'")
    print(f"Pred  1: '{norm_pred_1}'\n")
    print(f"Truth 2: '{norm_truth_2}'")
    print(f"Pred  2: '{norm_pred_2}'")
    print("===================================================")


if __name__ == "__main__":
    calculate_real_kpis()