# AuraSync Edge Gateway User Guide

Welcome to the deployment, provisioning, and operational manual for the AuraSync offline smart assistant framework. This guide provides step-by-step instructions on how to install, verify, and run the real-time multi-speaker isolation pipeline on local edge gateways.

---

## 🛠️ Environmental & System Requirements

Before deploying the firmware pipeline, verify that your edge gateway or local machine meets the following computational baseline:

| Requirement | Specification |
|------------|--------------|
| **Operating System** | Windows 10/11 (64-bit) or macOS Sonoma+ |
| **Python Runtime** | Python 3.10 (strictly enforced due to PyTorch C++ extension bindings and Vosk Kaldi wheels) |
| **Hardware Architecture** | x86_64 or ARM64 (Apple Silicon / Raspberry Pi 4+) |
| **Minimum Compute Footprint** | Dual-Core CPU @ 2.0GHz, 4GB RAM, and 1.5GB free local storage |

---

# 🚀 Step-by-Step Installation & Environment Provisioning

## 1. Repository Clone & Staging

Open your terminal application and clone the source code repository while avoiding unnecessary model cache directories.

```bash
git clone https://github.com/garvitp06/hackathon.git
cd samsung-ennovatex-project
```

---

## 2. Isolate the Python Runtime Environment

Initialize a clean Python virtual environment. This isolates execution from global packages and prevents system path contamination.

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate on Windows

```bash
venv\Scripts\activate
```

### Activate on macOS / Linux

```bash
source venv/bin/activate
```

---

## 3. Install Open-Source Hardware Dependencies

Deploy the dependency manifest. This stages PyTorch, Asteroid, Vosk, and TorchAudio binaries optimized for edge hardware.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Verify Local Offline Model Weights

Ensure the lightweight Vosk Indian English acoustic model is extracted and placed in the project root under:

```text
vosk-model/
```

Verify that the following subdirectories exist:

```text
vosk-model/
├── am/
├── conf/
├── graph/
└── ivector/
```

---

# 📊 Live Production Execution

## 1. Run the Mathematical Evaluation Benchmark Suite

To independently verify system performance metrics and reproduce the reported KPIs, execute:

```bash
python src/evaluate_kpi.py
```

### What This Script Does

- Verifies structural graph parameter limits
- Processes overlapping audio streams through the separation network
- Calculates SI-SNR improvement
- Profiles the Real-Time Factor (xRT)
- Generates reproducible benchmark metrics

---

## 2. Launch the Active Edge Firmware Pipeline

Start the real-time orchestration loop that:

- Monitors ambient speech streams
- Performs multi-user speaker separation
- Routes transcribed text to the intent engine
- Executes local edge actions

Run:

```bash
python src/main_pipeline.py
```

---

# 🔍 Understanding the Telemetry Logger Dashboard

While running, AuraSync outputs real-time diagnostics directly to the terminal.

## [PARAM AUDIT]

Monitors active graph layer operations.

**Expected Output:**

```text
[PARAM AUDIT] Total Parameters: 5,050,545
```

**Purpose:**

- Confirms full compliance with hackathon edge-device constraints
- Verifies the model architecture loaded successfully

---

## [BRIDGE LOG]

Displays tracking information from the Acoustic Bridge.

**Pipeline Flow:**

```text
8 kHz Audio
      ↓
Low-Compute Masking
      ↓
16 kHz Reconstruction
      ↓
Speech Recognition
```

**Purpose:**

- Verifies downsampling for computational efficiency
- Confirms clean upsampling for transcription accuracy

---

## [THREAD MATRIX]

Confirms that asynchronous worker threads are active.

**Purpose:**

- Handles speech recognition in parallel
- Reduces blocking operations
- Improves real-time responsiveness

**Example Output:**

```text
[THREAD MATRIX] Worker Threads Initialized: 4
```

---

## [ROUTER NODE]

Displays execution targets selected by the `Tier1IntentRouter`.

**Example Output:**

```text
[ROUTER NODE]
Routed Action Node: [temperature]
Execution Node: Local Firmware Bus
```

**Purpose:**

- Shows intent classification results
- Verifies routing decisions
- Confirms local action execution

---

# ✅ Deployment Verification Checklist

Before deployment, confirm the following:

- [ ] Python 3.10 installed
- [ ] Virtual environment activated
- [ ] Dependencies installed successfully
- [ ] Vosk model extracted correctly
- [ ] KPI benchmark executed successfully
- [ ] Main pipeline launched without errors
- [ ] Parameter count shows **5,050,545**
- [ ] Acoustic Bridge logs are active
- [ ] Worker threads initialized
- [ ] Intent routing functioning correctly

---

# 📌 Support Notes

If the pipeline fails to initialize:

1. Verify Python version:

```bash
python --version
```

2. Confirm dependencies:

```bash
pip list
```

3. Verify model folder structure:

```text
vosk-model/
├── am/
├── conf/
├── graph/
└── ivector/
```

4. Reinstall dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

**AuraSync Edge Gateway Framework**  
*Offline • Private • Real-Time • Edge AI*
