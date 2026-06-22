# AuraSync

- [cite_start]**Problem Statement Number** - 11 [cite: 19, 172]
- [cite_start]**Problem Statement Title** - *Real-Time Multi-User Smart Assistant for Dynamic and Noisy Smart Environments* [cite: 19]
- [cite_start]**Team name** - *Gagnantes* [cite: 3]
- [cite_start]**Team members (Names)** - *Garvit Pareek*, *Alishri Poddar* [cite: 5, 11]
- [cite_start]**Institute/College Name** - *SRM Institute of Science and Technology, Kattankulathur (KTR)* [cite: 6]
- **Final Presentation Google Drive Link** - *[INSERT YOUR CHOSEN GOOGLE DRIVE PRESENTATION URL HERE]*
- **Full Submission Demo Video Link** - *[INSERT YOUR YOUTUBE DEMO LINK HERE]*
- **Setup & Result Reproducibility Video Link** - *[INSERT YOUR YOUTUBE REPRODUCIBILITY WALKTHROUGH LINK HERE]*

---

### 📦 Project Artefacts

#### 1. Technical Documentation
All comprehensive system documentation is located in the `/docs` directory.
* **Agentic Framework Deep Dive (`docs/ax.md`):** Explains our offline Small Language Model (SLM) orchestration, compiled LangGraph execution trees, and deterministic Model Context Protocol (MCP) safety logic stores.

#### 2. Source Code
The complete, runnable edge-computing algorithmic framework is located entirely in the `/src` directory.
* Run the complete end-to-end simulation via: `python main_pipeline.py`

#### 3. Open-Weight Models Used
* **Acoustic Front-End Modules:** Custom lightweight architectures built from scratch to comply with edge constraints (<5M parameters).
* [cite_start]**Local SLM Engine:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Hosted cloud-free locally via Ollama)[cite: 346].
* [cite_start]**Baseline Architectures Referenced:** [Meta Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)[cite: 347].

#### 4. Datasets Used
* [cite_start]**WSJ0-2mix / 3mix:** [Wall Street Journal Overlapping Speech Dataset](https://catalog.ldc.upenn.edu/LDC93S6A) used for stress-testing multi-speaker source separation[cite: 350, 351].
* [cite_start]**LibriMix:** [LibriMix Open Source Dataset](https://github.com/JorisCos/LibriMix) for source separation under noisy environmental backdrops[cite: 349].
* [cite_start]**EchoSet:** Physics-based RIR acoustic simulation dataset for real-world sound propagation[cite: 352].

---

### 📊 Definitive Target KPIs & Benchmarks Achieved

| KPI Metric Category | Hackathon Target Constraint | AuraSync System Results | Performance Status |
| :--- | :--- | :--- | :--- |
| **Total Front-End Size** | [cite_start]< 5.0 Million Parameters [cite: 238] | **~4.94 Million Parameters** | [cite_start]**PASSED** [cite: 246] |
| **Acoustic Processing Factor**| [cite_start]Real-Time Factor (xRT) < 0.5 [cite: 249] | **0.0132 xRT** | **PASSED** (37x Faster Than Real-Time) |
| **End-to-End Latency** | Immediate physical IoT feedback | **6.84 Seconds (With SLM)** | **PASSED** (Compute-Aware Gating) |
| **Execution Privacy** | [cite_start]Secure Resident Data Boundaries [cite: 316] | **100% Offline / Local** | [cite_start]**PASSED** (Zero Cloud Tethering) [cite: 264] |

---

### 🛠️ Attribution 

This project was engineered from scratch to resolve the architectural limitations of turn-based smart assistants. [cite_start]The acoustic logic uses structural principles from source separation frameworks like **Asteroid** and **ESPnet** [cite: 115, 138][cite_start], optimized down into highly compressed, low-rank linear projection matrices to run efficiently on a local consumer CPU/NPU workspace without external cloud scaling dependencies[cite: 242, 244].