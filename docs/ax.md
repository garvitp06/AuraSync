# Agentic AI Setup & Workflows Documentation (Track 2)

This document provides a comprehensive breakdown of the localized agentic orchestration engine developed for the AuraSync platform, explaining our design choices, tool-use routing pipelines, and optimization milestones.

---

## 🧠 Agentic AI Architecture

The system utilizes a compiled, state-directed orchestration loop to process text data streaming from the edge acoustic frontend.

### 1. Compiled LangGraph State Workflow
Instead of utilizing standard, brittle python conditional loops, the orchestration layer uses the `langgraph` framework to compile a resilient state routing tree. This configuration links our active nodes explicitly:
`SupervisorRouter Node` ➔ `IntentPlanner Node` ➔ `WatchdogCritic Node` ➔ `END`

### 2. Cloud-Free Local SLM Core
To guarantee absolute data sovereignty and zero external round-trip network latency, all semantic evaluation passes through a local instance of the **Qwen2.5-1.5B-Instruct** small language model running offline via an on-device Ollama server endpoint.

### 3. Adversarial Safety Watchdog
The system does not rely on probabilistic language model predictions to handle dangerous smart home device execution states. Instead, a deterministic **Watchdog Critic Agent** cross-checks every single planned action string against prior conversational state history variables before any tool-chain payload can fire.

---

## 📊 Developer Experience Insights: What Worked vs. What Did Not

### ❌ What Did NOT Work (The Sequential Latency Trap)
* **The Problem:** In our initial design phase, the text output from every single 160ms acoustic frame chunk was fed directly into the language model for safety clearance. 
* **The Consequence:** Because local small language model text inference on a consumer CPU requires roughly 1 to 2 seconds per generation loop, hitting the model sequentially on every split-second audio chunk created a severe latency bottleneck, causing the system execution runtime to spike past **70 seconds ($xRT = 8.33$)**. This severely breached our hackathon runtime constraints.

###  What WORKED (Compute-Aware Token Compression Gating)
* **The Solution:** We designed and deployed a **Compute-Aware Edge Ingest Gate**. The fast acoustic front-end (operating at a blazing **0.0132 xRT**) buffers text silently. The local language model agent is **only** invoked when a significant conversational shift is detected (such as a text context change, a new resident speaker profile signature, or an overlapping multi-user voice collision).
* **The Result:** This optimization eliminated redundant generation cycles, dropping the aggregate end-to-end execution duration from **70.6 seconds down to a crisp 6.84 seconds**—compressing our processing timeline by over **90%**!