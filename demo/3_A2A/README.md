# Building An Agentic Solution Using (Fine-Tuned and Quantized) SQL & Python Agents With Nemotron-Orchestrator

This project provides an end-to-end build of an Agentic solution Using (Fine-Tuned and Quantized) SQL and Python Agents with Nemotron-Orchestrator

## Prerequisites

- Python 3.12+
- CPU or Apple Silicon

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Prerequisite Models

If you did end up fine-tuning and quantizing the SQL or Python models, you can download these prebuild models below (place them in this example folder).

PRIMARY DOWNLOAD:
- Python LLM (Fine-Tuned & Quantized) - [https://drive.google.com/file/d/1xZaZttqhkKaSWM8OrH6BMizZ5l3WrKWY/view?usp=drive_link](https://drive.google.com/file/d/1xZaZttqhkKaSWM8OrH6BMizZ5l3WrKWY/view?usp=drive_link)
- SQL LLM (Fine-Tuned & Quantized) - [https://drive.google.com/file/d/1rLUd1N6mUkaTGtt6Pv5JB4MtgEkufPoA/view?usp=drive_link](https://drive.google.com/file/d/1rLUd1N6mUkaTGtt6Pv5JB4MtgEkufPoA/view?usp=drive_link)

BACKUP DOWNLOAD:
- Python LLM (Fine-Tuned & Quantized) - [https://drive.google.com/file/d/1kJQ4jy2kcHG3Adjgy_A6jUZsMdlNcY_g/view?usp=drive_link](https://drive.google.com/file/d/1kJQ4jy2kcHG3Adjgy_A6jUZsMdlNcY_g/view?usp=drive_link)
- SQL LLM (Fine-Tuned & Quantized) - [https://drive.google.com/file/d/1n1jsVjuYYIEpWvb4KPnlFoSpuY1T9byJ/view?usp=drive_link](https://drive.google.com/file/d/1n1jsVjuYYIEpWvb4KPnlFoSpuY1T9byJ/view?usp=drive_link)

NVIDIA Nemotron-Orchestrator-8B Model Download From Huggingface (Place this in `~/models`):
[https://huggingface.co/Mungert/Nemotron-Orchestrator-8B-GGUF](https://huggingface.co/Mungert/Nemotron-Orchestrator-8B-GGUF)

## Usage

### Step 1: Start Your SQL Agent

This starts the Fine-Tuned (and Quantized) SQL Agent with an Agent2Agent (and REST interface).

Use:
- CPU if you are using a non-Apple based system
- MLX if you are using Apple Silicon

```bash
python sql_service.py
```

### Step 2: Start Your Python Coding Agent

This starts the Fine-Tuned (and Quantized) Python Coding Agent with an Agent2Agent (and REST interface).

Use:
- CPU if you are using a non-Apple based system
- MLX if you are using Apple Silicon

```bash
python python_service.py
```

### Step 3: Start Your Agentic Orcheestrator

This starts the Orchestrator Agent with an Agent2Agent (and REST interface). This mimics a CLI and doesn't need the client below.

```bash
python orchestrator_service.py
```

To start the orchestrator with a REST interface:

```bash
python orchestrator_service.py --serve
```

### (Optional) Step 4: Agentic Orcheestrator Using REST

If you started the Orchestrator Agent using the REST interface `--serve`, you can start the client below:

```bash
python orchestrator_client.py
```


