# Model Bridge: Chinese Interaction with English-Only LLMs

A translation bridge enabling Chinese conversation with English-only models (Phi-3) using ChatGLM as translator.

## Features

- CN->EN -> Phi-3 -> EN->CN pipeline
- Streaming token generation
- Dual-panel UI
- INT4 AWQ quantized Phi-3

## Tech Stack

- Python, ChatGLM.cpp, ONNX Runtime GenAI
- Phi-3 (DirectML), PySide6

## Setup

1. Install dependencies from requirements.txt (if available)
2. Download model weights (not included in this repo)
3. Run: python main.py

> Model weight files and datasets are not included due to size.
