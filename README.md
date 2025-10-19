# Chain of Experience

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Learning from human-computer interaction histories to predict when AI agents need intervention**

## Goal

Train agents to recognize when they're about to make mistakes by learning from patterns in successful vs. failed interaction sequences.

**Context**: Prior work (e.g., memory-augmented LLM agents, generative human-behavior simulators) shows that agents can integrate past interactions into behavior. However, they lack full capture of multimodal human-agent interaction traces, correction loops, and lightweight adaptation in UI contexts. Chain of Experience fills this gap by logging UI state (DOM structure, screenshots, interactive elements), recording user interventions and corrections, and training experience encoders for agents to learn from how users interact with them, not just what they say.

## Quick Start

### Run the Agent

```bash
python agent/run_agent.py
```

This runs the agent with:
- LangGraph + Async Playwright
- MP4 video recording with visible cursor
- Intervention logging in SQLite
- ReAct pattern (Reasoning → Action → Observation)
- Observation (interactive elements + accessibility tree)

## Overview

### The Problem
AI agents can get stuck or make mistakes without realizing it. They need to know when to ask for human help, or even better, avoid those mistakes accordingly.

### Approach
1. **Collect interaction data** - Record agent browsing sessions with interventions
2. **Learn patterns** - Train models to recognize pre-intervention states
3. **Predict interventions** - Detect when agent is about to need help
4. **Improve autonomy** - Agents learn when to ask vs. when to continue

Instead of just recording what the agent did, we record when humans would have intervened, creating training data for learning failure patterns.

## Agent

### Architecture
- **LangGraph framework**
- **Async Playwright**
- **ReAct pattern** - Think -> Act -> Observe loop
- **Task tracking** - SQLite database logging

### Video Recording
- Visible cursor that persists across pages
- Smooth scrolling and element highlighting
- WebM -> MP4 conversion

### Data Collection
- Intervention logging
- Page state at each step
- Action history tracking
- Success/failure metrics

## Output

Running the agent produces:
1. **MP4 videos** (`data/videos/`) - For demos and papers
2. **Task logs** (`data/tasks.db`) - For training models
3. **Statistics** - Intervention patterns and success rates

## Installation

```bash
# Install dependencies
poetry install

# Install Playwright browsers
poetry run playwright install chromium

# Install ffmpeg for video conversion
brew install ffmpeg  # macOS
# or: apt-get install ffmpeg  # Linux
```

## Documentation

See **[agent/README.md](agent/README.md)** for full documentation.

## Example Tasks

Try these to see what the agent can do:

**E-commerce:**
- "Find the best wireless mouse under $50"
- "Compare noise-cancelling headphones on Amazon"
- "Look for a mirrorless camera with good reviews"

**Research:**
- "Find the latest Python release notes"
- "Search for papers about LLMs on arXiv"
- "Look up Playwright documentation"

**Information:**
- "Get the weather forecast for New York"
- "Find top-rated pizza restaurants in Chicago"
- "Search for news about AI developments"

**Navigation:**
- "Find the PyTorch installation guide"
- "Look up LangGraph tutorials"
- "Search for async Playwright examples"

## Usage

```python
import asyncio
from agent.run_agent import run_consolidated_agent

async def main():
    result = await run_consolidated_agent(
        url="https://en.wikipedia.org",
        goal="Find information about Python programming language",
        max_steps=20,
        headless=False
    )
    print(f"Status: {result['status']}")
    print(f"Video: {result['video_path']}")

asyncio.run(main())
```

##  Research & Training

### Phase 1: Collecting Training Data

Run the agent on diverse tasks to build a dataset of intervention patterns:

```bash
# Initialize database
poetry run python scripts/init_db.py

# Run agent on various tasks
poetry run python agent/run_agent.py "Find the best wireless mouse"
poetry run python agent/run_agent.py "Search for Python documentation"
# ... run 20-50 diverse tasks

# Export training data
poetry run python scripts/export_training_data.py
```

This creates:
- **Intervention states** - When agent gets stuck or needs help
- **Success states** - When agent completes tasks smoothly
- **Video recordings** - Visual context for each state
- **Page states** - DOM structure and interaction history

### Phase 2: Training Models

Train the intervention prediction model using contrastive learning:

```bash
# Train (2-phase training)
poetry run python train.py --epochs-contrastive 20 --epochs-predictor 10

# Evaluate
poetry run python evaluate.py
```

**Model Architecture:**
1. **Experience Encoder** - Multimodal encoder combining:
   - Visual (CLIP on screenshots/video frames)
   - Actions (transformer over action sequences)
   - Page state (DOM structure, accessibility tree)

2. **Contrastive Learning** - Learn to distinguish:
   - Similar states (both successful or both needing intervention)
   - Dissimilar states (successful vs. intervention needed)

3. **Intervention Predictor** - Binary classifier predicting when agent needs help

### Evaluation Metrics

The evaluation script computes:
- **Accuracy** - Overall prediction correctness
- **Precision/Recall** - Intervention prediction quality
- **F1 Score** - Harmonic mean of precision/recall
- **ROC AUC** - Model discrimination ability
- **Confusion Matrix** - Breakdown of prediction types

Results are saved to `data/visualizations/` with plots and metrics.

## References

- LangGraph: https://github.com/langchain-ai/langgraph
- Playwright: https://playwright.dev
- ReAct Pattern: https://arxiv.org/abs/2210.03629
- talk2browser: https://github.com/talk2silicon/talk2browser

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: Yonatan Tussa
