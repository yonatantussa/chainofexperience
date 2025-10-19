# Agent Documentation

Web browsing agent built with LangGraph and Playwright for multimodal interaction trace collection.

## Quick Start

```bash
python agent/run_agent.py
```

Interactive mode prompts for task goal and automatically selects optimal website.

## Overview

Research-grade web automation agent featuring:
- Async Playwright + LangGraph for robust state management
- MP4 video recording with cursor tracking
- ReAct pattern (Reasoning → Action → Observation)
- Enhanced observation (interactive elements + accessibility tree)
- SQLite logging of task outcomes and intervention points

## Architecture

```
Plan → Act → Observe → Check → [Loop]
         ↓
   Playwright Browser (async + video)
         ↓
   SQLite Logger (interventions)
```

**Files:**
- `web_agent.py` - LangGraph ReAct agent
- `browser_tools.py` - Browser automation tools
- `video_recorder.py` - Video recording with cursor
- `task_tracker.py` - SQLite intervention logging
- `run_agent.py` - Interactive CLI entry point

## Usage

```python
import asyncio
from agent.run_agent import run_consolidated_agent

async def main():
    result = await run_consolidated_agent(
        url="https://www.python.org",
        goal="Find the latest Python release version",
        max_steps=15,
        headless=False
    )
    print(f"Status: {result['status']}")
    print(f"Video: {result['video_path']}")

asyncio.run(main())
```

**Output:**
- MP4 video recording (`data/videos/`)
- Task logs in SQLite (`data/tasks.db`)
- Success/failure statistics

## Main Features

**LangGraph Execution**
- Async Playwright API
- Proper state management and graph execution

**Video Recording**
- Visible cursor persisting across pages
- Element highlighting on interactions
- WebM to MP4 conversion

**Data Collection**
- Action and observation tracking
- Intervention point logging (when agent needs help)
- Page state capture at each step

## Technical Notes

**ReAct Pattern**: Agent follows Reasoning + Acting loop:
1. Plan - LLM decides next action
2. Act - Execute browser action
3. Observe - Capture new page state
4. Check - Evaluate completion or intervention need

**Dependencies:**
- `playwright` - Browser automation
- `langchain-openai` - LLM reasoning
- `langgraph` - Agent framework
- `ffmpeg` - Video conversion
