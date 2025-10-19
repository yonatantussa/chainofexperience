# Contributing to Chain of Experience

Thank you for your interest in contributing! This guide will help you get started.

## Installation

### Prerequisites

- **Python 3.11+** - Download from [python.org](https://www.python.org/downloads/)
- **Poetry** - Python dependency manager
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
- **ffmpeg** - Video processing (for MP4 conversion)
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt-get install ffmpeg

  # Windows (using chocolatey)
  choco install ffmpeg
  ```

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ytussa/chainofexperience.git
   cd chainofexperience
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Install Playwright browsers**
   ```bash
   poetry run playwright install chromium
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=sk-your-api-key-here
   OPENAI_MODEL=gpt-4o-mini  # or gpt-4o for better performance
   ```

5. **Verify installation**
   ```bash
   poetry run python agent/run_agent.py
   ```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model to use for agent reasoning |

Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys).

## Project Structure

```
chainofexperience/
├── agent/
│   ├── run_agent.py        # Main entry point
│   ├── web_agent.py        # LangGraph agent
│   ├── browser_tools.py    # Playwright tools
│   ├── video_recorder.py   # Video recording
│   └── task_tracker.py     # SQLite logging
├── data/
│   ├── videos/             # Generated videos
│   └── tasks.db            # Task logs
├── README.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## Development

### Running the Agent

**Interactive mode:**
```bash
poetry run python agent/run_agent.py
```

**Command line:**
```bash
poetry run python agent/run_agent.py "Find the best wireless mouse under $50"
```

**Programmatic:**
```python
import asyncio
from agent.run_agent import run_consolidated_agent

async def main():
    result = await run_consolidated_agent(
        url="https://www.amazon.com",
        goal="Find the best wireless mouse under $50",
        max_steps=20,
        headless=False
    )
    print(f"Status: {result['status']}")
    print(f"Video: {result['video_path']}")

asyncio.run(main())
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public functions
- Keep functions focused and single-purpose

### Testing

Run tests before submitting:
```bash
poetry run pytest
```

## Common Issues

### "OPENAI_API_KEY not set"
Create a `.env` file with your API key:
```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### "Playwright not installed"
Install Playwright browsers:
```bash
poetry run playwright install chromium
```

### "ffmpeg not found"
Install ffmpeg for your platform (see Prerequisites above).

### Video conversion fails
- Ensure ffmpeg is in your PATH: `ffmpeg -version`
- Check output in `data/videos/` for error messages
- Try running without headless mode for debugging

### Agent gets stuck
- Increase `max_steps` to 30-50 for complex tasks
- Check the task logs in `data/tasks.db` for intervention data
- These failures become valuable training data!

## Contributing Code

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## Research Contributions

This project collects training data for learning when agents need intervention. You can contribute by:

1. **Running diverse tasks** - Try different websites and goals
2. **Logging interventions** - Note when agent gets stuck
3. **Sharing datasets** - Contribute anonymized task logs
4. **Testing models** - Help evaluate intervention prediction

See the [Research](#) section in README.md for details.

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Be respectful and constructive in discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
