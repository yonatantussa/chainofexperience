"""
Main entry point for running the async agent.
Combines video recording, async LangGraph agent, and task tracking.
"""

import sys
import asyncio
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agent.video_recorder import VideoRecorder
from agent.browser_tools import BrowserTools
from agent.web_agent import WebAgent
from agent.task_tracker import TaskTracker


def check_environment():
    """Check if required environment variables and dependencies are set."""
    issues = []

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append(
            "❌ OPENAI_API_KEY not set\n"
            "   Fix: Add to .env file or export OPENAI_API_KEY=sk-..."
        )

    # Check if playwright is installed
    try:
        import playwright
    except ImportError:
        issues.append(
            "❌ Playwright not installed\n"
            "   Fix: poetry run playwright install chromium"
        )

    # Check if ffmpeg is available
    import shutil
    if not shutil.which("ffmpeg"):
        issues.append(
            "⚠️  ffmpeg not found (video conversion will fail)\n"
            "   Fix: brew install ffmpeg  # macOS\n"
            "        sudo apt-get install ffmpeg  # Linux"
        )

    if issues:
        print("\n" + "="*70)
        print("Environment Check Failed")
        print("="*70 + "\n")
        for issue in issues:
            print(issue + "\n")
        print("See CONTRIBUTING.md for setup instructions.")
        print("="*70 + "\n")
        return False

    return True


async def run_consolidated_agent(
    url: str,
    goal: str,
    session_id: str = None,
    max_steps: int = 10,
    headless: bool = False
):
    """
    Run the async consolidated publication-ready agent.

    This combines:
    - Video recording with cursor and smooth visuals
    - Async LangGraph ReAct agent with proper reasoning
    - Task tracking for measuring success/failure
    - Intervention logging for training data

    Args:
        url: Starting URL
        goal: Task goal/objective
        session_id: Session identifier (generated if not provided)
        max_steps: Maximum steps before stopping
        headless: Run browser in headless mode

    Returns:
        dict with results and paths
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*70}")
    print(f"Agent (Async + LangGraph)")
    print(f"{'='*70}")
    print(f"Session: {session_id}")
    print(f"URL: {url}")
    print(f"Goal: {goal}")
    print(f"Max steps: {max_steps}")
    print(f"Headless: {headless}")
    print(f"{'='*70}\n")

    # Initialize components
    task_tracker = TaskTracker()
    video_recorder = VideoRecorder(session_id=session_id, headless=headless)

    # Track task start
    task_tracker.start_task(session_id=session_id, goal=goal, url=url)

    try:
        # Start video recording and get page (async!)
        page = await video_recorder.start()

        # Initialize browser tools (async!)
        browser_tools = BrowserTools(page=page, session_id=session_id)

        # Create agent (async!)
        agent = WebAgent(
            browser_tools=browser_tools,
            goal=goal,
            url=url,
            session_id=session_id
        )

        # Run agent (async!)
        final_state = await agent.run(max_steps=max_steps)

        # Determine status
        if final_state.get('is_complete'):
            status = 'complete'
        elif final_state.get('needs_intervention'):
            status = 'intervention'

            # Log intervention
            task_tracker.log_intervention(
                session_id=session_id,
                step=final_state['step'],
                reason='max_steps' if final_state['step'] >= max_steps else 'agent_stuck',
                page_state=final_state.get('page_state', {}),
                actions_before=final_state.get('actions_taken', [])
            )
        else:
            status = 'failed'

        # Stop video recording (async!)
        video_path = await video_recorder.stop()

        # Track task end
        task_tracker.end_task(
            session_id=session_id,
            status=status,
            total_steps=final_state['step'],
            video_path=video_path
        )

        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 Session Summary")
        print(f"{'='*70}")
        print(f"Status: {status}")
        print(f"Steps taken: {final_state['step']}")
        print(f"Video: {video_path}")
        print(f"{'='*70}\n")

        # Print overall stats
        task_tracker.print_stats()

        return {
            'session_id': session_id,
            'status': status,
            'steps': final_state['step'],
            'video_path': video_path,
            'final_state': final_state
        }

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

        # Try to save video anyway
        try:
            video_path = await video_recorder.stop()
        except:
            video_path = None

        # Track as failed
        task_tracker.end_task(
            session_id=session_id,
            status='failed',
            total_steps=0,
            video_path=video_path
        )

        return {
            'session_id': session_id,
            'status': 'failed',
            'error': str(e),
            'video_path': video_path
        }


async def choose_website_for_goal(goal: str) -> str:
    """
    Use LLM to determine the best website for a given goal.

    Args:
        goal: The user's goal/task

    Returns:
        Best URL to start from
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    import os

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    messages = [
        SystemMessage(content="""You are a helpful assistant that determines the best website to visit for a given task.

Respond with ONLY the URL, nothing else. Choose the most relevant website.

Examples:
- "Find a camera to buy" → https://www.amazon.com
- "Look up Python documentation" → https://docs.python.org
- "Find weather in Tokyo" → https://weather.com
- "Search for news about AI" → https://news.google.com
- "Find restaurants in SF" → https://www.yelp.com
- "General web search" → https://www.google.com
"""),
        HumanMessage(content=f"What website should I visit to: {goal}")
    ]

    response = await llm.ainvoke(messages)
    url = response.content.strip()

    # Ensure it's a valid URL
    if not url.startswith("http"):
        url = f"https://{url}"

    return url


# Example usage
async def main():
    """Main entry point for async execution."""
    # Check environment first
    if not check_environment():
        sys.exit(1)

    # Interactive prompting
    print("\n" + "="*70)
    print("Chain of Experience - Web Agent")
    print("="*70 + "\n")

    # Get goal from user
    if len(sys.argv) > 1:
        # Goal provided as command line argument
        goal = " ".join(sys.argv[1:])
    else:
        # Interactive prompt
        goal = input("What would you like the agent to do?\n> ").strip()

    if not goal:
        print("❌ No goal provided. Exiting.")
        return

    # Get URL from user (optional)
    url_input = input("\nStarting URL (press Enter to let agent choose): ").strip()

    if url_input:
        url = url_input
        if not url.startswith("http"):
            url = f"https://{url}"
    else:
        print(f"\n🔍 Choosing best website for: '{goal}'...")
        url = await choose_website_for_goal(goal)
        print(f"✅ Selected: {url}\n")

    # Get max steps (optional)
    steps_input = input("Max steps (press Enter for default 20): ").strip()
    max_steps = int(steps_input) if steps_input.isdigit() else 20

    # Get headless mode
    headless_input = input("Run headless (no visible window)? (y/N): ").strip().lower()
    headless = headless_input in ['y', 'yes']

    print("\n" + "="*70)
    print("Starting agent...")
    print("="*70 + "\n")

    # Run agent
    result = await run_consolidated_agent(
        url=url,
        goal=goal,
        max_steps=max_steps,
        headless=headless
    )

    print(f"\n✅ Agent finished!")
    print(f"Session: {result['session_id']}")
    print(f"Status: {result['status']}")
    print(f"Video: {result.get('video_path')}")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
