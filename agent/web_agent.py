"""
Async LangGraph-based ReAct agent for browser automation.
Uses Playwright async API to avoid threading conflicts.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

from agent.browser_tools import BrowserTools

load_dotenv()


class AgentState(TypedDict):
    """State tracked throughout agent execution."""
    goal: str
    url: str
    session_id: str

    # Current state
    page_state: dict
    step: int
    max_steps: int

    # History
    actions_taken: list[str]
    observations: list[str]
    thoughts: list[str]

    # Completion
    is_complete: bool
    needs_intervention: bool
    next_action: str  # For passing action between nodes


class WebAgent:
    """
    Async LangGraph-based web browsing agent using ReAct pattern.

    Architecture:
        Plan → Act → Observe → Check → [Loop or End]
    """

    def __init__(self, browser_tools: BrowserTools, goal: str, url: str, session_id: str):
        """
        Initialize web agent.

        Args:
            browser_tools: Browser automation tools instance (async)
            goal: Agent's goal/task
            url: Starting URL
            session_id: Session identifier
        """
        self.tools = browser_tools
        self.goal = goal
        self.url = url
        self.session_id = session_id

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("check", self._check_node)

        # Define edges
        workflow.set_entry_point("observe")  # Start by observing initial state
        workflow.add_edge("observe", "check")

        # Conditional edge from check
        workflow.add_conditional_edges(
            "check",
            self._should_continue,
            {
                "plan": "plan",
                "complete": END,
                "intervention": END
            }
        )

        workflow.add_edge("plan", "act")
        workflow.add_edge("act", "observe")

        return workflow.compile()

    async def _plan_node(self, state: AgentState) -> dict:
        """
        Plan next action based on current state.
        Uses LLM to reason about what to do next.
        """
        print(f"\n🤔 Step {state['step']}/{state['max_steps']} - Planning...")

        # Build context for LLM
        system_prompt = """You are a web browsing agent. You can take actions to accomplish goals.

Available actions:
1. scroll_down - Scroll down 500px smoothly
2. scroll_up - Scroll up 500px smoothly
3. click:[text] - Click on element containing text (e.g., "click:Search" or "click:Python")
4. type:[text] - Type text into first visible input field (e.g., "type:Python programming")
5. press_enter - Press Enter key (submit form)
6. screenshot - Take screenshot of current view
7. go_back - Navigate to previous page
8. done - Goal is complete

You will see a list of interactive elements on the page with their text/labels.
Use the element text to target your click actions precisely.

Think step-by-step about:
1. What is the goal?
2. What's the current page state?
3. What interactive elements are visible?
4. What have I tried so far?
5. What should I do next to make progress?

Respond with:
THOUGHT: [Your reasoning about current situation and next step]
ACTION: [One action from the list above]

Examples:
- If you see an input field and want to search, use: type:Python programming
- If you want to submit a search, use: press_enter
- If you see a link with text "About", use: click:About
"""

        # Prepare context with enhanced page state
        page_state = state.get('page_state', {})
        basic_state = page_state.get('state', {})
        interactive = page_state.get('interactive_elements', [])
        accessibility = page_state.get('accessibility_tree', {})

        # Format interactive elements for LLM
        elements_text = ""
        if interactive:
            elements_text = "\n\nInteractive elements visible on page:"
            for i, elem in enumerate(interactive[:15]):  # Limit to 15 for token efficiency
                elem_desc = f"\n  [{i}] {elem.get('tag', '')} "
                if elem.get('text'):
                    elem_desc += f"'{elem['text'][:50]}'"
                elif elem.get('ariaLabel'):
                    elem_desc += f"aria-label='{elem['ariaLabel'][:50]}'"
                elif elem.get('placeholder'):
                    elem_desc += f"placeholder='{elem['placeholder'][:30]}'"
                if elem.get('href'):
                    elem_desc += f" -> {elem['href'][:60]}"
                elements_text += elem_desc

        # Format accessibility tree
        accessibility_text = ""
        if accessibility and accessibility.get('role'):
            accessibility_text = f"\n\nPage structure:\n{self._format_a11y_tree(accessibility, depth=0)}"

        # Show only last 10 actions to prevent context overflow
        recent_actions = state['actions_taken'][-10:]
        actions_text = chr(10).join(f"  {i+1}. {a}" for i, a in enumerate(recent_actions))
        if len(state['actions_taken']) > 10:
            actions_text = f"  ... ({len(state['actions_taken']) - 10} earlier actions)\n" + actions_text

        context = f"""Goal: {state['goal']}

Current page state:
- URL: {basic_state.get('url', 'unknown')}
- Title: {basic_state.get('title', 'unknown')}
- Scroll: {basic_state.get('scrollY', 0)}px / {basic_state.get('scrollHeight', 0)}px
- Visible text (first 500 chars):
{basic_state.get('visibleText', '')[:500]}{elements_text}{accessibility_text}

Recent actions:
{actions_text}

What should you do next?"""

        # Get LLM decision
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        # Parse response
        thought = ""
        action = "scroll_down"  # Default

        for line in response_text.split('\n'):
            if line.startswith("THOUGHT:"):
                thought = line.replace("THOUGHT:", "").strip()
            elif line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip().lower()

        print(f"💭 Thought: {thought[:100]}...")
        print(f"🎯 Action: {action}")

        # Return ONLY the keys we want to update
        return {
            'thoughts': state['thoughts'] + [thought],
            'next_action': action
        }

    async def _act_node(self, state: AgentState) -> dict:
        """
        Execute the planned action using browser tools.
        """
        action = state.get('next_action', 'scroll_down')

        print(f"⚡ Executing: {action}")

        result = None
        is_complete = False

        # Parse action with parameters
        if ":" in action:
            action_type, action_param = action.split(":", 1)
            action_type = action_type.strip().lower()
            action_param = action_param.strip()
        else:
            action_type = action.strip().lower()
            action_param = None

        # Execute action (all async now!)
        if action_type == "scroll_down" or action_type == "scroll":
            result = await self.tools.scroll_smooth(pixels=500)

        elif action_type == "scroll_up":
            result = await self.tools.scroll_smooth(pixels=-500)

        elif action_type == "click":
            if action_param:
                # Click element by text
                result = await self.tools.click_element(text=action_param)
            else:
                # Click first visible element
                result = await self.tools.click_element()

        elif action_type == "type":
            if action_param:
                result = await self.tools.type_text(text=action_param)
            else:
                result = {"action": "type", "success": False, "error": "No text provided"}

        elif action_type == "press_enter":
            result = await self.tools.press_key(key="Enter")

        elif action_type == "screenshot":
            result = await self.tools.take_screenshot()

        elif action_type == "go_back":
            result = await self.tools.go_back()

        elif action_type == "done":
            result = {"action": "done", "success": True}
            is_complete = True

        else:
            # Default: scroll down
            result = await self.tools.scroll_smooth(pixels=500)

        # Log action
        action_log = f"{action}: {result.get('success', False)}"

        if result and result.get('success'):
            print(f"✅ {action} succeeded")
        else:
            print(f"⚠️  {action} failed: {result.get('error', 'unknown')}")

        # Return ONLY the keys we want to update
        return {
            'actions_taken': state['actions_taken'] + [action_log],
            'step': state['step'] + 1,
            'is_complete': is_complete
        }

    async def _observe_node(self, state: AgentState) -> dict:
        """
        Observe current page state after action.
        """
        print(f"👁️  Observing page state...")

        # Get current page state (async!)
        page_state = await self.tools.get_page_state()

        if page_state.get('success'):
            # Create observation summary
            obs = f"On page: {page_state['state']['title']}, scrolled to {page_state['state']['scrollY']}px"

            print(f"📊 {obs}")

            # Return ONLY the keys we want to update
            return {
                'page_state': page_state,
                'observations': state['observations'] + [obs]
            }
        else:
            print(f"⚠️  Failed to observe: {page_state.get('error')}")
            return {}

    async def _check_node(self, state: AgentState) -> dict:
        """
        Check if task is complete or needs intervention.
        """
        needs_intervention = False
        is_complete = state.get('is_complete', False)

        # Check max steps
        if state['step'] >= state['max_steps']:
            print(f"\n⏱️  Max steps ({state['max_steps']}) reached")
            needs_intervention = True

        # Check if marked as done by agent
        if is_complete:
            print(f"\n✅ Agent marked task as complete")

        # Smart completion detection: check if goal keywords appear on page
        if not is_complete and state.get('page_state'):
            page_text = state['page_state'].get('state', {}).get('visibleText', '').lower()
            goal_lower = state['goal'].lower()

            # Extract key terms from goal (remove common words)
            stop_words = {'find', 'search', 'look', 'get', 'show', 'the', 'a', 'an', 'for', 'to', 'in', 'on', 'about', 'information'}
            goal_keywords = [word for word in goal_lower.split() if word not in stop_words and len(word) > 3]

            # Check if majority of keywords are present
            if goal_keywords:
                matches = sum(1 for keyword in goal_keywords if keyword in page_text)
                keyword_ratio = matches / len(goal_keywords)

                if keyword_ratio >= 0.7:  # 70% of keywords present
                    print(f"\n🎯 Goal keywords found on page ({int(keyword_ratio*100)}% match)")

        # Detect if stuck (repeated actions)
        if len(state['actions_taken']) >= 5:
            recent_actions = state['actions_taken'][-5:]
            # Check if all recent actions are the same
            if len(set(recent_actions)) == 1:
                print(f"\n🔄 Stuck: Repeating same action 5 times")
                needs_intervention = True

        # Return ONLY the keys we want to update
        return {
            'needs_intervention': needs_intervention
        }

    def _format_a11y_tree(self, node: dict, depth: int = 0, max_depth: int = 2) -> str:
        """Format accessibility tree for LLM context."""
        if not node or depth > max_depth:
            return ""

        indent = "  " * depth
        role = node.get('role', '')
        name = node.get('name', '')

        if not role and not name:
            return ""

        result = f"{indent}- {role}"
        if name:
            result += f": '{name[:60]}'"
        result += "\n"

        # Add children
        children = node.get('children', [])
        for child in children[:5]:  # Limit to 5 children
            result += self._format_a11y_tree(child, depth + 1, max_depth)

        return result

    def _should_continue(self, state: AgentState) -> Literal["plan", "complete", "intervention"]:
        """
        Decide whether to continue or end.
        """
        if state.get('is_complete'):
            return "complete"

        if state.get('needs_intervention'):
            return "intervention"

        if state['step'] >= state['max_steps']:
            return "intervention"

        return "plan"

    async def run(self, max_steps: int = 10) -> dict:
        """
        Run the agent until completion or intervention needed.

        Args:
            max_steps: Maximum steps before stopping

        Returns:
            Final state with results
        """
        print(f"\n{'='*70}")
        print(f"🤖 Web Agent Starting (Async + LangGraph)")
        print(f"{'='*70}")
        print(f"Session: {self.session_id}")
        print(f"Goal: {self.goal}")
        print(f"URL: {self.url}")
        print(f"Max steps: {max_steps}")
        print(f"{'='*70}\n")

        # Initialize state
        initial_state: AgentState = {
            "goal": self.goal,
            "url": self.url,
            "session_id": self.session_id,
            "page_state": {},
            "step": 0,
            "max_steps": max_steps,
            "actions_taken": [],
            "observations": [],
            "thoughts": [],
            "is_complete": False,
            "needs_intervention": False,
            "next_action": ""
        }

        # Navigate to starting URL (async!)
        print(f"🌍 Navigating to {self.url}...")
        await self.tools.page.goto(self.url, wait_until="domcontentloaded")

        # Run graph (async invoke!)
        final_state = await self.graph.ainvoke(
            initial_state,
            config={"recursion_limit": max_steps * 5}
        )

        # Print summary
        print(f"\n{'='*70}")
        print(f"🏁 Agent Finished")
        print(f"{'='*70}")
        print(f"Total steps: {final_state['step']}")
        print(f"Completed: {final_state.get('is_complete', False)}")
        print(f"Needed intervention: {final_state.get('needs_intervention', False)}")
        print(f"\n📝 Actions taken:")
        for i, action in enumerate(final_state['actions_taken'], 1):
            print(f"  {i}. {action}")
        print(f"{'='*70}\n")

        return final_state
