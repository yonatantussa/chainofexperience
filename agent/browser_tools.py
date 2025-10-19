"""
Async browser automation tools that work with LangGraph and video recording.
Modular, testable, and visually smooth.
"""

from playwright.async_api import Page
from pathlib import Path
import asyncio
import time


class BrowserTools:
    """
    Encapsulates all browser automation tools (async version).
    Maintains reference to the active page.
    """

    def __init__(self, page: Page, session_id: str):
        """
        Initialize browser tools.

        Args:
            page: Active Playwright page (async)
            session_id: Unique session identifier
        """
        self.page = page
        self.session_id = session_id
        self.screenshot_dir = Path("data/screens")
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def scroll_smooth(self, pixels: int = 500) -> dict:
        """
        Smooth scroll by specified pixels.

        Args:
            pixels: Pixels to scroll (positive = down, negative = up)

        Returns:
            dict with action result
        """
        direction = "down" if pixels > 0 else "up"

        await self.page.evaluate(f"""
            () => {{
                window.scrollTo({{
                    top: window.pageYOffset + {pixels},
                    behavior: 'smooth'
                }});
            }}
        """)

        await asyncio.sleep(1.5)  # Wait for smooth scroll

        return {
            "action": "scroll",
            "direction": direction,
            "pixels": abs(pixels),
            "success": True
        }

    async def scroll_to_element(self, selector: str) -> dict:
        """Scroll element into view smoothly."""
        try:
            await self.page.locator(selector).first.scroll_into_view_if_needed()
            await asyncio.sleep(0.5)
            return {"action": "scroll_to_element", "selector": selector, "success": True}
        except Exception as e:
            return {"action": "scroll_to_element", "selector": selector, "success": False, "error": str(e)}

    async def click_element(self, text: str = None, selector: str = None, element_index: int = None) -> dict:
        """
        Click an element with smooth cursor movement and highlighting.

        Args:
            text: Text content to match (searches visible links)
            selector: CSS selector (if specific element known)
            element_index: Index from interactive_elements list (recommended)

        Returns:
            dict with click result
        """
        try:
            if selector:
                # Direct selector click
                element = self.page.locator(selector).first
                box = await element.bounding_box()

                if box:
                    x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
                    await self._move_cursor_to(x, y)
                    await self._highlight_element(selector=selector)
                    await self.page.mouse.click(x, y)
                    await asyncio.sleep(1)

                    return {
                        "action": "click",
                        "selector": selector,
                        "success": True
                    }

            else:
                # Find visible clickable elements
                clickable = await self.page.evaluate("""
                    () => {
                        const elements = [];
                        document.querySelectorAll('a[href], button').forEach(el => {
                            const rect = el.getBoundingClientRect();
                            const text = el.innerText.trim();

                            if (rect.width > 0 && rect.height > 0 &&
                                rect.top >= 0 && rect.top < window.innerHeight &&
                                text.length > 0) {
                                elements.push({
                                    text: text,
                                    href: el.href || '',
                                    x: rect.left + rect.width / 2,
                                    y: rect.top + rect.height / 2
                                });
                            }
                        });
                        return elements.slice(0, 10);
                    }
                """)

                if not clickable:
                    return {"action": "click", "success": False, "error": "No clickable elements found"}

                # Pick first match (or by text if provided)
                target = clickable[0]
                if text:
                    for el in clickable:
                        if text.lower() in el['text'].lower():
                            target = el
                            break

                # Click with visual feedback
                await self._move_cursor_to(target['x'], target['y'])
                await self._highlight_at_coords(target['x'], target['y'])
                await self.page.mouse.click(target['x'], target['y'])
                await asyncio.sleep(2)  # Wait for navigation

                return {
                    "action": "click",
                    "text": target['text'][:50],
                    "success": True
                }

        except Exception as e:
            return {"action": "click", "success": False, "error": str(e)}

    async def type_text(self, text: str, selector: str = None) -> dict:
        """
        Type text into an input field.

        Args:
            text: Text to type
            selector: CSS selector for input field (defaults to first visible input)

        Returns:
            dict with typing result
        """
        try:
            if selector:
                # Type into specific selector
                await self.page.locator(selector).first.fill(text)
            else:
                # Find first visible text/search input or textarea
                # Use more specific selector to avoid checkboxes, radio buttons, etc.
                text_inputs = self.page.locator('input[type="text"], input[type="search"], input:not([type]), textarea')

                # Filter to visible inputs only
                count = await text_inputs.count()
                input_found = None

                for i in range(count):
                    elem = text_inputs.nth(i)
                    if await elem.is_visible():
                        input_found = elem
                        break

                if not input_found:
                    return {"action": "type_text", "success": False, "error": "No visible text input found"}

                # Type into the found input
                await input_found.fill(text)

            await asyncio.sleep(0.5)
            return {
                "action": "type_text",
                "text": text[:50],
                "success": True
            }
        except Exception as e:
            return {"action": "type_text", "success": False, "error": str(e)}

    async def press_key(self, key: str = "Enter") -> dict:
        """
        Press a keyboard key.

        Args:
            key: Key to press (e.g., "Enter", "Tab", "Escape")

        Returns:
            dict with key press result
        """
        try:
            await self.page.keyboard.press(key)
            await asyncio.sleep(1)
            return {
                "action": "press_key",
                "key": key,
                "success": True
            }
        except Exception as e:
            return {"action": "press_key", "success": False, "error": str(e)}

    async def take_screenshot(self, name: str = None) -> dict:
        """Take a screenshot of current viewport."""
        try:
            if not name:
                name = f"{self.session_id}_{int(time.time()*1000)}.png"

            path = self.screenshot_dir / name
            await self.page.screenshot(path=str(path))

            return {
                "action": "screenshot",
                "path": str(path),
                "success": True
            }
        except Exception as e:
            return {"action": "screenshot", "success": False, "error": str(e)}

    async def get_page_state(self, include_screenshot: bool = False) -> dict:
        """
        Get comprehensive page state for agent observation.

        Args:
            include_screenshot: Whether to include a screenshot (adds latency)

        Returns:
            dict with page state including:
            - Basic info (URL, title, scroll position)
            - Interactive elements (links, buttons, inputs)
            - Accessibility tree (structured page content)
            - Screenshot (optional)
        """
        try:
            # 1. Get basic page info
            basic_state = await self.page.evaluate("""
                () => ({
                    url: window.location.href,
                    title: document.title,
                    scrollY: window.pageYOffset,
                    scrollHeight: document.body.scrollHeight,
                    viewportHeight: window.innerHeight,
                    visibleText: document.body.innerText.substring(0, 2000)
                })
            """)

            # 2. Get interactive elements (similar to talk2browser)
            interactive_elements = await self._get_interactive_elements()

            # 3. Get accessibility tree (structured content)
            accessibility_tree = await self._get_accessibility_snapshot()

            # 4. Optional: Take screenshot
            screenshot_path = None
            if include_screenshot:
                screenshot_path = await self._take_observation_screenshot()

            return {
                "action": "observe",
                "state": basic_state,
                "interactive_elements": interactive_elements,
                "accessibility_tree": accessibility_tree,
                "screenshot_path": screenshot_path,
                "success": True
            }
        except Exception as e:
            return {"action": "observe", "success": False, "error": str(e)}

    async def _get_interactive_elements(self) -> list[dict]:
        """
        Get all interactive elements currently visible in viewport.
        Returns element type, text, position, and attributes.
        """
        try:
            elements = await self.page.evaluate("""
                () => {
                    const elements = [];
                    const selectors = 'a[href], button, input, select, textarea, [role="button"], [role="link"]';

                    document.querySelectorAll(selectors).forEach((el, index) => {
                        const rect = el.getBoundingClientRect();

                        // Only include visible elements in viewport
                        if (rect.width > 0 && rect.height > 0 &&
                            rect.top >= 0 && rect.top < window.innerHeight) {

                            const text = el.innerText?.trim() || el.textContent?.trim() || '';
                            const value = el.value || '';

                            elements.push({
                                index: index,
                                tag: el.tagName.toLowerCase(),
                                type: el.type || '',
                                text: text.substring(0, 100),
                                value: value.substring(0, 100),
                                href: el.href || '',
                                ariaLabel: el.getAttribute('aria-label') || '',
                                placeholder: el.placeholder || '',
                                position: {
                                    x: Math.round(rect.left + rect.width / 2),
                                    y: Math.round(rect.top + rect.height / 2)
                                },
                                boundingBox: {
                                    x: Math.round(rect.left),
                                    y: Math.round(rect.top),
                                    width: Math.round(rect.width),
                                    height: Math.round(rect.height)
                                },
                                isVisible: true
                            });
                        }
                    });

                    return elements.slice(0, 50); // Limit to 50 elements
                }
            """)
            return elements
        except Exception as e:
            print(f"⚠️  Failed to get interactive elements: {e}")
            return []

    async def _get_accessibility_snapshot(self) -> dict:
        """
        Get accessibility tree snapshot for structured page understanding.
        Returns a tree of semantic elements (headings, landmarks, etc.).
        """
        try:
            # Get accessibility snapshot from Playwright
            snapshot = await self.page.accessibility.snapshot()

            # Simplify the tree to reduce token usage
            simplified = self._simplify_accessibility_tree(snapshot)
            return simplified
        except Exception as e:
            print(f"⚠️  Failed to get accessibility tree: {e}")
            return {}

    def _simplify_accessibility_tree(self, node: dict, depth: int = 0, max_depth: int = 3) -> dict:
        """
        Simplify accessibility tree to reduce token usage.
        Keep only important semantic information.
        """
        if not node or depth > max_depth:
            return {}

        # Only keep nodes with meaningful roles or names
        important_roles = {
            'heading', 'link', 'button', 'textbox', 'search', 'navigation',
            'main', 'article', 'region', 'form', 'list', 'listitem'
        }

        role = node.get('role', '')
        name = node.get('name', '')

        # Skip unimportant nodes
        if not role and not name:
            return {}

        if role not in important_roles and not name:
            return {}

        result = {
            'role': role,
            'name': name[:100] if name else '',  # Limit text length
        }

        # Process children
        children = node.get('children', [])
        if children and depth < max_depth:
            simplified_children = []
            for child in children[:10]:  # Limit to 10 children per node
                simplified_child = self._simplify_accessibility_tree(child, depth + 1, max_depth)
                if simplified_child:
                    simplified_children.append(simplified_child)

            if simplified_children:
                result['children'] = simplified_children

        return result

    async def _take_observation_screenshot(self) -> str:
        """Take a screenshot for observation (separate from action screenshots)."""
        try:
            timestamp = int(time.time() * 1000)
            name = f"{self.session_id}_obs_{timestamp}.png"
            path = self.screenshot_dir / name
            await self.page.screenshot(path=str(path))
            return str(path)
        except Exception as e:
            print(f"⚠️  Failed to take observation screenshot: {e}")
            return None

    async def go_back(self) -> dict:
        """Navigate back to previous page."""
        try:
            await self.page.go_back(wait_until="domcontentloaded", timeout=5000)
            await asyncio.sleep(1)
            return {"action": "go_back", "success": True}
        except Exception as e:
            return {"action": "go_back", "success": False, "error": str(e)}

    async def _move_cursor_to(self, x: float, y: float):
        """Smoothly move cursor to coordinates."""
        await self.page.mouse.move(x, y, steps=10)
        await asyncio.sleep(0.3)

    async def _highlight_element(self, selector: str = None, href: str = None):
        """Highlight element being interacted with."""
        try:
            if selector:
                await self.page.evaluate(f"""
                    () => {{
                        const el = document.querySelector('{selector}');
                        if (el) {{
                            el.style.outline = '3px solid red';
                            el.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
                            setTimeout(() => {{
                                el.style.outline = '';
                                el.style.backgroundColor = '';
                            }}, 500);
                        }}
                    }}
                """)
            elif href:
                await self.page.evaluate(f"""
                    () => {{
                        const el = document.querySelector('a[href="{href}"]');
                        if (el) {{
                            el.style.outline = '3px solid red';
                            el.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
                            setTimeout(() => {{
                                el.style.outline = '';
                                el.style.backgroundColor = '';
                            }}, 500);
                        }}
                    }}
                """)
        except:
            pass

    async def _highlight_at_coords(self, x: float, y: float):
        """Create temporary highlight at coordinates."""
        try:
            await self.page.evaluate(f"""
                () => {{
                    const highlight = document.createElement('div');
                    highlight.style.cssText = `
                        position: fixed;
                        left: {x - 30}px;
                        top: {y - 30}px;
                        width: 60px;
                        height: 60px;
                        border: 3px solid red;
                        border-radius: 50%;
                        pointer-events: none;
                        z-index: 999998;
                        animation: pulse 0.5s ease-out;
                    `;
                    document.body.appendChild(highlight);
                    setTimeout(() => highlight.remove(), 500);
                }}
            """)
        except:
            pass
