"""
Async video recording wrapper for Playwright.
Handles video recording, cursor injection, and MP4 conversion.
"""

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright
from pathlib import Path
import shutil
import subprocess
import asyncio


class VideoRecorder:
    """
    Manages browser video recording with cursor and visual enhancements (async version).
    """

    def __init__(self, session_id: str, headless: bool = False):
        """
        Initialize video recorder.

        Args:
            session_id: Unique session identifier
            headless: Run browser in headless mode
        """
        self.session_id = session_id
        self.headless = headless
        self.video_dir = Path("data/videos")
        self.temp_dir = Path("data/videos/temp")
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.playwright: Playwright = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.page: Page = None

    async def start(self) -> Page:
        """
        Start browser with video recording enabled.

        Returns:
            Playwright Page object ready for use
        """
        self.playwright = await async_playwright().start()

        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=800  # Slow down for visibility
        )

        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            record_video_dir=str(self.temp_dir),
            record_video_size={"width": 1280, "height": 800}
        )

        self.page = await self.context.new_page()

        # Set up cursor injection on page loads
        self.page.on("load", lambda: asyncio.create_task(self._inject_cursor()))
        self.page.on("domcontentloaded", lambda: asyncio.create_task(self._inject_cursor()))

        print(f"🎥 Video recording started")
        print(f"📹 Browser: {'Headless' if self.headless else 'Visible'}")

        return self.page

    async def stop(self) -> str:
        """
        Stop recording and save video.

        Returns:
            Path to saved MP4 video file
        """
        print(f"\n🎬 Stopping video recording...")

        # Close browser to finalize video
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

        # Find and process video
        video_files = list(self.temp_dir.glob("*.webm"))

        if not video_files:
            print("⚠️  No video file created")
            return None

        webm_file = video_files[0]
        webm_path = self.video_dir / f"{self.session_id}.webm"
        shutil.move(str(webm_file), str(webm_path))

        # Convert to MP4
        mp4_path = self._convert_to_mp4(webm_path)

        if mp4_path:
            print(f"\n✅ Video saved: {mp4_path}")
            print(f"📁 Size: {mp4_path.stat().st_size / 1024 / 1024:.1f} MB")
            return str(mp4_path)
        else:
            print(f"\n⚠️  Keeping WebM format: {webm_path}")
            return str(webm_path)

    async def _inject_cursor(self):
        """Inject visible cursor into the page."""
        try:
            await self.page.evaluate("""
                () => {
                    // Remove old cursor if exists
                    const oldCursor = document.getElementById('ai-cursor');
                    if (oldCursor) oldCursor.remove();

                    // Create cursor element
                    const cursor = document.createElement('div');
                    cursor.id = 'ai-cursor';
                    cursor.style.cssText = `
                        position: fixed;
                        width: 24px;
                        height: 24px;
                        border: 4px solid #ff0000;
                        border-radius: 50%;
                        pointer-events: none;
                        z-index: 999999;
                        transition: all 0.2s ease;
                        box-shadow: 0 0 15px rgba(255, 0, 0, 0.8);
                        background: rgba(255, 0, 0, 0.2);
                    `;
                    document.body.appendChild(cursor);

                    // Update cursor position on mouse move
                    document.addEventListener('mousemove', (e) => {
                        cursor.style.left = e.clientX - 12 + 'px';
                        cursor.style.top = e.clientY - 12 + 'px';
                    });

                    // Pulse effect on click
                    document.addEventListener('click', () => {
                        cursor.style.transform = 'scale(1.8)';
                        cursor.style.borderWidth = '6px';
                        setTimeout(() => {
                            cursor.style.transform = 'scale(1)';
                            cursor.style.borderWidth = '4px';
                        }, 300);
                    });
                }
            """)
        except Exception as e:
            # Page might not be ready, that's OK
            pass

    def _convert_to_mp4(self, webm_path: Path) -> Path:
        """
        Convert WebM to MP4 using ffmpeg.

        Args:
            webm_path: Path to WebM file

        Returns:
            Path to MP4 file, or None if conversion failed
        """
        mp4_path = webm_path.with_suffix('.mp4')

        print(f"🔄 Converting to MP4...")

        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(webm_path),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-y",  # Overwrite
                    str(mp4_path)
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and mp4_path.exists():
                # Remove WebM, keep MP4
                webm_path.unlink()
                print(f"✅ Converted to MP4")
                return mp4_path
            else:
                print(f"⚠️  Conversion failed, keeping WebM")
                return None

        except FileNotFoundError:
            print(f"⚠️  ffmpeg not found (install with: brew install ffmpeg)")
            return None
        except Exception as e:
            print(f"⚠️  Conversion error: {e}")
            return None

    async def __aenter__(self):
        """Context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()
