"""
Task tracking and intervention logging for agents.
Measures task completion and records when human intervention is needed.
"""

from pathlib import Path
from datetime import datetime
import sqlite3
import json
from typing import Optional


class TaskTracker:
    """
    Tracks task completion and logs interventions for training data.
    """

    def __init__(self, db_path: str = "data/tasks.db"):
        """
        Initialize task tracker.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    url TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status TEXT,  -- 'complete', 'intervention', 'failed'
                    total_steps INTEGER,
                    video_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS interventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    reason TEXT,  -- 'stuck', 'wrong_direction', 'max_steps', etc.
                    page_state TEXT,  -- JSON of page state at intervention
                    actions_before TEXT,  -- JSON of actions leading to intervention
                    screenshot_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    success BOOLEAN,
                    page_state_before TEXT,  -- JSON
                    page_state_after TEXT,   -- JSON
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def start_task(self, session_id: str, goal: str, url: str) -> None:
        """
        Record start of a new task.

        Args:
            session_id: Unique session identifier
            goal: Task goal/objective
            url: Starting URL
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO tasks (session_id, goal, url, start_time)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, goal, url, datetime.now())
            )
            conn.commit()

        print(f"📋 Task started: {goal}")

    def end_task(
        self,
        session_id: str,
        status: str,
        total_steps: int,
        video_path: Optional[str] = None
    ) -> None:
        """
        Record task completion.

        Args:
            session_id: Session identifier
            status: 'complete', 'intervention', or 'failed'
            total_steps: Number of steps taken
            video_path: Path to recorded video (if any)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE tasks
                SET end_time = ?, status = ?, total_steps = ?, video_path = ?
                WHERE session_id = ?
                """,
                (datetime.now(), status, total_steps, video_path, session_id)
            )
            conn.commit()

        status_emoji = {
            'complete': '✅',
            'intervention': '🚨',
            'failed': '❌'
        }

        print(f"{status_emoji.get(status, '📋')} Task ended: {status} ({total_steps} steps)")

    def log_intervention(
        self,
        session_id: str,
        step: int,
        reason: str,
        page_state: dict,
        actions_before: list[str],
        screenshot_path: Optional[str] = None
    ) -> None:
        """
        Log an intervention point (where human help was needed).

        Args:
            session_id: Session identifier
            step: Step number when intervention occurred
            reason: Why intervention was needed
            page_state: Page state at intervention
            actions_before: Actions taken before intervention
            screenshot_path: Path to screenshot (if any)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO interventions
                (session_id, step, reason, page_state, actions_before, screenshot_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    step,
                    reason,
                    json.dumps(page_state),
                    json.dumps(actions_before),
                    screenshot_path
                )
            )
            conn.commit()

        print(f"🚨 Intervention logged: {reason} at step {step}")

    def log_action(
        self,
        session_id: str,
        step: int,
        action: str,
        success: bool,
        page_state_before: dict,
        page_state_after: dict
    ) -> None:
        """
        Log an individual action for training data.

        Args:
            session_id: Session identifier
            step: Step number
            action: Action taken
            success: Whether action succeeded
            page_state_before: Page state before action
            page_state_after: Page state after action
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO actions
                (session_id, step, action, success, page_state_before, page_state_after)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    step,
                    action,
                    success,
                    json.dumps(page_state_before),
                    json.dumps(page_state_after)
                )
            )
            conn.commit()

    def get_task_stats(self) -> dict:
        """
        Get statistics about tracked tasks.

        Returns:
            dict with task statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'intervention' THEN 1 ELSE 0 END) as interventions,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(total_steps) as avg_steps
                FROM tasks
            """)

            row = cursor.fetchone()

            return {
                "total_tasks": row[0] or 0,
                "completed": row[1] or 0,
                "interventions": row[2] or 0,
                "failed": row[3] or 0,
                "avg_steps": round(row[4], 1) if row[4] else 0
            }

    def get_intervention_reasons(self) -> dict:
        """
        Get breakdown of intervention reasons.

        Returns:
            dict mapping reasons to counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT reason, COUNT(*) as count
                FROM interventions
                GROUP BY reason
                ORDER BY count DESC
            """)

            return {row[0]: row[1] for row in cursor.fetchall()}

    def print_stats(self) -> None:
        """Print task statistics to console."""
        stats = self.get_task_stats()
        reasons = self.get_intervention_reasons()

        print(f"\n{'='*70}")
        print(f"📊 Task Tracker Statistics")
        print(f"{'='*70}")
        print(f"Total tasks: {stats['total_tasks']}")
        print(f"  ✅ Completed: {stats['completed']}")
        print(f"  🚨 Interventions: {stats['interventions']}")
        print(f"  ❌ Failed: {stats['failed']}")
        print(f"Average steps: {stats['avg_steps']}")

        if reasons:
            print(f"\n🚨 Intervention Reasons:")
            for reason, count in reasons.items():
                print(f"  - {reason}: {count}")

        print(f"{'='*70}\n")


# Example usage
if __name__ == "__main__":
    tracker = TaskTracker()

    # Start a task
    tracker.start_task(
        session_id="test_001",
        goal="Find Python documentation",
        url="https://www.google.com"
    )

    # Log some actions
    tracker.log_action(
        session_id="test_001",
        step=1,
        action="scroll_down",
        success=True,
        page_state_before={"scrollY": 0},
        page_state_after={"scrollY": 500}
    )

    # Log intervention
    tracker.log_intervention(
        session_id="test_001",
        step=5,
        reason="stuck_scrolling",
        page_state={"scrollY": 2500, "url": "https://www.google.com"},
        actions_before=["scroll_down", "scroll_down", "scroll_down", "scroll_down"]
    )

    # End task
    tracker.end_task(
        session_id="test_001",
        status="intervention",
        total_steps=5,
        video_path="data/videos/test_001.mp4"
    )

    # Print stats
    tracker.print_stats()
