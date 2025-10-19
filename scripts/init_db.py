"""
Initialize the tasks database.

Creates the database schema and directories for data collection.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from agent.task_tracker import TaskTracker


def main():
    """Initialize database and data directories."""
    print("\n" + "="*70)
    print("Initializing Chain of Experience Database")
    print("="*70 + "\n")

    # Create data directories
    data_dir = Path("data")
    (data_dir / "videos").mkdir(parents=True, exist_ok=True)
    (data_dir / "screens").mkdir(parents=True, exist_ok=True)
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    (data_dir / "training").mkdir(parents=True, exist_ok=True)
    (data_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    print("✅ Created data directories:")
    print(f"  - {data_dir / 'videos'}")
    print(f"  - {data_dir / 'screens'}")
    print(f"  - {data_dir / 'models'}")
    print(f"  - {data_dir / 'training'}")
    print(f"  - {data_dir / 'visualizations'}")

    # Initialize database
    tracker = TaskTracker(db_path="data/tasks.db")

    print(f"\n✅ Initialized database: data/tasks.db")
    print("\n" + "="*70)
    print("Database Ready!")
    print("="*70 + "\n")

    print("Next steps:")
    print("  1. Run agent: poetry run python agent/run_agent.py")
    print("  2. Collect data by running various tasks")
    print("  3. Export training data: poetry run python scripts/export_training_data.py")
    print("  4. Train models: poetry run python train.py")
    print("  5. Evaluate: poetry run python evaluate.py")
    print()


if __name__ == "__main__":
    main()
