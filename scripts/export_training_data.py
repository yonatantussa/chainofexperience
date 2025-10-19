"""
Export training data from SQLite database.

Extracts intervention points and successful task states for contrastive learning.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict
import argparse


def export_intervention_data(db_path: str = "data/tasks.db", output_dir: str = "data/training"):
    """
    Export intervention data for training.

    Creates two datasets:
    1. Intervention states (negative examples - agent needs help)
    2. Successful states (positive examples - agent doing well)

    Args:
        db_path: Path to tasks database
        output_dir: Directory to save exported data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Export intervention data
    print("\n" + "="*70)
    print("Exporting Intervention Data")
    print("="*70 + "\n")

    interventions = []
    cursor = conn.execute("""
        SELECT
            i.*,
            t.goal,
            t.url as start_url,
            t.video_path,
            t.total_steps
        FROM interventions i
        JOIN tasks t ON i.session_id = t.session_id
        ORDER BY i.created_at DESC
    """)

    for row in cursor:
        intervention = {
            "session_id": row["session_id"],
            "step": row["step"],
            "total_steps": row["total_steps"],
            "reason": row["reason"],
            "goal": row["goal"],
            "start_url": row["start_url"],
            "page_state": json.loads(row["page_state"]) if row["page_state"] else {},
            "actions_before": json.loads(row["actions_before"]) if row["actions_before"] else [],
            "screenshot_path": row["screenshot_path"],
            "video_path": row["video_path"],
            "timestamp": row["created_at"],
            "label": "intervention"  # Negative example
        }
        interventions.append(intervention)

    print(f"Found {len(interventions)} intervention states")

    # Export successful task states
    successes = []
    cursor = conn.execute("""
        SELECT
            session_id,
            goal,
            url as start_url,
            video_path,
            total_steps
        FROM tasks
        WHERE status = 'complete'
        ORDER BY created_at DESC
    """)

    for row in cursor:
        success = {
            "session_id": row["session_id"],
            "goal": row["goal"],
            "start_url": row["start_url"],
            "video_path": row["video_path"],
            "total_steps": row["total_steps"],
            "label": "success"  # Positive example
        }
        successes.append(success)

    print(f"Found {len(successes)} successful task completions")

    # Get intervention reason breakdown
    cursor = conn.execute("""
        SELECT reason, COUNT(*) as count
        FROM interventions
        GROUP BY reason
        ORDER BY count DESC
    """)

    print("\nIntervention reasons:")
    for row in cursor:
        print(f"  - {row['reason']}: {row['count']}")

    # Save to JSON
    interventions_file = output_path / "interventions.json"
    successes_file = output_path / "successes.json"

    with open(interventions_file, "w") as f:
        json.dump(interventions, f, indent=2)

    with open(successes_file, "w") as f:
        json.dump(successes, f, indent=2)

    print(f"\n✅ Exported data:")
    print(f"  - Interventions: {interventions_file}")
    print(f"  - Successes: {successes_file}")

    # Create training manifest
    manifest = {
        "num_interventions": len(interventions),
        "num_successes": len(successes),
        "total_examples": len(interventions) + len(successes),
        "intervention_ratio": len(interventions) / (len(interventions) + len(successes)) if interventions or successes else 0,
        "files": {
            "interventions": str(interventions_file),
            "successes": str(successes_file)
        }
    }

    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  - Manifest: {manifest_file}")
    print("\n" + "="*70 + "\n")

    conn.close()

    return manifest


def extract_video_frames(video_path: str, output_dir: str, num_frames: int = 10):
    """
    Extract frames from video for visual encoding.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        num_frames: Number of frames to extract
    """
    import cv2

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"⚠️  No frames found in {video_path}")
        return []

    # Extract frames at regular intervals
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    extracted_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            frame_path = output_path / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append(str(frame_path))

    cap.release()

    return extracted_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export training data from task database")
    parser.add_argument("--db", default="data/tasks.db", help="Path to tasks database")
    parser.add_argument("--output", default="data/training", help="Output directory")

    args = parser.parse_args()

    manifest = export_intervention_data(db_path=args.db, output_dir=args.output)

    print("Dataset Summary:")
    print(f"  Total examples: {manifest['total_examples']}")
    print(f"  Intervention ratio: {manifest['intervention_ratio']:.2%}")
    print(f"\nNext steps:")
    print("  1. Run: python scripts/train.py --data data/training")
    print("  2. Or: python scripts/visualize_data.py --data data/training")
