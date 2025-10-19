"""
Train intervention prediction model.

Two-phase training:
1. Phase 1: Contrastive learning on experience embeddings
2. Phase 2: Supervised learning for intervention prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import cv2
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent))

from models.experience_encoder import (
    ExperienceEncoder,
    InterventionPredictor,
    ContrastiveLoss,
    action_to_id
)


class ExperienceDataset(Dataset):
    """
    Dataset for agent experience states.
    """

    def __init__(
        self,
        interventions_file: str,
        successes_file: str,
        video_dir: str = "data/videos",
        use_video: bool = True
    ):
        """
        Initialize dataset.

        Args:
            interventions_file: Path to interventions JSON
            successes_file: Path to successes JSON
            video_dir: Directory containing videos
            use_video: Whether to load video frames
        """
        self.video_dir = Path(video_dir)
        self.use_video = use_video

        # Load data
        with open(interventions_file) as f:
            interventions = json.load(f)

        with open(successes_file) as f:
            successes = json.load(f)

        # Combine into single dataset
        self.examples = []

        # Add interventions (label=1)
        for item in interventions:
            self.examples.append({
                "session_id": item["session_id"],
                "video_path": item.get("video_path"),
                "actions": item.get("actions_before", []),
                "page_state": item.get("page_state", {}),
                "label": 1  # Intervention needed
            })

        # Add successes (label=0)
        for item in successes:
            self.examples.append({
                "session_id": item["session_id"],
                "video_path": item.get("video_path"),
                "actions": [],  # Success states don't have action sequences yet
                "page_state": {},
                "label": 0  # No intervention
            })

        print(f"Loaded {len(self.examples)} examples ({len(interventions)} interventions, {len(successes)} successes)")

    def __len__(self):
        return len(self.examples)

    def load_video_frame(self, video_path: str, frame_idx: int = -10) -> Image.Image:
        """
        Load a frame from video.

        Args:
            video_path: Path to video file
            frame_idx: Frame index (negative = from end)

        Returns:
            PIL Image
        """
        if not video_path or not Path(video_path).exists():
            # Return blank image if video not found
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Get frame index (negative indexing from end)
        if frame_idx < 0:
            frame_idx = max(0, total_frames + frame_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get example.

        Returns:
            (image, actions, page_state, label)
        """
        example = self.examples[idx]

        # Load video frame
        if self.use_video and example["video_path"]:
            image = self.load_video_frame(example["video_path"])
        else:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Convert actions to IDs
        action_ids = [action_to_id(str(a)) for a in example["actions"]]
        if not action_ids:
            action_ids = [0]  # Default action

        return (
            image,
            action_ids,
            example["page_state"],
            example["label"]
        )


def collate_fn(batch):
    """Custom collate function for batching."""
    images, actions, page_states, labels = zip(*batch)

    return {
        "images": list(images),
        "actions": list(actions),
        "page_states": list(page_states),
        "labels": torch.tensor(labels, dtype=torch.float32)
    }


def train_contrastive(
    encoder: ExperienceEncoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: ContrastiveLoss,
    device: str = "cpu",
    epochs: int = 10
):
    """
    Phase 1: Train encoder with contrastive learning.

    Args:
        encoder: Experience encoder model
        dataloader: Training data
        optimizer: Optimizer
        loss_fn: Contrastive loss function
        device: Device to train on
        epochs: Number of epochs
    """
    encoder.train()
    encoder.to(device)

    print("\n" + "="*70)
    print("Phase 1: Contrastive Learning")
    print("="*70 + "\n")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            # Get embeddings
            embeddings = encoder(
                images=batch["images"],
                action_sequences=batch["actions"],
                page_states=batch["page_states"]
            )

            # Compute contrastive loss
            loss = loss_fn(embeddings, batch["labels"])

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{total_loss/num_batches:.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    print("\n✅ Contrastive learning complete")


def train_predictor(
    encoder: ExperienceEncoder,
    predictor: InterventionPredictor,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.BCELoss,
    device: str = "cpu",
    epochs: int = 10
):
    """
    Phase 2: Train intervention predictor.

    Args:
        encoder: Trained experience encoder (frozen)
        predictor: Intervention predictor
        dataloader: Training data
        optimizer: Optimizer
        loss_fn: Binary cross-entropy loss
        device: Device to train on
        epochs: Number of epochs
    """
    encoder.eval()  # Freeze encoder
    predictor.train()

    encoder.to(device)
    predictor.to(device)

    print("\n" + "="*70)
    print("Phase 2: Intervention Prediction")
    print("="*70 + "\n")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            # Get embeddings (no gradients)
            with torch.no_grad():
                embeddings = encoder(
                    images=batch["images"],
                    action_sequences=batch["actions"],
                    page_states=batch["page_states"]
                )

            # Predict interventions
            predictions = predictor(embeddings).squeeze(-1)
            labels = batch["labels"]

            # Handle single-element batches
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)

            # Compute loss
            loss = loss_fn(predictions, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            accuracy = correct / total
            pbar.set_postfix({
                "loss": f"{total_loss/(pbar.n+1):.4f}",
                "acc": f"{accuracy:.2%}"
            })

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Accuracy = {accuracy:.2%}")

    print("\n✅ Intervention prediction training complete")


def main():
    parser = argparse.ArgumentParser(description="Train intervention prediction model")
    parser.add_argument("--data", default="data/training", help="Training data directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs-contrastive", type=int, default=20, help="Contrastive learning epochs")
    parser.add_argument("--epochs-predictor", type=int, default=10, help="Predictor training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output", default="data/models", help="Output directory for models")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Training Intervention Prediction Model")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*70 + "\n")

    # Load dataset
    dataset = ExperienceDataset(
        interventions_file=f"{args.data}/interventions.json",
        successes_file=f"{args.data}/successes.json",
        use_video=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize models
    encoder = ExperienceEncoder(embedding_dim=512)
    predictor = InterventionPredictor(embedding_dim=512)

    # Phase 1: Contrastive learning
    contrastive_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    contrastive_loss = ContrastiveLoss(temperature=0.07)

    train_contrastive(
        encoder=encoder,
        dataloader=dataloader,
        optimizer=contrastive_optimizer,
        loss_fn=contrastive_loss,
        device=args.device,
        epochs=args.epochs_contrastive
    )

    # Save encoder
    encoder_path = output_dir / "experience_encoder.pt"
    torch.save(encoder.state_dict(), encoder_path)
    print(f"\n💾 Saved encoder: {encoder_path}")

    # Phase 2: Intervention prediction
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=args.lr)
    predictor_loss = nn.BCELoss()

    train_predictor(
        encoder=encoder,
        predictor=predictor,
        dataloader=dataloader,
        optimizer=predictor_optimizer,
        loss_fn=predictor_loss,
        device=args.device,
        epochs=args.epochs_predictor
    )

    # Save predictor
    predictor_path = output_dir / "intervention_predictor.pt"
    torch.save(predictor.state_dict(), predictor_path)
    print(f"💾 Saved predictor: {predictor_path}")

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"\nModels saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Evaluate: python evaluate.py --model-dir data/models")
    print("  2. Integrate: Use predictor in agent loop")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
