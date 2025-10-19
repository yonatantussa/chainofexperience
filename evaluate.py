"""
Evaluate intervention prediction model.

Metrics:
- Accuracy
- Precision/Recall
- F1 Score
- Confusion Matrix
- Intervention Prediction Timing (how early can we predict?)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))

from models.experience_encoder import ExperienceEncoder, InterventionPredictor
from train import ExperienceDataset, collate_fn


def evaluate_model(
    encoder: ExperienceEncoder,
    predictor: InterventionPredictor,
    dataloader: DataLoader,
    device: str = "cpu"
):
    """
    Evaluate intervention prediction model.

    Args:
        encoder: Trained experience encoder
        predictor: Trained intervention predictor
        dataloader: Evaluation data
        device: Device to evaluate on

    Returns:
        dict with evaluation metrics
    """
    encoder.eval()
    predictor.eval()

    encoder.to(device)
    predictor.to(device)

    all_predictions = []
    all_probabilities = []
    all_labels = []

    print("\n" + "="*70)
    print("Evaluating Model")
    print("="*70 + "\n")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get embeddings
            embeddings = encoder(
                images=batch["images"],
                action_sequences=batch["actions"],
                page_states=batch["page_states"]
            )

            # Predict
            probabilities = predictor(embeddings).squeeze(-1)

            # Handle single-element batches
            if probabilities.dim() == 0:
                probabilities = probabilities.unsqueeze(0)

            # Store results
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_predictions.extend((probabilities > 0.5).cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())

    # Convert to arrays
    import numpy as np
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    # ROC AUC
    auc = roc_auc_score(labels, probabilities)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "samples": {
            "total": len(labels),
            "interventions": int(labels.sum()),
            "successes": int((1 - labels).sum())
        }
    }

    return metrics, predictions, probabilities, labels


def plot_confusion_matrix(cm: dict, output_path: Path):
    """Plot confusion matrix."""
    matrix = [
        [cm["true_negatives"], cm["false_positives"]],
        [cm["false_negatives"], cm["true_positives"]]
    ]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Success', 'Intervention'],
        yticklabels=['Success', 'Intervention']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 Saved confusion matrix: {output_path}")


def plot_roc_curve(labels, probabilities, auc: float, output_path: Path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Intervention Prediction')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 Saved ROC curve: {output_path}")


def print_metrics(metrics: dict):
    """Print evaluation metrics."""
    print("\n" + "="*70)
    print("📊 Evaluation Results")
    print("="*70 + "\n")

    print(f"Overall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1 Score:  {metrics['f1']:.2%}")
    print(f"  ROC AUC:   {metrics['auc']:.2%}")

    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  True Negatives:  {cm['true_negatives']}")
    print(f"  False Positives: {cm['false_positives']}")
    print(f"  False Negatives: {cm['false_negatives']}")
    print(f"  True Positives:  {cm['true_positives']}")

    print(f"\nDataset:")
    print(f"  Total samples: {metrics['samples']['total']}")
    print(f"  Interventions: {metrics['samples']['interventions']}")
    print(f"  Successes:     {metrics['samples']['successes']}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate intervention prediction model")
    parser.add_argument("--data", default="data/training", help="Training data directory")
    parser.add_argument("--model-dir", default="data/models", help="Model directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output", default="data/visualizations", help="Output directory for plots")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Model Evaluation")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Models: {args.model_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    # Load models
    encoder = ExperienceEncoder(embedding_dim=512)
    predictor = InterventionPredictor(embedding_dim=512)

    encoder_path = Path(args.model_dir) / "experience_encoder.pt"
    predictor_path = Path(args.model_dir) / "intervention_predictor.pt"

    if not encoder_path.exists() or not predictor_path.exists():
        print(f"\n❌ Model files not found!")
        print(f"Expected:")
        print(f"  - {encoder_path}")
        print(f"  - {predictor_path}")
        print(f"\nRun training first: python train.py")
        return

    encoder.load_state_dict(torch.load(encoder_path, map_location=args.device))
    predictor.load_state_dict(torch.load(predictor_path, map_location=args.device))

    print(f"\n✅ Loaded models from {args.model_dir}")

    # Load dataset
    dataset = ExperienceDataset(
        interventions_file=f"{args.data}/interventions.json",
        successes_file=f"{args.data}/successes.json",
        use_video=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Evaluate
    metrics, predictions, probabilities, labels = evaluate_model(
        encoder=encoder,
        predictor=predictor,
        dataloader=dataloader,
        device=args.device
    )

    # Print results
    print_metrics(metrics)

    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved metrics: {metrics_file}")

    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        output_dir / "confusion_matrix.png"
    )

    # Plot ROC curve
    plot_roc_curve(
        labels,
        probabilities,
        metrics['auc'],
        output_dir / "roc_curve.png"
    )

    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
