# Training Guide

Complete guide for training intervention prediction models.

## Overview

The training pipeline learns to predict when the agent needs human intervention by analyzing patterns in successful vs. failed interaction sequences.

## Quick Start

```bash
# 1. Initialize database
poetry run python scripts/init_db.py

# 2. Collect data (run 20-50 diverse tasks)
poetry run python agent/run_agent.py

# 3. Export training data
poetry run python scripts/export_training_data.py

# 4. Install additional dependencies
poetry add scikit-learn

# 5. Train model
poetry run python train.py --epochs-contrastive 20 --epochs-predictor 10

# 6. Evaluate
poetry run python evaluate.py
```

## Data Collection Best Practices

### Task Diversity

Run the agent on diverse tasks across different categories:

**E-commerce:**
- Product searches with price constraints
- Comparison shopping
- Reading reviews

**Research:**
- Finding documentation
- Academic paper searches
- Tutorial lookups

**Information:**
- Weather lookups
- Restaurant searches
- News queries

**Goal:** Collect 20-50 tasks with a mix of successes and interventions.

### Balancing the Dataset

Current ratio: **16 interventions : 1 success (94% intervention)**

For good training, aim for:
- **60-70% interventions** - Shows agent struggling (what we want to predict)
- **30-40% successes** - Shows agent working well (contrast examples)

**Tips to get more successes:**
1. Use simpler tasks initially
2. Increase `max_steps` to 30-50 for complex tasks
3. Choose websites agent performs well on
4. Start with familiar domains (documentation sites, search engines)

### What Gets Logged

**Interventions (label=1):**
- Final step when agent hits max_steps
- States where agent gets stuck in loops
- Moments before user needs to intervene

**Successes (label=0):**
- Final states of completed tasks
- Intermediate states from successful runs
- Smooth navigation patterns

## Model Architecture

### Experience Encoder

**Multimodal architecture** combining:

1. **Visual Encoding** (CLIP)
   - Input: Screenshots/video frames
   - Output: 512-dim visual embeddings
   - Frozen weights (pre-trained CLIP)

2. **Action Encoding** (Transformer)
   - Input: Sequence of actions (click, scroll, type, etc.)
   - Output: 128-dim action embeddings
   - Learned representations

3. **Page State Encoding** (MLP)
   - Input: DOM features, scroll position, URL, num elements
   - Output: 128-dim state embeddings
   - Simple feature extraction

4. **Fusion Layer**
   - Concatenates all modalities
   - Projects to 512-dim final embedding
   - L2 normalized for contrastive learning

### Training Phases

**Phase 1: Contrastive Learning (20 epochs)**
- Learn rich experience representations
- Pull together similar states
- Push apart different states (success vs. intervention)
- Unsupervised on experience embeddings

**Phase 2: Intervention Prediction (10 epochs)**
- Freeze encoder weights
- Train binary classifier on top
- Supervised with intervention labels
- Optimize for accuracy and recall

## Training Parameters

### Default Configuration

```bash
python train.py \
  --data data/training \
  --batch-size 4 \
  --epochs-contrastive 20 \
  --epochs-predictor 10 \
  --lr 1e-4 \
  --device cuda  # or cpu
```

### Hyperparameter Tuning

**Learning Rate:**
- Default: `1e-4`
- Lower (1e-5): More stable, slower convergence
- Higher (1e-3): Faster, risk of instability

**Batch Size:**
- Default: `4`
- Larger (8-16): Better gradient estimates, needs more memory
- Smaller (2): Works with limited data, noisier updates

**Contrastive Temperature:**
- Default: `0.07`
- Lower (0.05): Harder negatives, stricter similarity
- Higher (0.1): Softer constraints

**Epochs:**
- Contrastive: 20-50 epochs
- Predictor: 10-20 epochs
- Watch for overfitting on small datasets

## Evaluation

### Metrics

The evaluation script (`evaluate.py`) computes:

1. **Accuracy** - Overall correctness
2. **Precision** - Of predicted interventions, how many were correct?
3. **Recall** - Of actual interventions, how many did we catch?
4. **F1 Score** - Harmonic mean of precision/recall
5. **ROC AUC** - Discrimination ability
6. **Confusion Matrix** - Detailed breakdown

### Interpreting Results

**Good Performance:**
- Accuracy > 80%
- F1 Score > 0.75
- High recall (catch most interventions)
- Acceptable precision (few false alarms)

## Outputs

### Exported Data
- `data/training/interventions.json` - Intervention states
- `data/training/successes.json` - Success states
- `data/training/manifest.json` - Dataset statistics

### Trained Models
- `data/models/experience_encoder.pt` - Encoder weights
- `data/models/intervention_predictor.pt` - Predictor weights

### Evaluation Results
- `data/visualizations/evaluation_metrics.json` - All metrics
- `data/visualizations/confusion_matrix.png` - Confusion matrix plot
- `data/visualizations/roc_curve.png` - ROC curve plot

## Research Extensions

### Multimodal Improvements
- Better visual encoding (video understanding)
- Semantic page state (DOM embeddings)
- User intent modeling

### Learning Approaches
- Self-supervised pre-training
- Few-shot adaptation
- Active learning for data collection

### Evaluation
- Online A/B testing
- User studies
- Task completion time analysis

## References

- **Contrastive Learning:** [SimCLR](https://arxiv.org/abs/2002.05709)
- **CLIP:** [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- **Web Agents:** [WebGPT](https://arxiv.org/abs/2112.09332)
