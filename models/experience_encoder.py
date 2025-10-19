"""
Experience Encoder for multimodal agent state representation.

Combines visual (video frames), textual (page state, actions), and structural
(DOM tree) information to create rich embeddings of agent interaction states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Optional
import json


class ExperienceEncoder(nn.Module):
    """
    Multimodal encoder for agent interaction states.

    Combines:
    1. Visual encoding (CLIP vision encoder on screenshots/video frames)
    2. Action sequence encoding (transformer over action embeddings)
    3. Page state encoding (DOM structure, accessibility tree)

    Output: Fixed-size embedding representing the agent's current state
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_clip: bool = True
    ):
        """
        Initialize experience encoder.

        Args:
            embedding_dim: Dimension of final embedding
            clip_model_name: HuggingFace CLIP model name
            freeze_clip: Whether to freeze CLIP weights
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Visual encoder (CLIP)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Get CLIP embedding dimension
        clip_dim = self.clip_model.config.projection_dim

        # Action sequence encoder
        self.action_embedding = nn.Embedding(
            num_embeddings=100,  # Max 100 unique action types
            embedding_dim=128
        )

        self.action_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        )

        # Page state encoder (simple MLP for now)
        self.page_state_encoder = nn.Sequential(
            nn.Linear(256, 256),  # Simplified page state features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )

        # Fusion layer - combines all modalities
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim + 128 + 128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def encode_visual(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode visual information using CLIP.

        Args:
            images: List of image tensors

        Returns:
            Visual embeddings (batch_size, clip_dim)
        """
        # Process images through CLIP
        inputs = self.clip_processor(
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Get image embeddings
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.clip_model.get_image_features(**inputs)

        return outputs

    def encode_actions(self, action_sequences: List[List[int]]) -> torch.Tensor:
        """
        Encode action sequences.

        Args:
            action_sequences: List of action ID sequences

        Returns:
            Action embeddings (batch_size, 128)
        """
        # Pad sequences
        max_len = max(len(seq) for seq in action_sequences)
        padded = torch.zeros(len(action_sequences), max_len, dtype=torch.long)

        for i, seq in enumerate(action_sequences):
            padded[i, :len(seq)] = torch.tensor(seq)

        # Embed actions
        embedded = self.action_embedding(padded)

        # Encode with transformer
        encoded = self.action_encoder(embedded)

        # Pool (mean over sequence)
        pooled = encoded.mean(dim=1)

        return pooled

    def encode_page_state(self, page_states: List[Dict]) -> torch.Tensor:
        """
        Encode page state information.

        Args:
            page_states: List of page state dictionaries

        Returns:
            Page state embeddings (batch_size, 128)
        """
        # Simple feature extraction from page state
        # In practice, you'd want more sophisticated encoding
        features = []

        for state in page_states:
            # Extract basic features
            feat = torch.zeros(256)

            # URL features (hash-based)
            if "url" in state:
                url_hash = hash(state["url"]) % 100
                feat[url_hash] = 1.0

            # Scroll position
            if "scrollY" in state:
                feat[100] = min(state["scrollY"] / 10000, 1.0)

            # Number of interactive elements
            if "num_interactive" in state:
                feat[101] = min(state["num_interactive"] / 100, 1.0)

            features.append(feat)

        features_tensor = torch.stack(features)
        encoded = self.page_state_encoder(features_tensor)

        return encoded

    def forward(
        self,
        images: Optional[List[torch.Tensor]] = None,
        action_sequences: Optional[List[List[int]]] = None,
        page_states: Optional[List[Dict]] = None
    ) -> torch.Tensor:
        """
        Forward pass - encode multimodal experience.

        Args:
            images: Visual information
            action_sequences: Action history
            page_states: Page state information

        Returns:
            Experience embeddings (batch_size, embedding_dim)
        """
        embeddings = []

        # Encode each modality
        if images is not None:
            visual_emb = self.encode_visual(images)
            embeddings.append(visual_emb)

        if action_sequences is not None:
            action_emb = self.encode_actions(action_sequences)
            embeddings.append(action_emb)

        if page_states is not None:
            page_emb = self.encode_page_state(page_states)
            embeddings.append(page_emb)

        # Concatenate and fuse
        combined = torch.cat(embeddings, dim=1)
        experience_embedding = self.fusion(combined)

        # L2 normalize for contrastive learning
        experience_embedding = F.normalize(experience_embedding, p=2, dim=1)

        return experience_embedding


class InterventionPredictor(nn.Module):
    """
    Predicts whether agent needs intervention based on experience embedding.
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize intervention predictor.

        Args:
            embedding_dim: Dimension of experience embeddings
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict intervention probability.

        Args:
            embeddings: Experience embeddings (batch_size, embedding_dim)

        Returns:
            Intervention probabilities (batch_size, 1)
        """
        return self.classifier(embeddings)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning experience representations.

    Pulls together similar states (both successful or both needing intervention)
    and pushes apart dissimilar states (successful vs intervention).
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Experience embeddings (batch_size, embedding_dim)
            labels: Binary labels (0=success, 1=intervention)

        Returns:
            Contrastive loss value
        """
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create label similarity matrix
        # Same label = similar (1), different label = dissimilar (0)
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Mask out diagonal (self-similarity)
        mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
        label_matrix = label_matrix.masked_fill(mask, 0)

        # Check if there are any positive pairs
        num_positives = label_matrix.sum(dim=1)
        if num_positives.sum() == 0:
            # No positive pairs - fall back to simple separation loss
            # Separate different labels, ignore same labels
            diff_label_matrix = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
            diff_label_matrix = diff_label_matrix.masked_fill(mask, 0)
            # Penalize high similarity for different labels
            loss = (diff_label_matrix * similarity).sum() / (diff_label_matrix.sum() + 1e-8)
            return loss

        # Compute loss
        exp_sim = torch.exp(similarity).masked_fill(mask, 0)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs
        loss = -(label_matrix * log_prob).sum(dim=1) / (num_positives + 1e-8)

        # Only average over samples that have positive pairs
        valid_samples = num_positives > 0
        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss


# Action vocabulary - maps action strings to IDs
ACTION_VOCAB = {
    "click": 0,
    "scroll_down": 1,
    "scroll_up": 2,
    "type": 3,
    "navigate": 4,
    "wait": 5,
    "go_back": 6,
    "search": 7,
    "hover": 8,
    "select": 9,
    # Add more as needed
}


def action_to_id(action: str) -> int:
    """Convert action string to ID."""
    # Simple mapping - extract action type
    for key in ACTION_VOCAB:
        if key in action.lower():
            return ACTION_VOCAB[key]
    return len(ACTION_VOCAB)  # Unknown action


if __name__ == "__main__":
    # Test the model
    print("Testing Experience Encoder...")

    encoder = ExperienceEncoder(embedding_dim=512)
    print(f"✅ Created encoder with {sum(p.numel() for p in encoder.parameters())} parameters")

    # Test forward pass
    batch_size = 4

    # Dummy data
    images = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    actions = [[0, 1, 2], [3, 4], [1, 1, 1, 1], [5]]
    page_states = [
        {"url": "https://example.com", "scrollY": 500, "num_interactive": 10},
        {"url": "https://test.com", "scrollY": 1000, "num_interactive": 20},
        {"url": "https://example.com", "scrollY": 2000, "num_interactive": 15},
        {"url": "https://demo.com", "scrollY": 100, "num_interactive": 5}
    ]

    # Encode
    embeddings = encoder(
        images=images,
        action_sequences=actions,
        page_states=page_states
    )

    print(f"✅ Output embeddings: {embeddings.shape}")
    print(f"✅ Embedding norm: {embeddings.norm(dim=1).mean():.2f} (should be ~1.0)")

    # Test predictor
    predictor = InterventionPredictor(embedding_dim=512)
    predictions = predictor(embeddings)

    print(f"✅ Predictions: {predictions.shape}")
    print(f"✅ Prediction values: {predictions.squeeze().tolist()}")

    # Test contrastive loss
    loss_fn = ContrastiveLoss()
    labels = torch.tensor([0, 1, 0, 1])  # 0=success, 1=intervention
    loss = loss_fn(embeddings, labels)

    print(f"✅ Contrastive loss: {loss.item():.4f}")
