"""
Custom PyTorch modules for embedding and feature processing.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

class SectorEmbedder(nn.Module):
    """
    A learnable embedding module for hierarchical sector data.
    Uses separate embedding layers for sectors and subsectors,
    and combines them through a configurable MLP.
    """

    def __init__(self,
                 sector_vocab_size: int,
                 subsector_vocab_size: int,
                 sector_embedding_dim: int = 32,
                 subsector_embedding_dim: int = 16,
                 mlp_hidden_layers: List[int] = [64, 48],
                 output_embedding_dim: int = 32,
                 dropout_rate: float = 0.1):
        """
        Initialize the SectorEmbedder.

        Args:
            sector_vocab_size (int): Size of the sector vocabulary.
            subsector_vocab_size (int): Size of the subsector vocabulary.
            sector_embedding_dim (int): Dimension for sector embeddings.
            subsector_embedding_dim (int): Dimension for subsector embeddings.
            mlp_hidden_layers (List[int]): List of hidden layer sizes for the MLP.
            output_embedding_dim (int): Final output embedding dimension.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()

        self.sector_embedding = nn.Embedding(sector_vocab_size, sector_embedding_dim)
        self.subsector_embedding = nn.Embedding(subsector_vocab_size, subsector_embedding_dim)

        # Total input dimension for the MLP
        mlp_input_dim = sector_embedding_dim + subsector_embedding_dim

        # Build the MLP layers
        mlp_layers = []
        current_dim = mlp_input_dim
        for hidden_dim in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Output layer
        mlp_layers.append(nn.Linear(current_dim, output_embedding_dim))

        self.mlp = nn.Sequential(*mlp_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for embedding and linear layers."""
        nn.init.xavier_uniform_(self.sector_embedding.weight)
        nn.init.xavier_uniform_(self.subsector_embedding.weight)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sector_ids: torch.Tensor, subsector_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedder.

        Args:
            sector_ids (torch.Tensor): Tensor of sector IDs (batch_size,).
            subsector_ids (torch.Tensor): Tensor of subsector IDs (batch_size,).

        Returns:
            torch.Tensor: The final embedding vector (batch_size, output_embedding_dim).
        """
        # Get embeddings
        sector_emb = self.sector_embedding(sector_ids)
        subsector_emb = self.subsector_embedding(subsector_ids)

        # Concatenate embeddings
        combined_emb = torch.cat([sector_emb, subsector_emb], dim=-1)

        # Pass through MLP
        output_embedding = self.mlp(combined_emb)

        return output_embedding

if __name__ == '__main__':
    # Example usage of the SectorEmbedder
    print("🧪 Testing SectorEmbedder...")

    # 1. Define parameters based on a tokenizer's output
    # In a real scenario, you would get this from `tokenizer.create_embedding_matrix_info()`
    tokenizer_info = {
        'sector_vocab_size': 50,
        'subsector_vocab_size': 200,
    }

    # 2. Instantiate the embedder with a custom configuration
    embedder = SectorEmbedder(
        sector_vocab_size=tokenizer_info['sector_vocab_size'],
        subsector_vocab_size=tokenizer_info['subsector_vocab_size'],
        sector_embedding_dim=24,
        subsector_embedding_dim=12,
        mlp_hidden_layers=[48, 32],
        output_embedding_dim=40,
        dropout_rate=0.15
    )
    print("\nModel Architecture:")
    print(embedder)

    # 3. Create some dummy input data (batch of 4 samples)
    batch_size = 4
    # Random sector/subsector IDs from the vocabulary
    sector_ids = torch.randint(0, tokenizer_info['sector_vocab_size'], (batch_size,))
    subsector_ids = torch.randint(0, tokenizer_info['subsector_vocab_size'], (batch_size,))

    print(f"\nInput sector_ids (shape: {sector_ids.shape}):\n{sector_ids}")
    print(f"Input subsector_ids (shape: {subsector_ids.shape}):\n{subsector_ids}")

    # 4. Perform a forward pass
    output_embeddings = embedder(sector_ids, subsector_ids)

    print(f"\nOutput embeddings (shape: {output_embeddings.shape}):\n{output_embeddings}")

    # 5. Verify output shape
    expected_shape = (batch_size, 40)
    assert output_embeddings.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {output_embeddings.shape}"

    print("\n✅ SectorEmbedder test completed successfully!")
