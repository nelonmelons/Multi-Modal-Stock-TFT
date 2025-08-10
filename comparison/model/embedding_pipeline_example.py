"""
Example pipeline demonstrating the use of SectorTokenizer and SectorEmbedder.
"""

import torch
import pandas as pd
from sector_tokenizer import SectorTokenizer
from modules import SectorEmbedder

def run_embedding_pipeline():
    """
    An example pipeline that:
    1. Creates a sample DataFrame with sector data.
    2. Initializes and uses a SectorTokenizer.
    3. Initializes a SectorEmbedder with tokenizer info.
    4. Performs a forward pass to get learnable embeddings.
    """
    print("🚀 Starting Sector Embedding Pipeline Example...")

    # 1. Sample DataFrame with stock and sector information
    data = {
        'symbol': ['AAPL', 'MSFT', 'NVDA', 'JPM', 'PFE', 'GOOGL', 'TSLA', 'XOM'],
        'sector': [
            'technology', 'technology', 'technology', 'finance',
            'healthcare', 'technology', 'consumer', 'energy'
        ],
        'subsector': [
            'hardware_devices', 'software_cloud', 'semiconductors', 'commercial_banks',
            'pharmaceuticals', 'software_cloud', 'automotive', 'oil_gas'
        ]
    }
    sample_df = pd.DataFrame(data)
    print("\n📄 Sample DataFrame:")
    print(sample_df)

    # 2. Create and configure the SectorTokenizer
    # In a real application, you might save/load this from a file
    tokenizer = SectorTokenizer()
    tokenizer.update_from_dataframe(sample_df, sector_col='sector', subsector_col='subsector')

    # 3. Tokenize the data from the DataFrame
    sectors = sample_df['sector'].tolist()
    subsectors = sample_df['subsector'].tolist()
    
    encoded_data = tokenizer.batch_encode(sectors, subsectors)
    
    # Convert to PyTorch tensors
    sector_ids = torch.tensor(encoded_data['sector_ids'], dtype=torch.long)
    subsector_ids = torch.tensor(encoded_data['subsector_ids'], dtype=torch.long)

    print("\n🔒 Tokenized Data (as Tensors):")
    print(f"   Sector IDs: {sector_ids.tolist()}")
    print(f"   Sub-sector IDs: {subsector_ids.tolist()}")

    # 4. Get embedding matrix info and initialize the SectorEmbedder
    embedding_info = tokenizer.create_embedding_matrix_info()
    
    embedder = SectorEmbedder(
        sector_vocab_size=embedding_info['sector_vocab_size'],
        subsector_vocab_size=embedding_info['subsector_vocab_size'],
        sector_embedding_dim=16,
        subsector_embedding_dim=8,
        mlp_hidden_layers=[32, 24],
        output_embedding_dim=20  # Final desired embedding size
    )
    print("\n🤖 SectorEmbedder Model Architecture:")
    print(embedder)

    # 5. Perform a forward pass to get the embeddings
    # In a real model, this would be part of your model's forward pass
    output_embeddings = embedder(sector_ids, subsector_ids)

    print(f"\n✨ Generated Embeddings (Batch Size: {len(sample_df)}, Embedding Dim: 20):")
    print(output_embeddings)
    print(f"   Shape: {output_embeddings.shape}")

    # You can now use these `output_embeddings` as input to your main model (e.g., a TFT)
    print("\n✅ Embedding pipeline example completed successfully!")

if __name__ == '__main__':
    run_embedding_pipeline()
