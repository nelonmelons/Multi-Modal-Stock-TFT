"""
Enhanced Sector Tokenizer with Hierarchical Encoding
Supports learnable embeddings with sector hierarchy and semantic relationships
"""

import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path


class SectorTokenizer:
    """
    Advanced sector tokenizer with hierarchical structure and semantic relationships.
    Supports multiple levels of sector classification and learnable embeddings.
    """
    
    def __init__(self, 
                 vocab_path: Optional[str] = None,
                 use_hierarchical: bool = True,
                 max_vocab_size: int = 1000):
        """
        Initialize the sector tokenizer.
        
        Args:
            vocab_path: Path to load existing vocabulary
            use_hierarchical: Whether to use hierarchical sector encoding
            max_vocab_size: Maximum vocabulary size
        """
        self.use_hierarchical = use_hierarchical
        self.max_vocab_size = max_vocab_size
        
        # Core vocabularies
        self.sector_to_id = {}
        self.id_to_sector = {}
        self.subsector_to_id = {}
        self.id_to_subsector = {}
        
        # Hierarchical mappings
        self.sector_hierarchy = {}  # sector -> list of subsectors
        self.subsector_parent = {}  # subsector -> parent sector
        
        # Special tokens
        self.UNKNOWN_TOKEN = "<UNK>"
        self.PAD_TOKEN = "<PAD>"
        self.MASK_TOKEN = "<MASK>"
        
        # Initialize special tokens
        self._init_special_tokens()
        
        # Load existing vocabulary if provided
        if vocab_path and Path(vocab_path).exists():
            self.load_vocabulary(vocab_path)
        else:
            self._build_default_vocabulary()
    
    def _init_special_tokens(self):
        """Initialize special tokens."""
        # Sector vocabulary
        self.sector_to_id = {
            self.PAD_TOKEN: 0,
            self.UNKNOWN_TOKEN: 1,
            self.MASK_TOKEN: 2
        }
        self.id_to_sector = {v: k for k, v in self.sector_to_id.items()}
        
        # Subsector vocabulary
        self.subsector_to_id = {
            self.PAD_TOKEN: 0,
            self.UNKNOWN_TOKEN: 1,
            self.MASK_TOKEN: 2
        }
        self.id_to_subsector = {v: k for k, v in self.subsector_to_id.items()}
    
    def _build_default_vocabulary(self):
        """Build default sector vocabulary based on common financial sectors."""
        
        # Main sector hierarchy
        default_hierarchy = {
            "technology": [
                "semiconductors", "software_cloud", "hardware_devices", 
                "networking_infra", "data_analytics", "cybersecurity",
                "artificial_intelligence", "fintech"
            ],
            "healthcare": [
                "pharmaceuticals", "biotechnology", "medical_devices",
                "healthcare_services", "health_insurance", "diagnostics"
            ],
            "finance": [
                "commercial_banks", "investment_banks", "insurance",
                "asset_management", "payment_processors", "reits"
            ],
            "energy": [
                "oil_gas", "renewable_energy", "utilities", "energy_storage",
                "nuclear", "coal", "solar", "wind"
            ],
            "consumer": [
                "retail", "restaurants", "automotive", "apparel",
                "consumer_electronics", "home_improvement", "luxury"
            ],
            "industrials": [
                "aerospace", "defense", "construction", "machinery",
                "transportation", "logistics", "waste_management"
            ],
            "materials": [
                "chemicals", "metals_mining", "paper_packaging",
                "construction_materials", "agriculture", "forestry"
            ],
            "telecommunications": [
                "telecom_services", "media", "entertainment", "broadcasting",
                "cable", "wireless", "satellite"
            ],
            "real_estate": [
                "residential", "commercial", "industrial_real_estate",
                "real_estate_services", "development"
            ],
            "utilities": [
                "electric_utilities", "gas_utilities", "water_utilities",
                "independent_power_producers"
            ]
        }
        
        self._build_vocabulary_from_hierarchy(default_hierarchy)
    
    def _build_vocabulary_from_hierarchy(self, hierarchy: Dict[str, List[str]]):
        """Build vocabulary from hierarchical structure."""
        sector_id = len(self.sector_to_id)  # Start after special tokens
        subsector_id = len(self.subsector_to_id)  # Start after special tokens
        
        for sector, subsectors in hierarchy.items():
            # Add main sector
            if sector not in self.sector_to_id:
                self.sector_to_id[sector] = sector_id
                self.id_to_sector[sector_id] = sector
                sector_id += 1
            
            # Add subsectors
            self.sector_hierarchy[sector] = []
            for subsector in subsectors:
                if subsector not in self.subsector_to_id:
                    self.subsector_to_id[subsector] = subsector_id
                    self.id_to_subsector[subsector_id] = subsector
                    subsector_id += 1
                
                # Build hierarchy mappings
                self.sector_hierarchy[sector].append(subsector)
                self.subsector_parent[subsector] = sector
    
    def tokenize_sector(self, sector: str) -> int:
        """
        Tokenize a single sector name.
        
        Args:
            sector: Sector name string
            
        Returns:
            Token ID for the sector
        """
        sector = sector.lower().strip()
        return self.sector_to_id.get(sector, self.sector_to_id[self.UNKNOWN_TOKEN])
    
    def tokenize_subsector(self, subsector: str) -> int:
        """
        Tokenize a single subsector name.
        
        Args:
            subsector: Subsector name string
            
        Returns:
            Token ID for the subsector
        """
        subsector = subsector.lower().strip()
        return self.subsector_to_id.get(subsector, self.subsector_to_id[self.UNKNOWN_TOKEN])
    
    def encode_hierarchical(self, sector: str, subsector: Optional[str] = None) -> Dict[str, int]:
        """
        Encode sector with hierarchical information.
        
        Args:
            sector: Main sector name
            subsector: Optional subsector name
            
        Returns:
            Dictionary with sector_id, subsector_id, and relationship info
        """
        sector_id = self.tokenize_sector(sector)
        
        if subsector:
            subsector_id = self.tokenize_subsector(subsector)
        else:
            # Try to infer subsector from sector mapping
            subsector_id = self.subsector_to_id[self.UNKNOWN_TOKEN]
        
        # Check if subsector belongs to sector
        is_valid_hierarchy = False
        if sector in self.sector_hierarchy and subsector:
            is_valid_hierarchy = subsector.lower() in [s.lower() for s in self.sector_hierarchy[sector]]
        
        return {
            'sector_id': sector_id,
            'subsector_id': subsector_id,
            'is_valid_hierarchy': int(is_valid_hierarchy),
            'hierarchy_depth': 2 if subsector else 1
        }
    
    def batch_encode(self, sectors: List[str], subsectors: Optional[List[str]] = None) -> Dict[str, List[int]]:
        """
        Encode batch of sectors.
        
        Args:
            sectors: List of sector names
            subsectors: Optional list of subsector names
            
        Returns:
            Dictionary with lists of encoded values
        """
        if subsectors is None:
            subsectors = [None] * len(sectors)
        
        batch_encoding = {
            'sector_ids': [],
            'subsector_ids': [],
            'is_valid_hierarchy': [],
            'hierarchy_depth': []
        }
        
        for i, sector in enumerate(sectors):
            subsector = subsectors[i] if subsectors and i < len(subsectors) else None
            encoding = self.encode_hierarchical(sector, subsector)
            batch_encoding['sector_ids'].append(encoding['sector_id'])
            batch_encoding['subsector_ids'].append(encoding['subsector_id'])
            batch_encoding['is_valid_hierarchy'].append(encoding['is_valid_hierarchy'])
            batch_encoding['hierarchy_depth'].append(encoding['hierarchy_depth'])
        
        return batch_encoding
    
    def decode_sector(self, sector_id: int) -> str:
        """Decode sector ID back to sector name."""
        return self.id_to_sector.get(sector_id, self.UNKNOWN_TOKEN)
    
    def decode_subsector(self, subsector_id: int) -> str:
        """Decode subsector ID back to subsector name."""
        return self.id_to_subsector.get(subsector_id, self.UNKNOWN_TOKEN)
    
    def get_sector_relationships(self, sector: str) -> Dict[str, Any]:
        """
        Get relationship information for a sector.
        
        Args:
            sector: Sector name
            
        Returns:
            Dictionary with relationship information
        """
        sector = sector.lower().strip()
        
        relationships = {
            'sector': sector,
            'subsectors': self.sector_hierarchy.get(sector, []),
            'sector_id': self.tokenize_sector(sector),
            'parent_sector': None,
            'sibling_subsectors': []
        }
        
        # If this is actually a subsector, get parent info
        if sector in self.subsector_parent:
            parent_sector = self.subsector_parent[sector]
            relationships['parent_sector'] = parent_sector
            relationships['sibling_subsectors'] = [
                s for s in self.sector_hierarchy.get(parent_sector, []) 
                if s.lower() != sector
            ]
        
        return relationships
    
    def create_embedding_matrix_info(self) -> Dict[str, Any]:
        """
        Create information needed for embedding matrix initialization.
        
        Returns:
            Dictionary with embedding matrix specifications
        """
        return {
            'sector_vocab_size': len(self.sector_to_id),
            'subsector_vocab_size': len(self.subsector_to_id),
            'sector_to_id': self.sector_to_id.copy(),
            'subsector_to_id': self.subsector_to_id.copy(),
            'hierarchy_mapping': self.sector_hierarchy.copy(),
            'special_tokens': {
                'pad_id': self.sector_to_id[self.PAD_TOKEN],
                'unk_id': self.sector_to_id[self.UNKNOWN_TOKEN],
                'mask_id': self.sector_to_id[self.MASK_TOKEN]
            }
        }
    
    def save_vocabulary(self, save_path: str):
        """Save vocabulary to file."""
        vocab_data = {
            'sector_to_id': self.sector_to_id,
            'id_to_sector': self.id_to_sector,
            'subsector_to_id': self.subsector_to_id,
            'id_to_subsector': self.id_to_subsector,
            'sector_hierarchy': self.sector_hierarchy,
            'subsector_parent': self.subsector_parent,
            'use_hierarchical': self.use_hierarchical,
            'max_vocab_size': self.max_vocab_size
        }
        
        with open(save_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"💾 Vocabulary saved to {save_path}")
        print(f"   Sectors: {len(self.sector_to_id)}")
        print(f"   Subsectors: {len(self.subsector_to_id)}")
    
    def load_vocabulary(self, vocab_path: str):
        """Load vocabulary from file."""
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.sector_to_id = vocab_data['sector_to_id']
        self.id_to_sector = {int(k): v for k, v in vocab_data['id_to_sector'].items()}
        self.subsector_to_id = vocab_data['subsector_to_id']
        self.id_to_subsector = {int(k): v for k, v in vocab_data['id_to_subsector'].items()}
        self.sector_hierarchy = vocab_data['sector_hierarchy']
        self.subsector_parent = vocab_data['subsector_parent']
        self.use_hierarchical = vocab_data.get('use_hierarchical', True)
        self.max_vocab_size = vocab_data.get('max_vocab_size', 1000)
        
        print(f"📚 Vocabulary loaded from {vocab_path}")
        print(f"   Sectors: {len(self.sector_to_id)}")
        print(f"   Subsectors: {len(self.subsector_to_id)}")
    
    def update_from_dataframe(self, df: pd.DataFrame, 
                             sector_col: str = 'sector', 
                             subsector_col: Optional[str] = None):
        """
        Update vocabulary from a DataFrame with actual sector data.
        
        Args:
            df: DataFrame containing sector information
            sector_col: Column name for sectors
            subsector_col: Optional column name for subsectors
        """
        print(f"🔄 Updating vocabulary from DataFrame...")
        
        # Get unique sectors from data
        if sector_col in df.columns:
            unique_sectors = df[sector_col].dropna().unique()
            sector_id = max(self.sector_to_id.values()) + 1
            
            for sector in unique_sectors:
                sector = sector.lower().strip()
                if sector not in self.sector_to_id:
                    self.sector_to_id[sector] = sector_id
                    self.id_to_sector[sector_id] = sector
                    sector_id += 1
                    print(f"   Added new sector: {sector}")
        
        # Get unique subsectors if provided
        if subsector_col and subsector_col in df.columns:
            unique_subsectors = df[subsector_col].dropna().unique()
            subsector_id = max(self.subsector_to_id.values()) + 1
            
            for subsector in unique_subsectors:
                subsector = subsector.lower().strip()
                if subsector not in self.subsector_to_id:
                    self.subsector_to_id[subsector] = subsector_id
                    self.id_to_subsector[subsector_id] = subsector
                    subsector_id += 1
                    print(f"   Added new subsector: {subsector}")
        
        print(f"✅ Vocabulary updated. Total sectors: {len(self.sector_to_id)}, subsectors: {len(self.subsector_to_id)}")
    
    def get_vocab_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        return {
            'total_sectors': len(self.sector_to_id),
            'total_subsectors': len(self.subsector_to_id),
            'main_sectors': len(self.sector_hierarchy),
            'avg_subsectors_per_sector': np.mean([len(subs) for subs in self.sector_hierarchy.values()]) if self.sector_hierarchy else 0,
            'special_tokens': 3,  # PAD, UNK, MASK
            'use_hierarchical': self.use_hierarchical
        }


def create_tokenizer_from_data(df: pd.DataFrame, 
                              sector_col: str = 'sector',
                              save_path: Optional[str] = None) -> SectorTokenizer:
    """
    Create a sector tokenizer from DataFrame data.
    
    Args:
        df: DataFrame with sector information
        sector_col: Column name containing sector data
        save_path: Optional path to save the tokenizer
        
    Returns:
        Configured SectorTokenizer
    """
    print("🏗️ Creating sector tokenizer from data...")
    
    tokenizer = SectorTokenizer()
    tokenizer.update_from_dataframe(df, sector_col)
    
    if save_path:
        tokenizer.save_vocabulary(save_path)
    
    stats = tokenizer.get_vocab_stats()
    print("📊 Tokenizer Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return tokenizer


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing SectorTokenizer...")
    
    # Create tokenizer
    tokenizer = SectorTokenizer()
    
    # Test encoding
    test_sectors = ["technology", "healthcare", "finance"]
    test_subsectors = ["semiconductors", "pharmaceuticals", "commercial_banks"]
    
    # Single encoding
    for sector, subsector in zip(test_sectors, test_subsectors):
        encoding = tokenizer.encode_hierarchical(sector, subsector)
        print(f"Encoded {sector}/{subsector}: {encoding}")
    
    # Batch encoding
    batch_encoding = tokenizer.batch_encode(test_sectors, test_subsectors)
    print(f"Batch encoding: {batch_encoding}")
    
    # Get relationships
    for sector in test_sectors:
        relationships = tokenizer.get_sector_relationships(sector)
        print(f"Relationships for {sector}: {relationships}")
    
    # Save vocabulary
    tokenizer.save_vocabulary("test_sector_vocab.json")
    
    print("✅ SectorTokenizer test completed!")
