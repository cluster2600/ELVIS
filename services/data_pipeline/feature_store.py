"""
Feature Store for managing and versioning ML model features.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import hashlib

class FeatureStore:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.store_path = config.get('store_path', 'data/feature_store')
        self.feature_sets = {}
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the feature store directory structure."""
        os.makedirs(self.store_path, exist_ok=True)
        os.makedirs(os.path.join(self.store_path, 'features'), exist_ok=True)
        os.makedirs(os.path.join(self.store_path, 'metadata'), exist_ok=True)
        os.makedirs(os.path.join(self.store_path, 'versions'), exist_ok=True)

    def create_feature_set(self, name: str, features: List[Dict[str, Any]], description: str = ""):
        """Create a new feature set."""
        feature_set = {
            'name': name,
            'features': features,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'version': 1
        }
        
        # Save feature set metadata
        metadata_path = os.path.join(self.store_path, 'metadata', f'{name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(feature_set, f, indent=2)
            
        self.feature_sets[name] = feature_set
        self.logger.info(f"Created feature set: {name}")

    def get_feature_set(self, name: str) -> Optional[Dict]:
        """Get a feature set by name."""
        if name in self.feature_sets:
            return self.feature_sets[name]
            
        metadata_path = os.path.join(self.store_path, 'metadata', f'{name}.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                feature_set = json.load(f)
                self.feature_sets[name] = feature_set
                return feature_set
                
        return None

    def save_features(self, name: str, data: pd.DataFrame, version: Optional[int] = None):
        """Save features to the store."""
        feature_set = self.get_feature_set(name)
        if not feature_set:
            raise ValueError(f"Feature set {name} not found")
            
        if version is None:
            version = feature_set['version']
            
        # Create version directory
        version_path = os.path.join(self.store_path, 'versions', name, str(version))
        os.makedirs(version_path, exist_ok=True)
        
        # Save features
        feature_path = os.path.join(version_path, 'features.parquet')
        data.to_parquet(feature_path)
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(data),
            'feature_columns': list(data.columns),
            'data_hash': self._calculate_data_hash(data)
        }
        
        with open(os.path.join(version_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Saved features for {name} version {version}")

    def load_features(self, name: str, version: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load features from the store."""
        feature_set = self.get_feature_set(name)
        if not feature_set:
            return None
            
        if version is None:
            version = feature_set['version']
            
        version_path = os.path.join(self.store_path, 'versions', name, str(version))
        if not os.path.exists(version_path):
            return None
            
        feature_path = os.path.join(version_path, 'features.parquet')
        if not os.path.exists(feature_path):
            return None
            
        return pd.read_parquet(feature_path)

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate a hash of the data for versioning."""
        # Convert DataFrame to bytes
        data_bytes = data.to_csv(index=False).encode()
        return hashlib.sha256(data_bytes).hexdigest()

    def get_feature_set_versions(self, name: str) -> List[Dict]:
        """Get all versions of a feature set."""
        versions = []
        version_dir = os.path.join(self.store_path, 'versions', name)
        
        if os.path.exists(version_dir):
            for version in os.listdir(version_dir):
                metadata_path = os.path.join(version_dir, version, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        versions.append(metadata)
                        
        return sorted(versions, key=lambda x: x['version'], reverse=True)

    def delete_feature_set(self, name: str):
        """Delete a feature set and all its versions."""
        if name in self.feature_sets:
            del self.feature_sets[name]
            
        metadata_path = os.path.join(self.store_path, 'metadata', f'{name}.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            
        version_dir = os.path.join(self.store_path, 'versions', name)
        if os.path.exists(version_dir):
            import shutil
            shutil.rmtree(version_dir)
            
        self.logger.info(f"Deleted feature set: {name}") 