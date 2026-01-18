"""
Configuration loader for the Academic Paper Collection Tool
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file"""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
    
    def get(self, *keys, default=None) -> Any:
        """Get nested configuration value"""
        value = self.data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
    
    @property
    def survey_topic(self) -> str:
        return self.get('survey', 'topic', default='')
    
    @property
    def survey_title(self) -> str:
        return self.get('survey', 'title', default='')
    
    @property
    def primary_terms(self) -> list:
        return self.get('search_terms', 'primary', default=[])
    
    @property
    def secondary_terms(self) -> list:
        return self.get('search_terms', 'secondary', default=[])
    
    @property
    def modifier_terms(self) -> list:
        return self.get('search_terms', 'modifiers', default=[])
    
    @property
    def all_search_terms(self) -> list:
        """Get all search terms combined"""
        return (self.primary_terms + self.secondary_terms + 
                self.modifier_terms)
    
    @property
    def max_papers_per_source(self) -> int:
        return self.get('collection', 'max_papers_per_source', default=100)
    
    @property
    def min_year(self) -> int:
        return self.get('collection', 'min_year', default=2020)
    
    @property
    def max_year(self) -> int:
        return self.get('collection', 'max_year', default=2025)
    
    @property
    def min_citations(self) -> int:
        return self.get('collection', 'min_citations', default=1)
    
    @property
    def db_path(self) -> str:
        return self.get('database', 'path', default='data/papers.db')
    
    @property
    def exclude_keywords(self) -> list:
        return self.get('quality_filters', 'exclude_keywords', default=[])
    
    @property
    def preferred_venues(self) -> list:
        return self.get('quality_filters', 'preferred_venues', default=[])
    
    @property
    def scoring_weights(self) -> Dict[str, float]:
        return self.get('quality_filters', 'scoring_weights', default={})
    
    @property
    def export_formats(self) -> list:
        return self.get('export', 'formats', default=['csv'])
    
    @property
    def output_dir(self) -> str:
        return self.get('export', 'output_dir', default='results')
    
    @property
    def ollama_enabled(self) -> bool:
        return self.get('ollama', 'enabled', default=False)
    
    @property
    def ollama_url(self) -> str:
        return self.get('ollama', 'url', default='http://localhost:11434')
    
    @property
    def ollama_model(self) -> str:
        return self.get('ollama', 'model', default='deepseek-r1:7b')
    
    @property
    def title_similarity_threshold(self) -> float:
        return self.get('deduplication', 'title_similarity_threshold', default=0.85)
