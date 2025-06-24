"""
Configuration for Semantic Scholar API integration
"""

import os
from typing import Optional

class SemanticScholarConfig:
    """Configuration class for Semantic Scholar API"""
    
    def __init__(self):
        # Try to get API key from environment variable first
        self.api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        
        # API endpoints
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.search_endpoint = f"{self.base_url}/paper/search/bulk"
        self.paper_endpoint = f"{self.base_url}/paper"
        
        # Rate limiting settings
        self.requests_per_second = 1
        self.max_retries = 3
        self.retry_delay = 2
        
        # Search settings
        self.default_fields = [
            'title', 'authors', 'abstract', 'year', 
            'journal', 'citationCount', 'url', 'externalIds'
        ]
        self.max_papers_per_query = 10
        self.year_filter = "2015-2024"  # Focus on recent papers
        
    def get_headers(self) -> dict:
        """Get headers for API requests"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'ScientificPaperSystem/1.0'
        }
        
        if self.api_key:
            headers['x-api-key'] = self.api_key
        
        return headers
    
    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return self.api_key is not None
    
    def set_api_key(self, api_key: str):
        """Set API key manually"""
        self.api_key = api_key
    
    def get_search_params(self, query: str, limit: int = 10) -> dict:
        """Get standard search parameters"""
        return {
            'query': query,
            'fields': ','.join(self.default_fields),
            'limit': min(limit, self.max_papers_per_query),
            'year': self.year_filter
        }

# Global configuration instance
semantic_scholar_config = SemanticScholarConfig()

def setup_semantic_scholar_api_key(api_key: Optional[str] = None):
    """
    Setup Semantic Scholar API key
    
    Args:
        api_key: API key string. If None, will try to get from environment
    """
    if api_key:
        semantic_scholar_config.set_api_key(api_key)
        print("✅ Semantic Scholar API key configured")
    elif semantic_scholar_config.is_configured():
        print("✅ Semantic Scholar API key found in environment")
    else:
        print("⚠️ No Semantic Scholar API key found")
        print("💡 You can:")
        print("   1. Set environment variable: export SEMANTIC_SCHOLAR_API_KEY='your_key'")
        print("   2. Request a free API key at: https://www.semanticscholar.org/product/api")
        print("   3. The system will work with rate limits without an API key")
    
    return semantic_scholar_config.is_configured() 