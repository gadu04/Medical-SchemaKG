"""
UMLS API Loader Module
=======================
Integrates UMLS (Unified Medical Language System) API for concept lookup.
Handles authentication and concept searching via NLM UMLS API.

As of 2022-05-02, UMLS deprecated TGT/ST authentication.
Now you simply include the API key directly in requests.
"""

import os
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("  ⚠ 'requests' library not installed!")
    print("    Install with: pip install requests")
    requests = None


class UMLSLoader:
    """
    Manages UMLS API concept lookup (simplified authentication).
    Simply includes API key in request headers - no TGT/ST needed.
    Requires UMLS_API_KEY environment variable or .env file.
    """
    
    SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
    CONTENT_URL = "https://uts-ws.nlm.nih.gov/rest/content/current"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize UMLS loader with API key.
        
        Args:
            api_key (Optional[str]): UMLS API key. If None, reads from UMLS_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("UMLS_API_KEY", "").strip()
        self.authenticated = False
        
        # Check if requests library is available
        if requests is None:
            print("  ⚠ UMLS disabled: 'requests' library not installed")
            print("    Install with: pip install requests")
            return
        
        if not self.api_key:
            print("  ⚠ UMLS_API_KEY not found in environment variables or .env file")
            print("    Skipping UMLS integration. Set UMLS_API_KEY to enable.")
            return
        
        # Try to validate API key
        self._validate_api_key()
    
    def _validate_api_key(self) -> bool:
        """
        Validate API key by making a simple request to UMLS.
        
        Returns:
            bool: True if API key is valid
        """
        try:
            print("  Validating UMLS API key...", end=" ", flush=True)
            
            params = {
                "apiKey": self.api_key,
                "string": "diabetes",
                "pageSize": 1
            }
            
            resp = requests.get(
                self.SEARCH_URL,
                params=params,
                timeout=10
            )
            
            if resp.status_code == 200:
                self.authenticated = True
                print(f"✓ Valid")
                return True
            elif resp.status_code == 401:
                print(f"✗ Invalid API key (HTTP 401)")
                return False
            else:
                print(f"✗ Validation failed (HTTP {resp.status_code})")
                return False
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Connection error: {e}")
            return False
        except Exception as e:
            print(f"✗ Validation error: {e}")
            return False
    
    def get_service_ticket(self) -> Optional[str]:
        """
        (Deprecated) UMLS no longer requires service tickets.
        Kept for backwards compatibility - just returns the API key.
        
        Returns:
            Optional[str]: API key (replaces old service ticket)
        """
        return self.api_key if self.authenticated else None
    
    def search_concept(self, concept: str) -> List[Dict]:
        """
        Search for a concept in UMLS.
        
        Args:
            concept (str): Concept/term to search for
        
        Returns:
            List[Dict]: List of matching concepts with metadata
        """
        if not self.authenticated:
            return []
        
        try:
            # Search UMLS - simply include apiKey in params (new method as of 2022-05-02)
            # MỚI (Sửa lại như sau)
            params = {
                "apiKey": self.api_key,
                "string": concept,
                "pageSize": 10,           
                "searchType": "words",   
            }
            
            resp = requests.get(
                self.SEARCH_URL,
                params=params,
                timeout=10
            )
            
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            results = []
            
            if "result" in data and "results" in data["result"]:
                for item in data["result"]["results"]:
                    results.append({
                        'umls_id': item.get('ui'),
                        'name': item.get('name'),
                        'source': item.get('rootSource'),
                        'semantic_type': item.get('semanticType'),
                        'score': self._calculate_match_score(concept, item.get('name', ''))
                    })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
        
        except requests.exceptions.RequestException as e:
            return []
        except Exception as e:
            return []
    
    def get_concept_details(self, umls_id: str) -> Optional[Dict]:
        """
        Get detailed information about a UMLS concept.
        
        Args:
            umls_id (str): UMLS Concept Unique Identifier (CUI)
        
        Returns:
            Optional[Dict]: Concept details or None if not found
        """
        if not self.authenticated:
            return None
        
        try:
            params = {
                "apiKey": self.api_key,
                "includeDefinitions": "true",
                "includeSynonyms": "true"
            }
            
            resp = requests.get(
                f"{self.CONTENT_URL}/CUI/{umls_id}",
                params=params,
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            if "result" not in data:
                return None
            
            concept = data["result"]
            
            return {
                'umls_id': umls_id,
                'name': concept.get('name'),
                'definition': concept.get('definitions', [None])[0] if concept.get('definitions') else None,
                'semantic_types': concept.get('semanticTypes'),
                'synonyms': [s.get('value') for s in concept.get('atoms', [])[:5]],
                'relationships': concept.get('relations', [])
            }
        
        except Exception as e:
            return None
    
    def get_best_match(self, concept: str, threshold: float = 0.5) -> Optional[Dict]:
        """
        Get best matching UMLS concept.
        
        Args:
            concept (str): Concept to search
            threshold (float): Minimum score threshold
        
        Returns:
            Optional[Dict]: Best match or None
        """
        results = self.search_concept(concept)
        
        if results and results[0]['score'] >= threshold:
            best = results[0]
            return {
                'umls_id': best['umls_id'],
                'name': best['name'],
                'source': best['source'],
                'semantic_type': best['semantic_type'],
                'score': best['score']
            }
        
        return None
    
    def get_all_matches(self, concept: str, threshold: float = 0.3) -> List[Dict]:
        """
        Get all matching UMLS concepts above threshold.
        
        Args:
            concept (str): Concept to search
            threshold (float): Minimum score threshold
        
        Returns:
            List[Dict]: All matches above threshold
        """
        results = self.search_concept(concept)
        return [r for r in results if r['score'] >= threshold]
    
    def _calculate_match_score(self, search_term: str, result_name: str) -> float:
        """Calculate match score (0-1)."""
        search_lower = search_term.lower()
        result_lower = result_name.lower()
        
        if search_lower == result_lower:
            return 1.0
        elif search_lower in result_lower or result_lower in search_lower:
            return 0.8
        elif any(word in result_lower for word in search_lower.split()):
            return 0.6
        else:
            return 0.0
    
    def is_available(self) -> bool:
        """Check if UMLS is available and authenticated."""
        return self.authenticated
