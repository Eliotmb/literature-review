"""
Deduplication module for identifying and removing duplicate papers
"""
import logging
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher


class Deduplicator:
    """Handles paper deduplication based on title similarity"""
    
    def __init__(self, title_threshold: float = 0.85, author_matching: bool = True, venue_matching: bool = True):
        """
        Initialize deduplicator
        
        Args:
            title_threshold: Similarity threshold for title matching (0.0 to 1.0)
            author_matching: Whether to consider author overlap
            venue_matching: Whether to consider venue matching
        """
        self.title_threshold = title_threshold
        self.author_matching = author_matching
        self.venue_matching = venue_matching
        self.logger = logging.getLogger(__name__)
    
    def calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles using SequenceMatcher
        
        Returns: Similarity score between 0.0 and 1.0
        """
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        t1 = self._normalize_title(title1)
        t2 = self._normalize_title(title2)
        
        # Calculate similarity
        return SequenceMatcher(None, t1, t2).ratio()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        # Convert to lowercase and remove extra whitespace
        normalized = ' '.join(title.lower().strip().split())
        
        # Remove common punctuation
        for char in [',', '.', ':', ';', '!', '?', '"', "'", '(', ')', '[', ']']:
            normalized = normalized.replace(char, '')
        
        return normalized
    
    def _normalize_author_list(self, authors) -> set:
        """Normalize author names for comparison"""
        if isinstance(authors, str):
            # Try to parse JSON string
            import json
            try:
                authors = json.loads(authors)
            except:
                authors = [authors]
        
        if not isinstance(authors, list):
            return set()
        
        # Extract last names (simple approach)
        normalized = set()
        for author in authors:
            if isinstance(author, str):
                parts = author.strip().split()
                if parts:
                    # Take last part as surname
                    normalized.add(parts[-1].lower())
        
        return normalized
    
    def calculate_author_overlap(self, authors1, authors2) -> float:
        """
        Calculate overlap between two author lists
        
        Returns: Jaccard similarity between 0.0 and 1.0
        """
        set1 = self._normalize_author_list(authors1)
        set2 = self._normalize_author_list(authors2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def are_duplicates(self, paper1: Dict, paper2: Dict) -> bool:
        """
        Determine if two papers are duplicates
        
        Returns: True if papers are likely duplicates
        """
        # Calculate title similarity
        title_sim = self.calculate_title_similarity(
            paper1.get('title', ''),
            paper2.get('title', '')
        )
        
        # If title similarity is below threshold, not duplicates
        if title_sim < self.title_threshold:
            return False
        
        # If title similarity is very high, likely duplicates
        if title_sim >= 0.95:
            return True
        
        # For borderline cases, check additional factors
        score = title_sim
        factors = 1
        
        # Check author overlap
        if self.author_matching:
            author_overlap = self.calculate_author_overlap(
                paper1.get('authors', []),
                paper2.get('authors', [])
            )
            score += author_overlap
            factors += 1
        
        # Check venue matching
        if self.venue_matching:
            venue1 = paper1.get('venue', '')
            venue2 = paper2.get('venue', '')
            
            # Convert to string if list
            if isinstance(venue1, list):
                venue1 = ' '.join(venue1) if venue1 else ''
            if isinstance(venue2, list):
                venue2 = ' '.join(venue2) if venue2 else ''
            
            venue1 = str(venue1).lower().strip()
            venue2 = str(venue2).lower().strip()
            
            if venue1 and venue2:
                venue_match = 1.0 if venue1 == venue2 else 0.0
                score += venue_match
                factors += 1
        
        # Average score across all factors
        avg_score = score / factors
        
        return avg_score >= self.title_threshold
    
    def find_duplicates_in_list(self, papers: List[Dict]) -> List[Tuple[int, int]]:
        """
        Find all duplicate pairs in a list of papers (optimized version)
        
        Returns: List of tuples containing indices of duplicate pairs
        """
        duplicates = []
        n = len(papers)
        
        self.logger.info(f"Checking {n} papers for duplicates (optimized algorithm)...")
        
        # Pre-normalize all titles for faster comparison
        normalized_titles = [self._normalize_title(p.get('title', '')) for p in papers]
        
        # Use a more efficient approach: group by first few words of title
        # This reduces comparisons significantly
        title_groups = {}
        for i, norm_title in enumerate(normalized_titles):
            if not norm_title:
                continue
            # Use first 3 words as a key for grouping
            key_words = ' '.join(norm_title.split()[:3]).lower()
            if key_words not in title_groups:
                title_groups[key_words] = []
            title_groups[key_words].append(i)
        
        # Only compare papers within the same group (much faster)
        comparisons = 0
        for key, indices in title_groups.items():
            if len(indices) > 1:
                # Compare papers in this group
                for idx_i, i in enumerate(indices):
                    for j in indices[idx_i + 1:]:
                        comparisons += 1
                        if self.are_duplicates(papers[i], papers[j]):
                            duplicates.append((i, j))
                            self.logger.debug(
                                f"Found duplicate: '{papers[i].get('title', '')[:50]}...' "
                                f"and '{papers[j].get('title', '')[:50]}...'"
                            )
        
        self.logger.info(f"Found {len(duplicates)} duplicate pairs (made {comparisons} comparisons instead of {n*(n-1)//2})")
        return duplicates
    
    def remove_duplicates(self, papers: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Remove duplicates from a list of papers
        
        Returns: (deduplicated_papers, num_removed)
        """
        if not papers:
            return [], 0
        
        # Find all duplicate pairs
        duplicate_pairs = self.find_duplicates_in_list(papers)
        
        # Build set of indices to remove (keep the one with more citations)
        to_remove = set()
        
        for i, j in duplicate_pairs:
            citations_i = papers[i].get('citations', 0) or 0
            citations_j = papers[j].get('citations', 0) or 0
            
            # Keep the paper with more citations
            if citations_i >= citations_j:
                to_remove.add(j)
            else:
                to_remove.add(i)
        
        # Create deduplicated list
        deduplicated = [paper for idx, paper in enumerate(papers) if idx not in to_remove]
        
        num_removed = len(papers) - len(deduplicated)
        self.logger.info(f"Removed {num_removed} duplicate papers")
        
        return deduplicated, num_removed
    
    def find_duplicate_in_existing(self, new_paper: Dict, existing_papers: List[Dict]) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a new paper is a duplicate of any existing paper
        
        Returns: (is_duplicate, matching_paper)
        """
        for existing in existing_papers:
            if self.are_duplicates(new_paper, existing):
                return True, existing
        
        return False, None
