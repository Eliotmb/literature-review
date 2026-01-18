"""
Paper collectors for various academic databases
"""
import requests
import time
import logging
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from urllib.parse import quote


class BaseCollector(ABC):
    """Abstract base class for paper collectors"""
    
    def __init__(self, rate_limit: float, timeout: int, retry_attempts: int):
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_request_time = 0
    
    def _simplify_query(self, query: str) -> str:
        """
        Simplify query for better API compatibility.
        Removes parentheses and complex boolean syntax.
        
        Examples:
            "optimization" AND "transport" AND ("aviation" OR "railway")
            -> "optimization" AND "transport" AND "aviation" OR "railway"
        """
        # Remove outer parentheses
        simplified = query
        while '(' in simplified or ')' in simplified:
            simplified = simplified.replace('(', '').replace(')', '')
        
        # Clean up extra spaces
        simplified = ' '.join(simplified.split())
        
        return simplified
    
    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and 429 handling"""
        self._rate_limit_wait()
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                
                # Handle rate limiting (429) specially
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s exponential backoff
                    self.logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{self.retry_attempts}")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All retry attempts failed for URL: {url}")
                    return None
    
    @abstractmethod
    def search(self, query: str, max_results: int, min_year: int, max_year: int) -> List[Dict]:
        """Search for papers - must be implemented by subclasses"""
        pass


class SemanticScholarCollector(BaseCollector):
    """Collector for Semantic Scholar API"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, rate_limit: float = 10.0, timeout: int = 30, retry_attempts: int = 5):
        super().__init__(rate_limit, timeout, retry_attempts)
    
    def search(self, query: str, max_results: int = 100, min_year: int = 2020, max_year: int = 2025) -> List[Dict]:
        """Search Semantic Scholar for papers"""
        papers = []
        
        # Simplify query for API compatibility
        simplified_query = self._simplify_query(query)
        
        url = f"{self.BASE_URL}/paper/search"
        params = {
            'query': simplified_query,
            'limit': min(max_results, 100),
            'fields': 'paperId,title,authors,year,venue,abstract,citationCount,externalIds,url',
            'year': f'{min_year}-{max_year}'
        }
        
        self.logger.info(f"Searching Semantic Scholar for: {simplified_query}")
        response = self._make_request(url, params)
        
        if not response:
            return papers
        
        try:
            data = response.json()
            
            if 'data' in data:
                for item in data['data']:
                    paper = self._parse_paper(item)
                    if paper:
                        papers.append(paper)
            
            self.logger.info(f"Found {len(papers)} papers from Semantic Scholar")
        
        except Exception as e:
            self.logger.error(f"Error parsing Semantic Scholar response: {e}")
        
        return papers
    
    def _parse_paper(self, item: Dict) -> Optional[Dict]:
        """Parse Semantic Scholar paper data"""
        try:
            authors = [author.get('name', '') for author in item.get('authors', [])]
            
            paper = {
                'paper_id': f"s2_{item.get('paperId', '')}",
                'title': item.get('title', ''),
                'authors': authors,
                'year': item.get('year'),
                'venue': item.get('venue', ''),
                'abstract': item.get('abstract', ''),
                'citations': item.get('citationCount', 0),
                'url': item.get('url', ''),
                'doi': item.get('externalIds', {}).get('DOI', ''),
                'source': 'semantic_scholar',
                'keywords': '',
                'relevance_score': 0.0
            }
            
            return paper
        except Exception as e:
            self.logger.error(f"Error parsing paper: {e}")
            return None


class ArXivCollector(BaseCollector):
    """Collector for arXiv API"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, rate_limit: float = 1.0, timeout: int = 25, retry_attempts: int = 3):
        super().__init__(rate_limit, timeout, retry_attempts)
    
    def search(self, query: str, max_results: int = 100, min_year: int = 2020, max_year: int = 2025) -> List[Dict]:
        """Search arXiv for papers"""
        papers = []
        
        # Simplify and prepare query for arXiv
        simplified_query = self._simplify_query(query)
        search_query = f'all:{simplified_query}'
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        self.logger.info(f"Searching arXiv for: {query}")
        response = self._make_request(self.BASE_URL, params)
        
        if not response:
            return papers
        
        try:
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = self._parse_paper(entry, ns, min_year, max_year)
                if paper:
                    papers.append(paper)
            
            self.logger.info(f"Found {len(papers)} papers from arXiv")
        
        except Exception as e:
            self.logger.error(f"Error parsing arXiv response: {e}")
        
        return papers
    
    def _parse_paper(self, entry: ET.Element, ns: Dict, min_year: int, max_year: int) -> Optional[Dict]:
        """Parse arXiv paper data"""
        try:
            # Extract publication date and filter by year
            published = entry.find('atom:published', ns)
            if published is not None:
                year = int(published.text[:4])
                if year < min_year or year > max_year:
                    return None
            else:
                year = None
            
            # Extract authors
            authors = [author.find('atom:name', ns).text 
                      for author in entry.findall('atom:author', ns)]
            
            # Extract ID
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            
            paper = {
                'paper_id': f"arxiv_{arxiv_id}",
                'title': entry.find('atom:title', ns).text.strip(),
                'authors': authors,
                'year': year,
                'venue': 'arXiv',
                'abstract': entry.find('atom:summary', ns).text.strip(),
                'citations': 0,  # arXiv doesn't provide citation counts
                'url': entry.find('atom:id', ns).text,
                'doi': '',
                'source': 'arxiv',
                'keywords': '',
                'relevance_score': 0.0
            }
            
            return paper
        
        except Exception as e:
            self.logger.error(f"Error parsing arXiv paper: {e}")
            return None


class DBLPCollector(BaseCollector):
    """Collector for DBLP API"""
    
    BASE_URL = "https://dblp.org/search/publ/api"
    
    def __init__(self, rate_limit: float = 5.0, timeout: int = 40, retry_attempts: int = 3):
        super().__init__(rate_limit, timeout, retry_attempts)
    
    def search(self, query: str, max_results: int = 100, min_year: int = 2020, max_year: int = 2025) -> List[Dict]:
        """Search DBLP for papers"""
        papers = []
        
        # Simplify query for DBLP
        simplified_query = self._simplify_query(query)
        
        params = {
            'q': simplified_query,
            'format': 'json',
            'h': min(max_results, 1000)  # DBLP allows up to 1000 results
        }
        
        self.logger.info(f"Searching DBLP for: {simplified_query}")
        response = self._make_request(self.BASE_URL, params)
        
        if not response:
            return papers
        
        try:
            data = response.json()
            
            if 'result' in data and 'hits' in data['result']:
                hits = data['result']['hits'].get('hit', [])
                
                for hit in hits:
                    info = hit.get('info', {})
                    paper = self._parse_paper(info, min_year, max_year)
                    if paper:
                        papers.append(paper)
            
            self.logger.info(f"Found {len(papers)} papers from DBLP")
        
        except Exception as e:
            self.logger.error(f"Error parsing DBLP response: {e}")
        
        return papers
    
    def _parse_paper(self, info: Dict, min_year: int, max_year: int) -> Optional[Dict]:
        """Parse DBLP paper data"""
        try:
            # Extract year and filter
            year_str = info.get('year', '')
            if year_str:
                year = int(year_str)
                if year < min_year or year > max_year:
                    return None
            else:
                year = None
            
            # Extract authors
            authors_data = info.get('authors', {}).get('author', [])
            if isinstance(authors_data, str):
                authors = [authors_data]
            elif isinstance(authors_data, list):
                authors = [a.get('text', a) if isinstance(a, dict) else a 
                          for a in authors_data]
            else:
                authors = []
            
            # Extract DOI
            doi = ''
            if 'ee' in info:
                ee = info['ee']
                if isinstance(ee, str) and 'doi.org' in ee:
                    doi = ee.split('doi.org/')[-1]
            
            paper = {
                'paper_id': f"dblp_{info.get('key', '').replace('/', '_')}",
                'title': info.get('title', ''),
                'authors': authors,
                'year': year,
                'venue': info.get('venue', ''),
                'abstract': '',  # DBLP doesn't provide abstracts
                'citations': 0,  # DBLP doesn't provide citation counts
                'url': info.get('url', ''),
                'doi': doi,
                'source': 'dblp',
                'keywords': '',
                'relevance_score': 0.0
            }
            
            return paper
        
        except Exception as e:
            self.logger.error(f"Error parsing DBLP paper: {e}")
            return None


class SpringerCollector(BaseCollector):
    """Collector for SpringerLink API"""
    
    BASE_URL = "https://api.springernature.com/meta/v2/json"
    
    def __init__(self, api_key: str = None, rate_limit: float = 5.0, timeout: int = 30, retry_attempts: int = 3):
        super().__init__(rate_limit, timeout, retry_attempts)
        self.api_key = api_key  # Requires API key from https://dev.springernature.com/
    
    def search(self, query: str, max_results: int = 100, min_year: int = 2020, max_year: int = 2025) -> List[Dict]:
        """Search SpringerLink for papers"""
        if not self.api_key:
            self.logger.warning("SpringerLink API key not provided - skipping")
            return []
        
        papers = []
        
        params = {
            'q': query,
            'api_key': self.api_key,
            'p': min(max_results, 100),
            's': 1
        }
        
        self.logger.info(f"Searching SpringerLink for: {query}")
        response = self._make_request(self.BASE_URL, params)
        
        if not response:
            return papers
        
        try:
            data = response.json()
            records = data.get('records', [])
            
            for record in records:
                paper = self._parse_paper(record, min_year, max_year)
                if paper:
                    papers.append(paper)
            
            self.logger.info(f"Found {len(papers)} papers from SpringerLink")
        except Exception as e:
            self.logger.error(f"Error parsing SpringerLink response: {e}")
        
        return papers
    
    def _parse_paper(self, record: Dict, min_year: int, max_year: int) -> Optional[Dict]:
        """Parse SpringerLink paper data"""
        try:
            year_str = record.get('publicationDate', '')
            if year_str:
                year = int(year_str[:4])
                if year < min_year or year > max_year:
                    return None
            else:
                year = None
            
            creators = record.get('creators', [])
            authors = [c.get('creator', '') for c in creators]
            
            paper = {
                'paper_id': f"springer_{record.get('identifier', '')}",
                'title': record.get('title', ''),
                'authors': authors,
                'year': year,
                'venue': record.get('publicationName', ''),
                'abstract': record.get('abstract', ''),
                'citations': 0,
                'url': record.get('url', [{}])[0].get('value', '') if record.get('url') else '',
                'doi': record.get('doi', ''),
                'source': 'springer',
                'keywords': '',
                'relevance_score': 0.0
            }
            
            return paper
        except Exception as e:
            self.logger.error(f"Error parsing SpringerLink paper: {e}")
            return None


class IEEECollector(BaseCollector):
    """Collector for IEEE Xplore API"""
    
    BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    
    def __init__(self, api_key: str = None, rate_limit: float = 5.0, timeout: int = 30, retry_attempts: int = 3):
        super().__init__(rate_limit, timeout, retry_attempts)
        self.api_key = api_key  # Requires API key from https://developer.ieee.org/
    
    def search(self, query: str, max_results: int = 100, min_year: int = 2020, max_year: int = 2025) -> List[Dict]:
        """Search IEEE Xplore for papers"""
        if not self.api_key:
            self.logger.warning("IEEE Xplore API key not provided - skipping")
            return []
        
        papers = []
        
        params = {
            'querytext': query,
            'apikey': self.api_key,
            'max_records': min(max_results, 200),
            'start_year': min_year,
            'end_year': max_year,
            'format': 'json'
        }
        
        self.logger.info(f"Searching IEEE Xplore for: {query}")
        response = self._make_request(self.BASE_URL, params)
        
        if not response:
            return papers
        
        try:
            data = response.json()
            articles = data.get('articles', [])
            
            for article in articles:
                paper = self._parse_paper(article)
                if paper:
                    papers.append(paper)
            
            self.logger.info(f"Found {len(papers)} papers from IEEE Xplore")
        except Exception as e:
            self.logger.error(f"Error parsing IEEE response: {e}")
        
        return papers
    
    def _parse_paper(self, article: Dict) -> Optional[Dict]:
        """Parse IEEE paper data"""
        try:
            authors_data = article.get('authors', {}).get('authors', [])
            authors = [a.get('full_name', '') for a in authors_data]
            
            paper = {
                'paper_id': f"ieee_{article.get('article_number', '')}",
                'title': article.get('title', ''),
                'authors': authors,
                'year': article.get('publication_year'),
                'venue': article.get('publication_title', ''),
                'abstract': article.get('abstract', ''),
                'citations': article.get('citing_paper_count', 0),
                'url': article.get('html_url', ''),
                'doi': article.get('doi', ''),
                'source': 'ieee',
                'keywords': '; '.join(article.get('index_terms', {}).get('author_terms', {}).get('terms', [])),
                'relevance_score': 0.0
            }
            
            return paper
        except Exception as e:
            self.logger.error(f"Error parsing IEEE paper: {e}")
            return None


class ACMCollector(BaseCollector):
    """Collector for ACM Digital Library"""
    
    # Note: ACM doesn't have a free public API
    # This is a placeholder for scraping or institutional access
    
    def __init__(self, rate_limit: float = 5.0, timeout: int = 30, retry_attempts: int = 3):
        super().__init__(rate_limit, timeout, retry_attempts)
        self.logger.warning("ACM Digital Library requires institutional access or web scraping")
    
    def search(self, query: str, max_results: int = 100, min_year: int = 2020, max_year: int = 2025) -> List[Dict]:
        """Search ACM Digital Library"""
        self.logger.warning("ACM collector not implemented - requires institutional access")
        return []


class ScienceDirectCollector(BaseCollector):
    """Collector for ScienceDirect API"""
    
    BASE_URL = "https://api.elsevier.com/content/search/sciencedirect"
    
    def __init__(self, api_key: str = None, rate_limit: float = 5.0, timeout: int = 30, retry_attempts: int = 3):
        super().__init__(rate_limit, timeout, retry_attempts)
        self.api_key = api_key  # Requires API key from https://dev.elsevier.com/
    
    def search(self, query: str, max_results: int = 100, min_year: int = 2020, max_year: int = 2025) -> List[Dict]:
        """Search ScienceDirect for papers"""
        if not self.api_key:
            self.logger.warning("ScienceDirect API key not provided - skipping")
            return []
        
        papers = []
        
        params = {
            'query': query,
            'apiKey': self.api_key,
            'count': min(max_results, 100),
            'date': f"{min_year}-{max_year}"
        }
        
        headers = {
            'X-ELS-APIKey': self.api_key,
            'Accept': 'application/json'
        }
        
        self.logger.info(f"Searching ScienceDirect for: {query}")
        
        try:
            response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('search-results', {}).get('entry', [])
            
            for result in results:
                paper = self._parse_paper(result, min_year, max_year)
                if paper:
                    papers.append(paper)
            
            self.logger.info(f"Found {len(papers)} papers from ScienceDirect")
        except Exception as e:
            self.logger.error(f"Error searching ScienceDirect: {e}")
        
        return papers
    
    def _parse_paper(self, result: Dict, min_year: int, max_year: int) -> Optional[Dict]:
        """Parse ScienceDirect paper data"""
        try:
            year_str = result.get('prism:coverDate', '')
            if year_str:
                year = int(year_str[:4])
                if year < min_year or year > max_year:
                    return None
            else:
                year = None
            
            authors_str = result.get('dc:creator', '')
            authors = [authors_str] if authors_str else []
            
            paper = {
                'paper_id': f"sciencedirect_{result.get('dc:identifier', '')}",
                'title': result.get('dc:title', ''),
                'authors': authors,
                'year': year,
                'venue': result.get('prism:publicationName', ''),
                'abstract': result.get('dc:description', ''),
                'citations': 0,
                'url': result.get('prism:url', ''),
                'doi': result.get('prism:doi', ''),
                'source': 'sciencedirect',
                'keywords': '',
                'relevance_score': 0.0
            }
            
            return paper
        except Exception as e:
            self.logger.error(f"Error parsing ScienceDirect paper: {e}")
            return None


def create_collectors(config) -> Dict[str, BaseCollector]:
    """Factory function to create all collectors"""
    collectors = {}
    
    # Semantic Scholar
    ss_config = config.get('apis', 'semantic_scholar', default={})
    collectors['semantic_scholar'] = SemanticScholarCollector(
        rate_limit=ss_config.get('rate_limit', 5.0),
        timeout=ss_config.get('timeout', 30),
        retry_attempts=ss_config.get('retry_attempts', 3)
    )
    
    # arXiv
    arxiv_config = config.get('apis', 'arxiv', default={})
    collectors['arxiv'] = ArXivCollector(
        rate_limit=arxiv_config.get('rate_limit', 1.0),
        timeout=arxiv_config.get('timeout', 25),
        retry_attempts=arxiv_config.get('retry_attempts', 3)
    )
    
    # DBLP
    dblp_config = config.get('apis', 'dblp', default={})
    collectors['dblp'] = DBLPCollector(
        rate_limit=dblp_config.get('rate_limit', 5.0),
        timeout=dblp_config.get('timeout', 40),
        retry_attempts=dblp_config.get('retry_attempts', 3)
    )
    
    return collectors
