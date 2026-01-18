"""
Export module for generating output files in various formats
"""
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class Exporter:
    """Handles exporting papers to various formats"""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize exporter"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def export_all(self, papers: List[Dict], formats: List[str], survey_title: str = ""):
        """Export papers to all specified formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"papers_{timestamp}"
        
        results = {}
        
        for fmt in formats:
            if fmt == 'csv':
                filepath = self.export_csv(papers, base_filename)
                results['csv'] = filepath
            elif fmt == 'json':
                filepath = self.export_json(papers, base_filename, survey_title)
                results['json'] = filepath
            elif fmt == 'bibtex':
                filepath = self.export_bibtex(papers, base_filename)
                results['bibtex'] = filepath
        
        return results
    
    def export_csv(self, papers: List[Dict], filename: str = "papers") -> str:
        """Export papers to CSV format"""
        filepath = self.output_dir / f"{filename}.csv"
        
        if not papers:
            self.logger.warning("No papers to export to CSV")
            return str(filepath)
        
        # Define CSV columns
        columns = [
            'title', 'authors', 'year', 'venue', 'citations', 
            'abstract', 'url', 'doi', 'source', 'relevance_score', 'ai_analysis'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            
            for paper in papers:
                # Convert authors list to string
                row = paper.copy()
                if isinstance(row.get('authors'), list):
                    row['authors'] = '; '.join(row['authors'])
                
                writer.writerow(row)
        
        self.logger.info(f"Exported {len(papers)} papers to CSV: {filepath}")
        return str(filepath)
    
    def export_json(self, papers: List[Dict], filename: str = "papers", survey_title: str = "") -> str:
        """Export papers to JSON format with metadata"""
        filepath = self.output_dir / f"{filename}.json"
        
        # Create structured JSON output
        output = {
            'metadata': {
                'survey_title': survey_title,
                'export_date': datetime.now().isoformat(),
                'total_papers': len(papers),
                'sources': self._count_by_source(papers),
                'year_range': self._get_year_range(papers)
            },
            'papers': papers
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(papers)} papers to JSON: {filepath}")
        return str(filepath)
    
    def export_bibtex(self, papers: List[Dict], filename: str = "papers") -> str:
        """Export papers to BibTeX format"""
        filepath = self.output_dir / f"{filename}.bib"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx, paper in enumerate(papers, 1):
                bibtex_entry = self._paper_to_bibtex(paper, idx)
                f.write(bibtex_entry)
                f.write('\n\n')
        
        self.logger.info(f"Exported {len(papers)} papers to BibTeX: {filepath}")
        return str(filepath)
    
    def _paper_to_bibtex(self, paper: Dict, idx: int) -> str:
        """Convert a paper to BibTeX entry"""
        # Generate citation key
        authors = paper.get('authors', [])
        if isinstance(authors, str):
            import json
            try:
                authors = json.loads(authors)
            except:
                authors = [authors]
        
        first_author = ""
        if authors and len(authors) > 0:
            first_author = authors[0].split()[-1] if authors[0] else "Unknown"
        else:
            first_author = "Unknown"
        
        year = paper.get('year', 'XXXX')
        key = f"{first_author}{year}_{idx}"
        
        # Determine entry type
        venue = paper.get('venue', '').lower()
        if 'arxiv' in venue or paper.get('source') == 'arxiv':
            entry_type = 'misc'
        elif any(conf in venue for conf in ['conference', 'proceedings', 'symposium', 'workshop']):
            entry_type = 'inproceedings'
        elif any(jour in venue for jour in ['journal', 'transactions', 'letters']):
            entry_type = 'article'
        else:
            entry_type = 'misc'
        
        # Build BibTeX entry
        bibtex = f"@{entry_type}{{{key},\n"
        
        # Title
        title = paper.get('title', '').replace('{', '\\{').replace('}', '\\}')
        bibtex += f"  title = {{{title}}},\n"
        
        # Authors
        if authors:
            author_str = ' and '.join(authors)
            bibtex += f"  author = {{{author_str}}},\n"
        
        # Year
        if year:
            bibtex += f"  year = {{{year}}},\n"
        
        # Venue/Journal/Booktitle
        venue_name = paper.get('venue', '')
        if venue_name:
            if entry_type == 'article':
                bibtex += f"  journal = {{{venue_name}}},\n"
            elif entry_type == 'inproceedings':
                bibtex += f"  booktitle = {{{venue_name}}},\n"
            else:
                bibtex += f"  howpublished = {{{venue_name}}},\n"
        
        # DOI
        doi = paper.get('doi', '')
        if doi:
            bibtex += f"  doi = {{{doi}}},\n"
        
        # URL
        url = paper.get('url', '')
        if url:
            bibtex += f"  url = {{{url}}},\n"
        
        # Abstract (optional, commented out by default)
        # abstract = paper.get('abstract', '')
        # if abstract:
        #     bibtex += f"  abstract = {{{abstract}}},\n"
        
        bibtex += "}"
        
        return bibtex
    
    def _count_by_source(self, papers: List[Dict]) -> Dict[str, int]:
        """Count papers by source"""
        counts = {}
        for paper in papers:
            source = paper.get('source', 'unknown')
            counts[source] = counts.get(source, 0) + 1
        return counts
    
    def _get_year_range(self, papers: List[Dict]) -> Dict[str, int]:
        """Get year range of papers"""
        years = [p.get('year') for p in papers if p.get('year')]
        if years:
            return {'min': min(years), 'max': max(years)}
        return {'min': None, 'max': None}
    
    def export_statistics(self, stats: Dict, filename: str = "statistics") -> str:
        """Export collection statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{filename}_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported statistics to: {filepath}")
        return str(filepath)
