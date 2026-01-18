"""
Database manager for storing and retrieving academic papers
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging


class DatabaseManager:
    """Manages SQLite database for paper storage"""
    
    def __init__(self, db_path: str):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.logger = logging.getLogger(__name__)
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.logger.info(f"Connected to database: {self.db_path}")
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Main papers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT UNIQUE,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                venue TEXT,
                abstract TEXT,
                citations INTEGER DEFAULT 0,
                url TEXT,
                doi TEXT,
                source TEXT,
                keywords TEXT,
                relevance_score REAL DEFAULT 0.0,
                ai_analysis TEXT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON papers(title)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON papers(year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations ON papers(citations)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON papers(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_id ON papers(paper_id)')
        
        # Search terms tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_term TEXT NOT NULL,
                source TEXT NOT NULL,
                papers_found INTEGER DEFAULT 0,
                searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Collection statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_papers INTEGER DEFAULT 0,
                duplicates_removed INTEGER DEFAULT 0,
                semantic_scholar_count INTEGER DEFAULT 0,
                dblp_count INTEGER DEFAULT 0,
                arxiv_count INTEGER DEFAULT 0,
                last_collection TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        self.logger.info("Database tables created/verified")
    
    def insert_paper(self, paper_data: Dict) -> Tuple[bool, str]:
        """
        Insert a paper into the database
        Returns: (success: bool, message: str)
        """
        try:
            cursor = self.conn.cursor()
            
            # Convert authors list to JSON string if it's a list
            authors = paper_data.get('authors', '')
            if isinstance(authors, list):
                authors = json.dumps(authors)
            
            cursor.execute('''
                INSERT INTO papers (
                    paper_id, title, authors, year, venue, abstract,
                    citations, url, doi, source, keywords, relevance_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_data.get('paper_id'),
                paper_data.get('title'),
                authors,
                paper_data.get('year'),
                paper_data.get('venue'),
                paper_data.get('abstract'),
                paper_data.get('citations', 0),
                paper_data.get('url'),
                paper_data.get('doi'),
                paper_data.get('source'),
                paper_data.get('keywords'),
                paper_data.get('relevance_score', 0.0)
            ))
            
            self.conn.commit()
            return True, f"Inserted paper: {paper_data.get('title', 'Unknown')}"
        
        except sqlite3.IntegrityError:
            return False, f"Duplicate paper: {paper_data.get('title', 'Unknown')}"
        except Exception as e:
            self.logger.error(f"Error inserting paper: {e}")
            return False, str(e)
    
    def bulk_insert_papers(self, papers: List[Dict]) -> Tuple[int, int]:
        """
        Insert multiple papers at once
        Returns: (inserted_count, duplicate_count)
        """
        inserted = 0
        duplicates = 0
        
        for paper in papers:
            success, _ = self.insert_paper(paper)
            if success:
                inserted += 1
            else:
                duplicates += 1
        
        self.logger.info(f"Bulk insert: {inserted} inserted, {duplicates} duplicates")
        return inserted, duplicates
    
    def get_all_papers(self) -> List[Dict]:
        """Retrieve all papers from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM papers ORDER BY citations DESC, year DESC')
        rows = cursor.fetchall()
        
        papers = []
        for row in rows:
            paper = dict(row)
            # Parse authors JSON if it's a string
            if paper.get('authors') and isinstance(paper['authors'], str):
                try:
                    paper['authors'] = json.loads(paper['authors'])
                except:
                    pass
            papers.append(paper)
        
        return papers
    
    def get_papers_by_year_range(self, min_year: int, max_year: int) -> List[Dict]:
        """Get papers within a specific year range"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM papers 
            WHERE year >= ? AND year <= ?
            ORDER BY citations DESC, year DESC
        ''', (min_year, max_year))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_papers_by_source(self, source: str) -> List[Dict]:
        """Get papers from a specific source"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE source = ?', (source,))
        return [dict(row) for row in cursor.fetchall()]
    
    def search_papers(self, keyword: str) -> List[Dict]:
        """Search papers by keyword in title or abstract"""
        cursor = self.conn.cursor()
        search_term = f"%{keyword}%"
        cursor.execute('''
            SELECT * FROM papers 
            WHERE title LIKE ? OR abstract LIKE ?
            ORDER BY relevance_score DESC, citations DESC
        ''', (search_term, search_term))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def update_ai_analysis(self, paper_id: str, analysis: str) -> bool:
        """Update AI analysis for a paper"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE papers 
                SET ai_analysis = ?, updated_at = CURRENT_TIMESTAMP
                WHERE paper_id = ?
            ''', (analysis, paper_id))
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error updating AI analysis: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total papers
        cursor.execute('SELECT COUNT(*) as count FROM papers')
        stats['total_papers'] = cursor.fetchone()['count']
        
        # Papers by source
        cursor.execute('SELECT source, COUNT(*) as count FROM papers GROUP BY source')
        stats['by_source'] = {row['source']: row['count'] for row in cursor.fetchall()}
        
        # Papers by year
        cursor.execute('SELECT year, COUNT(*) as count FROM papers GROUP BY year ORDER BY year')
        stats['by_year'] = {row['year']: row['count'] for row in cursor.fetchall()}
        
        # Average citations
        cursor.execute('SELECT AVG(citations) as avg_citations FROM papers')
        stats['avg_citations'] = cursor.fetchone()['avg_citations'] or 0
        
        # Top cited papers
        cursor.execute('SELECT title, citations FROM papers ORDER BY citations DESC LIMIT 10')
        stats['top_cited'] = [dict(row) for row in cursor.fetchall()]
        
        return stats
    
    def check_duplicate_by_title(self, title: str, threshold: float = 0.85) -> Optional[Dict]:
        """Check if a paper with similar title exists"""
        # Simple exact match check (more sophisticated similarity in deduplication module)
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE LOWER(title) = LOWER(?)', (title,))
        result = cursor.fetchone()
        
        if result:
            return dict(result)
        return None
    
    def get_all_titles(self) -> List[str]:
        """Get all paper titles for deduplication"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT title FROM papers')
        return [row['title'] for row in cursor.fetchall()]
    
    def vacuum(self):
        """Optimize database"""
        self.conn.execute('VACUUM')
        self.logger.info("Database vacuumed")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
