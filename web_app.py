"""
Flask Web Application for Academic Paper Collection
"""
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import threading
import uuid
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_loader import Config
from database import DatabaseManager
from collectors import SemanticScholarCollector, ArXivCollector, DBLPCollector
from deduplication import Deduplicator
from exporter import Exporter

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Store collection tasks
collection_tasks = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global error handler
@app.errorhandler(Exception)
def handle_error(error):
    """Handle all errors and return JSON"""
    logger.exception("Unhandled error")
    return jsonify({'error': f'Server error: {str(error)}'}), 500


class CollectionTask:
    """Represents a paper collection task"""
    def __init__(self, task_id):
        self.task_id = task_id
        self.status = 'pending'  # pending, running, completed, error
        self.progress = 0
        self.total_papers = 0
        self.message = ''
        self.results = {}
        self.error = None


def collect_papers_background(task_id, query, databases, year_min, year_max, max_papers, 
                              min_citations):
    """Background task for collecting papers"""
    task = collection_tasks[task_id]
    task.status = 'running'
    task.message = 'Initializing collection...'
    logger.info(f"Starting collection task {task_id} with query: {query}")
    
    try:
        # Create temporary database for this task
        db_path = f"data/temp_{task_id}.db"
        db = DatabaseManager(db_path)
        
        # Initialize collectors based on selected databases
        collectors = {}
        logger.info(f"Initializing collectors for databases: {databases}")
        
        if 'semantic_scholar' in databases:
            try:
                collectors['semantic_scholar'] = SemanticScholarCollector()
                logger.info("✓ Semantic Scholar collector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Semantic Scholar collector: {e}")
        
        if 'arxiv' in databases:
            try:
                collectors['arxiv'] = ArXivCollector()
                logger.info("✓ arXiv collector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize arXiv collector: {e}")
        
        if 'dblp' in databases:
            try:
                collectors['dblp'] = DBLPCollector()
                logger.info("✓ DBLP collector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DBLP collector: {e}")
        
        if 'springer' in databases:
            try:
                from collectors import SpringerCollector
                api_key = os.getenv('SPRINGER_API_KEY')  # Set via environment variable
                collectors['springer'] = SpringerCollector(api_key=api_key)
                logger.info("✓ Springer collector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Springer collector: {e}")
        
        if 'ieee' in databases:
            try:
                from collectors import IEEECollector
                api_key = os.getenv('IEEE_API_KEY')  # Set via environment variable
                collectors['ieee'] = IEEECollector(api_key=api_key)
                logger.info("✓ IEEE collector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize IEEE collector: {e}")
        
        if 'sciencedirect' in databases:
            try:
                from collectors import ScienceDirectCollector
                api_key = os.getenv('SCIENCEDIRECT_API_KEY')  # Set via environment variable
                collectors['sciencedirect'] = ScienceDirectCollector(api_key=api_key)
                logger.info("✓ ScienceDirect collector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ScienceDirect collector: {e}")
        
        if 'acm' in databases:
            try:
                from collectors import ACMCollector
                collectors['acm'] = ACMCollector()
                logger.info("✓ ACM collector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ACM collector: {e}")
        
        logger.info(f"Successfully initialized {len(collectors)} collector(s)")
        
        task.message = f'Collecting from {len(collectors)} databases...'
        all_papers = []
        
        # Log the actual query being used
        logger.info(f"Original query: {query}")
        logger.info(f"Selected databases: {list(collectors.keys())}")
        
        # Collect from all sources in parallel
        total_sources = len(collectors)
        logger.info(f"Starting PARALLEL collection from {total_sources} databases for up to {max_papers} papers each")
        task.message = f'Collecting from {total_sources} databases simultaneously (up to {max_papers} papers each)...'
        
        def collect_from_source(source_name, collector):
            """Helper function to collect papers from a single source"""
            try:
                logger.info(f"Starting parallel collection from {source_name}...")
                
                papers = collector.search(
                    query=query,
                    max_results=max_papers,
                    min_year=year_min,
                    max_year=year_max
                )
                
                # Ensure papers is a list (handle None case)
                if papers is None:
                    papers = []
                    logger.warning(f"{source_name} returned None instead of a list")
                
                logger.info(f"✓ Completed {source_name}: Found {len(papers)} papers")
                return (source_name, papers, None)
                
            except Exception as e:
                logger.error(f"Error collecting from {source_name}: {e}", exc_info=True)
                return (source_name, [], str(e))
        
        # Use ThreadPoolExecutor for parallel collection
        results = {}
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=min(total_sources, 5)) as executor:
            # Submit all collection tasks
            future_to_source = {
                executor.submit(collect_from_source, source_name, collector): source_name
                for source_name, collector in collectors.items()
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_source):
                source_name, papers, error = future.result()
                results[source_name] = {'papers': papers, 'error': error}
                completed_count += 1
                
                # Update progress
                task.progress = min(80, (completed_count / total_sources) * 80)
                if error:
                    task.message = f'Collected from {completed_count}/{total_sources} databases (error in {source_name})'
                else:
                    task.message = f'Collected from {completed_count}/{total_sources} databases ({len(papers)} from {source_name})'
        
        # Combine all papers
        for source_name, result in results.items():
            if result['error']:
                logger.warning(f"Skipping {source_name} due to error: {result['error']}")
            else:
                all_papers.extend(result['papers'])
                logger.info(f"Added {len(result['papers'])} papers from {source_name}")
        
        logger.info(f"Total papers collected from all sources: {len(all_papers)}")
        
        task.message = 'Applying quality filters...'
        # Apply filters - but be lenient with sources that don't provide citations
        # DBLP and some other sources don't have citation counts, so we shouldn't filter them out
        filtered_papers = []
        papers_without_citations = 0
        for p in all_papers:
            citations = p.get('citations', 0) or 0
            source = p.get('source', '')
            
            # Some sources (like DBLP) don't provide citation counts, so include them anyway
            if source in ['dblp', 'arxiv'] and citations == 0:
                # Include papers from sources that don't track citations
                filtered_papers.append(p)
                papers_without_citations += 1
            elif citations >= min_citations:
                filtered_papers.append(p)
        
        logger.info(f"After filtering: {len(filtered_papers)} papers (included {papers_without_citations} without citation data)")
        
        task.message = f'Removing duplicates from {len(filtered_papers)} papers...'
        logger.info(f"Starting deduplication of {len(filtered_papers)} papers...")
        
        # Deduplicate with progress updates for large datasets
        deduplicator = Deduplicator(title_threshold=0.85)
        
        # For very large datasets, we can skip deduplication or use a faster method
        if len(filtered_papers) > 1000:
            logger.info(f"Large dataset ({len(filtered_papers)} papers), using optimized deduplication...")
        
        deduplicated_papers, num_removed = deduplicator.remove_duplicates(filtered_papers)
        logger.info(f"After deduplication: {len(deduplicated_papers)} papers (removed {num_removed} duplicates)")
        task.message = f'Deduplication complete: {len(deduplicated_papers)} unique papers'
        
        task.message = 'Saving to database...'
        # Insert into database
        inserted, db_duplicates = db.bulk_insert_papers(deduplicated_papers)
        logger.info(f"Database insert: {inserted} inserted, {db_duplicates} duplicates")
        
        # Get all papers
        all_papers_db = db.get_all_papers()
        task.total_papers = len(all_papers_db)
        
        task.message = 'Exporting results...'
        task.progress = 90
        
        # Export results
        output_dir = f"results/web_{task_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        exporter = Exporter(output_dir=output_dir)
        export_results = exporter.export_all(all_papers_db, ['csv', 'json', 'bibtex'], 
                                            survey_title='Web Collection')
        
        # Get statistics
        stats = db.get_statistics()
        
        db.close()
        
        # Store results
        task.results = {
            'export_files': export_results,
            'statistics': stats,
            'db_path': db_path
        }
        
        task.status = 'completed'
        task.progress = 100
        task.message = f'Collection complete! Found {task.total_papers} papers.'
        
    except Exception as e:
        task.status = 'error'
        task.error = str(e)
        task.message = f'Error: {str(e)}'
        logger.exception("Collection task failed")


@app.route('/')
def index():
    """Main page with collection form"""
    return render_template('index.html')


@app.route('/start_collection', methods=['POST'])
def start_collection():
    """Start a new paper collection task"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        logger.info(f"Received request with data: {data}")
        
        # Get search query
        query = data.get('search_query', '').strip()
        databases = data.get('databases', [])
        
        logger.info(f"Query: {query}, Databases: {databases}")
        
        if not query:
            return jsonify({'error': 'Please provide a search query'}), 400
        
        if not databases:
            return jsonify({'error': 'Please select at least one database'}), 400
        
        # Create task
        task_id = str(uuid.uuid4())
        task = CollectionTask(task_id)
        collection_tasks[task_id] = task
        
        # Get parameters
        year_min = int(data.get('year_min', 2020))
        year_max = int(data.get('year_max', 2025))
        max_papers = int(data.get('max_papers', 150))
        min_citations = int(data.get('min_citations', 1))
        
        logger.info(f"Collection parameters: max_papers={max_papers}, databases={len(databases)}, year_range={year_min}-{year_max}")
        
        # Start background thread
        thread = threading.Thread(
            target=collect_papers_background,
            args=(task_id, query, databases, year_min, year_max, 
                  max_papers, min_citations)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'status': 'started'
        })
        
    except Exception as e:
        logger.exception("Error starting collection")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/task_status/<task_id>')
def task_status(task_id):
    """Get status of a collection task"""
    try:
        task = collection_tasks.get(task_id)
        
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        response_data = {
            'task_id': task_id,
            'status': task.status,
            'progress': task.progress,
            'message': task.message,
            'total_papers': task.total_papers,
            'error': task.error,
            'results': task.results if task.status == 'completed' else None
        }
        logger.debug(f"Status response for {task_id}: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        logger.exception(f"Error getting task status for {task_id}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/results/<task_id>')
def results(task_id):
    """Display results page"""
    task = collection_tasks.get(task_id)
    
    if not task:
        return "Task not found", 404
    
    if task.status != 'completed':
        return redirect(url_for('index'))
    
    return render_template('results.html', task=task)


@app.route('/download/<task_id>/<file_type>')
def download_file(task_id, file_type):
    """Download exported file"""
    task = collection_tasks.get(task_id)
    
    if not task or task.status != 'completed':
        return "File not found", 404
    
    file_path = task.results['export_files'].get(file_type)
    
    if not file_path or not os.path.exists(file_path):
        return "File not found", 404
    
    return send_file(file_path, as_attachment=True)


@app.route('/statistics/<task_id>')
def statistics(task_id):
    """Get statistics for a task"""
    task = collection_tasks.get(task_id)
    
    if not task or task.status != 'completed':
        return jsonify({'error': 'Task not found or not completed'}), 404
    
    return jsonify(task.results['statistics'])


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Ensure directories exist
    Path('data').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    
    # Run the app
    print("\n" + "="*60)
    print("  Academic Paper Collection Web App")
    print("="*60)
    print("\n  Starting server at: http://127.0.0.1:5000")
    print("  Press CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
