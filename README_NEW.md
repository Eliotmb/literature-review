# 📚 Academic Paper Collector - Web App

A powerful web-based application for collecting and managing academic papers from multiple sources for systematic literature reviews.

## ✨ Features

- **🔍 Flexible Search Queries** - Use AND, OR, and exact phrase matching in one powerful query field
  - Example: `"critical infrastructure" AND (optimization OR resilience)`
  - Supports complex boolean search logic

- **🗄️ Multi-Source Collection** - Gather papers from 7 major academic databases:
  - Semantic Scholar (free)
  - arXiv (free)
  - DBLP (free)
  - IEEE Xplore (requires API key)
  - SpringerLink (requires API key)
  - ScienceDirect (requires API key)
  - ACM Digital Library (requires API key)

- **📊 Smart Deduplication** - Automatically removes duplicate papers using:
  - Title similarity matching (85% threshold)
  - Author overlap analysis
  - Venue comparison

- **📥 Multiple Export Formats**:
  - CSV (Excel-compatible)
  - JSON (structured data with metadata)
  - BibTeX (for LaTeX)

- **⚙️ Advanced Filtering**:
  - Publication year range
  - Minimum citation count
  - Maximum papers per search

- **💻 Real-Time Progress Tracking** - Monitor collection status as it runs

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Flask 3.0+

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/paper-finding.git
cd paper-finding
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python web_app.py
```

4. **Open in browser:**
```
http://127.0.0.1:5000
```

## 📖 Usage

### Basic Collection

1. Enter your search query in the "Search Query" field
   - Use AND/OR operators and quotes for phrases
   - Example: `machine learning AND security`

2. Select databases to search:
   - At least one database must be selected
   - Mix free and premium sources as needed

3. Configure parameters:
   - **Year Range**: Filter by publication year (default: 2020-2025)
   - **Max Papers**: Maximum papers to retrieve (default: 150)
   - **Min Citations**: Minimum citation count (default: 1)

4. Click "🚀 Start Collection"

5. Monitor progress in real-time

6. Download results in your preferred format

### Query Examples

- **Broad Search**: `optimization` (finds papers with any mention)
- **Specific Topic**: `"machine learning" AND security`
- **Multiple Options**: `(AI OR "artificial intelligence") AND (privacy OR security)`
- **Exact Phrase**: `"critical infrastructure"`

## 📁 Project Structure

```
paper-finding/
├── web_app.py                 # Flask application
├── config_loader.py           # Configuration management
├── collectors.py              # Database API collectors
├── database.py                # SQLite database manager
├── deduplication.py           # Duplicate detection
├── exporter.py                # Export functionality
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── templates/
│   ├── index.html            # Main form page
│   └── results.html          # Results display page
├── static/
│   └── style.css             # Styling
└── data/
    └── papers.db             # SQLite database
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Search terms and keywords
- Quality filters
- Rate limiting
- API settings

## 🔑 API Keys (Optional)

For premium databases, set environment variables:

```bash
# Linux/Mac
export IEEE_API_KEY="your_key_here"
export SPRINGER_API_KEY="your_key_here"
export SCIENCEDIRECT_API_KEY="your_key_here"

# Windows (PowerShell)
$env:IEEE_API_KEY="your_key_here"
$env:SPRINGER_API_KEY="your_key_here"
$env:SCIENCEDIRECT_API_KEY="your_key_here"
```

## 📦 Dependencies

- **Flask** - Web framework
- **PyYAML** - Configuration parsing
- **requests** - HTTP client for API calls
- **sqlite3** - Database (built-in)

## 📊 Output Formats

### CSV
- Excel-compatible format
- Includes: Title, Authors, Year, Venue, Abstract, Citations, URL

### JSON
- Structured data with metadata
- Includes collection statistics
- Suitable for data processing pipelines

### BibTeX
- LaTeX citation format
- Auto-detected entry types (@article, @inproceedings, @misc)
- Ready for bibliography compilation

## 🐛 Troubleshooting

**No results found:**
- Try simpler search terms
- Check database selection
- Verify API keys for premium sources

**Rate limiting errors:**
- Wait a few moments and try again
- API providers have rate limits

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version is 3.8+

## 📝 License

MIT License - feel free to use and modify

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Support

For issues and feature requests, please open an issue on GitHub.

---

**Happy researching! 🔬📚**
