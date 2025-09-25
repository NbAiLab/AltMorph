# AltMorph: Context-Aware Norwegian Morphological Alternative Generator

**AltMorph** is a sophisticated tool for expanding Norwegian text by finding morphological alternatives for each word. It combines the Ordbank API with advanced NLP techniques to provide contextually appropriate word alternatives.

## ✨ Features

- **🎯 Context-sensitive filtering**: Intelligently handles ambiguous cases using BERT-based acceptability scoring
- **📚 Comprehensive lemma coverage**: Finds all valid morphological forms from multiple lemmas
- **🔍 Position-specific analysis**: Analyzes each word occurrence in its specific syntactic context  
- **⚡ Intelligent caching**: Dramatic performance improvements with persistent file-based caching
- **🗣️ Multiple verbosity levels**: From silent operation to detailed pipeline insights
- **🌐 Language support**: Norwegian Bokmål (`nob`) and Nynorsk (`nno`)
- **🧠 POS-aware**: Uses NbAiLab BERT models for accurate part-of-speech tagging
- **🚀 Parallel processing**: Efficient concurrent API calls

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Ordbank API key (free registration at [Ordbank](https://www.ordbank.no/))

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Get API Key
1. Register at [https://www.ordbank.no/](https://www.ordbank.no/)
2. Obtain your API key from your account dashboard
3. Set the environment variable:
   ```bash
   export ORDBANK_API_KEY="your_api_key_here"
   ```
   Or pass it directly with `--api_key` flag

## 🚀 Quick Start

### Basic Usage
```bash
python altmorph.py --sentence "Katta ligger på matta." --lang nob
```
**Output:**
```
"{Katta, Katten} ligger på {matten, matta}."
```

### With API Key
```bash
python altmorph.py \
  --sentence "Katta ligger på matta." \
  --lang nob \
  --api_key "your_api_key_here"
```

## 📖 Usage Examples

### Context-Sensitive Intelligence
The tool demonstrates sophisticated linguistic understanding:

**Simple example:**
```bash
python altmorph.py --sentence "Katta ligger på matta." --lang nob
# Output: "{Katta, Katten} ligger på {matta, matten}."
# Shows different morphological forms for the same words
```

**Complex context:**
```bash
python altmorph.py --sentence "Katta ligger på matta i stua." --lang nob  
# Output: "{Katta, Katten} ligger på {matta, matten} i stua."
# BERT contextual filtering ensures appropriate alternatives in context
```

### Position-Specific Analysis
```bash
python altmorph.py --sentence "Katta ligger på matta." --lang nob
# Each word occurrence is analyzed in its specific syntactic context
```

## 🎛️ Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sentence` | *required* | Input sentence to process |
| `--lang` | `nob` | Language code (`nob` or `nno`) |
| `--api_key` | `$ORDBANK_API_KEY` | Ordbank API key |
| `--verbosity` | `0` | Verbosity level (0-3) |
| `--logit-threshold` | `3.0` | BERT acceptability threshold |
| `--timeout` | `6.0` | HTTP timeout per request |
| `--max_workers` | `4` | Parallel API requests |
| `--no-cache` | `False` | Disable caching |
| `--delete-cache` | `False` | Clear cache and exit |

## 🔊 Verbosity Levels

### Level 0: Quiet (Default)
```bash
python altmorph.py --sentence "Katta ligger på matta." --verbosity 0
```
**Output:** Just the final result
```
"{Katta, Katten} ligger på {matta, matten}."
```

### Level 1: Normal  
```bash
python altmorph.py --sentence "Katta ligger på matta." --verbosity 1
```
**Output:** Basic progress information
```
2025-XX-XX 12:00:00 INFO Loading POS tagger...
2025-XX-XX 12:00:02 INFO POS tagger loaded
"{Katta, Katten} ligger på {matta, matten}."
```

### Level 2: Verbose
```bash
python altmorph.py --sentence "Katta ligger på matta." --verbosity 2
```
**Output:** Processing details (POS tags, API lookups, alternatives found)
```
🎯 PROCESSING: Katta ligger på matta.
📝 WORDS: ['katta', 'ligger', 'på', 'matta']
🏷️ POS TAGS:
   katta: NOUN
   ligger: VERB
   på: ADP
   matta: NOUN
📡 API LOOKUP: katta (POS: NOUN)
   ✅ katta: 2 alternatives: ['katta', 'katten']
...
✨ RESULT: "{Katta, Katten} ligger på {matta, matten}."
```

### Level 3: Very Verbose
```bash
python altmorph.py --sentence "Katta ligger på matta." --verbosity 3
```
**Output:** Everything including cache operations, lemma analysis, BERT filtering
```
🎯 PROCESSING: Katta ligger på matta.
📝 FOUND 2 LEMMAS for katta
💾 CACHE HIT: lemmas for 'katta' (POS: NOUN)
🧠 ACCEPTABILITY FILTERING (threshold: 3.00)
🔍 ANALYZING: katta (position 0)
   Context: [Katta] ligger på matta.
   Alternatives: ['katta', 'katten']
📊 CACHE STATS: 8 hits, 0 misses (100.0% hit rate)
...
```

## 🗂️ Caching System

AltMorph includes an intelligent caching system that dramatically improves performance:

- **Cache location:** `~/.ordbank_cache/`
- **Cache types:** Lemma searches and inflection data
- **Performance:** ~95%+ hit rate for repeated usage
- **Management:** 
  - `--no-cache`: Disable caching
  - `--delete-cache`: Clear all cache files

**Performance impact:**
- First run: ~3-4 seconds (API calls)
- Cached runs: ~0.5 seconds (near-instant)

## 🧠 Technical Details

### Code Architecture Deep-Dive
📖 **[Complete Code Walkthrough](code_explanation.md)** - Detailed technical explanation of how AltMorph works, perfect for developers wanting to understand the implementation.

### Architecture
1. **Input Processing**: Tokenization preserving whitespace and punctuation
2. **POS Tagging**: NbAiLab/nb-bert-base-pos for accurate grammatical analysis
3. **Lemma Discovery**: Comprehensive search across all relevant Ordbank lemmas
4. **Inflection Analysis**: Full morphological paradigm extraction
5. **Acceptability Scoring**: NbAiLab/nb-bert-base for context-sensitive filtering
6. **Output Generation**: Case-preserving alternative presentation

### Models Used
- **POS Tagging**: `NbAiLab/nb-bert-base-pos`
- **Acceptability**: `NbAiLab/nb-bert-base` 
- **API**: [Ordbank](https://www.ordbank.no/) - Norwegian morphological database

### Key Algorithms
- **Comprehensive lemma matching**: Finds all lemmas containing target word
- **Position-specific analysis**: Each word occurrence analyzed in context
- **Logit-based filtering**: Robust acceptability thresholding (default: 3.0)
- **Intelligent prioritization**: Balances morphological completeness with contextual appropriateness

## 📊 Performance

### Typical Performance
- **Single sentence**: 0.5-4 seconds (depending on cache state)
- **Cache hit rate**: Typically 95%+ for repeated usage
- **API efficiency**: Parallel requests with intelligent batching
- **Memory usage**: ~500MB (loaded BERT models)

### Scaling Considerations
- **Concurrent requests**: Configurable via `--max_workers`
- **Timeout handling**: Robust error recovery with retries
- **Rate limiting**: Respectful API usage patterns

## 🔧 Development

### Project Structure
```
altmorph/
├── altmorph.py              # Main application
├── tools/
│   └── pos_tester.py        # POS tagging comparison tool
├── README.md                # Documentation
├── setup.py                 # Legacy packaging
├── pyproject.toml          # Modern packaging
├── requirements.txt         # Dependencies
└── ~/.ordbank_cache/        # Cache directory (auto-created)
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows existing style
5. Submit a pull request

### Testing
```bash
# Test basic functionality
python altmorph.py --sentence "Katta ligger på matta." --lang nob

# Test cache functionality  
python altmorph.py --delete-cache
python altmorph.py --sentence "Katta ligger på matta." --lang nob --verbosity 3

# Test without cache
python altmorph.py --sentence "Katta ligger på matta." --lang nob --no-cache

# Test POS comparison tool
python tools/pos_tester.py --text "Katta ligger på matta."
```

## 🤝 Related Projects

- **[AltWER](https://github.com/peregilk/altwer)**: Depends on AltMorph's output format for Norwegian text evaluation

## 📄 License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)


## 🙏 Acknowledgments

- **Ordbank Team**: For providing the comprehensive Norwegian morphological API
- **Clarino/UiB**: For hosting the API infrastructure
- **NbAiLab**: For the Norwegian BERT models
- **AltMorph**: Idea and coding by Magnus Breder Birkenes and Per Egil Kummervold
