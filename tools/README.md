# AltMorph Tools

This directory contains utility tools for working with AltMorph and Norwegian text processing.

## 🛠️ Available Tools

### `process_jsonl.py` - JSONL Batch Processor

Processes JSON Lines files by adding morphological alternatives to text fields using AltMorph.

**Purpose**: Enable batch processing of large datasets with Norwegian text, adding morphological alternatives to each text entry while preserving all original metadata.

#### Features
- ✅ Batch processing of JSONL files
- ✅ Adds "alt" field with morphological alternatives
- ✅ Preserves all original JSON fields and metadata
- ✅ Robust error handling (continues on individual failures)
- ✅ Progress reporting with 4 verbosity levels
- ✅ Parallel processing support
- ✅ All AltMorph parameters configurable

#### Quick Start
```bash
# Basic usage
python tools/process_jsonl.py --input_file data/sample_input.jsonl --output_file output.jsonl

# With custom settings
python tools/process_jsonl.py \
  --input_file data/texts.jsonl \
  --output_file results.jsonl \
  --api_key "your_api_key" \
  --verbosity 2 \
  --max_workers 8
```

#### Input/Output Format
**Input JSONL:**
```json
{"id": 1, "text": "Katta ligger på matten.", "source": "example"}
{"id": 2, "text": "Gutten løper fort.", "category": "sports"}
```

**Output JSONL:**
```json
{"id": 1, "text": "Katta ligger på matten.", "source": "example", "alt": "{Katta, Katten} ligger på {matten, matta}."}
{"id": 2, "text": "Gutten løper fort.", "category": "sports", "alt": "Gutten løper fort."}
```

#### Command Line Options
| Option | Default | Description |
|--------|---------|-------------|
| `--input_file` | *required* | Input JSONL file with 'text' fields |
| `--output_file` | *required* | Output JSONL file with added 'alt' fields |
| `--lang` | `nob` | Language code (`nob` or `nno`) |
| `--api_key` | `$ORDBANK_API_KEY` | Ordbank API key |
| `--verbosity` | `1` | Verbosity level (0-3) |
| `--logit_threshold` | `3.0` | BERT acceptability threshold |
| `--timeout` | `6.0` | HTTP timeout per request |
| `--max_workers` | `4` | Parallel API requests |

### `pos_tester.py` - POS Tagging Comparison Tool  

Compares Part-of-Speech tagging across multiple Norwegian NLP models.

**Purpose**: Test and compare POS tagging accuracy across different models to help choose the best tagger for your use case or debug tagging issues.

#### Features
- ✅ Tests 3 different POS tagging approaches:
  - **HuggingFace Transformers**: `NbAiLab/nb-bert-base-pos` (Norwegian BERT)
  - **spaCy**: `nb_core_news_lg` (Norwegian language model)
  - **Flair**: `flair/upos-multi` (Multilingual Universal POS)
- ✅ Outputs comparison table in Markdown format
- ✅ Configurable models and text input
- ✅ Shows confidence scores for each prediction
- ✅ Robust handling of different model versions

#### Quick Start
```bash
# Test with default Norwegian sentence
python tools/pos_tester.py

# Test with custom text
python tools/pos_tester.py --text "Katta ligger på matten og sover."

# Test only specific model
python tools/pos_tester.py --which hf --text "Min egen setning."
```

#### Example Output
```
Model | Text | POS
---|---|---
NbAiLab/nb-bert-base-pos (agg=none) | Katta ligger på matten. | Katta/NOUN ligger/VERB på/ADP matten/NOUN ./PUNCT
nb_core_news_lg | Katta ligger på matten. | Katta/NOUN:NN ligger/VERB:VERB på/ADP:PREP matten/NOUN:NN ./.:/
flair/upos-multi (label_type=pos) | Katta ligger på matten. | Katta/NOUN:0.99 ligger/VERB:0.98 på/ADP:0.99 matten/NOUN:0.97 ./PUNCT:1.00
```

#### Command Line Options  
| Option | Default | Description |
|--------|---------|-------------|
| `--text` | Norwegian test sentence | Input text to analyze |
| `--which` | `all` | Models to test: `all`, `hf`, `spacy`, `flair` |
| `--hf_model` | `NbAiLab/nb-bert-base-pos` | HuggingFace model name |
| `--hf_agg` | `none` | HuggingFace aggregation strategy |
| `--spacy_model` | `nb_core_news_lg` | spaCy model name |
| `--flair_model` | `flair/upos-multi` | Flair model name |

## 📁 Project Structure

```
tools/
├── README.md              # This file
├── process_jsonl.py       # JSONL batch processor  
├── pos_tester.py         # POS tagging comparison
└── example_usage.md      # Detailed JSONL processing examples

data/
└── sample_input.jsonl    # Sample data for testing
```

## 🚀 Usage Workflows

### 1. Dataset Processing Pipeline
```bash
# 1. Prepare your JSONL data (each line: {"text": "Norwegian sentence", ...})
# 2. Process through AltMorph
python tools/process_jsonl.py \
  --input_file data/my_dataset.jsonl \
  --output_file results/enhanced_dataset.jsonl \
  --verbosity 2

# 3. Results include original data + "alt" field with morphological alternatives
```

### 2. Model Comparison Workflow  
```bash
# Test POS tagging quality on your text
python tools/pos_tester.py --text "Your Norwegian text here."

# Compare results and choose best model for your needs
# Use the insights to configure AltMorph POS settings if needed
```

## 📋 Requirements

### For `process_jsonl.py`:
- All AltMorph dependencies (see main README.md)
- Ordbank API key
- Input JSONL file with "text" fields

### For `pos_tester.py`:
- **HuggingFace**: `transformers` library
- **spaCy**: `spacy` + `nb_core_news_lg` model  
- **Flair**: `flair` library
- *Note: Script will attempt to auto-install missing spaCy models*

## 🐛 Troubleshooting

### JSONL Processor Issues
- **Import Error**: Ensure `altmorph.py` is in the parent directory
- **API Errors**: Check your Ordbank API key and internet connection  
- **Memory Issues**: Reduce `--max_workers` for large files
- **Invalid JSON**: Use `--verbosity 1+` to see which lines are problematic

### POS Tester Issues
- **Model Download Errors**: Run with good internet connection, models are large
- **Missing Dependencies**: Install required libraries (`pip install transformers spacy flair`)
- **spaCy Model Issues**: Script will try auto-download, or run `python -m spacy download nb_core_news_lg`

## 🔗 Integration

Both tools are designed to work alongside the main AltMorph application:

- **`process_jsonl.py`** enables production batch processing of datasets
- **`pos_tester.py`** helps validate and debug the POS tagging that AltMorph relies on

See the main project README.md for more details about AltMorph itself.

## 🔧 Additional Utilities

### `hf_probe_fields.py` - HuggingFace Dataset Inspector
Quick utility to probe HuggingFace dataset fields and stream sample rows.

### `stream_ncc_text.py` - NCC Speech Text Streamer  
Streams text from NbAiLab/ncc_speech_v7 dataset without downloading audio files.
