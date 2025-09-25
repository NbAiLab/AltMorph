# JSONL Processing with AltMorph

## Overview
The `process_jsonl.py` script processes JSON Lines files by adding morphological alternatives to text fields using AltMorph.

## Quick Start

### 1. Basic Usage
```bash
python process_jsonl.py --input_file sample_input.jsonl --output_file output.jsonl
```

### 2. With API Key
```bash
python process_jsonl.py \
  --input_file sample_input.jsonl \
  --output_file output.jsonl \
  --api_key "your_api_key_here"
```

### 3. Verbose Processing
```bash
python process_jsonl.py \
  --input_file sample_input.jsonl \
  --output_file output.jsonl \
  --verbosity 2
```

## Input Format
Each line should be a JSON object with a "text" field:
```json
{"id": 1, "text": "Katta ligger på matten.", "source": "example"}
{"id": 2, "text": "Gutten løper fort.", "category": "sports"}
```

## Output Format
Same as input but with added "alt" field:
```json
{"id": 1, "text": "Katta ligger på matten.", "source": "example", "alt": "{Katta, Katten} ligger på {matten, matta}."}
{"id": 2, "text": "Gutten løper fort.", "category": "sports", "alt": "Gutten løper fort."}
```

## Options
- `--verbosity 0`: Quiet (errors only)
- `--verbosity 1`: Normal progress info (default)
- `--verbosity 2`: Verbose processing details  
- `--verbosity 3`: Very verbose with full AltMorph output
- `--max_workers N`: Parallel API requests (default: 4)
- `--lang nno`: Use Nynorsk instead of Bokmål
