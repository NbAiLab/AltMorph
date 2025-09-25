#!/usr/bin/env python3
"""
Process JSONL files with AltMorph: Add morphological alternatives to text fields.

This script reads a JSONL (JSON Lines) file where each line contains a JSON object
with a "text" field. It processes each text through AltMorph to generate morphological
alternatives and adds the result as an "alt" field.

Usage Examples:
    python process_jsonl.py --input_file data.jsonl --output_file enhanced.jsonl
    python process_jsonl.py --input_file texts.jsonl --api_key your_key --lang nno
    python process_jsonl.py --input_file large.jsonl --verbosity 2 --max_workers 8

Requirements:
    - Input JSONL file with "text" field in each JSON object
    - Ordbank API key (via --api_key or ORDBANK_API_KEY environment variable)
    - altmorph.py in the same directory

Output:
    - JSONL file with original fields plus "alt" field containing alternatives
    - Progress information based on verbosity level
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Import altmorph functions from parent directory
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from altmorph import process_sentence
except ImportError:
    print("Error: Cannot import altmorph. Ensure altmorph.py is in the parent directory.")
    sys.exit(1)


def process_jsonl_file(input_file: str, output_file: str, lang: str, api_key: str,
                      timeout: float, max_workers: int, verbosity: int, 
                      logit_threshold: float) -> None:
    """
    Process JSONL file by adding morphological alternatives to each text field.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        lang: Language code (nob or nno)
        api_key: Ordbank API key
        timeout: HTTP timeout per request
        max_workers: Number of parallel API requests
        verbosity: Verbosity level (0-3)
        logit_threshold: BERT acceptability threshold
    """
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not api_key.strip():
        raise ValueError("API key required. Set ORDBANK_API_KEY environment variable or use --api_key")
    
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON line
                data = json.loads(line)
                
                # Check if "text" field exists
                if "text" not in data:
                    if verbosity >= 1:
                        print(f"Warning: Line {line_num} missing 'text' field, skipping")
                    continue
                
                text = data["text"]
                if not isinstance(text, str) or not text.strip():
                    if verbosity >= 1:
                        print(f"Warning: Line {line_num} has empty/invalid text, skipping")
                    continue
                
                # Process text with AltMorph
                if verbosity >= 2:
                    print(f"Processing line {line_num}: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                alt_text = process_sentence(
                    sentence=text,
                    lang=lang,
                    api_key=api_key,
                    timeout=timeout,
                    max_workers=max_workers,
                    verbosity=max(0, verbosity - 2),  # Reduce verbosity for process_sentence
                    logit_threshold=logit_threshold
                )
                
                # Add "alt" field to data
                data["alt"] = alt_text
                
                # Write enhanced JSON line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
                if verbosity >= 3:
                    print(f"Line {line_num} result: {alt_text}")
                    
            except json.JSONDecodeError as e:
                error_count += 1
                if verbosity >= 1:
                    print(f"Error: Line {line_num} invalid JSON: {e}")
                continue
                
            except Exception as e:
                error_count += 1
                if verbosity >= 1:
                    print(f"Error processing line {line_num}: {e}")
                continue
    
    if verbosity >= 1:
        print(f"Processing complete: {processed_count} lines processed, {error_count} errors")


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process JSONL files with AltMorph morphological alternatives"
    )
    
    parser.add_argument("--input_file", required=True,
                       help="Input JSONL file with 'text' fields")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSONL file with added 'alt' fields")
    parser.add_argument("--lang", default="nob", choices=["nob", "nno"],
                       help="Language code (default: nob)")
    parser.add_argument("--api_key", default=os.getenv("ORDBANK_API_KEY", ""),
                       help="Ordbank API key (or set ORDBANK_API_KEY)")
    parser.add_argument("--timeout", type=float, default=6.0,
                       help="HTTP timeout per request (default: 6.0)")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Parallel API requests (default: 4)")
    parser.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2, 3],
                       help="Verbosity level: 0=quiet, 1=normal, 2=verbose, 3=very verbose (default: 1)")
    parser.add_argument("--logit_threshold", type=float, default=3.0,
                       help="BERT acceptability threshold (default: 3.0)")
    
    args = parser.parse_args()
    
    # Configure logging
    log_levels = {
        0: logging.ERROR,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG
    }
    
    logging.basicConfig(
        level=log_levels.get(args.verbosity, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    try:
        process_jsonl_file(
            input_file=args.input_file,
            output_file=args.output_file,
            lang=args.lang,
            api_key=args.api_key,
            timeout=args.timeout,
            max_workers=args.max_workers,
            verbosity=args.verbosity,
            logit_threshold=args.logit_threshold
        )
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
