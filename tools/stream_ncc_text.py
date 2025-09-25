#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream {id, text} from NbAiLab/ncc_speech_v7 without downloading/decoding audio.

Hardcoded filters:
  - source == "stortinget"
  - language == "no"

Hardcoded behavior:
  - streaming=True
  - trust_remote_code=True
  - token=True (uses cached HF auth or HF_TOKEN env)
  - strip text and skip empty/whitespace entries
  - select only [id, text, source, language] to avoid audio overhead

Usage (your desired default):
  ./stream_ncc_text.py \
    --dataset NbAiLab/ncc_speech_v7 \
    --config no \
    --split train \
    --output train.jsonl

Optional:
  --max-rows N   # If omitted or 0, processes the entire (filtered) split.
  --debug
"""
from __future__ import annotations
import argparse
import io
import json
import logging
import os
import sys
from typing import Any, Dict

# Hardcoded field names and filters
ID_FIELD = "id"
TEXT_FIELD = "text"
SOURCE_FIELD = "source"
SOURCE_VALUE = "stortinget"
LANG_FIELD = "language"
LANG_VALUE = "no"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream {id, text} filtered by source=='stortinget' and language=='no' (JSONL out)"
    )
    p.add_argument("--dataset", required=True, help="HF dataset id, e.g. NbAiLab/ncc_speech_v7")
    p.add_argument("--config", required=True, help="Config name, e.g. 'no'")
    p.add_argument("--split", required=True, help="Split to stream, e.g. 'train'")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--max-rows", type=int, default=0,
                   help="Max matches to write (0 = no limit; default: 0)")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args()

def _load_streaming(dataset: str, config: str, split: str):
    from datasets import load_dataset
    # Hardcode streaming, trust_remote_code, and token usage
    kw: Dict[str, Any] = {
        "name": config,
        "split": split,
        "streaming": True,
        "trust_remote_code": True,
        "token": True,  # newer datasets; fallback below if older API
    }
    try:
        return load_dataset(dataset, **kw)
    except TypeError:
        kw.pop("token", None)
        kw["use_auth_token"] = True
        return load_dataset(dataset, **kw)

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(message)s",
        force=True,
    )

    # Optional: enable faster hub downloader where applicable
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    try:
        ds = _load_streaming(args.dataset, args.config, args.split)
    except Exception as e:
        logging.error("Failed to open dataset '%s' (config=%s, split=%s): %s",
                      args.dataset, args.config, args.split, e)
        sys.exit(3)

    wanted_cols = sorted({ID_FIELD, TEXT_FIELD, SOURCE_FIELD, LANG_FIELD})
    try:
        ds = ds.select_columns(wanted_cols)
        logging.debug("Selected columns: %s", wanted_cols)
    except Exception as e:
        logging.warning("select_columns failed (%s). Proceeding without pruning.", e)

    # Sanity check against features, if present
    try:
        feats = getattr(ds, "features", None)
        if feats is not None:
            missing = [c for c in wanted_cols if c not in feats]
            if missing:
                logging.warning("Expected columns missing in features: %s. Streaming best-effort.",
                                ", ".join(missing))
    except Exception:
        pass

    limit = int(args.max_rows) if args.max_rows and args.max_rows > 0 else 0
    written = 0
    checked = 0

    # Buffered text IO for throughput
    with io.open(args.output, "w", encoding="utf-8", buffering=1024 * 1024) as fout:
        try:
            for ex in ds:
                checked += 1
                # Strict filters
                if ex.get(SOURCE_FIELD) != SOURCE_VALUE:
                    continue
                if ex.get(LANG_FIELD) != LANG_VALUE:
                    continue

                _id = ex.get(ID_FIELD)
                _text = ex.get(TEXT_FIELD)
                if _id is None or _text is None:
                    continue

                _text = _text.strip()
                if not _text:
                    continue

                fout.write(json.dumps({"id": _id, "text": _text}, ensure_ascii=False) + "\n")
                written += 1
                if limit and written >= limit:
                    break
        except KeyboardInterrupt:
            logging.warning("Interrupted. Wrote %d rows.", written)
            sys.exit(130)
        except Exception as e:
            logging.error("Write failed after %d rows: %s", written, e)
            sys.exit(4)

    logging.info(
        "Done. Wrote %d rows to %s. Scanned %d records. Filters: %s=='%s', %s=='%s'.%s",
        written, args.output, checked,
        SOURCE_FIELD, SOURCE_VALUE, LANG_FIELD, LANG_VALUE,
        "" if limit == 0 else f" (stopped at max_rows={limit})"
    )

if __name__ == "__main__":
    main()
