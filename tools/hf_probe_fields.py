#!/usr/bin/env python3
"""
Probe HF dataset fields (features, splits) and stream a few rows.

Examples:
  ./hf_probe_fields.py --dataset NbAiLab/ncc_speech_v7 --split train --max-rows 5 \
    --use-token --trust-remote-code
"""
from __future__ import annotations
import argparse, json, logging, sys
from itertools import islice

def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--max-rows", type=int, default=3)
    p.add_argument("--use-token", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--debug", action="store_true")

def _builder(args):
    from datasets import load_dataset_builder, get_dataset_config_names
    kw = {}
    if args.trust_remote_code:
        kw["trust_remote_code"] = True
    if args.use_token:
        # Prefer new param if present, else legacy kw
        try:
            kw["token"] = True
        except TypeError:
            kw["use_auth_token"] = True
    try:
        b = load_dataset_builder(args.dataset, name=args.config, **kw)
    except TypeError:
        # Fallback for older versions
        if args.use_token:
            b = load_dataset_builder(args.dataset, name=args.config, use_auth_token=True,
                                     **({"trust_remote_code": True} if args.trust_remote_code else {}))
        else:
            b = load_dataset_builder(args.dataset, name=args.config,
                                     **({"trust_remote_code": True} if args.trust_remote_code else {}))
    # Config list
    try:
        cfgs = get_dataset_config_names(args.dataset, **({ "token": True } if args.use_token else {}),
                                        **({ "trust_remote_code": True } if args.trust_remote_code else {}))
    except TypeError:
        cfgs = get_dataset_config_names(args.dataset,
                                        **({ "use_auth_token": True } if args.use_token else {}),
                                        **({ "trust_remote_code": True } if args.trust_remote_code else {}))
    return b, cfgs

def _stream(args):
    from datasets import load_dataset
    kw = dict(streaming=True)
    if args.config:
        kw["name"] = args.config
    if args.split:
        kw["split"] = args.split
    if args.trust_remote_code:
        kw["trust_remote_code"] = True
    if args.use_token:
        try:
            kw["token"] = True
        except TypeError:
            kw["use_auth_token"] = True
    return load_dataset(args.dataset, **kw)

def _short(v, n=160):
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."

def main():
    ap = argparse.ArgumentParser(description="HF dataset field probe")
    _add_args(ap); args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s | %(message)s")

    try:
        builder, cfgs = _builder(args)
    except Exception as e:
        logging.error("Failed to load dataset builder: %s", e); sys.exit(2)

    info = builder.info
    print("# === DATASET BASIC INFO ===")
    print(json.dumps({
        "dataset": args.dataset,
        "available_configs": cfgs,
        "splits": {k: dict(num_examples=v.num_examples) for k, v in (info.splits or {}).items()},
    }, ensure_ascii=False, indent=2))

    print("\n# === FEATURES (schema) ===")
    try:
        feat_dict = info.features.to_dict() if info.features is not None else None
    except Exception:
        feat_dict = None
    print(json.dumps(feat_dict, ensure_ascii=False, indent=2))

    if args.split:
        try:
            ds = _stream(args)
        except Exception as e:
            logging.error("Failed to open streamed split '%s': %s", args.split, e); sys.exit(3)
        print(f"\n# === FIRST {args.max_rows} ROWS FROM split='{args.split}' (streamed) ===")
        for i, ex in enumerate(islice(ds, args.max_rows), 1):
            preview = {}
            for k, v in ex.items():
                if isinstance(v, (str, bytes, int, float, bool)) or v is None:
                    preview[k] = _short(v)
                else:
                    preview[k] = f"<{type(v).__name__}>"
            print(json.dumps({"row": i, "keys": list(ex.keys()), "preview": preview},
                             ensure_ascii=False, indent=2))
    else:
        print("\n# (No split specified — skip row preview.)")

    # Heuristics for id/text
    cand_text, cand_id = [], []
    if info.features is not None:
        for name, feat in info.features.items():
            tname = type(feat).__name__
            nlow = name.lower()
            if tname == "Value" and getattr(feat, "dtype", "string").startswith("string"):
                if nlow in ("text","transcription","sentence","normalized_text","target_text"):
                    cand_text.append(name)
            if nlow in ("id","audio_id","utt_id","utterance_id","segment_id","sample_id","guid"):
                cand_id.append(name)
    print("\n# === HEURISTICS ===")
    print(json.dumps({"possible_text_fields": cand_text, "possible_id_fields": cand_id},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
