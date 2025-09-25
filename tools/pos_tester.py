#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kompakt POS-tester for tre modeller:
  - NbAiLab/nb-bert-base-pos (HF Transformers)
  - nb_core_news_lg          (spaCy)
  - flair/upos-multi         (Flair)

Skriver en enkel tabell: Model | Text | POS
"""
import argparse

def run_hf(text, model_name, agg):
    from transformers import pipeline
    nlp = pipeline("token-classification", model=model_name, aggregation_strategy=agg)
    out = nlp(text)
    parts = []
    for x in out:
        w   = x.get("word", x.get("token", ""))
        lab = x.get("entity", x.get("entity_group", ""))
        sc  = x.get("score", None)
        parts.append(f"{w}/{lab}" + (f":{sc:.2f}" if sc is not None else ""))
    return f"{model_name} (agg={agg})", " ".join(parts)

def run_spacy(text, model_name):
    import spacy
    try:
        nlp = spacy.load(model_name)
    except Exception:
        import spacy.cli
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    doc = nlp(text)
    parts = [f"{t.text}/{t.pos_}:{t.tag_}" for t in doc if not t.is_space]
    return model_name, " ".join(parts)

def run_flair(text, model_name):
    from flair.models import SequenceTagger
    from flair.data import Sentence
    tagger = SequenceTagger.load(model_name)
    sent = Sentence(text, use_tokenizer=True)
    tagger.predict(sent)
    lt = getattr(tagger, "label_type", "pos")

    parts = []
    for tok in sent:
        # Robust mellom ulike Flair-versjoner
        val, score = None, None
        try:
            lab = tok.get_labels(lt)
            if lab:
                val, score = lab[0].value, float(lab[0].score)
        except Exception:
            try:
                lab = tok.get_label(lt)
                if lab:
                    val, score = lab.value, float(lab.score)
            except Exception:
                try:
                    lab = tok.get_tag(lt)  # eldre API
                    val, score = lab.value, float(lab.score)
                except Exception:
                    val, score = "UNK", 0.0
        parts.append(f"{tok.text}/{val}:{score:.2f}")
    return f"{model_name} (label_type={lt})", " ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Jenta kasta ballen til gutten. Hinduen syntes den kasta han var i var grei")
    ap.add_argument("--which", choices=["all", "hf", "spacy", "flair"], default="all")
    ap.add_argument("--hf_model", default="NbAiLab/nb-bert-base-pos")
    ap.add_argument("--hf_agg", choices=["none", "simple"], default="none")
    ap.add_argument("--spacy_model", default="nb_core_news_lg")
    ap.add_argument("--flair_model", default="flair/upos-multi")
    args = ap.parse_args()

    rows = []
    if args.which in ("all", "hf"):
        m, res = run_hf(args.text, args.hf_model, args.hf_agg)
        rows.append((m, args.text, res))
    if args.which in ("all", "spacy"):
        m, res = run_spacy(args.text, args.spacy_model)
        rows.append((m, args.text, res))
    if args.which in ("all", "flair"):
        m, res = run_flair(args.text, args.flair_model)
        rows.append((m, args.text, res))

    # Enkel, kopierbar tabell (Markdown)
    print("Model | Text | POS")
    print("---|---|---")
    for m, txt, res in rows:
        print(f"{m} | {txt} | {res}")

if __name__ == "__main__":
    main()
