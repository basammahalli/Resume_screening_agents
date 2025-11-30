#!/usr/bin/env python3
"""
Resume screening using sentence-transformers embeddings (local).
Usage:
  python rank_resumes_embeddings.py --jd job.txt --resumes resumes_folder --out results_embeddings.csv
"""
import argparse, os, glob, re
try:
    import pdfplumber
except Exception:
    pdfplumber = None
import pandas as pd
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"

def extract_text_from_pdf(path):
    text = []
    if pdfplumber is None:
        return ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text.append(txt)
    return "\\n".join(text)

def load_resumes(folder):
    docs = {}
    for ext in ('*.pdf','*.txt'):
        for p in glob.glob(os.path.join(folder, ext)):
            name = os.path.basename(p)
            if p.lower().endswith('.pdf'):
                txt = extract_text_from_pdf(p)
            else:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
            docs[name] = txt
    return docs

def chunk_sentences(text):
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if len(s.strip())>20]
    return sents if len(sents)>0 else [text[:200]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", required=True)
    parser.add_argument("--resumes", required=True)
    parser.add_argument("--out", default="results_embeddings.csv")
    args = parser.parse_args()

    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    resumes = load_resumes(args.resumes)
    if len(resumes) == 0:
        print("No resumes found in", args.resumes)
        return

    model = SentenceTransformer(MODEL_NAME)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    rows = []
    for name, text in resumes.items():
        resume_emb = model.encode(text, convert_to_tensor=True)
        score = float(util.cos_sim(jd_emb, resume_emb).item())
        sents = chunk_sentences(text)
        sent_embs = model.encode(sents, convert_to_tensor=True)
        sims = util.cos_sim(jd_emb, sent_embs)[0].cpu().numpy()
        top_idx = sims.argsort()[-3:][::-1]
        top_sents = [sents[i] for i in top_idx]
        rows.append({"resume": name, "score": score, 
                     "top_sent_1": top_sents[0] if len(top_sents)>0 else "", 
                     "top_sent_2": top_sents[1] if len(top_sents)>1 else ""})
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(args.out, index=False)
    print("Wrote", args.out)
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
