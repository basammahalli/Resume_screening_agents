#!/usr/bin/env python3
"""
Simple Resume Screening (TF-IDF).
Usage:
  python rank_resumes_tfidf.py --jd job.txt --resumes resumes_folder --out results.csv
"""
import argparse, os, glob, re
try:
    import pdfplumber
except Exception:
    pdfplumber = None
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional: basic stopword list to avoid large nltk dependency
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me
more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've this those through to too under until
up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while
who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

def extract_text_from_pdf(path):
    text = []
    if pdfplumber is None:
        return ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\\n".join(text)

def clean_text(t):
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r'[\\r\\n]+', ' ', t)
    t = re.sub(r'[^a-z0-9\\s]', ' ', t)
    tokens = [w for w in t.split() if w not in STOPWORDS and len(w)>2]
    return " ".join(tokens)

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
            docs[name] = clean_text(txt)
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", required=True)
    parser.add_argument("--resumes", required=True)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    with open(args.jd, 'r', encoding='utf-8') as f:
        jd = clean_text(f.read())

    resumes = load_resumes(args.resumes)
    if len(resumes) == 0:
        print("No resumes found in", args.resumes)
        return
    names = list(resumes.keys())
    docs = [resumes[n] for n in names]

    # TF-IDF on resumes + JD
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    X = vect.fit_transform(docs + [jd])  # last row is JD
    jd_vec = X[-1]
    res_vecs = X[:-1]
    sims = cosine_similarity(res_vecs, jd_vec).reshape(-1)
    ranked = sorted(zip(names, sims), key=lambda x: x[1], reverse=True)

    # For explainability: find top matching keywords per resume
    feature_names = vect.get_feature_names_out()
    jd_tokens = set(jd.split())
    rows = []
    for name, score in ranked:
        vec = vect.transform([resumes[name]]).toarray()[0]
        top_idx = vec.argsort()[-10:][::-1]
        top_feats = [feature_names[i] for i in top_idx if vec[i] > 0]
        matched = [w for w in top_feats if any(tok in w for tok in jd_tokens)]
        rows.append({"resume": name, "score": float(score), "top_matches": ";".join(matched[:5])})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Wrote", args.out)
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
