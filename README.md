# Resume Screening Agent (MVP)

This package contains an MVP Resume Screening Agent with two scoring options:
- TF-IDF baseline (`rank_resumes_tfidf.py`) — fast, works offline.
- Sentence-transformer embeddings (`rank_resumes_embeddings.py`) — better accuracy.

Also includes a minimal Streamlit app (`app.py`) for quick demo.

## Quick start

1. Create a Python virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Prepare `job.txt` (job description) and a folder `resumes/` with PDF or TXT resumes.

3. Run TF-IDF scorer:
   ```
   python rank_resumes_tfidf.py --jd job.txt --resumes resumes/ --out results.csv
   ```

4. Run embeddings scorer:
   ```
   python rank_resumes_embeddings.py --jd job.txt --resumes resumes/ --out results_embeddings.csv
   ```

5. Optional: run Streamlit demo:
   ```
   streamlit run app.py
   ```

## Notes
- `sentence-transformers` will download a model (~50-100MB) on first run.
- If you cannot install `pdfplumber`, use TXT resumes or convert PDFs to text first.
- The TF-IDF script uses a simple built-in stopword list to avoid extra NLTK setup.
