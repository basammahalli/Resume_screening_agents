import streamlit as st
import pandas as pd
import subprocess, tempfile, os

st.title("Resume Screening (MVP)")

jd = st.file_uploader("Upload Job Description (txt)", type=["txt"])
res_files = st.file_uploader("Upload Resumes (pdf/txt)", accept_multiple_files=True)
method = st.selectbox("Method", ["tfidf","embeddings"])
if st.button("Run"):
    if not jd or not res_files:
        st.error("Upload JD and resumes")
    else:
        tmpdir = tempfile.mkdtemp()
        jd_path = os.path.join(tmpdir, "job.txt")
        with open(jd_path, "wb") as f:
            f.write(jd.getvalue())
        rdir = os.path.join(tmpdir, "resumes")
        os.makedirs(rdir, exist_ok=True)
        for rf in res_files:
            with open(os.path.join(rdir, rf.name), "wb") as f:
                f.write(rf.getvalue())
        if method=="tfidf":
            cmd = f"python rank_resumes_tfidf.py --jd {jd_path} --resumes {rdir} --out {tmpdir}/out.csv"
        else:
            cmd = f"python rank_resumes_embeddings.py --jd {jd_path} --resumes {rdir} --out {tmpdir}/out.csv"
        try:
            subprocess.check_call(cmd, shell=True)
            df = pd.read_csv(os.path.join(tmpdir,"out.csv"))
            st.dataframe(df)
            st.success("Done")
        except Exception as e:
            st.error(str(e))
