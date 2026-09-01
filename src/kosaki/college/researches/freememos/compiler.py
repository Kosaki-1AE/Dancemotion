import subprocess
import sys

import streamlit as st


# pip install all required packages
def install(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

st.title("ローカルコード実行環境")

language = st.selectbox("言語を選択", ["Python", "C", "JavaScript", "html", "css"])
code = st.text_area("コードを入力")

if st.button("実行"):
    if language == "Python":
        try:
            exec(code)
        except Exception as e:
            st.error(str(e))
    elif language == "C":
        with open("temp.c", "w") as f:
            f.write(code)
        subprocess.run(["gcc", "temp.c", "-o", "temp.out"])
        result = subprocess.run(["./temp.out"], capture_output=True, text=True)
        st.text(result.stdout)
    elif language == "JavaScript":
        JS_FILE = "temp.js"
        with open(JS_FILE, "w") as f:
            f.write(code)
        result = subprocess.run(["node", JS_FILE], capture_output=True, text=True)
        st.text(result.stdout)
    elif language == "html":
        with open("temp.html", "w") as f:
            f.write(code)
        result = subprocess.run(["html", "temp.html"], capture_output=True, text=True)
        st.text(result.stdout)
    elif language == "css":
        with open("temp.css", "w") as f:
            f.write(code)
        st.text("CSS code executed, but no output to display.")
