import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import tiktoken
import os
import fitz  # PyMuPDF

load_dotenv()
# === Token Utility ===

def num_tokens(text, model_name="llama3-8b-8192"):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Approximate
    return len(encoding.encode(text))

# === PDF Parser ===

def extract_text_from_pdf(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === Text Chunker ===

def chunk_text(text, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: num_tokens(x),
    )
    docs = splitter.create_documents([text])
    return docs

# === Strategy Selector ===

def select_strategy(token_count, context_window):
    if token_count < context_window * 0.8:
        return "stuff"
    elif token_count < context_window * 4:
        return "refine"
    else:
        return "map_reduce"

# === Streamlit UI ===

st.set_page_config(page_title="📄 Adaptive Text Summarizer", layout="wide")
st.title("📄 Adaptive Text Summarizer (Groq + LangChain)")
st.caption("Summarize long `.txt` or `.pdf` files using **LLaMA 3 (8B)** on Groq")

uploaded_file = st.file_uploader("Upload a `.txt` or `.pdf` file", type=["txt", "pdf"])
context_window = 8192
model_name = "llama3-8b-8192"

if uploaded_file:
    filetype = uploaded_file.name.split(".")[-1].lower()

    if filetype == "txt":
        text = uploaded_file.read().decode("utf-8")
    elif filetype == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    token_count = num_tokens(text)
    strategy = select_strategy(token_count, context_window)

    st.write(f"📏 Estimated tokens: `{token_count}`")
    st.write(f"🧠 Chosen summarization strategy: **{strategy}**")

    if st.button("🔍 Summarize Text"):
        with st.spinner("Summarizing using LLaMA 3 8B..."):
            llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=model_name)
            docs = chunk_text(text)

            chain = load_summarize_chain(llm, chain_type=strategy)
            summary = chain.run(docs)

        st.subheader("📝 Final Summary")
        st.write(summary)

        st.download_button("⬇️ Download Summary", summary, file_name="summary.txt", mime="text/plain")
