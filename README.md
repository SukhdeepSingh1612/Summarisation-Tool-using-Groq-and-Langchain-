# 📄 Adaptive Text Summarizer (LangChain + Groq + Streamlit)

This is a simple Streamlit app that performs **adaptive text summarization** using **LangChain**, **Groq API**, and **Meta’s LLaMA 3 8B** model. It supports `.txt` and `.pdf` files, dynamically selecting the best summarization strategy based on input size.

---

## 🚀 Features

✅ Summarize `.txt` and `.pdf` files  
✅ Uses Groq's blazing-fast **LLaMA 3 8B** model  
✅ Automatically chooses the best LangChain summarization chain:
- `stuff` for short text  
- `refine` for medium text  
- `map_reduce` for long documents  
✅ Downloads summary as `.txt`  
✅ Handles Groq rate limits gracefully  

---

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/adaptive-summarizer
cd adaptive-summarizer
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit langchain langchain-groq pymupdf tiktoken
```

3. **Set your Groq API key:**

```bash
export GROQ_API_KEY=your_groq_api_key
```

---

## 🧠 Model Used

- `llama3-8b-8192`
- Provided by [Groq API](https://console.groq.com)
- Context window: 8192 tokens

---

## ▶️ Run the App

```bash
streamlit run adaptive_summarizer_app.py
```

---

## 📂 File Structure

```
adaptive_summarizer_app.py     # Main app code
README.md                      # You are here
requirements.txt               # Package dependencies
```

---

## 📄 License

MIT License. Use freely with credit.

---
