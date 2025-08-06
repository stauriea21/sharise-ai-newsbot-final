# ✨ L02 – NLP Preprocessing Techniques: From Raw Text to Real Insight

**Course:** ITAI 2373 – Natural Language Processing  
**Student:** Sharise Griggs  
**Assignment:** Lab 02 – Text Cleaning & Preprocessing  
**Instructor:** Patricia McManus  
**Score:** ✅ 98 / 100

---

## 📖 Project Overview

This lab was my official dive into the world of **text preprocessing** — the behind-the-scenes magic that transforms messy, human language into clean, usable data for machines to understand.

I learned quickly that this isn’t just a “setup step” in NLP... it's where **meaning is made or destroyed**. Every decision here — from removing stopwords to choosing lemmatization over stemming — can make or break your results.

---

## 💡 What I Explored

| 🔧 Task                        | 💬 My Insight |
|-------------------------------|----------------|
| Tokenization                  | Split text into usable units using NLTK & spaCy |
| Stopword Filtering            | Realized that removing “not” can destroy sentiment |
| Stemming vs. Lemmatization    | Saw how aggressive stemming distorts meaning |
| spaCy vs. NLTK                | spaCy handled slang, emojis, and hashtags better |
| Error Handling (NLTK `punkt`) | Learned how to debug tokenizer failures in Colab |
| Custom Cleaning Functions     | Built toggleable pipelines for "light" or "heavy" preprocessing |

---

## 🌟 Reflection Snippet

> “If you mess up here, your whole model might fall apart — no matter how fancy it is.”

This lab made it clear that preprocessing isn't boring boilerplate — it's where your model learns what *not* to forget. I now approach cleaning with respect and flexibility depending on context (e.g., preserving emojis in mental health journaling apps).

---

## 🧠 Tools & Technologies Used

- 🐍 Python 3.x  
- 📚 NLTK (Natural Language Toolkit)  
- 💬 spaCy  
- 🧪 Google Colab (Jupyter environment)

---

## 🧪 How to Run This Notebook

### 🔧 Installation Requirements

```bash
pip install nltk
pip install spacy
python -m nltk.downloader all
python -m spacy download en_core_web_sm
