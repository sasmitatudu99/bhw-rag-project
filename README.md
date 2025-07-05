## 🔍 RAG Pipeline for Intelligent Recommendations on BlackHatWorld

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** to generate intelligent, context-aware responses from discussions on [BlackHatWorld](https://www.blackhatworld.com/), a popular digital marketing forum. It combines **web scraping**, **natural language processing (NLP)**, **vector similarity search**, and **large language models (LLMs)** to enhance information retrieval and recommendation accuracy.

---

### 🚀 Features

- 🔗 **Web Scraping**: Extracts posts, comments, usernames, timestamps, and engagement metrics from BlackHatWorld using **Selenium** and **BeautifulSoup**.
- 🧹 **Data Cleaning**: Removes noise, special characters, and irrelevant HTML content.
- 📦 **Structured Storage**: Stores cleaned data in **JSON** and **MongoDB** format.
- 🧠 **Embeddings Generation**: Converts text into vector representations using **Hugging Face Sentence Transformers** (`all-MiniLM-L6-v2`).
- 🧮 **Vector Database (ChromaDB)**: Enables fast semantic similarity search.
- 💬 **RAG-based Query System**: Accepts user queries, retrieves relevant BHW discussions using **ChromaDB**, and generates natural responses via **OLLamA 3.2**.
- ⚙️ **Prompt Engineering**: Ensures high-quality, relevant responses through carefully structured prompts.
- 📊 **Use Cases**: Keyword/topic analysis, user-specific queries, sentiment insights, SEO strategy mining.

---

### 🛠️ Tech Stack

| Component            | Technology                         |
|---------------------|------------------------------------|
| Language             | Python                             |
| Web Scraping         | Selenium, BeautifulSoup            |
| Data Storage         | JSON, MongoDB                      |
| Vector DB            | ChromaDB                           |
| Embeddings           | Hugging Face Transformers (MiniLM) |
| LLM for Generation   | OLLamA 3.2                         |
| Frameworks/Libraries | LangChain, Transformers, Pandas    |

---

### 📌 Project Workflow

1. **Data Collection** → Scrape BHW forum threads using Selenium + BeautifulSoup  
2. **Preprocessing** → Clean and format text  
3. **Embedding** → Use Sentence Transformers to vectorize the text  
4. **Storage** → Store embeddings in ChromaDB  
5. **Query Handling** → Convert user query into a vector and retrieve similar discussions  
6. **LLM Response Generation** → Feed context to Ollama LLM to generate user-friendly response  
7. **Output** → Present context-aware answer to the user

---

### 📈 Results

- 🔍 Improved retrieval accuracy over traditional keyword search  
- ⚡ Response time < 500ms  
- ✅ Handles single/multiple user queries, keyword-based, and timestamp queries effectively

---

### 📚 Future Enhancements

- Support for **multi-modal content** (images, videos)
- Add **user feedback loop** for continuous model improvement
- Expand to other digital marketing forums

---




