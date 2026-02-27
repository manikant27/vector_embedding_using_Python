# vector_embedding_using_Python
# 🧠 Word Embedding Project (Mini GPT-Style Language Model)

This project demonstrates the foundational working of a GPT-style language model built from scratch using **PyTorch** and a **Hindi language dataset**.

The goal of this project is to understand how modern language models convert text into numerical representations and perform **next-token prediction** using embeddings.

---
Dataset Link:   https://www.kaggle.com/datasets/rushikeshdarge/hindi-text-corpus?resource=download
## 🚀 Project Overview

In this project, we implement:

- 📄 Loading and preprocessing Hindi text data
- 🔢 Tokenization using GPT-2 tokenizer (`tiktoken`)
- 🧩 Custom GPT-style Dataset class
- 📦 DataLoader for batch processing
- 🧠 Embedding layer for word representation
- 🔁 Linear output layer (Language Model Head)
- 🔍 Forward pass demonstration
- 🧾 Token shift verification (Next-token prediction logic)

This project focuses on understanding the **core pipeline behind GPT models**, without using high-level transformer libraries.

---

## 📂 Project Structure

```
project/
│
├── word_embedding.ipynb     # Main notebook
├── hi_part_1.txt            # Hindi dataset
├── README.md                # Project documentation
└── venv/                    # Virtual environment (not pushed to GitHub)
```

---

## 🛠️ Technologies Used

- Python 3.x
- PyTorch
- tiktoken (GPT-2 tokenizer)
- Jupyter Notebook

---

## 🔄 How It Works

### 1️⃣ Load Hindi Dataset
We load text data from a file (`hi_part_1.txt`) and preprocess it.

### 2️⃣ Tokenization
Using GPT-2 tokenizer, we convert Hindi text into numerical token IDs.

Example:
```
"मैं मशीन लर्निंग सीख रहा हूँ"
↓
[1234, 5678, 9101, ...]
```

---

### 3️⃣ GPT-Style Dataset Creation

We create input-target pairs using a sliding window approach:

- Input: 64 tokens  
- Target: Same sequence shifted by one token  

This enables **next-token prediction training**.

---

### 4️⃣ Embedding Layer

Each token ID is converted into a 128-dimensional vector using:

```python
nn.Embedding(vocab_size, embedding_dim)
```

Embedding matrix shape:
```
[vocab_size, 128]
```

---

### 5️⃣ Language Model Head

A linear layer converts embeddings back to vocabulary space:

```python
nn.Linear(embedding_dim, vocab_size)
```

This produces logits for next-token prediction.

---

### 6️⃣ Forward Pass

Pipeline:

```
Input Tokens
     ↓
Embedding Layer
     ↓
Linear Layer (LM Head)
     ↓
Logits (Next-token scores)
```

Tensor shape transformations:

- Input: `[batch_size, sequence_length]`
- Embedding: `[batch_size, sequence_length, 128]`
- Logits: `[batch_size, sequence_length, vocab_size]`

---

## 📊 Key Concepts Demonstrated

✔ Word Embeddings  
✔ Tokenization  
✔ Next-token prediction  
✔ Sliding window dataset  
✔ Batch processing  
✔ Tensor shape understanding  

---

## 🎯 Learning Objective

This project builds a strong foundation for understanding:

- GPT-style language models  
- Transformer architectures  
- Text-to-number representation  
- Neural network-based language modeling  

---

## 🚀 Future Improvements

- Add loss function and optimizer  
- Train model on larger dataset  
- Add positional encoding  
- Implement self-attention layer  
- Convert into full Transformer model  
- Implement text generation  

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install torch tiktoken jupyter
```

4. Run Jupyter Notebook:

```bash
jupyter notebook
```

Open `word_embedding.ipynb` and run all cells.

---

## 📌 Author

**Manikant Kumar**  
B.Tech CSE Student  
Interested in AI, Machine Learning, and NLP  

---

## ⭐ Why This Project Matters

Understanding embeddings and next-token prediction is the foundation of modern Large Language Models like GPT.

This project demonstrates the core mechanics behind those systems in a simple and educational way.
