# 🗡️ Hamlet — Next Word Oracle

An NLP deep learning project that predicts the next word in a sequence using a Recurrent Neural Network (LSTM/GRU) trained on the full original text of **William Shakespeare's *Hamlet***. Deployed as an interactive, parchment-themed web application using Streamlit.



---

## 📸 Preview

<img width="1830" height="842" alt="Screenshot 2026-08-04 115232" src="https://github.com/user-attachments/assets/eb3a179a-bf49-42cd-af44-6c685d547877" />

*An interactive divination chamber powered by LSTM and decorated in candlelit Elsinore aesthetics.*

---

## 📌 Features

*   **Live Web Application:** Test the model live in your browser without any installation.
*   **Custom Deep Learning Model:** Built using Keras/TensorFlow featuring Embedding layers, Stacked LSTM/GRU units, and Dropout layers for regularization.
*   **Shakespearean Dataset:** Trained on the complete raw text of *Hamlet* from the NLTK Gutenberg Corpus.
*   **Interactive UI:** Styled with custom HTML/CSS to recreate a candlelit castle atmosphere.
*   **Preset Famous Quotes:** Select classic lines from *Hamlet* directly within the app sidebar to test predictions.

---

## 🌐 Live Web App

You can access the hosted application here:  
👉 **[Hamlet - Next Word Oracle Live](https://bardlstm-nextwordprediction-fjxzxxbwuxdmstborhqnbi.streamlit.app/)**

---

## 📂 Project Structure

```text
.
├── assets/
│   └── preview.png         # App screenshot for README
├── hamlet.txt              # Raw dataset from NLTK Gutenberg corpus
├── Hamlet_Next_Word.ipynb  # Jupyter notebook (Data prep, Training, Evaluation)
├── next_word_lstm.h5       # Saved trained Keras model
├── tokenizer.pickle        # Saved Keras Tokenizer
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
