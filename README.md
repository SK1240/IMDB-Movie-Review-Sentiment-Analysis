# 🎬 IMDB Movie Review Sentiment Analysis – Simple RNN

A **deep learning** project that performs **sentiment classification** on **IMDB movie reviews** using a **Simple Recurrent Neural Network (RNN)**.
Developed with **TensorFlow/Keras** and deployed as an interactive **Streamlit web app** for live predictions.

## ✅ Project Overview

Dataset: **IMDB Movie Reviews** (`tensorflow.keras.datasets.imdb`)

Architecture: **SimpleRNN with word embeddings**

Goal: **Binary text classification — Positive or Negative**

Interface: **Streamlit app** (app.py)

Extras: Jupyter notebooks for **training**, **testing**, and **manual prediction**

## 📁 Project Structure

```
IMDB-Movie-Review-Sentiment-Analysis/
├── app.py                               # 🚀 Streamlit UI for real-time sentiment analysis
├── IMDB_sentiment_analysis.ipynb        # Model training and saving notebook
├── prediction.ipynb                     # Notebook for custom text predictions
├── model.h5                             # Pre-trained SimpleRNN model file
├── requirements.txt                     # Required dependencies
```

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository
```
git clone https://github.com/SK1240/IMDB-Movie-Review-Sentiment-Analysis.git
cd IMDB-Movie-Review-Sentiment-Analysis
```

### 2️⃣ Create and activate a virtual environment
```
python -m venv .venv
```
Activation:
* Windows: `.venv\Scripts\activate`
* Mac/Linux: `source .venv/bin/activate`

### 3️⃣ Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
💡 If your system doesn’t have GPU support, you can install `tensorflow-cpu` instead.

### 🚀 Launching the Streamlit App
To start the web app:
```
streamlit run main.py
```
A local Streamlit server will open at [localhost:8501](http://localhost:8501)

Enter a review in the input box, click “**Classify**”, and see the sentiment output:

* ✅ Positive Review

* ❌ Negative Review
  
You’ll also see the model’s confidence score.


## 🧠 Model Pipeline

🔹 Data Preprocessing

   * Load the IMDB dataset (`num_words=10000`)

   * Pad or truncate reviews to uniform length (`maxlen=500`)

🔹 Model Design (in `IMDB_sentiment_analysis.ipynb`)

   * Embedding Layer: Vocabulary size = 10,000, vector size = 128

   * SimpleRNN Layer: 128 hidden units, activation = **ReLU**

   * Dense Layer: 1 unit, activation = **Sigmoid**

   * Optimizer: **Adam**

   * Loss Function: **Binary Crossentropy**

   * Model saved as `model.h5`

🔹 Prediction Process (`app.py` / `prediction.ipynb`)

   * Convert text → sequence of indices

   * Pad sequence to consistent input size

   * Feed into trained model → Output predicted sentiment


## 🌟 Example
Input:
> “Absolutely loved this movie — the cast and story were brilliant!”

Output:
```
Sentiment: Positive ✅
Confidence Score: 0.95
```
