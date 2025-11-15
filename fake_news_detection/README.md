# Fake News Detection using Natural Language Processing

## Project Overview

This repository contains a Data Science project focused on developing and evaluating machine learning models for the automatic detection of fake news articles. Leveraging Natural Language Processing (NLP) techniques, the goal is to distinguish between credible and unreliable sources based on textual content.


## Project Description

The rapid spread of misinformation poses a significant challenge to society. This project addresses this by implementing and comparing several classification algorithms to accurately flag articles as 'REAL' or 'FAKE'. The core steps involved are:

* **Data Cleaning and Preprocessing:** Tokenization, stemming/lemmatization, and stop-word removal.
* **Feature Engineering:** Converting text data into numerical features using techniques like TF-IDF or Count Vectorization.
* **Model Training and Evaluation:** Training a robust machine learning model like the **Passive Aggressive Classifier** and evaluating its performance using metrics such as accuracy, precision, recall, and F1-score.

## Dataset

This project typically utilizes a publicly available fake news dataset containing two classes: **Real** and **Fake**. Common datasets include titles, texts, and corresponding labels.

* **Expected Features:** `title`, `text`, `label`
* **Data Source:** _(Please update this section with the exact source of your dataset, e.g., Kaggle, or a specific research paper.)_

## Methodology

The primary approach implemented in the Jupyter notebook focuses on highly efficient text classification:

1. **Classical Machine Learning:**  
   * **Vectorization:** **TF-IDF (Term Frequency-Inverse Document Frequency)**, used to weight word importance in the document corpus.  
   * **Model:** **Passive Aggressive Classifier (PAC)**, which is particularly suitable for large datasets and is highly efficient for online learning scenarios.  
   * **Evaluation:** Model performance is assessed using a confusion matrix and a detailed report including **Accuracy, Precision, Recall, and F1-Score.**
2. **Deep Learning (If applicable):**  
   * _If you plan to add deep learning:_ Embeddings (Word2Vec, GloVe) and Models (RNN, LSTM, or BERT) can be implemented here for comparative analysis.

## Requirements

To run the notebooks and scripts in this repository, you will need to install the following libraries. It is highly recommended to use a virtual environment.

```
pip install pandas numpy scikit-learn nltk matplotlib jupyter
# Add other specific libraries if used, e.g.,
# pip install tensorflow keras transformers

```

## Usage and Setup

### 1\. Clone the repository

```
git clone [https://github.com/olliewh00/Data_Science_Projects/tree/main/fake_news_detection](https://github.com/olliewh00/Data_Science_Projects/tree/main/fake_news_detection)
cd fake_news_detection
```

### 2\. Install dependencies

```
pip install -r requirements.txt # (Assuming you create this file)
```

### 3\. Run the analysis

Open the main Jupyter Notebook to step through the entire process, from data loading to model evaluation and prediction.

```
jupyter notebook
```

Navigate to the primary notebook file, **`fake_news.ipynb`**, and execute the cells sequentially.

## Results

The primary objective is to achieve the highest possible classification accuracy. The results summary can be found in the dedicated notebook, but the key performance metrics are typically:

| Model                             | Vectorizer      | Accuracy  | F1-Score  | Best Parameters   |
| --------------------------------- | --------------- | --------- | --------- | ----------------- |
| **Passive Aggressive Classifier** | TF-IDF          | **92.98%** | **93.0%** | Max Iterations=50 |







