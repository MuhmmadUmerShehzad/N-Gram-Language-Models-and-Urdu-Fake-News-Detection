# N-Gram Language Models & Urdu Fake News Detection

## Overview

This project explores natural language processing techniques, focusing on N-Gram language modeling and text classification. It is divided into two major components:
1. **Language Modeling:** Constructing, generating text from, and evaluating Bigram and Trigram language models using various text corpora.
2. **Urdu Fake News Detection:** Building a custom Naive Bayes classifier from scratch to identify fake news articles in Urdu.

## Features

* **Advanced Text Preprocessing:** Custom pipelines for both English and Urdu text. Features include heuristic sentence segmentation, tokenization, punctuation removal, and Urdu text normalization (standardizing character variants, removing diacritics/tatweel, and handling stop words).
* **Language Models:** Implementation of Bigram and Trigram models. Supports text generation using Shannon's Method (for diverse outputs) and Simple Probability (for high-confidence predictions).
* **Out-of-Domain Evaluation:** Cross-evaluating language models using Perplexity, Coverage, and Out-of-Vocabulary (OOV) rates to test generalization.
* **Custom Naive Bayes Classifier:** Built entirely from scratch with:
    * Log-space arithmetic to prevent underflow.
    * Unified, configurable smoothing (Laplace and Add-k).
    * Boolean Mode for set-based word counting (ignoring frequency).
    * Negation handling (prefixing tokens with `NOT_`).

## Key Findings & Results

### Language Modeling
* **Generalization:** Bigram models generalize significantly better than trigram models on unseen, out-of-domain text due to reduced data sparsity.
* **Vocabulary Overlap:** Models trained on larger, more diverse corpora (e.g., Google Scholar extracts) exhibit the best vocabulary coverage and lowest OOV rates for general tasks.

### Fake News Detection
* **Best Configuration:** The **Boolean Naive Bayes** model (with stop words retained) achieved the highest performance: **75.19% Accuracy** and **75.15% F1-Score**.
* **Boolean vs. Standard:** Ignoring word frequencies (Boolean mode) improved accuracy by up to 4.2%. Fake news detection often depends more on *which* specific words are used rather than how often they are repeated.
* **Stop Words Matter:** Retaining Urdu stop words actually improved performance, as the differences in formal vs. informal function word usage correlate with article authenticity.
* **Optimal Smoothing:** Add-k smoothing with `k=0.05` was found to be the optimal parameter, outperforming standard Laplace smoothing (`k=1`).

## Usage

**Note on Data:** You can place your PDF files in a `.zip` archive in the same folder as the Jupyter Notebook to easily load and process them.

To run the project:
1. Ensure all your data archives and `.zip` files are in the same directory as the notebook.
2. Open the Jupyter Notebook (`M_Umer_Shehzad_BS_23_IB_101047.ipynb`).
3. Execute the cells sequentially to run the preprocessing pipelines, train the models, and view the evaluation results.
