# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## how its Works
This project detects whether a news article is real or fake using Machine Learning and Natural Language Processing (NLP).
Here’s a clear step-by-step explanation of how the system works:

1️⃣ Data Collection

The system uses a labeled dataset containing real and fake news articles.

Each record contains:

Title

Text/body

Label (Real/Fake)

2️⃣ Data Cleaning & Preprocessing

To prepare the text for machine learning, the project applies these NLP steps:

Removing punctuation

Converting text to lowercase

Removing stopwords (like the, is, at)

Stemming / Lemmatization to reduce words to their base form

Tokenization

Converting raw text into clean, meaningful input for modeling

This improves accuracy and reduces noise in the data.

3️⃣ Feature Extraction (TF-IDF)

The cleaned text is converted into numerical features using TF-IDF Vectorization.

TF-IDF assigns importance scores to words based on how frequently they appear in real vs fake news.

This converts text into a format that machine learning models can understand.

4️⃣ Model Training

A Logistic Regression classifier (or whichever you use) is trained on the TF-IDF vectors.

The model learns patterns in language that are common in fake news.

After training, accuracy and other metrics are calculated to evaluate performance.

5️⃣ Saving Model & Vectorizer

The trained items are saved for reuse:

model.pkl → stores the ML model

vectorizer.pkl → stores TF-IDF vectorizer

This allows predictions without retraining the model.

6️⃣ Prediction Workflow

When the user enters a new news article:

Text is cleaned using the same preprocessing steps

Text is transformed into TF-IDF vector

Model predicts Real or Fake

Result is displayed on the UI

7️⃣ User Interface (Streamlit / Flask)

A simple UI allows users to input news text

After clicking “Predict”, the model instantly returns the result

The entire pipeline runs in the background seamlessly

⭐ In Short
Data → Preprocessing → TF-IDF → ML Model → Prediction → Result
