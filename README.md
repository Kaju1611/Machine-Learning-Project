# Twitter Sentiment Analysis using Sentiment140 Dataset 🚀

This project performs sentiment analysis on a large dataset of 1.6 million tweets (Sentiment140). The goal is to classify tweets into positive or negative sentiments using natural language processing (NLP) techniques and machine learning models.

---

## Features ✨

- **Sentiment140 Dataset**: Pre-labeled tweets for supervised learning.
- Comprehensive text preprocessing pipeline.
- Machine learning model training for sentiment classification.
- Visualizations of sentiment trends and insights.

---

## Dataset 📂

We use the **Sentiment140 dataset**, which contains 1.6 million labeled tweets:

- **Columns:**
  - Sentiment: `0` (negative), `4` (positive)
  - Tweet ID
  - Date
  - Query (unused)
  - Username
  - Tweet text

- **Source**: [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).

---

## Prerequisites 📋

Make sure you have:

- A **Google Account** (for Colab).
- A stable internet connection to download the dataset.

---

## How to Use 🛠️

### Open the Colab Notebook:

1. Click this link to open the notebook: **[Sentiment Analysis Colab](https://colab.research.google.com/)**.
2. Alternatively, upload the notebook file (`Twitter_Sentiment_Analysis.ipynb`) to your Google Drive and open it in Colab.

### Download the Dataset:

1. Download the Sentiment140 dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).
2. Upload the dataset (`training.1600000.processed.noemoticon.csv`) to your Colab environment.

### Install Dependencies:

The notebook installs the required libraries automatically. Alternatively, you can run the following command:

```bash
!pip install nltk scikit-learn pandas matplotlib seaborn
```

### Run the Notebook:

Follow the steps in the notebook to preprocess the dataset, train the model, and evaluate its performance.

---

## Workflow Overview 🔄

1. **Load Dataset**: Import the Sentiment140 CSV file.
2. **Preprocess Tweets**: Clean text by removing special characters, URLs, mentions, hashtags, and stopwords.
3. **Feature Extraction**: Convert text into numerical features using techniques like TF-IDF or Bag of Words.
4. **Model Training**: Train machine learning models like Logistic Regression, SVM, or Random Forest.
5. **Evaluation**: Measure model performance using metrics like accuracy, precision, and recall.
6. **Visualization**: Generate sentiment distribution charts.

---

## Example Output 📊

- **Model Accuracy**: 85%
- **Positive Sentiment**: 😀 52%
- **Negative Sentiment**: 😢 48%

---


## Libraries Used 📚

- **Pandas**: Data manipulation and analysis.
- **NLTK**: Text preprocessing and tokenization.
- **Scikit-learn**: Machine learning models and feature extraction.
- **Matplotlib/Seaborn**: Data visualization.

---

## License 📜

This project is licensed under the MIT License.

---

## Acknowledgements 🙏

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) for providing a large-scale labeled dataset.
- Open-source libraries like NLTK, Scikit-learn, Pandas, and Matplotlib.
- Google Colab for a free computational environment.

---

