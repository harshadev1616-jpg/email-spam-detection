# Email Spam Detection

A Streamlit app that classifies an email as **Spam** or **Not Spam** using a
TF-IDF vectorizer and a trained Multinomial Naive Bayes model.

## Run locally

Use Python 3.10 or newer. From the project directory, install the dependencies
and the NLTK language data required by the text-cleaning step:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader punkt_tab stopwords
streamlit run nano_app.py
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead.

Keep `spam_model.pkl` and `tfidf_vectorizer.pkl` beside `nano_app.py`; the app
loads these saved artifacts when it starts.

## Model evaluation

The **Model performance** page recalculates accuracy, precision, recall, F1,
and the confusion matrix from `sample_email_ai_training_large.csv`. That data
file is intentionally excluded from this public upload because its
redistribution permission has not been established. Without the CSV, email
scanning still works, but dataset evaluation is unavailable.

The notebook `email_smap_file.ipynb` is also kept out of the public upload
because it contains saved training outputs. Add the dataset and notebook to a
public release only after verifying that the data is safe to share and its
license permits redistribution.

## Limitations and safety

The displayed evaluation describes a small holdout from the supplied training
dataset; it is not a guarantee of accuracy on new or real-world email. Treat
the classification as a signal, and verify unexpected requests and links.

Avoid entering confidential email in the demo. The app does not save submitted
messages to a database, but email text is processed locally by the running app.

The `.pkl` files use Python's pickle-based serialization. Only load model
artifacts from a source you trust.
