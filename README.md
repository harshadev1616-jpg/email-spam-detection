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
streamlit run nano_app.py
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead.

The app downloads the NLTK `punkt_tab` and `stopwords` data on first use and
caches it in a temporary directory. An internet connection is required the
first time the scanner is used.

Keep `spam_model.pkl` and `tfidf_vectorizer.pkl` beside `nano_app.py`; the app
loads these saved artifacts when it starts.

## Model evaluation

The **Model performance** page displays accuracy, precision, recall, F1, and
the confusion matrix from the verified stratified holdout in
`model_metrics.json`. The source CSV is intentionally excluded because its
redistribution permission has not been established. Metrics are therefore
reported from the recorded test run rather than recomputed by the hosted app.

The notebook `email_smap_file.ipynb` is also kept out of the public upload
because it contains saved training outputs. Add the dataset and notebook to a
public release only after verifying that the data is safe to share and its
license permits redistribution.

## Limitations and safety

The displayed evaluation describes a small holdout from the supplied training
dataset; it is not a guarantee of accuracy on new or real-world email. Treat
the classification as a signal, and verify unexpected requests and links.

Avoid entering confidential email in the demo. The app does not save submitted
messages to a database. When hosted on Streamlit Community Cloud, submitted
email text is processed on Streamlit's servers, not on the visitor's device.

The `.pkl` files use Python's pickle-based serialization. Only load model
artifacts from a source you trust.
