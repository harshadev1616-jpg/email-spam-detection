import json
from pathlib import Path
from tempfile import gettempdir

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

APP_DIR = Path(__file__).resolve().parent
METRICS_FILE = APP_DIR / "model_metrics.json"
NLTK_DATA_DIR = Path(gettempdir()) / "mailguard_nltk_data"

st.set_page_config(
    page_title="MailGuard | Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17253c;
        --muted: #687992;
        --teal: #0d8b83;
        --line: #e5ebf2;
        --surface: #ffffff;
        --background: #f5f8fc;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 88% 3%, rgba(51, 186, 171, .11), transparent 28%),
            var(--background);
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
        letter-spacing: -.035em;
    }
    [data-testid="stSidebar"] {
        background: #101d32;
        border-right: 1px solid rgba(255,255,255,.07);
    }
    [data-testid="stSidebar"] * { color: #e7effb; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: .35rem .25rem;
        font-weight: 600;
    }
    .brand {
        font-family: Inter, ui-sans-serif, system-ui, sans-serif;
        color: white;
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: -.04em;
        padding: .35rem 0 1.5rem;
    }
    .brand span { color: #61d5c5; }
    .sidebar-note {
        color: #9cacc3;
        font-size: .82rem;
        line-height: 1.6;
        padding-top: 1rem;
    }
    .eyebrow {
        color: var(--teal);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .hero-title {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.05rem);
        line-height: 1.08;
        font-weight: 800;
        margin: 0 0 .8rem;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1.04rem;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 1.5rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--line) !important;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(26, 49, 79, .045);
    }
    .panel-kicker {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(105deg, #087d79, #13a092);
        border: 0;
        border-radius: 11px;
        color: white;
        font-weight: 700;
        min-height: 3rem;
        box-shadow: 0 8px 18px rgba(13, 139, 131, .18);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(105deg, #076d69, #0d8b83);
        border: 0;
        color: white;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 22px rgba(26, 49, 79, .035);
    }
    [data-testid="stMetricValue"] { color: var(--ink) !important; }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid #d6e0eb !important;
        border-radius: 11px !important;
        caret-color: var(--teal);
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #8190a4 !important;
        opacity: 1;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(13, 139, 131, .12) !important;
    }
    .fine-print { color: var(--muted); font-size: .83rem; line-height: 1.55; }
    div[data-testid="stProgress"] > div > div { background: #10a091; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts() -> tuple[MultinomialNB, TfidfVectorizer]:
    model = joblib.load(APP_DIR / "spam_model.pkl")
    vectorizer = joblib.load(APP_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer


@st.cache_resource
def load_stop_words() -> set[str]:
    nltk_data_path = str(NLTK_DATA_DIR)
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)

    for resource, package in (
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab/english/", "punkt_tab"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(
                package,
                download_dir=nltk_data_path,
                quiet=True,
            ):
                raise LookupError(
                    f"Could not download required NLTK data: {package}."
                )

    return set(stopwords.words("english"))


def clean_email(text: str, stop_words: set[str]) -> str:
    tokens = word_tokenize(text.lower())
    return " ".join(
        token for token in tokens if token.isalpha() and token not in stop_words
    )


@st.cache_data
def load_model_metrics() -> dict[str, object]:
    with METRICS_FILE.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    required_fields = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion",
        "samples",
        "test_samples",
        "spam_samples",
        "not_spam_samples",
        "class_counts",
    }
    missing_fields = required_fields.difference(metrics)
    if missing_fields:
        raise ValueError(
            f"Model metrics are missing fields: {', '.join(sorted(missing_fields))}"
        )

    metrics["class_counts"] = (
        pd.Series(metrics["class_counts"], name="Emails")
        .rename_axis("Category")
        .to_frame()
    )
    return metrics


def show_page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="eyebrow">{kicker}</div>
        <h1 class="hero-title">{title}</h1>
        <div class="hero-copy">{description}</div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="brand">🛡️ Mail<span>Guard</span></div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Email scanner", "Model performance", "About"],
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="sidebar-note">
        <strong>No app database storage</strong><br>
        On Streamlit Cloud, email text is processed on Streamlit's servers. This
        app does not save it to a database. Avoid entering confidential email.
        </div>
        """,
        unsafe_allow_html=True,
    )

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError as error:
    st.error(
        f"Could not find {error.filename!r}. Keep spam_model.pkl and "
        "tfidf_vectorizer.pkl beside nano_app.py."
    )
    st.stop()
except (EOFError, OSError, ValueError, ImportError) as error:
    st.error(f"Could not load the saved spam model: {error}")
    st.stop()

try:
    evaluation = load_model_metrics()
except FileNotFoundError as error:
    evaluation = None
    st.warning(
        f"Model evaluation file {error.filename!r} is missing. "
        "Email scanning remains available."
    )
except LookupError:
    evaluation = None
    st.warning(
        "Model evaluation needs the NLTK English tokenizer and stop-word data. "
        "Install the NLTK data used by the training notebook to enable metrics."
    )
except (ValueError, KeyError, OSError) as error:
    evaluation = None
    st.warning(f"Could not load model metrics: {error}")

if page == "Email scanner":
    show_page_header(
        "Email security, made simple",
        "Know what’s in your inbox.",
        "Check an email with your trained spam classifier. Add the subject and message "
        "for the best match with the model’s training format.",
    )

    if evaluation is not None:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Holdout accuracy", f"{evaluation['accuracy']:.0%}")
        metric_columns[1].metric("Precision", f"{evaluation['precision']:.0%}")
        metric_columns[2].metric("Recall", f"{evaluation['recall']:.0%}")
        metric_columns[3].metric("Test emails", f"{evaluation['test_samples']:,}")
        st.caption(
            "Measured on a stratified holdout from the project dataset—not a guarantee "
            "of performance on future or real-world email."
        )

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-kicker">Message analysis</div>', unsafe_allow_html=True)
            subject = st.text_input("Subject", placeholder="e.g. Your account needs attention")
            body = st.text_area(
                "Email message",
                placeholder="Paste the email message here…",
                height=210,
                max_chars=20000,
            )
            analyze = st.button(
                "Analyze email  →",
                type="primary",
                width="stretch",
            )
            st.markdown(
                '<div class="fine-print">Tip: include the subject and full message for a '
                'more informative analysis. The model’s score is not a certainty.</div>',
                unsafe_allow_html=True,
            )

        if analyze:
            email_text = f"{subject.strip()} {body.strip()}".strip()
            if not email_text:
                st.warning("Add a subject or message before analyzing.")
            else:
                try:
                    prepared_text = clean_email(email_text, load_stop_words())
                    features = vectorizer.transform([prepared_text])
                    prediction = int(model.predict(features)[0])
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(max(probabilities))
                    st.session_state["last_result"] = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "characters": len(email_text),
                        "words": len(email_text.split()),
                    }
                except LookupError:
                    st.error(
                        "The NLTK English tokenizer or stop-word data is missing. "
                        "Install the language data used by the training notebook, then "
                        "restart the app."
                    )

        result = st.session_state.get("last_result")
        if result:
            if result["prediction"] == 1:
                st.error("🚨 **Likely spam** — review the sender and links carefully.")
            else:
                st.success("✅ **Likely not spam** — still check unexpected links and requests.")
            st.progress(result["confidence"], text=f"Model score: {result['confidence']:.0%}")
            st.caption(
                f"Analyzed {result['words']:,} words · {result['characters']:,} characters. "
                "This score is the model’s estimate, not a calibrated guarantee."
            )

    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-kicker">How it works</div>', unsafe_allow_html=True)
            st.markdown("### A quick, local check")
            st.write(
                "Your text is cleaned with the same English tokenization used during "
                "training, converted into TF-IDF features, and evaluated by the saved "
                "Multinomial Naive Bayes model."
            )
            st.markdown("---")
            st.markdown("**What the score means**")
            st.write(
                "The score reflects the model’s predicted class probabilities. It can be "
                "overconfident, so use it as a signal—not proof that a message is safe."
            )
            st.markdown("---")
            if evaluation is not None:
                st.markdown("**Training dataset**")
                st.write(
                    f"{evaluation['samples']:,} labeled emails · "
                    f"{evaluation['spam_samples']:,} spam · "
                    f"{evaluation['not_spam_samples']:,} not spam"
                )
            else:
                st.write("Dataset evaluation is currently unavailable.")

elif page == "Model performance":
    show_page_header(
        "Transparent evaluation",
        "Measure the model, not the hype.",
        "These results were measured on the model’s stratified holdout from the "
        "project dataset. They are not a live evaluation or a guarantee.",
    )
    if evaluation is None:
        st.info("Model metrics are unavailable because the published metrics file could not be loaded.")
    else:
        st.caption(
            f"Previously evaluated on {evaluation['test_samples']:,} holdout emails "
            f"from a {evaluation['samples']:,}-email dataset · stratified 80/20 split."
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric("Accuracy", f"{evaluation['accuracy']:.0%}")
        metric_columns[1].metric("Precision · spam", f"{evaluation['precision']:.0%}")
        metric_columns[2].metric("Recall · spam", f"{evaluation['recall']:.0%}")
        metric_columns[3].metric("F1 · spam", f"{evaluation['f1']:.0%}")

        matrix_col, balance_col = st.columns(2, gap="large")
        with matrix_col:
            st.subheader("Confusion matrix")
            st.caption("Rows are actual labels; columns are model predictions.")
            matrix = evaluation["confusion"]
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual · Not Spam", "Actual · Spam"],
                    columns=["Predicted · Not Spam", "Predicted · Spam"],
                ),
                width="stretch",
            )
            st.markdown(
                f"- Correctly identified spam: **{matrix[1][1]}**\n"
                f"- Spam missed: **{matrix[1][0]}**\n"
                f"- Legitimate email flagged: **{matrix[0][1]}**\n"
                f"- Correctly identified legitimate: **{matrix[0][0]}**"
            )
        with balance_col:
            st.subheader("Dataset label balance")
            st.caption(
                f"Counts across all {evaluation['samples']:,} labeled examples "
                "(not just the holdout)."
            )
            st.bar_chart(evaluation["class_counts"], color="#0d8b83")
            st.markdown(
                "Only the **Spam** category is treated as spam; every other category "
                "is grouped as not spam."
            )

        st.info(
            "A perfect score on a small, synthetic or repetitive dataset may not "
            "generalize. Test with varied, real-world examples before relying on this "
            "classifier for important decisions."
        )

elif page == "About":
    show_page_header(
        "Built around your project",
        "A clearer view of email classification.",
        "MailGuard wraps your saved training artifacts in a simple scanner and makes "
        "the model’s evaluation visible.",
    )
    about_left, about_right = st.columns(2, gap="large")
    with about_left:
        st.markdown("### The pipeline")
        st.markdown(
            "1. Combine the email subject and body.\n"
            "2. Tokenize the text and remove English stop words.\n"
            "3. Convert the cleaned text to TF-IDF features.\n"
            "4. Predict with the saved Multinomial Naive Bayes classifier."
        )
        st.markdown("### Labels")
        st.write(
            "The training notebook maps the dataset’s `Spam` category to Spam. "
            "All other categories are grouped as Not Spam."
        )
    with about_right:
        st.markdown("### Important limitations")
        st.write(
            "Evaluation scores summarize one held-out split of the supplied dataset. "
            "They do not establish performance on different inboxes, languages, "
            "new spam campaigns, or malicious links."
        )
        st.write(
            "Avoid pasting confidential or sensitive messages into this demo. The app "
            "does not save submitted text, but your local machine and browser still "
            "process it."
        )
