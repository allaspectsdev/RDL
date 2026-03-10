"""
Text Analytics Module - Text exploration, sentiment analysis, and term frequency.
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re

from modules.ui_helpers import section_header, empty_state, help_tip


def render_text_analytics(df: pd.DataFrame):
    """Main entry point for Text Analytics module."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset with text columns to begin.")
        return

    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not text_cols:
        empty_state("No text columns found.", "Your dataset needs at least one text/string column.")
        return

    tabs = st.tabs(["Text Explorer", "Sentiment Analysis", "Term Frequency"])

    with tabs[0]:
        _render_text_explorer(df, text_cols)
    with tabs[1]:
        _render_sentiment(df, text_cols)
    with tabs[2]:
        _render_term_frequency(df, text_cols)


# ─── Tokenization Helpers ────────────────────────────────────────────────────

_STOP_WORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very", "just",
    "about", "up", "out", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "each", "all", "both", "few",
    "more", "most", "other", "some", "such", "only", "own", "same",
    "also", "how", "what", "which", "who", "whom", "when", "where", "why",
])


def _tokenize(text: str, remove_stopwords: bool = True) -> list:
    """Simple whitespace + punctuation tokenizer."""
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]
    return tokens


def _get_ngrams(tokens: list, n: int) -> list:
    """Generate n-grams from token list."""
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ─── Text Explorer ───────────────────────────────────────────────────────────

def _render_text_explorer(df, text_cols):
    section_header("Text Explorer")
    help_tip("Text Explorer", "Explore text data: document length, term frequency, TF-IDF.")

    col = st.selectbox("Text column:", text_cols, key="te_col")
    texts = df[col].dropna().astype(str)

    if texts.empty:
        empty_state("Selected column has no text data.")
        return

    remove_sw = st.checkbox("Remove stop words", value=True, key="te_stop")

    # Basic stats
    lengths = texts.str.len()
    word_counts = texts.str.split().str.len()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", f"{len(texts):,}")
    c2.metric("Avg Length (chars)", f"{lengths.mean():.0f}")
    c3.metric("Avg Words", f"{word_counts.mean():.1f}")
    c4.metric("Unique Tokens", f"{len(set(t for doc in texts for t in _tokenize(doc, remove_sw))):,}")

    # Document length distribution
    import plotly.express as px
    fig = px.histogram(x=word_counts, nbins=30, title="Document Length Distribution (Words)")
    fig.update_layout(xaxis_title="Word Count", yaxis_title="Frequency", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # TF-IDF table
    section_header("TF-IDF")
    all_tokens = [_tokenize(doc, remove_sw) for doc in texts]
    n_docs = len(all_tokens)

    # Term frequency across corpus
    tf_counter = Counter()
    df_counter = Counter()
    for tokens in all_tokens:
        tf_counter.update(tokens)
        df_counter.update(set(tokens))

    top_n = st.slider("Top N terms:", 10, 100, 30, key="te_topn")
    tfidf_data = []
    for term, tf in tf_counter.most_common(top_n * 3):
        doc_freq = df_counter[term]
        idf = np.log(n_docs / (1 + doc_freq))
        tfidf_data.append({
            "Term": term,
            "Term Freq": tf,
            "Doc Freq": doc_freq,
            "IDF": round(idf, 4),
            "TF-IDF": round(tf * idf, 4),
        })

    tfidf_df = pd.DataFrame(tfidf_data).sort_values("TF-IDF", ascending=False).head(top_n)
    st.dataframe(tfidf_df, use_container_width=True, hide_index=True)

    # Word cloud
    section_header("Word Cloud")
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        all_text = " ".join(texts)
        if remove_sw:
            wc = WordCloud(width=800, height=400, background_color="white",
                           colormap="cool", stopwords=_STOP_WORDS, max_words=100)
        else:
            wc = WordCloud(width=800, height=400, background_color="white",
                           colormap="cool", max_words=100)
        wc.generate(all_text)

        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
    except ImportError:
        st.info("Install `wordcloud` package for word cloud visualization: `pip install wordcloud`")


# ─── Sentiment Analysis ─────────────────────────────────────────────────────

def _render_sentiment(df, text_cols):
    section_header("Sentiment Analysis")
    help_tip("Sentiment Analysis",
             "Uses TextBlob to compute polarity (-1 to 1) and subjectivity (0 to 1) per document.")

    col = st.selectbox("Text column:", text_cols, key="sent_col")
    texts = df[col].dropna().astype(str)

    if texts.empty:
        empty_state("Selected column has no text data.")
        return

    try:
        from textblob import TextBlob
    except ImportError:
        st.info("Install `textblob` package for sentiment analysis: `pip install textblob`")
        return

    with st.spinner("Computing sentiment..."):
        sentiments = []
        for text in texts:
            blob = TextBlob(text)
            sentiments.append({
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity,
            })
        sent_df = pd.DataFrame(sentiments, index=texts.index)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Polarity", f"{sent_df['polarity'].mean():.3f}")
    c2.metric("Mean Subjectivity", f"{sent_df['subjectivity'].mean():.3f}")
    positive_pct = (sent_df["polarity"] > 0).mean() * 100
    c3.metric("% Positive", f"{positive_pct:.1f}%")

    import plotly.express as px

    # Polarity histogram
    fig = px.histogram(sent_df, x="polarity", nbins=30,
                       title="Sentiment Polarity Distribution",
                       color_discrete_sequence=["#6366f1"])
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Subjectivity histogram
    fig2 = px.histogram(sent_df, x="subjectivity", nbins=30,
                        title="Subjectivity Distribution",
                        color_discrete_sequence=["#22c55e"])
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # Polarity vs Subjectivity scatter
    fig3 = px.scatter(sent_df, x="polarity", y="subjectivity",
                      title="Polarity vs Subjectivity",
                      opacity=0.6)
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Top positive/negative
    combined = df[[col]].loc[texts.index].copy()
    combined["Polarity"] = sent_df["polarity"].values
    combined["Subjectivity"] = sent_df["subjectivity"].values

    col1, col2 = st.columns(2)
    with col1:
        section_header("Most Positive")
        top_pos = combined.nlargest(5, "Polarity")
        st.dataframe(top_pos, use_container_width=True, hide_index=True)
    with col2:
        section_header("Most Negative")
        top_neg = combined.nsmallest(5, "Polarity")
        st.dataframe(top_neg, use_container_width=True, hide_index=True)

    # Sentiment by group
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != col]
    if cat_cols:
        group_col = st.selectbox("Group sentiment by:", [None] + cat_cols, key="sent_group")
        if group_col:
            combined[group_col] = df[group_col].loc[texts.index].values
            fig4 = px.box(combined, x=group_col, y="Polarity",
                          title=f"Sentiment by {group_col}",
                          color=group_col)
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)


# ─── Term Frequency ──────────────────────────────────────────────────────────

def _render_term_frequency(df, text_cols):
    section_header("Term Frequency")
    help_tip("Term Frequency", "Analyze top terms, bigrams, and trigrams in your text data.")

    col = st.selectbox("Text column:", text_cols, key="tf_col")
    texts = df[col].dropna().astype(str)

    if texts.empty:
        empty_state("Selected column has no text data.")
        return

    remove_sw = st.checkbox("Remove stop words", value=True, key="tf_stop")
    top_n = st.slider("Top N:", 10, 50, 20, key="tf_topn")

    all_tokens = []
    for doc in texts:
        all_tokens.extend(_tokenize(doc, remove_sw))

    import plotly.express as px

    # Unigrams
    section_header("Top Unigrams")
    uni_counts = Counter(all_tokens).most_common(top_n)
    if uni_counts:
        uni_df = pd.DataFrame(uni_counts, columns=["Term", "Count"])
        fig = px.bar(uni_df, x="Count", y="Term", orientation="h",
                     title=f"Top {top_n} Unigrams",
                     color_discrete_sequence=["#6366f1"])
        fig.update_layout(height=max(350, top_n * 22), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Bigrams
    section_header("Top Bigrams")
    bigrams = []
    for doc in texts:
        tokens = _tokenize(doc, remove_sw)
        bigrams.extend(_get_ngrams(tokens, 2))
    bi_counts = Counter(bigrams).most_common(top_n)
    if bi_counts:
        bi_df = pd.DataFrame(bi_counts, columns=["Bigram", "Count"])
        fig2 = px.bar(bi_df, x="Count", y="Bigram", orientation="h",
                      title=f"Top {top_n} Bigrams",
                      color_discrete_sequence=["#818cf8"])
        fig2.update_layout(height=max(350, top_n * 22), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    # Trigrams
    section_header("Top Trigrams")
    trigrams = []
    for doc in texts:
        tokens = _tokenize(doc, remove_sw)
        trigrams.extend(_get_ngrams(tokens, 3))
    tri_counts = Counter(trigrams).most_common(top_n)
    if tri_counts:
        tri_df = pd.DataFrame(tri_counts, columns=["Trigram", "Count"])
        fig3 = px.bar(tri_df, x="Count", y="Trigram", orientation="h",
                      title=f"Top {top_n} Trigrams",
                      color_discrete_sequence=["#a78bfa"])
        fig3.update_layout(height=max(350, top_n * 22), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)
