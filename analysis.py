# analysis.py
"""
CORD-19 Research Metadata Analysis
----------------------------------
Loads, cleans, and analyzes the Kaggle CORD-19 metadata dataset.
Generates visualizations and provides an interactive Streamlit app.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

# ---------------------------------
# Load and clean the dataset
# ---------------------------------
@st.cache_data
def load_data(file="metadata.csv"):
    try:
        df = pd.read_csv(file)

        # Ensure publish_time is datetime
        if "publish_time" in df.columns:
            df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
            df["year"] = df["publish_time"].dt.year

        # Drop rows with missing titles
        df = df.dropna(subset=["title"])

        return df
    except FileNotFoundError:
        st.error(f"File {file} not found. Please make sure it exists.")
        return pd.DataFrame()

# ---------------------------------
# Visualization Functions
# ---------------------------------
def plot_publications_over_time(df):
    yearly_counts = df["year"].value_counts().sort_index()
    fig, ax = plt.subplots()
    yearly_counts.plot(kind="line", marker="o", ax=ax)
    ax.set_title("Publications Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    st.pyplot(fig)

def plot_top_journals(df, top_n=10):
    top_journals = df["journal"].value_counts().head(top_n)
    fig, ax = plt.subplots()
    sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax)
    ax.set_title(f"Top {top_n} Journals Publishing COVID-19 Research")
    ax.set_xlabel("Number of Papers")
    st.pyplot(fig)

def plot_wordcloud(df):
    text = " ".join(df["title"].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def plot_sources(df):
    source_counts = df["source"].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=source_counts.values, y=source_counts.index, ax=ax)
    ax.set_title("Top Sources of Publications")
    ax.set_xlabel("Number of Papers")
    st.pyplot(fig)

# ---------------------------------
# Streamlit App Layout
# ---------------------------------
def main():
    st.title("CORD-19 Research Data Explorer")
    st.write("Explore COVID-19 research papers from the CORD-19 dataset.")

    df = load_data()

    if df.empty:
        return

    # Dataset overview
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Total rows: {len(df)}")

    # Filters
    if "year" in df.columns:
        min_year, max_year = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    # Visualizations
    st.subheader("Publications Over Time")
    plot_publications_over_time(df)

    st.subheader("Top Journals")
    plot_top_journals(df)

    st.subheader("Word Cloud of Titles")
    plot_wordcloud(df)

    st.subheader("Top Sources")
    plot_sources(df)

if __name__ == "__main__":
    main()
