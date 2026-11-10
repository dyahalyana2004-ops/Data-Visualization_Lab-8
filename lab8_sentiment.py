# lab8_sentiment.py
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# ---------------------------------------------------------------
# 1. SAMPLE DATA (synthetic product comments, no download needed)
# ---------------------------------------------------------------
reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality.",
]

df = pd.DataFrame({
    "id": [f"R{i:02}" for i in range(1, 19)],
    "text": reviews,
})

# ---------------------------------------------------------------
# 2. SENTIMENT ANALYSIS (VADER compound score)
# ---------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score

df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text))
)

# ---------------------------------------------------------------
# ASSIGNMENT 4: main topic per comment (manual mapping)
# ---------------------------------------------------------------
topics = [
    "insulation", "carry", "design/lid", "leak", "lid",
    "cleaning", "insulation", "general", "insulation", "cleaning",
    "capacity", "general", "lid", "delivery", "insulation",
    "price", "handle", "general",
]
df["topic"] = topics

# ---------------------------------------------------------------
# 3. STREAMLIT DASHBOARD
# ---------------------------------------------------------------
st.set_page_config(page_title="Lab8 Feedback", layout="wide")
st.title("Product Feedback: A Text Summary")
st.caption("Practice data only. Automated labels need human review.")

left, right = st.columns(2)

with left:
    st.subheader("Comment counts by sentiment")
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
    st.bar_chart(counts, horizontal=True)

with right:
    st.subheader("Words used in the comments")
    # ASSIGNMENT 2: removed "bottle" (repeated, uninformative for THIS dataset),
    # kept "cold" so readers notice the insulation theme.
    stop_words = set(STOPWORDS)
    stop_words.update(["bottle", "drinks", "drink"])  # removed "bottle", kept "cold"
    cloud = WordCloud(
        width=900, height=500, background_color="white",
        colormap="viridis", stopwords=stop_words, collocations=False,
    ).generate(" ".join(df["text"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Word size reflects frequency only — position and color do not "
        "represent sentiment, and common words are removed."
    )

# ASSIGNMENT 4: sentiment by topic
st.subheader("Sentiment by topic (small sample — indicative only)")
topic_counts = (
    df.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
    .reindex(columns=order, fill_value=0)
)
st.dataframe(topic_counts, use_container_width=True)
st.caption(
    "Topics are manually assigned and each has only 1–3 comments, so "
    "differences should not be treated as statistically meaningful."
)

st.subheader("Review the original comments")
st.dataframe(
    df[["id", "text", "sentiment", "compound", "topic"]],
    use_container_width=True, hide_index=True,
)
st.caption("A compound score is a model output, not a confidence rating.")

# ASSIGNMENT 3: negation test
st.subheader("Negation test")
neg_text = "The bottle is not easy to clean."
neg_label, neg_score = label_sentiment(neg_text)
st.write(f"**Comment:** {neg_text}")
st.write(f"**Label:** {neg_label} | **Compound score:** {neg_score:.3f}")
st.caption(
    "VADER handles simple negation, but this is a single check — review "
    "more borderline comments before trusting the labels."
)
