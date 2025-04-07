import os
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob



def text_to_word_sankey(tt, word_list=None, k=5):
    """
    Creates an interactive Sankey diagram mapping each text to its most frequent words.
    Link thickness represents word frequency.

    Parameters:
        tt (TextAnalysis): Parsed text analysis object.
        word_list (list[str], optional): Specific words to include (overrides top-k selection).
        k (int): Number of top words to include per article if word_list is not provided.
    """
    labels = []
    sources, targets, values = [], [], []
    label_indices = {}

    text_labels = list(tt.data["word_freqs"].keys())

    # Select words based on user input or top-k per text
    selected_words = set(word_list) if word_list else set()
    if not word_list:
        for label in text_labels:
            top_words = tt.data["word_freqs"][label].most_common(k)
            selected_words.update(word for word, _ in top_words)

    # Assign indices to text labels
    for label in text_labels:
        label_indices[label] = len(labels)
        labels.append(label)

    # Assign indices to word labels
    for word in sorted(selected_words):
        label_indices[word] = len(labels)
        labels.append(word)

    # Create links from text to words with frequency as value
    for label in text_labels:
        freqs = tt.data["word_freqs"][label]
        for word in selected_words:
            if word in freqs:
                sources.append(label_indices[label])
                targets.append(label_indices[word])
                values.append(freqs[word])

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    # Save and open diagram
    os.makedirs("output", exist_ok=True)
    html_path = "output/sankey_diagram.html"
    png_path = "output/sankey_diagram.png"

    fig.write_html(html_path)
    fig.write_image(png_path)
    webbrowser.open(f"file://{os.path.abspath(html_path)}")

def sentiment_distribution_subplots(tt):
    """
    Creates subplots of sentence-level sentiment histograms for each article.
    Each subplot shows the emotional distribution within an article.
    """
    print("📊 Building sentiment distribution subplots...")

    article_texts = tt.data["text"]
    num_articles = len(article_texts)
    cols = 3
    rows = -(-num_articles // cols)  # ceil division

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(article_texts.keys()),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    for i, (label, text) in enumerate(article_texts.items()):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentiments = [TextBlob(s).sentiment.polarity for s in sentences]

        row = (i // cols) + 1
        col = (i % cols) + 1

        fig.add_trace(
            go.Histogram(
                x=sentiments,
                nbinsx=20,
                marker_color='mediumslateblue',
                showlegend=False
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text="Sentiment", row=row, col=col)
        fig.update_yaxes(title_text="Sentence Count", row=row, col=col)

    fig.update_layout(
        height=300 * rows,
        width=1000,
        title_text="Sentence-Level Sentiment Distribution per Article",
        title_x=0.5,
        plot_bgcolor="white"
    )

    fig.show()