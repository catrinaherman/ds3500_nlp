## requires installation into ds environment

import nltk
from nltk.corpus import stopwords
import re
import string
from collections import Counter, defaultdict
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

nltk.download('stopwords')


class TextAnalysis:

    def __init__(self):
        self.stopwords = set()
        self.data = defaultdict(dict)

    # List of stop words
    def load_stop_words(self):
        # A list of common stop words from NLTK library
        self.stopwords = set(stopwords.words('english'))

    # Our generic parser for pre-processing text
    def generic_parser(self, text):
        text = text.lower()

        # record the sentences
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # get the sentence sentiments and average sentiment
        sentence_sentiments = [TextBlob(s).sentiment.polarity for s in sentences if s.strip()]
        avg_sentiment = sum(sentence_sentiments) / len(sentence_sentiments)

        # analyze the polarity of each individual sentence, whether positive, negative or neutral
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for s in sentences:
            polarity = TextBlob(s).sentiment.polarity
            if polarity > .1:
                sentiment_counts['positive'] += 1
            elif polarity < -.1:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1

        # get the percentages of each articles sentiment
        sentiment_percentages = {
            sentiment: round((count / len(sentence_sentiments) * 100), 3) if len(sentence_sentiments) else 0
            for sentiment, count in sentiment_counts.items()
        }

        # Clean the text -- white space and punctuation
        clean_text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Split up words
        words = clean_text.split()
        words = [word for word in words if word not in self.stopwords]

        # Get sentiment for raw text
        sentiment = TextBlob(text).sentiment.polarity

        # Create the dictionary
        text_dict = {
            "text": clean_text,
            "word_freqs": Counter(words),
            "num_words": len(words),
            "num_sentences": len(sentences),
            "avg_word_len": sum(len(word) for word in words) / len(words) if words else 0,
            "avg_sentence_len": len(words) / len(sentences),
            "sentiment": sentiment,
            "avg_sentiment": avg_sentiment,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages
        }

        return text_dict

    def load_text(self, filename, label=None, parser=None):
        # Register a text file with the library. The label is an optional label you’ll use in your
        # visualizations to identify the text
        with open(filename, 'r') as f:
            text = f.read()

        if not label:
            label = filename

        if parser:
            parsed = parser(text)
        else:
            parsed = self.generic_parser(text)

        for k, v in parsed.items():
            self.data[k][label] = v

    # Visualization 1: Sankey
    def wordcount_sankey(self, word_list=None, k=5):
        labels = []  # All nodes in the diagram: text names + words
        sources = []  # Index of source node (text label)
        targets = []  # Index of target node (word)
        values = []  # Thickness of the connection = word count
        label_indices = {}  # Mapping from label to node index

        # Getting all the text labels
        text_labels = list(self.data["word_freqs"].keys())

        # Picks which words to use:
        if word_list:
            selected_words = set(word_list)
        else:
            selected_words = set()
            for label in text_labels:
                most_common = self.data["word_freqs"][label].most_common(k)
                selected_words.update([w for w, _ in most_common])

        # Add text labels first
        for label in text_labels:
            label_indices[label] = len(labels)
            labels.append(label)

        # Then add word labels
        for word in sorted(selected_words):
            label_indices[word] = len(labels)
            labels.append(word)

        # Build text to word connections
        for label in text_labels:
            word_freq = self.data["word_freqs"][label]
            for word in selected_words:
                if word in word_freq:
                    sources.append(label_indices[label])
                    targets.append(label_indices[word])
                    values.append(word_freq[word])

        # Create sankey diagram
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

        fig.update_layout(title_text="Text-to-Word Sankey Diagram", font_size=10)
        fig.show()


    def sentiment_distribution_subplots(self):
        """
        Creates subplots of sentence-level sentiment histograms for each article.
        Each subplot shows the emotional distribution within an article.
        """

        num_articles = len(self.data["text"])
        cols = 2
        rows = -(-num_articles // cols)  # ceil division


        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(self.data['text'].keys()),
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )

        for i, label in enumerate(self.data['text'].keys()):
            sentiment_percentages = self.data['sentiment_percentages'][label]

            labels = ['negative', 'neutral', 'positive']

            sentiments = [
            sentiment_percentages.get('negative', 0),
            sentiment_percentages.get('neutral', 0),
            sentiment_percentages.get('positive', 0),
            ]

            colors = ['red', 'gray', 'green']

            row = (i // cols) + 1
            col = (i % cols) + 1

            fig.add_trace(
                go.Bar(
                    x=labels,
                    y = sentiments,
                    marker_color=colors,
                    showlegend=False,
                    textposition='auto'
                ),
                row=row,
                col=col,
            )

            fig.update_xaxes(
                title_text="Sentiment",
                tickvals=["positive", "neutral", "negative"],  # Set the tick values to the sentiment labels
                row=row,
                col=col
            )

            # Set y-axis properties with the same range across all subplots
            fig.update_yaxes(
                title_text="Sentence Percentage",
                range=[0, 80],  # Set the y-axis range to the max sentence count
                row=row,
                col=col
            )

        fig.update_layout(
            height=300 * rows,
            width=1000,
            title_text="Sentence-Level Sentiment Distribution per Article",
            title_x=0.5,
            plot_bgcolor="white"
        )
        fig.show()

    def article_sentiment_comparison(self, sentiment_type='positive'):
        """
        Creates a bar chart comparing the percentage of a selected sentiment type
        (positive, neutral, or negative) for each article.
        """
        if sentiment_type not in ['positive', 'negative', 'neutral']:
            raise ValueError("sentiment_type must be 'positive', 'negative', or 'neutral'")

        article_labels = list(self.data['sentiment_percentages'].keys())
        percentages = []

        for label in article_labels:
            percent = self.data['sentiment_percentages'][label].get(sentiment_type, 0)
            percentages.append(percent)

        color_map_dict = {
            'positive': 'green',
            'negative': 'red',
            'neutral': 'gray'
        }

        fig = go.Figure(data=[
            go.Bar(
                x=article_labels,
                y=percentages,
                marker_color=color_map_dict[sentiment_type],
                text=[f"{p:.1f}%" for p in percentages],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title=f"Percentage of {sentiment_type.capitalize()} Sentences per Article",
            xaxis_title="Article",
            yaxis_title=f"% {sentiment_type.capitalize()} Sentences",
            yaxis=dict(range=[0, 100]),
            bargap=0.3
        )

        fig.show()

    def sentiment_scatter(self):
        """
        Creates a scatterplot of positive vs. negative sentiment percentages for each article.
        """
        x_vals = []
        y_vals = []
        labels = []

        for label, percentages in self.data['sentiment_percentages'].items():
            pos_percent = percentages.get('positive', 0)
            neg_percent = percentages.get('negative', 0)
            x_vals.append(pos_percent)
            y_vals.append(neg_percent)
            labels.append(label)

        fig = go.Figure(data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            text=labels,
            textposition='top center',
            marker=dict(
                size=10,
                color='royalblue',
                line=dict(width=1, color='black')
            )
        ))

        fig.update_layout(
            title="Positive vs. Negative Sentiment (Percentage) per Article",
            xaxis_title="Positive Sentiment (%)",
            yaxis_title="Negative Sentiment (%)",
            xaxis=dict(range=[0, 45]),
            yaxis=dict(range=[0, 25])
        )

        fig.show()

    def radar_chart(self):
        """
        Generates a radar chart comparing article metrics like sentiment, word length, etc.
        """

        metrics = {
            "avg_word_len": [],
            "avg_sentence_len": [],
            "sentiment": [],
            "num_words": [],
            "%_positive": [],
            "%_negative": [],
            "%_neutral": []
        }

        article_labels = list(self.data["text"].keys())

        for label in article_labels:
            media = self.data
            sent_count = media["sentiment_counts"][label]
            total = sum(sent_count.values())

            metrics["avg_word_len"].append(media["avg_word_len"][label])
            metrics["avg_sentence_len"].append(media["avg_sentence_len"][label])
            metrics["sentiment"].append(media["sentiment"][label])
            metrics["num_words"].append(media["num_words"][label])
            metrics["%_positive"].append(sent_count["positive"] / total * 100)
            metrics["%_negative"].append(sent_count["negative"] / total * 100)
            metrics["%_neutral"].append(sent_count["neutral"] / total * 100)

        categories = list(metrics.keys())
        fig = go.Figure()

        # Normalize non-percentage metrics (first 4)
        normalized_metrics = {}
        for key in list(metrics.keys())[:4]:
            max_val = max(metrics[key])
            normalized_metrics[key] = [v / max_val * 100 if max_val else 0 for v in metrics[key]]

            # Create the radar chart for each article
        for i, label in enumerate(article_labels):
            values = [
                normalized_metrics["avg_word_len"][i],
                normalized_metrics["avg_sentence_len"][i],
                normalized_metrics["sentiment"][i],
                normalized_metrics["num_words"][i],
                metrics["%_positive"][i],
                metrics["%_negative"][i],
                metrics["%_neutral"][i]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=label
            ))

        fig.update_layout(
            title="Radar Chart of Text Metrics by Article",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True
        )

        fig.show()

