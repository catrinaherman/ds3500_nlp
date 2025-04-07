from text_analysis_class import TextAnalysis
import pprint as pp
import plotly.graph_objects as go
from sankey_subplots import text_to_word_sankey
from sankey_subplots import sentiment_distribution_subplots


def main():
    tt = TextAnalysis()
    tt.load_stop_words()
    tt.load_text("ABC_1.txt", label="ABC1")
    tt.load_text("ABC_2.txt", label="ABC2")
    tt.load_text("BBC_1.txt", label="BBC1")
    tt.load_text("CNN_1.txt", label="CNN1")
    tt.load_text("CNN_2.txt", label="CNN2")
    tt.load_text("Fox_1.txt", label="FOX1")
    tt.load_text("Fox_2.txt", label="FOX2")
    tt.load_text("Fox_3.txt", label="FOX3")
    tt.load_text("NPR.txt", label="NPR")
    tt.load_text("USA_Today.txt", label="USA_Today")

    pp.pprint(tt.data)

    text_to_word_sankey(tt, k=5)
    sentiment_distribution_subplots(tt)


if __name__ == "__main__":
    main()

# Sankey Visualization

def wordcount_sankey(self, word_list=None, k=5):
    labels = []           # All nodes in the diagram: text names + words
    sources = []          # Index of source node (text label)
    targets = []          # Index of target node (word)
    values = []           # Thickness of the connection = word count
    label_indices = {}    # Mapping from label to node index

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

    # Example
    tt.wordcount_sankey(k=5)





