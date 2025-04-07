## requires installation into ds environment

import nltk
from nltk.corpus import stopwords
import re
import string
from collections import Counter, defaultdict
from textblob import TextBlob

nltk.download('stopwords')


class TextAnalysis:

    def __init__(self):
        self.stopwords = set()
        self.data = defaultdict(dict)

    def load_stop_words(self):
        # A list of common stop words from NLTK library
        self.stopwords = set(stopwords.words('english'))


    def generic_parser(self, text):
        text = text.lower()

        # record the sentences
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_sentiments = [TextBlob(s).sentiment.polarity for s in sentences if s.strip()]
        avg_sentiment = sum(sentence_sentiments) / len(sentence_sentiments)

        # Clean the text -- white space and punctuation
        clean_text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Split up words
        words = clean_text.split()
        words = [word for word in words if word not in self.stopwords]

        # Get sentiment for raw text
        sentiment = TextBlob(text).sentiment.polarity

        text_dict = {
            "text": clean_text,
            "word_freqs": Counter(words),
            "num_words": len(words),
            "num_sentences": len(sentences),
            "avg_word_len": sum(len(word) for word in words) / len(words) if words else 0,
            "avg_sentence_len": len(words) / len(sentences),
            "sentiment": sentiment,
            "avg_sentiment": avg_sentiment
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


    def wordcount_sankey(self, word_list=None, k=5):
        # Map each text to words using a Sankey diagram, where the thickness of the line
        # is the number of times that word occurs in the text. Users can specify a particular
        # set of words, or the words can be the union of the k most common words across
        # each text file (excluding stop words).
        pass

    def your_second_visualization(self, misc_parameters):
        # A visualization array of subplots with one subplot for each text file.
        # Rendering subplots is a good, advanced skill to know!
        pass

    def your_third_visualization(self, misc_parameters):
        # A single visualization that overlays data from each of the text files. Make sure your
        # visualization distinguishes the data from each text file using labels or a legend
        pass
