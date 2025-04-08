from text_analysis_class import TextAnalysis

def main():
    # Initialize the text analysis object
    tt = TextAnalysis()

    # Loading common English stopwords to filter out from the text
    tt.load_stop_words()

    # Loading and labeling each news article file
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
    tt.load_text("The_Hill.txt", label="The_Hill")

<<<<<<< Updated upstream
    # Generate a sankey diagram to show which words are most commonly used
    tt.wordcount_sankey(k=3)

    # Create subplots showing sentiment distribution (positve and negative) per article
    tt.sentiment_distribution_subplots()

    # Create barcharts comparing the % of positive sentiment across all articles
    tt.article_sentiment_comparison(sentiment_type='positive')

    #Create barcharts comparing the % of negative sentiment across all articles
    tt.article_sentiment_comparison(sentiment_type='negative')

    # Plotting positive vs. negative sentiment % in a scatter plot
    tt.sentiment_scatter()

    # Create a radar chart to compare multiple features (ex: avg sentence length, positive/negative sentiment, etc)
=======
    # tt.wordcount_sankey(k=3)
    # tt.sentiment_distribution_subplots()
    # tt.article_sentiment_comparison(sentiment_type='positive')
    # tt.article_sentiment_comparison(sentiment_type='negative')
    # tt.sentiment_scatter()
>>>>>>> Stashed changes
    tt.radar_chart()

# Runs the analysis when the script gets executed
if __name__ == "__main__":
    main()







