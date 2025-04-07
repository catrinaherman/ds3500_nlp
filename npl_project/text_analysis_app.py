from text_analysis_class import TextAnalysis

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
    tt.load_text("The_Hill.txt", label="The_Hill")

    tt.wordcount_sankey(k=5)
    tt.sentiment_distribution_subplots()
    tt.article_sentiment_comparison()
    tt.article_sentiment_comparison(sentiment_type='negative')
    tt.article_sentiment_comparison(sentiment_type='neutral')
    tt.sentiment_scatter()

if __name__ == "__main__":
    main()







