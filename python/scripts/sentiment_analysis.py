import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


feedback = pd.read_csv('../data/customer_feedback.csv') 


nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


feedback['sentiment'] = feedback['review'].apply(lambda x: sia.polarity_scores(x)['compound'])


feedback['sentiment_label'] = feedback['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')


feedback['sentiment_label'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.show()
