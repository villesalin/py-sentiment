from random import shuffle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from statistics import mean

import numpy as np
import pandas as pd
import nltk

nltk.download([
    "names",
    "stopwords",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])


def isPostive_Negative_or_Neutral(text: str) -> str:
    sentiment = sia.polarity_scores(text)["compound"]
    if sentiment >= 0.5:
        return str(sentiment)+" Positive"
    elif sentiment <= -0.5:
        return str(sentiment)+" Negative"
    else:
        return str(sentiment)+" Neutral"


example_input1 = "Jere is great teacher but school is hectic and there is not always time to do all homework"
example_input2 = "Jere is THE BEST! teacher ever and I want to learn everything from him!!"
example_input3 = "Jere is an asshole and no one likes him!!"
example_input4 = "Jere is a fucking IDIOT and asshole and no one likes him!!"

stopwords = nltk.corpus.stopwords.words("english")

sia = SentimentIntensityAnalyzer()

df = pd.read_csv(
    "C:/Users/Aapo/py-sentiment/python/TextFiles/reviews.csv", sep="\t")

print(df.head(99))
print("")
print(isPostive_Negative_or_Neutral(example_input1))
print(isPostive_Negative_or_Neutral(example_input2))
print(isPostive_Negative_or_Neutral(example_input3))
print(isPostive_Negative_or_Neutral(example_input4))