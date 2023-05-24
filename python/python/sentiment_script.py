from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

import re
import string
import random

""" 
remove_noise(tweet_tokens, stop_words=()):
    Input:  tweet_tokens (a list of tokens representing a tweet), 
            stop_words (optional list of stopwords to be removed)
    Output: cleaned_tokens (a list of cleaned tokens)

    This function removes noise from the tweet tokens by performing the following operations:
        -Removing URLs using regular expressions.
        -Removing Twitter handles (user mentions).
        -Lemmatizing the tokens based on their part of speech (noun, verb, adjective).
        -Filtering out tokens that are punctuation or stopwords.
"""
def remove_noise(tweet_tokens, stop_words=()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

"""
get_tweets_for_model(cleaned_tokens_list):
    Input: cleaned_tokens_list (a list of lists of cleaned tokens)
    Output: Generator that yields a dictionary of token:True pairs for each tweet tokens list

    This function iterates over the cleaned tokens list and 
    yields a dictionary where each token is a key with a value of True.
"""
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

"""
is_pos_or_neg(custom_string):
    Input: custom_string (a string representing a custom tweet)
    Output: "positive" or "negative" (the sentiment classification for the custom tweet)

    This function performs sentiment classification for a custom tweet
    by using the Naive Bayes classifier trained on the positive and negative tweet datasets.
    It tokenizes the custom string, removes noise from the tokens, 
    and then uses the classifier to predict the sentiment.
"""
def is_pos_or_neg(custom_string):
    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    positive_tokens_for_model = get_tweets_for_model(
        positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(
        negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]

    classifier = NaiveBayesClassifier.train(train_data)

    custom_tokens = remove_noise(word_tokenize(custom_string))

    return classifier.classify(dict([token, True] for token in custom_tokens))

