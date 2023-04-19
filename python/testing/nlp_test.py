import nltk
import re, string, random
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize



# nltk.download('twitter_samples')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

# 5000 tweets with positive sentiment
#positive_tweets = twitter_samples.strings('positive_tweets.json')
# 5000 tweets with negative sentiment
#negative_tweets = twitter_samples.strings('negative_tweets.json')
# 20000 tweets with no sentiment
#text = twitter_samples.strings('tweets.20150430-223406.json')

#positive tweets tokenized
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
"""
    tokenize() is a method provided by the Natural Language Toolkit 
    (NLTK) library in Python, which is used to break a given text into 
    individual words or tokens. It is a common step in natural language 
    processing (NLP) and text analysis tasks, where text data needs to 
    be processed and analyzed at the word level.

    tokenized() method in the "tweet" module is used for tokenizing 
    tweets into individual words, taking into account the unique 
    characteristics of tweets.
"""
# print(tweet_tokens[99]) # ['My', 'birthday', 'is', 'a', 'week', 'today', '!', ':D']

# print(pos_tag(tweet_tokens[99]))
''' give words position in sentence tag.
    NN: Noun, common, singular or mass
    VBG: Verb, gerund or present participle
    etc...'''


def lemmatize_sentence(tokens):
    ''' This code imports the WordNetLemmatizer class 
    and initializes it to a variable, lemmatizer.
    The function lemmatize_sentence first gets the position tag 
    of each token of a tweet. Within the if statement, 
    if the tag starts with NN, the token is assigned as a noun. 
    Similarly, if the tag starts with VB, the token is assigned 
    as a verb.'''
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

# print(tweet_tokens[99])
# print(lemmatize_sentence(tweet_tokens[99]))

stop_words = stopwords.words('english') #english stopwords to remove

def remove_noise(tweet_tokens, stop_words = ()):
    ''' This code creates a remove_noise() function that removes noise 
    and incorporates the normalization and lemmatization mentioned in 
    the previous section. The code takes two arguments: the tweet tokens
    and the tuple of stop words.
    The function removes all @ mentions, stop words, and converts the words to lowercase.'''
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

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

# print(remove_noise(tweet_tokens[99], stop_words))

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

# print(positive_tweet_tokens[500])
# print(positive_cleaned_tokens_list[500])

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)

#print("Accuracy is:", classify.accuracy(classifier, test_data))

#print(classifier.show_most_informative_features(10))

custom_tweet = 'After the access to Champions League semi finals, AC Milan are prepared to confirm and announce Olivier Giroud’s contract extension. 🔴⚫️🤝🏻 #MilanNew deal agreed weeks ago and set to be signed — it will be valid until June 2024, one more season.'
# Some Tweets
# Last few days I’m really wondering if people see my tweets on their timeline.. If you see this tweet, please give it a like. 🙏
# Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies
# Pure happiness.. 😊
# After the access to Champions League semi finals, AC Milan are prepared to confirm and announce Olivier Giroud’s contract extension. 🔴⚫️🤝🏻 #MilanNew deal agreed weeks ago and set to be signed — it will be valid until June 2024, one more season.

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token, True] for token in custom_tokens)))