from statistics import mean
from random import shuffle
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

import numpy as np
import pandas as pd
from pprint import pprint

nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

stopwords = nltk.corpus.stopwords.words("english")

text = """
Python 3 made backwards-incompatible changes and affected major parts of the language, like unifying string and Unicode, adding typing (in later versions), and even affected print(), the most basic way of creating some output.

However, there were libraries that helped, so someone could write Python code that worked on both versions of Python. Notably, one library (aptly named) was “six”. Six would act as a translator between the two versions and run a command the correct way, depending on whether it detected that the runtime was Python 2 or Python 3.

Unfortunately, moving everyone over to Python 3 was not an easy task. Many third-party libraries had to move over as well, and operating systems often packaged both versions of Python at the same time. This lead to more than a little confusion.

Recall that Python 3.0 was first released in 2008, with nearly one version every year since then. A “final” version of Python 2, version 2.7, was released in 2010. However, it received security fixes until 2019 and was supported until January 1, 2020. There was even a website with a countdown made by the community.

"""

# words: list[str] = nltk.word_tokenize(text)                 #tokenizening the sample text
# words = [w for w in words if w.lower() not in stopwords]    #removing stopwords
# print(words)
# fd = nltk.FreqDist(words)                                   #creating frequency distribution object from sample text
# lower_fd = nltk.FreqDist([w.lower() for w in words])
# print(fd.most_common(3))
# fd.tabulate(9)
# lower_fd.tabulate(9)

# print(fd["Python"])  # 9 times appears
# print(fd["python"])  # 0 times appears
# print(fd["PYTHON"])  # 0 times appears
# print(lower_fd["Python"])  # 0 times appears
# print(lower_fd["python"])  # 9 times appears
# print(lower_fd["PYTHON"])  # 0 times appears

text = nltk.Text(nltk.corpus.state_union.words())

# words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
# words_no_stopwords = [w for w in words if w.lower() not in stopwords]

# text = nltk.Text(nltk.corpus.state_union.words())
# text.concordance("america", lines=5)
''' 
In the context of NLP, a concordance is a collection of word locations along with their context. You can use concordances to find:
    1.How many times a word appears
    2.Where each occurrence appears
    3.What words surround each occurrence

Note that .concordance() already ignores case, allowing you to see the context of all case variants of a word in order of appearance.
Note also that this function doesn’t show you the location of each word in the text.

Additionally, since .concordance() only prints information to the console, it’s not ideal for data manipulation. 
To obtain a usable list that will also give you information about the location of each occurrence, use .concordance_list():
'''
concordance_list = text.concordance_list("america", lines=2)
for entry in concordance_list:
    print(entry.line)
"""
.concordance_list() gives you a list of ConcordanceLine objects,
which contain information about where each word occurs as well as a few more properties worth exploring. 
The list is also sorted in order of appearance.
"""

words: list[str] = nltk.word_tokenize(
    """Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex."""
)
text = nltk.Text(words)
fd = text.vocab()  # Equivalent to fd = nltk.FreqDist(words)
fd.tabulate(3)
"""
.vocab() is essentially a shortcut to create a frequency distribution from an instance of nltk.Text. 
That way, you don’t have to make a separate call to instantiate a new nltk.FreqDist object.
"""
"""
Another powerful feature of NLTK is its ability to quickly find collocations with simple function calls. 
Collocations are series of words that frequently appear together in a given text. In the State of the Union corpus, 
for example, you’d expect to find the words United and States appearing next to each other very often. 
Those two words appearing together is a collocation.

Collocations can be made up of two or more words. NLTK provides classes to handle several types of collocations:

    Bigrams: Frequent two-word combinations
    Trigrams: Frequent three-word combinations
    Quadgrams: Frequent four-word combination
"""
# words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
# finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
# # BigramCollocationFinder and QuadgramCollocationFinde exist too
# print(finder.ngram_fd.most_common(2))
# finder.ngram_fd.tabulate(2)

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("Wow, NLTK is really powerful!")
# {'neg': 0.0, 'neu': 0.295, 'pos': 0.705, 'compound': 0.8012}
print(sia.polarity_scores("Wow, NLTK is really powerful!"))

# tweets = [t.replace("://", "//")
#           for t in nltk.corpus.twitter_samples.strings()]


# def is_positive(tweet: str) -> bool:
#     """True if tweet has positive compound sentiment, False otherwise.
#     In this case, is_positive() uses only the positivity of the compound score to make the call.
#     You can choose any combination of VADER scores to tweak the classification to your needs."""
#     return sia.polarity_scores(tweet)["compound"] > 0


# shuffle(tweets)
# for tweet in tweets[:10]:
#     print(">", is_positive(tweet), tweet)

positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids


# def is_positive(review_id: str) -> bool:
#     """True if the average of all sentence compound scores is positive."""
#     text = nltk.corpus.movie_reviews.raw(review_id)
#     scores = [
#         sia.polarity_scores(sentence)["compound"]
#         for sentence in nltk.sent_tokenize(text)
#     ]
#     return mean(scores) > 0


# shuffle(all_review_ids)
# correct = 0
# for review_id in all_review_ids:
#     if is_positive(review_id):
#         if review_id in positive_review_ids:
#             correct += 1
#     else:
#         if review_id in negative_review_ids:
#             correct += 1

# print(F"{correct / len(all_review_ids):.2%} correct") #64% correct

unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])


def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True


positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
)]
negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
)]

positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}

# df = pd.read_csv(
#     "C:/Users/Aapo/py-sentiment/python/TextFiles/reviews.csv", sep="\t")

# print(df.head(99))
