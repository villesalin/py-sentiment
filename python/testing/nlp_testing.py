import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tree import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

example_text = "Hello there, how are you doing today? The weather is great and python is awesome."
example_text2 = """NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

Thanks to a hands-on guide introducing programming fundamentals alongside topics in computational linguistics, plus comprehensive API documentation, NLTK is suitable for linguists, engineers, students, educators, researchers, and industry users alike. NLTK is available for Windows, Mac OS X, and Linux. Best of all, NLTK is a free, open source, community-driven project. """

example_text3 = """
Muad'Dib learned rapidly because his first training was in how to learn.
And the first lesson of all was the basic trust that he could learn.
It's shocking to find how many people do not believe they can learn,
and how many more believe learning to be difficult."""

worf_quote = "Sir, I protest. I am not a merry man!"

words_in_quote = word_tokenize(worf_quote)

stop_words = set(stopwords.words('english'))

filtered_list = []

for word in words_in_quote:
    if word.casefold() not in stop_words:
        filtered_list.append(word)

# alternate way
# filtered_list = [
#      word for word in words_in_quote if word.casefold() not in stop_words
# ]

#print(filtered_list)



string_for_stemming = """
The crew of the USS Discovery discovered many discoveries.
Discovering is what explorers do."""
stemmer = PorterStemmer()
words = word_tokenize(string_for_stemming)
stemmed_words = [stemmer.stem(word) for word in words]
#print(stemmed_words)


carl_sagan_quote = """
If you wish to make an apple pie from scratch,
you must first invent the universe."""
words_in_carl_sagan_quote = word_tokenize(carl_sagan_quote)
#print(nltk.pos_tag(words_in_carl_sagan_quote))
# All the words in the quote are now in a separate tuple, 
# with a tag that represents their part of speech. 
# But what do the tags mean? Here’s how to get a list of tags and their meanings:
# nltk.help.upenn_tagset()

jabberwocky_excerpt = """
'Twas brillig, and the slithy toves did gyre and gimble in the wabe:
all mimsy were the borogoves, and the mome raths outgrabe."""
words_in_excerpt = word_tokenize(jabberwocky_excerpt)
#print(nltk.pos_tag(words_in_excerpt))

lemmatizer = WordNetLemmatizer()
#print(lemmatizer.lemmatize("scarves")) #scarf

string_for_lemmatizing = "The friends of DeSoto love scarves."
words = word_tokenize(string_for_lemmatizing)
#print(words)
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
#print(lemmatized_words)


lotr_quote = "It's a dangerous business, Frodo, going out your door."
words_in_lotr_quote = word_tokenize(lotr_quote)
#print(words_in_lotr_quote)
nltk.download("averaged_perceptron_tagger")
lotr_pos_tags = nltk.pos_tag(words_in_lotr_quote)
#print(lotr_pos_tags)

grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)
#tree = chunk_parser.parse(lotr_pos_tags)
#tree.draw()

grammar2 = """
Chunk: {<.*>+}
       }<JJ>{"""
chunk_parser = nltk.RegexpParser(grammar2)
#tree = chunk_parser.parse(lotr_pos_tags)
#tree.draw()


nltk.download("maxent_ne_chunker")
nltk.download("words")
#tree = nltk.ne_chunk(lotr_pos_tags)
#tree.draw()

# tree = nltk.ne_chunk(lotr_pos_tags, binary=True)
# tree.draw()


quote = """
Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2."""

def extract_ne(quote):
    words = word_tokenize(quote, language="english")
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )

# print(extract_ne(quote))

nltk.download("book")
from nltk.book import *

#text8.concordance("man")
#text8.concordance("woman")

# text8.dispersion_plot(
#     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
# )

#text2.dispersion_plot(["Allenham", "Whitwell", "Cleveland", "Combe"])


from nltk import FreqDist

# frequency_distribution = FreqDist(text8)
# print(frequency_distribution)

#print(frequency_distribution.most_common(20))

meaningful_words = [
    word for word in text8 if word.casefold() not in stop_words
]

# frequency_distribution_meaningful_words = FreqDist(meaningful_words)
# print(frequency_distribution_meaningful_words)
# print(frequency_distribution_meaningful_words.most_common(20))
# frequency_distribution_meaningful_words.plot(20, cumulative=True)


text8.collocations()
lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]
new_text = nltk.Text(lemmatized_words)
new_text.collocations()

# for i in sent_tokenize(example_text3):
#     print(i)
#     print("")

# for i in word_tokenize(example_text3):
#     print(i)


# stopwords

# stop_words = set(stopwords.words('english'))

# print(stop_words)



