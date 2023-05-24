import zipfile
import contextlib
import nltk

nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

with contextlib.closing(zipfile.ZipFile('py-sentiment.zip', "r")) as z:
   z.extractall("C:\\FragFrog\\py-sentiment")
   