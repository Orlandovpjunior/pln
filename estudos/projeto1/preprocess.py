from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

class Preprocess:
    def __init__(self, text):
        self.text = text
        self.processed_text = text
        self.tokenizer = RegexpTokenizer(r'(\w+)')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('portuguese'))

    def normalize(self):
        self.processed_text = self.processed_text.lower()
        return self

    def tokenize(self):
        self.processed_text = self.tokenizer.tokenize(self.processed_text)
        return self

    def remove_big_small_words(self):
        self.processed_text = [word for word in self.processed_text if len(word) > 1 and len(word) < 15]
        return self

    def remove_stopwords(self):
        self.processed_text = [word for word in self.processed_text if word not in self.stop_words]
        return self

    def stem(self):
        self.processed_text = [self.stemmer.stem(word) for word in self.processed_text]
        return self

    def lemmatize(self):
        self.processed_text = [self.lemmatizer.lemmatize(word) for word in self.processed_text]
        return self

    def get_result(self):
        return self.processed_text

    def process(self):
        return (self.normalize()
                .tokenize()
                .remove_big_small_words()
                .remove_stopwords()
                .lemmatize()
                .get_result())
    
    
    
    
    
        