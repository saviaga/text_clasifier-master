import pandas
import pickle
import cPickle
import numpy
import re
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.externals import joblib
from sklearn import svm,metrics

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def preprocess(sstr):
    x_text = clean_str(sstr.strip())

    vectorizer_path = "vectorizer.pkl"
    vectorizer = joblib.load(vectorizer_path)
    x_text_transformed  = vectorizer.transform([x_text])
    selector_path = "selector.pkl"
    selector = joblib.load(selector_path)
    x_text_transformed  = selector.transform(x_text_transformed).toarray()
    return x_text_transformed
def predict(feature):
    clf_path = "clasifier.pkl"
    clf = joblib.load(clf_path)
    return clf.predict(feature)

if __name__ == '__main__':
    sstr=raw_input("Give me a tweet: ")
    feature=preprocess(sstr)
    print(predict(feature))
