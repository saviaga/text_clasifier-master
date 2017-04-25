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

def load_data_and_labels(data_path):
    """
    Loads Data and labels
    Returns split sentences and labels.
    """
    #data to dataframe
    # Load data from files
    df=pandas.read_csv(data_path)
    x_text=df.Text.tolist()
    # Data preprocessing
    x_text = [clean_str(s.strip()) for s in x_text]
    # Generate labels
    labels=df.Class_Tweet.tolist()

    return x_text,labels
def preprocess(X,y):
    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)


    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)
    joblib.dump(vectorizer, 'vectorizer.pkl')

    ### feature selection, because text is super high dimensional and
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    joblib.dump(selector, 'selector.pkl')
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()
    return features_train_transformed, features_test_transformed, labels_train, labels_test

def train(X_train, X_test, y_train, y_test):
    clf = svm.SVC(C=1,kernel='linear')
    clf.fit(X_train, y_train)
    acc=metrics.accuracy_score(y_test,clf.predict(X_test))
    print("End traning")
    print("model acc:"+str(acc))
    joblib.dump(clf, 'clasifier.pkl')
if __name__ == '__main__':
    print("Loading Data...")
    X,y=load_data_and_labels("../data/data.csv")
    print("Data preprocessing...")
    X_train, X_test, y_train, y_test=preprocess(X,y)
    print len(y_train[0])
    print len(X_train[0])
    print("Start training")
    train(X_train, X_test, y_train, y_test)
