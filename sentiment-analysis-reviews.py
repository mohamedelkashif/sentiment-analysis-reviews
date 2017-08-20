import pandas as pd
import csv
import json
import nltk
import numpy as np
from pandas.io.json import json_normalize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

file = "C:\\Users\\user\\Desktop\\semantics.json"
file1 = "C:\\Users\\user\\Desktop\\reviews1.json"  #reviews1.json
df = pd.read_json(file1,typ='series')
df_rev = json_normalize(df['Reviews'])
df_rev['full_content'] = df_rev['Content'] + '. ' + df_rev['Title']
pos = pd.read_json(file,typ='series')

newcol = df_rev['full_content']
newcol.to_csv('content_reviews.csv',header=False, index=False, encoding='utf-8')

df = pd.read_csv('C:\Users\user\Documents\sentiment-analysis-reviews\content_reviews.csv',encoding='utf-8', header=None)
scores = []

for x in df[0]:
    sen = x.lower()
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(sen)
    
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    #print wordsFiltered
    posScore = 0
    negScore = 0
    for i in range(0,len(pos['positive'])):
        for s in wordsFiltered:
            if s in pos['positive'][i]['phrase']:
                posScore = posScore+pos['positive'][i]['value']
                
    for j in range(0,len(pos['negative'])):
        for s in wordsFiltered:
            if s in pos['negative'][j]['phrase']:
                negScore = negScore+pos['negative'][j]['value']
                
    if posScore > negScore:
        scores.append('positive')
        
    elif posScore < negScore:
        scores.append('negative')
    else:
        scores.append('neutral')
        
            
df['scores'] = scores
#df

df.to_csv('hotels_scores.csv', header=False, index=False,encoding='utf-8')

#After having csv containg reviews and their coresponding sentiment, pass them to ML model to train and get their accuracy

#loading file

def load_file():
    with open('C:\Users\user\Documents\sentiment-analysis-reviews\hotels_scores.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            if row[0] and row[1]:
                data.append(row[0])
                target.append(row[1])

        return data,target

#creating the trem frequency matrix for the dataset
def preprocess():
    data,target = load_file()
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    
    return tfidf_data

def learning_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.3,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    evaluate_model(target_test,predicted)
    

def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))
        
def main():
    data,target = load_file()
    tf_idf = preprocess()
    learning_model(tf_idf,target)
main()
