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
import matplotlib.pyplot as plt



file = "C:\\Users\\user\\Desktop\\semantics.json"
file1 = "C:\\Users\\user\\Desktop\\reviews1.json"  #reviews1.json
df1 = pd.read_json(file1,typ='series')

df_rev = json_normalize(df1['Reviews'])
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

# Some statistics about the reviews:
print("Number of reviews in reviews1.json = " + str(len(df.index)))
positive_rev = (df['scores'] == 'positive').sum()
negative_rev = (df['scores'] == 'negative').sum()
neutral_rev = (df['scores'] == 'neutral').sum()
print("Number of positive reviews =" + " "+str(positive_rev))
print("Number of negative reviews =" + " "+str(negative_rev))
print("Number of neutral reviews =" +" "+ str(neutral_rev))

df.describe()


df_X = pd.read_json(file1,typ='series')
df_XX = json_normalize(df_X['Reviews'])


df2 = json_normalize(df1['Reviews'])
df2 = df2.convert_objects(convert_numeric=True)

#mean_ratings = df2.groupby('ReviewID').mean()
#min_ratings = df2.groupby('Ratings.Overall').min()
#max_ratings = df2.groupby('Ratings.Overall').max()




#print ('\nThe mean values are: ')
#print (mean_ratings)

#print ('\nThe min values are: ')
#print (min_ratings)

#print ('\nThe max values are: ')
#print (max_ratings)


print ('\nThe max values are: ')
print ("Max Ratings.Overall " + str(max(json_normalize(df1['Reviews'])['Ratings.Overall'])))
print ("Max Ratings.Rooms " + str(max(json_normalize(df1['Reviews'])['Ratings.Rooms'])))
print ("Max Ratings.Value " + str(max(json_normalize(df1['Reviews'])['Ratings.Value'])))
print ("Max Ratings.Cleanliness " + str(max(json_normalize(df1['Reviews'])['Ratings.Cleanliness'])))
print ("Max Ratings.Location " + str(max(json_normalize(df1['Reviews'])['Ratings.Location'])))
print ("Max Ratings.Sleep Quality " + str(max(json_normalize(df1['Reviews'])['Ratings.Sleep Quality'])))

print ('\nThe min values are: ')
print ("Min Ratings.Overall " + str(min(json_normalize(df1['Reviews'])['Ratings.Overall'])))
print ("Min Ratings.Rooms " + str(min(json_normalize(df1['Reviews'])['Ratings.Rooms'])))
print ("Min Ratings.Value " + str(min(json_normalize(df1['Reviews'])['Ratings.Value'])))
print ("Min Ratings.Cleanliness " + str(min(json_normalize(df1['Reviews'])['Ratings.Cleanliness'])))
print ("Min Ratings.Location " + str(min(json_normalize(dfm['Reviews'])['Ratings.Location'])))
print ("Min Ratings.Sleep Quality " + str(min(json_normalize(df1['Reviews'])['Ratings.Sleep Quality'])))


#Authors and their comments ordered by date of comment 
review_comments = pd.DataFrame(df1['Reviews'], columns=['Author', 'Date', 'Content'])
review_comments.set_index(['Author'])
print (review_comments)
# json_normalize(df1['Reviews'])