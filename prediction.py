import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

categories = list(range(0,20))
random_state = 10

def prediction_rule(x):
  ret = [0 for i in range(0,len(x))]
  arg = np.array(x).argsort()
  ret[arg[-1]] = 1
  for idx in arg[-3:-1]:
    if x[idx] >= 1:
      ret[idx] = 1

  return ret

  
def init_train(datapath):
  data = open(datapath,'r').readlines()
  data = [line.strip().split('\t') for line in data]
  df = pd.DataFrame.from_records(data, columns=['id','labels','features'])
  df['features'] = df['features'].apply(
      lambda x: " ".join([tok.split('/')[0] for tok in x.split()[:300]]))
  df['labels'] = df['labels'].apply(lambda x: [int(y) for y in x.split()])
  for i in categories:
    df[i] = df['labels'].apply(lambda x: 1 if i in x else 0)
  
  return df


def train(traindata):  
  trainedmodel = {}
  
  x_train = traindata.features
  vectorizer = TfidfVectorizer()
  vectorizer.fit(x_train)
  trainedmodel['vectorizer'] = vectorizer
  
  x_train_v = vectorizer.transform(x_train)
  
  for category in categories:
    models = [['LSVC', LinearSVC(C=0.5, random_state = random_state)],
              ['LORG', LogisticRegression(solver='sag', random_state = random_state)]]
    for model in models:
      model[1].fit(x_train_v, traindata[category])
      trainedmodel['{}_{}'.format(model[0], category)] = model[1]
      
  return trainedmodel


def predict_and_print(trainedmodel, testdata):
  x_test = testdata.features
  x_test_v = trainedmodel['vectorizer'].transform(x_test)
  
  for category in categories:
    testdata['pred_{}'.format(category)] = np.sum([
        trainedmodel['LSVC_{}'.format(category)].predict(x_test_v),
        np.array(trainedmodel['LORG_{}'.format(category)].predict_proba(x_test_v))[:, 1]
    ], axis=0)
    
  y_pred = testdata[['pred_{}'.format(i) for i in range(0,20)]]
  y_pred['prediction'] = y_pred.apply(lambda x: prediction_rule(x), axis=1)
  y_pred['labels'] = y_pred['prediction'].apply(
		  lambda x: " ".join([str(idx) for idx in np.where(np.array(x) > 0)[0]]))
  y_pred['labels'] = y_pred['prediction'].apply(lambda x: print(x))
  

def init_test(datapath):
  testdata = open(datapath, 'r').readlines()
  testdata = [[i, line.strip()] for i, line in enumerate(testdata)]
  df = pd.DataFrame.from_records(testdata, columns=['id', 'features'])
  df['features'] = df['features'].apply(lambda x: " ".join([tok.split('/')[0] for tok in x.split(' ')[:300]]))
  
  return df
  

if __name__ == "__main__":
  trainpath = sys.argv[1]
  testpath = sys.argv[2]

  df_train = init_train(trainpath)
  trainedmodel = train(df_train)

  df_test = init_test(testpath)
  predict_and_print(trainedmodel, df_test)
