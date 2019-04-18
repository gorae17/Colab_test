import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import Kfold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score

categories = list(range(0,20))
random_state = 30

def prediction_rule(x):
  ret = [0 for i in range(0,len(x))]
  arg = np.array(x).argsort()
  ret[arg[-1]] = 1
  for idx in arg[-3:-1]:
    if x[idx] >= 1:
      ret[idx] = 1

  return ret

  
def init(datapath):
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


def test(testdata, trainedmodel):
  x_test = testdata.features
  x_test_v = trainedmodel['vectorizer'].transform(x_test)
  
  for category in categories:
    testdata['pred_{}'.format(category)] = np.sum([
        trainedmodel['LSVC_{}'.format(category)].predict(x_test_v),
        np.array(trainedmodel['LORG_{}'.format(category)].predict_proba(x_test_v))[:, 1]
    ], axis=0)
    
  y_test = testdata[[i for i in range(0,20)]]
  y_pred = testdata[['pred_{}'.format(i) for i in range(0,20)]]
  y_pred = np.array([res for res in y_pred.apply(lambda x: prediction_rule(x), axis=1)])
  
  return fbeta_score(np.array(y_test), y_pred, 0.5, average='micro')


def cv(df, fold=5, random_state=10):
  fscore_list = []
  kf = KFold(n_splits=fold, shuffle=True, random_state = random_state)
  for train_idx, test_idx in kf.shuffle(df):
	df_train = df.iloc[train_idx]
	df_test = df.iloc[test_idx]
	trainedmodel = train(df_train)
	fscore_list.append(test(df_test, trainedmodel))

  return fscore_list
                     
  
if __name__ == "__main__":
  datapath = sys.argv[1]
  df = init(datapath)
  fscore_list = cv(df, random_state=random_state)
  print('F0.5 Score (mean) :', np.mean(fscore_list))
  print(fscore_list)
