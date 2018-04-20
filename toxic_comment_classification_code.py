import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

start = time.time()


train_data=pd.read_csv('train.csv')  #nrows=1000
test_data=pd.read_csv('test.csv')

# toxic=train_data.loc[train_data['toxic'] == 1]
# severe_toxic=train_data.loc[train_data['severe_toxic'] == 1]
# obscene=train_data.loc[train_data['obscene'] == 1]
# threat=train_data.loc[train_data['threat'] == 1]
# insult=train_data.loc[train_data['insult'] == 1]
# identity_hate=train_data.loc[train_data['identity_hate'] == 1]


y_toxic = train_data[['toxic']]
y_s_toxic = train_data[['severe_toxic']]
y_obscene = train_data[['obscene']]
y_threat = train_data[['threat']]
y_insult = train_data[['insult']]
y_i_hate = train_data[['identity_hate']]

y_toxic_train = train_data[['toxic']]
y_s_toxic_train = train_data[['severe_toxic']]
y_obscene_train = train_data[['obscene']]
y_threat_train = train_data[['threat']]
y_insult_train = train_data[['insult']]
y_i_hate_train = train_data[['identity_hate']]


# # Print number of tokens of vec_basic
#print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))
#

vectorizer = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',
                        stop_words= 'english',ngram_range=(1,4),dtype=np.float32) #
x_tfidf = vectorizer.fit_transform(train_data['comment_text'])

x_tfidf_train=x_tfidf

x_tfidf_test = vectorizer.transform(test_data['comment_text'])

#print("There are {} tokens in the dataset".format(len(x_tfidf.get_feature_names())))

X_train_toxic, X_test_toxic, y_train_toxic, y_test_toxic = train_test_split(x_tfidf, y_toxic, test_size=0.33, random_state=42)
X_train_s_toxic, X_test_s_toxic, y_train_s_toxic, y_test_s_toxic = train_test_split(x_tfidf, y_s_toxic, test_size=0.33, random_state=42)
X_train_obscene, X_test_obscene, y_train_obscene, y_test_obscene = train_test_split(x_tfidf, y_obscene, test_size=0.33, random_state=42)
X_train_threat, X_test_threat, y_train_threat, y_test_threat = train_test_split(x_tfidf, y_threat, test_size=0.33, random_state=42)
X_train_insult, X_test_insult, y_train_insult, y_test_insult = train_test_split(x_tfidf, y_insult, test_size=0.33, random_state=42)
X_train_i_hate, X_test_i_hate, y_train_i_hate, y_test_i_hate = train_test_split(x_tfidf, y_i_hate, test_size=0.33, random_state=42)


model_toxic = LogisticRegressionCV(cv=10)
model_toxic.fit(X_train_toxic, np.ravel(y_train_toxic))
pred_toxic=model_toxic.predict_proba(X_test_toxic)
#df_toxic = pd.DataFrame(pred_toxic[:,1])
#df_toxic.to_csv('pred_toxic.csv')

model_s_toxic = LogisticRegressionCV(cv=10)
model_s_toxic.fit(X_train_s_toxic, np.ravel(y_train_s_toxic))
pred_s_toxic=model_toxic.predict_proba(X_test_s_toxic)
#df_s_toxic = pd.DataFrame(pred_s_toxic[:,1])
#df_s_toxic.to_csv('pred_s_toxic.csv')

model_obscene = LogisticRegressionCV(cv=10)
model_obscene.fit(X_train_obscene, np.ravel(y_train_obscene))
pred_obscene=model_obscene.predict_proba(X_test_obscene)
#df_obscene = pd.DataFrame(pred_obscene[:,1])
#df_obscene.to_csv('pred_obscene.csv')

model_threat = LogisticRegressionCV(cv=10)
model_threat.fit(X_train_threat, np.ravel(y_train_threat))
pred_threat=model_threat.predict_proba(X_test_threat)
#df_threat = pd.DataFrame(pred_threat[:,1])
#df_threat.to_csv('pred_threat.csv')

model_insult = LogisticRegressionCV(cv=10)
model_insult.fit(X_train_insult, np.ravel(y_train_insult))
pred_insult=model_insult.predict_proba(X_test_insult)
#df_insult = pd.DataFrame(pred_insult[:,1])
#df_insult.to_csv('pred_insult.csv')

model_i_hate = LogisticRegressionCV(cv=10)
model_i_hate.fit(X_train_i_hate, np.ravel(y_train_i_hate))
pred_i_hate=model_i_hate.predict_proba(X_test_i_hate)
#df_i_hate = pd.DataFrame(pred_i_hate[:,1])
#df_i_hate.to_csv('pred_i_hate.csv')



model_toxic_real = LogisticRegressionCV(cv=10)
model_toxic_real.fit(x_tfidf_train, np.ravel(y_toxic_train))
pred_toxic_real=model_toxic_real.predict_proba(x_tfidf_test)
#df_toxic = pd.DataFrame(pred_toxic[:,1])
#df_toxic.to_csv('pred_toxic.csv')

model_s_toxic_real = LogisticRegressionCV(cv=10)
model_s_toxic_real.fit(x_tfidf_train, np.ravel(y_s_toxic_train))
pred_s_toxic_real=model_s_toxic_real.predict_proba(x_tfidf_test)
#df_s_toxic = pd.DataFrame(pred_s_toxic[:,1])
#df_s_toxic.to_csv('pred_s_toxic.csv')

model_obscene_real = LogisticRegressionCV(cv=10)
model_obscene_real.fit(x_tfidf_train, np.ravel(y_obscene_train))
pred_obscene_real=model_obscene_real.predict_proba(x_tfidf_test)
#df_obscene = pd.DataFrame(pred_obscene[_real:,1])
#df_obscene.to_csv('pred_obscene.csv')

model_threat_real = LogisticRegressionCV(cv=10)
model_threat_real.fit(x_tfidf_train, np.ravel(y_threat_train))
pred_threat_real=model_threat_real.predict_proba(x_tfidf_test)
#df_threat = pd.DataFrame(pred_threat[:,1])
#df_threat.to_csv('pred_threat.csv')

model_insult_real = LogisticRegressionCV(cv=10)
model_insult_real.fit(x_tfidf_train, np.ravel(y_insult_train))
pred_insult_real=model_insult.predict_proba(x_tfidf_test)
#df_insult = pd.DataFrame(pred_insult[:,1])
#df_insult.to_csv('pred_insult.csv')

model_i_hate_real = LogisticRegressionCV(cv=10)
model_i_hate_real.fit(x_tfidf_train, np.ravel(y_i_hate_train))
pred_i_hate_real=model_i_hate.predict_proba(x_tfidf_test)
#df_i_hate = pd.DataFrame(pred_i_hate[:,1])


score_toxic=roc_auc_score(y_test_toxic, pred_toxic[:,1])
print('roc_auc_score toxic',score_toxic)

score_s_toxic=roc_auc_score(y_test_s_toxic, pred_s_toxic[:,1])
print('roc_auc_score severe toxic',score_s_toxic)

score_obscene=roc_auc_score(y_test_obscene, pred_obscene[:,1])
print('roc_auc_score obscene',score_obscene)

score_threat=roc_auc_score(y_test_threat, pred_threat[:,1])
print('roc_auc_score threat',score_threat)

score_insult=roc_auc_score(y_test_insult, pred_insult[:,1])
print('roc_auc_score insult',score_insult)

score_i_hate=roc_auc_score(y_test_i_hate, pred_i_hate[:,1])
print('roc_auc_score identity_hate',score_i_hate)


columns_names=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
columns_values_train_t=[list(train_data.values[:,0]),list(pred_toxic[:,1]),list(pred_s_toxic[:,1])
                         ,list(pred_obscene[:,1]),list(pred_threat[:,1]),list(pred_insult[:,1])
                         ,list(pred_i_hate[:,1])]
columns_values_train=list(zip(*columns_values_train_t))
print(columns_names)

df_result_train = pd.DataFrame(columns_values_train,columns=columns_names)
df_result_train.to_csv('results_train.csv',index=False)

columns_values_t=[list(test_data.values[:,0]),list(pred_toxic_real[:,1]),list(pred_s_toxic_real[:,1])
                         ,list(pred_obscene_real[:,1]),list(pred_threat_real[:,1]),list(pred_insult_real[:,1])
                         ,list(pred_i_hate_real[:,1])]
columns_values=list(zip(*columns_values_t))


df_result = pd.DataFrame(columns_values,columns=columns_names)
df_result.to_csv('results.csv',index=False)


end=time.time()
temp = end-start
#print(temp)
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('%d:%d:%d' %(hours,minutes,seconds))