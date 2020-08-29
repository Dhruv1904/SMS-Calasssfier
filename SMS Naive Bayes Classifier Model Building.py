#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection  import train_test_split
from sklearn import metrics
import pickle
#docs = pd.read_excel('SMSSpamCollection.xls',header=None,names=['Class', 'SMS']) 
docs = pd.read_table('SMSSpamCollection+(1)', header=None, names=['Class', 'sms'])

#classifier in column 1, sms in column 2.
docs.head()


# In[2]:


# counting spam and ham instances
# df.column_name.value_counts() - gives no. of unique inputs in the columns

ham_spam=docs.Class.value_counts()
ham_spam


# In[3]:


print("Spam % is ",(ham_spam[1]/float(ham_spam[0]+ham_spam[1]))*100)


# In[4]:


# mapping labels to 0 and 1
docs['label'] = docs.Class.map({'ham':0, 'spam':1})


# In[5]:


docs.head()


# In[6]:


X=docs.sms
y=docs.label


# In[7]:


X = docs.sms
y = docs.label
print(X.shape)
print(y.shape)


# In[8]:


# splitting into test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[9]:


X_train.head()


# In[10]:


# vectorizing the sentences; removing stop words
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')


# In[11]:


vect.fit(X_train)
# X_train_dtm = vect.transform(X_train)


# In[12]:


# printing the vocabulary
vect.vocabulary_


# In[13]:


# transforming the train and test datasets
X_train_transformed = vect.transform(X_train)
X_test_transformed =vect.transform(X_test)


# In[14]:


# note that the type is transformed matrix
print(type(X_train_transformed))
print(X_train_transformed)


# In[15]:


# training the NB model and making predictions

mnb = MultinomialNB()

# fit
mnb.fit(X_train_transformed,y_train)

# predict class
y_pred_class = mnb.predict(X_test_transformed)

# predict probabilities
y_pred_proba =mnb.predict_proba(X_test_transformed)


# printing the overall accuracy

metrics.accuracy_score(y_test, y_pred_class)


# In[16]:


y_pred_class


# In[17]:


# note that alpha=1 is used by default for smoothing
mnb


# In[18]:


# confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[19]:


confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
#[row, column]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]


# In[20]:


sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)


# In[21]:


specificity = TN / float(TN + FP)

print("specificity",specificity)


# In[22]:


precision = TP / float(TP + FP)

print("precision",precision)
print(metrics.precision_score(y_test, y_pred_class))


# In[23]:


print("precision",precision)
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))


# In[24]:


y_pred_class


# In[25]:


y_pred_proba


# In[26]:


# creating an ROC curve
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[27]:


# area under the curve
print (roc_auc)


# In[28]:


print(true_positive_rate)


# In[29]:


print(false_positive_rate)


# In[30]:


print(thresholds)


# In[31]:


# matrix of thresholds, tpr, fpr
pd.DataFrame({'Threshold': thresholds, 
              'TPR': true_positive_rate, 
              'FPR':false_positive_rate
             })


# In[32]:


# plotting the ROC curve

get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)


# In[33]:


n=str(' win lottery of 100000$')


# In[34]:


n


# In[35]:


y=mnb.predict_proba(vect.transform([n]))


# In[36]:


pickle.dump(mnb,open('naivebayes.pkl','wb'))


model = pickle.load(open('naivebayes.pkl','rb'))


# In[37]:


z=vect.transform([n])


# In[38]:


z=model.predict_proba(vect.transform([n]))


# In[39]:


pickle.dump(vect, open("vect.pkl", "wb")) 
vectorizer=pickle.load(open("vect.pkl", 'rb')) 


# In[40]:


y=mnb.predict_proba(vectorizer.transform([n]))


# In[41]:


y


# In[42]:


X_train


# In[ ]:





# In[ ]:




