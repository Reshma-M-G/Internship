#!/usr/bin/env python
# coding: utf-8

# ## Project Description

# #### Build a model that classifies the side effects of a drug 

# ## Dataset Description

# * Name (categorical)       : Name of the patient
# * Age (numerical)          : Age group range of user
# * Race (categorical)       : Race of the patients 
# * Condition (categorical)  : Name of condition
# * Date (date)              : date of review entry
# * Drug (categorical)       : Name of drug
# * EaseOfUse (numerical)    : 5 star rating
# * Effectiveness (numerical): 5 star rating
# * Sex (categorical)        : gender of user
# * Side (text)              : side effects associated with drug (if any)
# 
# 
# 
# 

# #### Loading the libraries 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('Drug_Side_effects.csv')


# ## Understanding the data 

# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.duplicated().value_counts()


# In[6]:


data.shape


# In[7]:


data.nunique()


# In[8]:


data.describe()


# In[9]:


data.describe(include='all')


# In[10]:


data.columns


# In[11]:


data.Age.value_counts()


# In[12]:


data.Race.value_counts()


# In[13]:


data.Condition.value_counts()


# In[14]:


data.Drug.value_counts()


# In[15]:


data.EaseofUse.value_counts()


# In[16]:


data.Effectiveness.value_counts()


# In[17]:


data.Sex.value_counts()


# In[18]:


data.Sides.value_counts()


# ### Checking missing values

# In[19]:


data.isnull().sum()


# ####  Dataset has missing values 

# In[20]:


#Filling the dataset with modes 
data['Age'] = data['Age'].fillna(data['Age'].mode()[0])
data['Race'] = data['Race'].fillna(data['Race'].mode()[0])
data['Condition'] = data['Condition'].fillna(data['Condition'].mode()[0])
data['Drug'] = data['Drug'].fillna(data['Drug'].mode()[0])
data['Sex'] = data['Sex'].fillna(data['Sex'].mode()[0])
data['Sides'] = data['Sides'].fillna(data['Sides'].mode()[0])


# In[21]:


data.isnull().sum()


# #### Missing values are filled using mode and now dataset contains no missing values.

# ## Checking for Outliers

# In[22]:


# Plot boxplot to find outliers
df_numerical = data.select_dtypes(exclude='object')
x=1
plt.figure(figsize = (20, 15))
for col in df_numerical.columns:
    plt.subplot(6,4,x)
    sns.boxplot(data[col])
    x+=1
plt.tight_layout()


# #### No outliers present in the dataset 

# ## Data Encoding 

# In[23]:


#Creating a copy of the dataset "data"
new_data=data.copy()


# In[24]:


new_data.head()


# In[25]:


from sklearn.preprocessing import LabelEncoder

le_race=LabelEncoder()
le_Condition=LabelEncoder()
le_Drug=LabelEncoder()
le_Sex=LabelEncoder()


# In[26]:


new_data['Race']=le_race.fit_transform(data['Race'])
new_data['Condition']=le_Condition.fit_transform(data['Condition'])
new_data['Drug']=le_Drug.fit_transform(data['Drug'])
new_data['Sex']=le_Sex.fit_transform(data['Sex'])


# In[27]:


new_data['Age'].astype(str)


# In[28]:


# Replacing age_range
def age_fun(new_data):
    if new_data['Age'] == '0-10':
        return 1
    elif new_data['Age'] == '11-20':
        return 2
    elif new_data['Age'] == '21-30':
        return 3
    elif new_data['Age'] == '31-40':
        return 4
    elif new_data['Age'] == '41-50':
        return 5
    elif new_data['Age'] == '51-60':
        return 6
    elif new_data['Age'] == '61-70':
        return 7
    elif new_data['Age'] == '71-80':
        return 8
    elif new_data['Age'] == '81-90':
        return 9
    elif new_data['Age'] == '91-100':
        return 10


# In[29]:


new_data['Age'] = new_data.apply(age_fun, axis = 1)


# In[30]:


new_data.shape


# In[31]:


new_data.head()


# ## Data Reduction

# In[32]:


new_data.drop(['Name','Date'],axis=1,inplace=True)


# In[33]:


corrmatrix=new_data.corr()
plt.subplots(figsize=(10,4))
sns.heatmap(corrmatrix,vmin=-2,vmax=1,annot=True,linewidths=0.2,cmap='Accent')


# In[34]:


new_data.drop(['EaseofUse'],axis=1,inplace=True)


# In[35]:


new_data.head()


# ### Scaling

# In[36]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
scale=['Effectiveness']
new_data[scale] = ss.fit_transform(new_data[scale])


# ## Exploratory Data Analysis

# #### Age vs Side effect

# In[37]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(data['Age'],order=data['Age'].value_counts().index)
plt.subplot(1, 2, 2)
sns.countplot(data=data, x = 'Age', hue='Sides')
plt.suptitle("Age vs Side effect",y=1.05,fontsize=20)
plt.tight_layout()
plt.show()


data.Age.value_counts(normalize=True)
data.groupby('Age')['Sides'].value_counts(normalize=True)


# #### * Around 19% of the people are from 71-80 age group.Teenagers participation was least as compared to the adults  
# 

# #### Race vs Side effect

# In[38]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(data['Race'],order=data['Race'].value_counts().index)
plt.subplot(1, 2, 2)
sns.countplot(data=data, x = 'Race', hue='Sides')
plt.suptitle("Race vs Side effect",y=1.05,fontsize=20)
plt.tight_layout()
plt.show()


data.Race.value_counts(normalize=True)
data.groupby('Race')['Sides'].value_counts(normalize=True)


# #### * 31% of people are whites followed by Hispanic(28%),Black(22%) and Asian(18%)

# #### Condition

# In[39]:


plt.figure(figsize=(8,4))
sns.countplot(data=data,x='Condition',order=data['Condition'].value_counts().index)
plt.suptitle("Condition",y=1,fontsize=15)

plt.xticks(rotation=45,fontsize=8);
plt.show()

data.Condition.value_counts(normalize=True)


# #### * Majority of the people was affected by Pain followed by Migraine Headache. Few people are affected by Chronic Trouble sleeping.

# #### Drug vs Side effect

# In[40]:


plt.figure(figsize = (10, 5))
sns.countplot(data=data, x = 'Drug', hue='Sides')
plt.suptitle("Drug vs Side effect",y=1.05,fontsize=25)
plt.xticks(rotation=90,fontsize=8);
plt.tight_layout()
plt.show()  


# #### Drug

# In[41]:


plt.figure(figsize=(10,4))
sns.countplot(data=data,x='Drug',order=data['Drug'].value_counts().index)
plt.suptitle("Drug",y=1,fontsize=15)
#plt.xlabel('Drug', fontsize=15)
plt.xticks(rotation=90,fontsize=8);
plt.show()

data.Drug.value_counts(normalize=True)


# #### * Lexapro is the most widely used drug which is used for Depression and Pain

# In[42]:


data[data['Drug'] == 'lexapro']['Condition'].unique()


# In[43]:


data.groupby('Drug')['Condition'].nunique().sort_values(ascending=False)


# #### Effectiveness vs Side effect

# In[44]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(data['Effectiveness'],order=data['Effectiveness'].value_counts().index)
plt.subplot(1, 2, 2)
sns.countplot(data=data, x = 'Effectiveness', hue='Sides')
plt.suptitle("Effectiveness vs Side effect",y=1.05,fontsize=25)
plt.tight_layout()
plt.show()


data.Effectiveness.value_counts(normalize=True)


# #### * Majority rated the effectiveness as the drug used was extremely effective.When comparing with the side effect, more rated as that drug consumed has no side effects and rated effectiveness as Extremely Effective

# #### Effectiveness vs Age

# In[45]:


plt.figure(figsize=(8,4))
sns.countplot(data=data,x='Effectiveness',order=data['Effectiveness'].value_counts().index,hue='Age')
plt.xlabel('Effectiveness vs Age', fontsize=15)
plt.xticks(rotation=45,fontsize=8);
plt.tight_layout()
plt.show()


# #### The drug was extremely effective in 71-80 age group and not effective for the age group 41-50

# #### Ease of use vs Side effect

# In[46]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(data['EaseofUse'],order=data['EaseofUse'].value_counts().index)
plt.subplot(1, 2, 2)
sns.countplot(data=data, x = 'EaseofUse', hue='Sides')
plt.suptitle("EaseofUse vs Side effect",y=1.05,fontsize=25)
plt.tight_layout()
plt.show()


data.EaseofUse.value_counts(normalize=True)


# #### * Majority rated as 5 which means strongly agrees.When comparing with the side effect, majority strongly agrees that the drugs they consumed has no side effects

# #### Ease of use vs Age

# In[47]:


plt.figure(figsize=(8,4))
sns.countplot(data=data,x='EaseofUse',order=data['EaseofUse'].value_counts().index,hue='Age')
plt.xlabel('Ease of Use vs Age', fontsize=15)
plt.xticks(rotation=45,fontsize=8);
plt.show()


# #### * Easiness of use is high for the age groups 71-80 and 91-80 with respect to other age group. Easiness of use was low for the age group 41-50

# #### Ease of use vs Gender

# In[48]:


plt.figure(figsize=(8,4))
sns.countplot(data=data,x='EaseofUse',order=data['EaseofUse'].value_counts().index,hue='Sex')
plt.xlabel('Ease of Use vs Gender', fontsize=15)
plt.xticks(rotation=45,fontsize=8);
plt.show()


# #### Easiness rating was high for females compared to males.

# #### Sex vs Side effect

# In[49]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(data['Sex'],order=data['Sex'].value_counts().index)
plt.subplot(1, 2, 2)
sns.countplot(data=data, x = 'Sex', hue='Sides')
plt.suptitle("Sex vs Side effect",y=1.05,fontsize=25)
plt.tight_layout()
plt.show()


data.Sex.value_counts(normalize=True)


# #### Females are more affected by the side effects than men.

# #### Drug vs Age 

# In[50]:


plt.figure(figsize=(25,10))
sns.countplot(data=data,x='Drug',order=data['Drug'].value_counts().index,hue='Age')
plt.xlabel('Drug vs Age', fontsize=20)
plt.xticks(rotation=90,fontsize=8);
plt.show()


# #### Consumption rate of drug is high for the people of age group 71-80 ,81-90 and 91-100 and it is followed by adults.Consumption rate for teenagers is low .Higher Consumption rate of drug for the people above 71 shows their fear/old age illeness.

# #### Gender Participation

# In[51]:


plt.figure(figsize=(6,6))
data['Sex'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)
plt.show()


# #### Majority of the people are females

# #### Compairing Ease of use and Effectiveness with Year

# In[52]:


df=data.copy()
data['Date']=pd.to_datetime(data['Date'])
data['year']=data['Date'].dt.year # create year
data['month'] = data['Date'].dt.month #create month


# In[53]:


yea = data['year'].value_counts().sort_index()
sns.barplot(yea.index,yea.values,color='blue')
plt.title('Number of Year ')
plt.show()


# #### * Most of the rating was in the year 2009 followed by 2010 and 2008

# In[54]:


rating_year = data.groupby('year')['Effectiveness'].mean()
sns.barplot(rating_year.index,rating_year.values,color='blue')
plt.title('Mean Effectiveness rating Yearly')
plt.show()


# #### * Mean effective rating was high for the year 2007. 

# In[55]:


rating_year = data.groupby('year')['EaseofUse'].mean()
sns.barplot(rating_year.index,rating_year.values,color='blue')
plt.title('Mean Easeofuse rating Yearly')
plt.show()


# #### * Mean yearly ease of use rating was above 4 for the year 2007 compaared to other years.

# In[56]:


#checking year wise conditions counts
plt.figure(figsize=(8,5))

sns.swarmplot(data.groupby('year')['Condition'].value_counts(),color='green')
plt.title('Conditions Year wise',fontsize=20)
plt.xticks(rotation=90,fontsize=8);
plt.tight_layout()
plt.show()


# In[57]:


data.groupby('year')['Condition'].value_counts()


# #### * Conditions was higher during the year 2007 and it began to decrease from 2008 onwards to 2020.

# In[58]:


#checking year wise drug counts 


sns.swarmplot(data.groupby('year')['Drug'].value_counts(),color='green')
plt.title('Druglist Year wise',fontsize=20)
plt.show()


# In[59]:


data.groupby('year')['Drug'].value_counts()


# #### * Drug usage was high during 2007 as one drug is used for treating different condition.Later onwards,we can see a decline in the drug usage.

# In[60]:


plt.figure(figsize=(10,6))
plt.title('% of Side effect')
tr = pd.DataFrame(data['Sides'].value_counts())
tr_names = tr.index
count = tr['Sides']
plt.style.use('ggplot')
plt.rc('font', size=12)
plt.pie(count, autopct='%1.1f%%', labels = tr_names, pctdistance=0.9, labeldistance=1.1,shadow=True, startangle=90)

plt.show()


# ####  Around 70% of the people reported as they have side effects ranging from mild to extreme.30% reported as they have no side effects from the drugs they consumed.
# 

# In[61]:


plt.figure(figsize=(10,6))
plt.title('% of Conditions')
tr = pd.DataFrame(data['Condition'].value_counts())
tr_names = tr.index
count = tr['Condition']
plt.style.use('ggplot')
plt.rc('font', size=8.7)
plt.pie(count, autopct='%1.1f%%', labels = tr_names, pctdistance=0.9, labeldistance=1.1,shadow=True, startangle=90)

plt.show()


# #### * Majority of the people suffered due to Pain,Migraine Headache and other conditions.

# In[62]:


fe=data.loc[(data['Sex']=='female')]
dr=pd.DataFrame(fe['Drug'].value_counts(normalize=True))
dr_id=dr.index
list1=dr['Drug']
plt.pie(list1,labels=dr_id,labeldistance=1.3,autopct='%1.0f%%')
plt.show()


# In[63]:


me=data.loc[(data['Sex']=='male')]
dr=pd.DataFrame(me['Drug'].value_counts(normalize=True))
dr_id=dr.index
list1=dr['Drug']
plt.pie(list1,labels=dr_id,labeldistance=1.3,autopct='%1.0f%%')
plt.show()


# #### Age vs Sex

# In[64]:


plt.figure(figsize=(8,4))
sns.countplot(data=data,x='Age',hue='Sex')
plt.xlabel('Age', fontsize=15)
plt.xticks(rotation=45,fontsize=8);
plt.show()


# #### * Majority of the females and males participation is from 71-80,51-60 and 31-40 age groups.

# ## Splitting dataset to train and test

# In[65]:


y=new_data['Sides']
x=new_data.drop('Sides',axis=1)


# In[66]:


x.head()


# In[67]:


y.head()


# ## Modelling

# In[68]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score


# In[69]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


# #### Logistic Regression

# In[70]:


from sklearn.linear_model import LogisticRegression


# In[71]:


logrs_model=LogisticRegression(max_iter=100)# setting multiclass to multinomial since we have three categories 
logrs_model.fit(x_train,y_train)
y_pred=logrs_model.predict(x_test)

print("Accuracy",accuracy_score(y_test,y_pred))
print("F1 score:",f1_score(y_test,y_pred,average='macro'))


# In[72]:


conf=confusion_matrix(y_test,y_pred)
conf_mat= pd.DataFrame(conf, index = ['no sideffect','mild','moderate','severe','extreme severe'], columns = ['no side effects','mild side effects','moderate side effects','severe side effects','extreme severe side effects'])
conf_mat


# In[ ]:





# #### KNN Algorithm

# In[73]:


from sklearn.neighbors import KNeighborsClassifier 


# In[74]:


acc_values=[]

neighbor=np.arange(3,18)
for k in neighbor:
    k_model=KNeighborsClassifier(n_neighbors=k,metric='minkowski')
    k_model.fit(x_train,y_train)
    y_pred=k_model.predict(x_test)
  
    acc=accuracy_score(y_test,y_pred)
   
    acc_values.append(acc)
 


# In[75]:


print(acc_values)


# In[76]:


plt.plot(neighbor,acc_values,'*-')
plt.show()


# In[77]:


classifier=KNeighborsClassifier(n_neighbors=16,metric='minkowski')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

acc=accuracy_score(y_pred,y_test)
print(acc)


# In[78]:


print("F1 score",f1_score(y_test,y_pred,average='macro'))
print("precision",precision_score(y_test,y_pred,average='macro'))
print("recall score",recall_score(y_test,y_pred,average='macro'))


# In[79]:


conf=confusion_matrix(y_test,y_pred)
conf_mat= pd.DataFrame(conf, index = ['no sideffect','mild','moderate','severe','extreme severe'], columns = ['no side effects','mild side effects','moderate side effects','severe side effects','extreme severe side effects'])
conf_mat


# #### RANDOM FOREST CLASSIFIER 

# In[80]:


from sklearn.ensemble import RandomForestClassifier


# In[81]:


rfc_model=RandomForestClassifier()
rfc_model.fit(x_train,y_train)
y_pred=rfc_model.predict(x_test)
print("F1_score",f1_score(y_pred,y_test,average='macro'))
print("Accuracy",accuracy_score(y_pred,y_test))


# In[82]:


conf=confusion_matrix(y_test,y_pred)
conf_mat= pd.DataFrame(conf, index = ['no sideffect','mild','moderate','severe','extreme severe'], columns = ['no side effects','mild side effects','moderate side effects','severe side effects','extreme severe side effects'])
conf_mat


# Checking feature importance 

# In[83]:


pd.Series(rfc_model.feature_importances_,index=x.columns).sort_values(ascending=False)*100


# Fine tuning

# In[84]:


y1=pd.DataFrame(new_data['Sides'])
x1=new_data.drop(['Sides','Sex'],axis=1)

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,random_state=1,test_size=.2)


# In[85]:


rft_tuning=RandomForestClassifier(bootstrap=True,n_estimators=1500,max_depth=10,criterion='entropy',random_state=42)
rft_tuning.fit(x1_train,y1_train)
y1_pred=rft_tuning.predict(x1_test)
print("F1_score",f1_score(y1_test,y1_pred,average='macro'))
print("Accuracy",accuracy_score(y1_test,y1_pred)) 
conf=confusion_matrix(y1_test,y1_pred)
conf_mat= pd.DataFrame(conf, index = ['no sideffect','mild','moderate','severe','extreme severe'], columns = ['no side effects','mild side effects','moderate side effects','severe side effects','extreme severe side effects'])
conf_mat


# #### Naive Bayes Classifier

# In[86]:


from sklearn.naive_bayes import GaussianNB


# In[87]:


classifier=GaussianNB()
classifier.fit(x_train,y_train)
pred=classifier.predict(x_test)


# In[88]:


print("Accuracy_score",accuracy_score(y_test,pred))
print("precision",precision_score(y_test,pred,average='macro'))
print("F1_score",f1_score(y_test,y_pred,average='macro'))


# In[89]:


conf=confusion_matrix(y_test,y_pred)
conf_mat= pd.DataFrame(conf, index = ['no sideffect','mild','moderate','severe','extreme severe'], columns = ['no side effects','mild side effects','moderate side effects','severe side effects','extreme severe side effects'])
conf_mat


# #### Gradient Boosting Classifier 

# In[90]:


from sklearn.ensemble import GradientBoostingClassifier


# In[91]:


lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(x_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(x_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(x_test, y_test)))


# #### Gradient Boosting classifier with the highest accuracy rate is chosen

# In[92]:


gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(x_train, y_train)
y_pred = gb_clf2.predict(x_test)

print("Accuracy_score",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred,average='macro'))
print("F1_score",f1_score(y_test,y_pred,average='macro'))


# In[93]:


conf=confusion_matrix(y_test,y_pred)
conf_mat= pd.DataFrame(conf, index = ['no sideffect','mild','moderate','severe','extreme severe'], columns = ['no side effects','mild side effects','moderate side effects','severe side effects','extreme severe side effects'])
conf_mat


# In[ ]:




