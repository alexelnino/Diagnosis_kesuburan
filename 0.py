import numpy as np
import pandas as pd

dfData=pd.read_csv(
    'fertility.csv'
)

# print(dfData.head(5))
# print(dfData.columns.values)
# ['Season' 'Age' 'Childish diseases' 'Accident or serious trauma'
#  'Surgical intervention' 'High fevers in the last year'
#  'Frequency of alcohol consumption' 'Smoking habit'
#  'Number of hours spent sitting per day' 'Diagnosis']

# Labelling
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

dfData['Childish diseases'] = label.fit_transform(dfData['Childish diseases'])
# print(label.classes_)   #['no' 'yes']

dfData['Accident or serious trauma'] = label.fit_transform(dfData['Accident or serious trauma'])
# print(label.classes_)   #['no' 'yes']

dfData['Surgical intervention'] = label.fit_transform(dfData['Surgical intervention'])
# print(label.classes_)   #['no' 'yes']

dfData['Frequency of alcohol consumption'] = label.fit_transform(dfData['Frequency of alcohol consumption'])
# print(label.classes_)   #['every day' 'hardly ever or never' 'once a week' 'several times a day'
                        #'several times a week']

dfData['Smoking habit'] = label.fit_transform(dfData['Smoking habit'])
# print(label.classes_)   #['daily' 'never' 'occasional']

dfData['Diagnosis'] = label.fit_transform(dfData['Diagnosis'])
# print(label.classes_)   #['Altered' 'Normal']

dfData = dfData.drop(
    ['Season', 'High fevers in the last year'],
    axis=1
)

# cleaning data outlier
dfData=dfData.drop([50], axis=0)
# print(dfData.head(5))

# Split: feature X & target Y
x = dfData.drop(['Diagnosis'], axis=1)
# print(x)
# print(x.iloc[0])
# Age                                      30
# Childish diseases                         0
# Accident or serious trauma                1
# Surgical intervention                     1
# Frequency of alcohol consumption          2
# Smoking habit                             2
# Number of hours spent sitting per day    16
y = dfData['Diagnosis']
# print(y)
# print(dfData.columns.values)
# ['Age' 'Childish diseases' 'Accident or serious trauma'
#  'Surgical intervention' 'Frequency of alcohol consumption*'
#  'Smoking habit*' 'Number of hours spent sitting per day' 'Diagnosis*']



# One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 5])],           
    remainder = 'passthrough'                               
)                                                         

x = np.array(coltrans.fit_transform(x))
# print(x[0])         # [ 0.              0.                      1.             0.             0.                       0.     0.         1.        30.        0.                       1.                         1.                  16.]
# read:             'every day' 'hardly ever or never' 'once a week' 'several times a day'  'several times a week' 'daily' 'never' 'occasional'   'Age' 'Childish diseases' 'Accident or serious trauma' 'Surgical intervention'  hour sitting]
#                   'Frequency of alcohol consumption*'                                                            'Smoking habit*

# Splitting
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size = .1
)

# print(xtrain[0])
# print(ytrain.iloc[0])
# print(xtest[0])
# print(ytest.iloc[0])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'liblinear')
lr.fit(xtrain, ytrain)

# print(round(lr.score(xtest, ytest) * 100, 2), '%')
# print(xtest[0])
# print(ytest.iloc[0])
# print(lr.predict(xtest[0].reshape(1, -1)))

# random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rf.fit(xtrain,ytrain)

# KNN(K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain,ytrain)


# data
arin=[1,0,0,0,0,1,0,0,29,0,0,0,5]
bebi=[0,0,1,0,0,0,1,0,31,0,1,1,8]
caca=[0,1,0,0,0,0,1,0,25,1,0,0,7]
dini=[0,1,0,0,0,1,0,0,28,0,1,1,15]
enno=[0,1,0,0,0,0,1,0,42,1,0,0,8]

# print('arin')

# print('logistic regression : ',lr.predict([arin]))
# print('random forest : ',rf.predict([arin]))
# print('K-nearest Neighbors : ',knn.predict([arin]))

# hasil

# arin
if lr.predict([arin])==[0]:
    print('Arin, prediksi kesuburan: ALTERED (Logistic Regression)')
else:
    print('Arin, prediksi kesuburan: NORMAL (Logistic Regression)')
if rf.predict([arin])==[0]:
    print('Arin, prediksi kesuburan: ALTERED (Random Forest)')
else:
    print('Arin, prediksi kesuburan: NORMAL (Random Forest)')
if knn.predict([arin])==[0]:
    print('Arin, prediksi kesuburan: ALTERED (K-nearest neighbors)')
else:
    print('Arin, prediksi kesuburan: NORMAL (K-nearest neighbors)')
print()
# bebi
if lr.predict([bebi])==[0]:
    print('Bebi, prediksi kesuburan: ALTERED (Logistic Regression)')
else:
    print('Bebi, prediksi kesuburan: NORMAL (Logistic Regression)')
if rf.predict([bebi])==[0]:
    print('Bebi, prediksi kesuburan: ALTERED (Random Forest)')
else:
    print('Bebi, prediksi kesuburan: NORMAL (Random Forest)')
if knn.predict([bebi])==[0]:
    print('Bebi, prediksi kesuburan: ALTERED (K-nearest neighbors)')
else:
    print('Bebi, prediksi kesuburan: NORMAL (K-nearest neighbors)')
print()
# caca
if lr.predict([caca])==[0]:
    print('Caca, prediksi kesuburan: ALTERED (Logistic Regression)')
else:
    print('Caca, prediksi kesuburan: NORMAL (Logistic Regression)')
if rf.predict([caca])==[0]:
    print('Caca, prediksi kesuburan: ALTERED (Random Forest)')
else:
    print('Caca, prediksi kesuburan: NORMAL (Random Forest)')
if knn.predict([caca])==[0]:
    print('Caca, prediksi kesuburan: ALTERED (K-nearest neighbors)')
else:
    print('Caca, prediksi kesuburan: NORMAL (K-nearest neighbors)')
print()
# dini
if lr.predict([dini])==[0]:
    print('Dini, prediksi kesuburan: ALTERED (Logistic Regression)')
else:
    print('Dini, prediksi kesuburan: NORMAL (Logistic Regression)')
if rf.predict([dini])==[0]:
    print('Dini, prediksi kesuburan: ALTERED (Random Forest)')
else:
    print('Dini, prediksi kesuburan: NORMAL (Random Forest)')
if knn.predict([dini])==[0]:
    print('Dini, prediksi kesuburan: ALTERED (K-nearest neighbors)')
else:
    print('Dini, prediksi kesuburan: NORMAL (K-nearest neighbors)')
print()
# enno
if lr.predict([enno])==[0]:
    print('Enno, prediksi kesuburan: ALTERED (Logistic Regression)')
else:
    print('Enno, prediksi kesuburan: NORMAL (Logistic Regression)')
if rf.predict([enno])==[0]:
    print('Enno, prediksi kesuburan: ALTERED (Random Forest)')
else:
    print('Enno, prediksi kesuburan: NORMAL (Random Forest)')
if knn.predict([enno])==[0]:
    print('Enno, prediksi kesuburan: ALTERED (K-nearest neighbors)')
else:
    print('Enno, prediksi kesuburan: NORMAL (K-nearest neighbors)')






