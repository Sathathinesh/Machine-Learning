import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix 
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import tree

#get the data from csv file

df = pd.read_csv("trainData.csv")

#replace missing values in A2&A14 with value 0 & 1
df['A2'].replace({'?':0}, inplace=True)      
df['A14'].replace({'?':1}, inplace=True)

#convert column A2 & A14 to numeric
df.A2=pd.to_numeric(df.A2)  
df.A14=pd.to_numeric(df.A14)

#replace zero and one with mean
df['A2'].replace({0:df.A2.mean()}, inplace=True)
df['A14'].replace({1:df.A14.mean()}, inplace=True)

#convert boolean to int
df.A8 = df.A8.astype(int)	
df.A11 = df.A11.astype(int)	
df.A13 = df.A13.astype(int)

#Label encode categorical variables
Label=LabelEncoder()
df['A1']=Label.fit_transform(df['A1'])
df['A3']=Label.fit_transform(df['A3'])
df['A4']=Label.fit_transform(df['A4'])
df['A6']=Label.fit_transform(df['A6'])
df['A9']=Label.fit_transform(df['A9'])
df['A15']=Label.fit_transform(df['A15'])
df['A16']=Label.fit_transform(df['A16'])


#prepare the training set
TrainingSet=df.drop(['A16'],axis='columns')


#train model
X=TrainingSet
Y=df.A16
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=0)
scaler = MinMaxScaler(feature_range = (0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#classification

#classify=svm.SVC(kernel='linear',C=1)
#classify=tree.DecisionTreeClassifier()
classify=KNeighborsClassifier(n_neighbors=100,p=2,metric='euclidean')

classify.fit(X_train,y_train)

y_pred=classify.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix: ")
print(cm)
print("Classification score: ",classify.score(X_test,y_test))
print("Classification report")
print(classification_report(y_test,y_pred))


###########################################################################################
#############     PRE PROCESS TEST DATA     ###############################################

df2 = pd.read_csv("testdata.csv")

#replace missing values in A2&A14 with value 0 and 1
df2['A2'].replace({'?':0}, inplace=True)      
df2['A14'].replace({'?':1}, inplace=True)


#convert column A2 & A14 to numeric
df2.A2=pd.to_numeric(df2.A2)  
df2.A14=pd.to_numeric(df2.A14)

#replace zero and one with mean
df2['A2'].replace({0:df2.A2.mean()}, inplace=True)
df2['A14'].replace({1:df2.A14.mean()}, inplace=True)

#convert boolean to int
df2.A8 = df2.A8.astype(int)	
df2.A11 = df2.A11.astype(int)	
df2.A13 = df2.A13.astype(int)

#Label encode categorical variables
Label_test=LabelEncoder()
df2['A1']=Label_test.fit_transform(df2['A1'])
df2['A3']=Label_test.fit_transform(df2['A3'])
df2['A4']=Label_test.fit_transform(df2['A4'])
df2['A6']=Label_test.fit_transform(df2['A6'])
df2['A9']=Label_test.fit_transform(df2['A9'])
df2['A15']=Label_test.fit_transform(df2['A15'])

#predict the test data
Test=df2
test_scaler = scaler.transform(Test)

out=classify.predict(test_scaler)
print(out)
out=np.where(out==1,'Success',out)
out=np.where(out=='0','Failure',out)
print(out)      
np.savetxt("outknn.csv",out,fmt='%s', delimiter=",")





