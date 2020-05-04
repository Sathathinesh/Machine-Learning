import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("trainData.csv")

#replace '?' in A2&A14 with value 0
df['A2'].replace({'?':0}, inplace=True)      
df['A14'].replace({'?':0}, inplace=True)

#convert column A2 & A14 to numeric
df.A2=pd.to_numeric(df.A2)  
df.A14=pd.to_numeric(df.A14)

#convert boolean to int
df.A8 = df.A8.astype(int)	
df.A11 = df.A11.astype(int)	
df.A13 = df.A13.astype(int)

#create dummy variables
A1_dummy=pd.get_dummies(df.A1).drop(['?'],axis='columns')
A3_dummy=pd.get_dummies(df.A3).drop(['l'],axis='columns')
A4_dummy=pd.get_dummies(df.A4).drop(['gg'],axis='columns')
A6_dummy=pd.get_dummies(df.A6).drop(['?'],axis='columns')
A9_dummy=pd.get_dummies(df.A9).drop(['dd'],axis='columns')
A9_dummy=A9_dummy.drop(['o'],axis='columns')
A15_dummy=pd.get_dummies(df.A15).drop(['p'],axis='columns')
A16_dummy=pd.get_dummies(df.A16).drop(['Failure'],axis='columns')
	

#drop categorical variables
df=df.drop(['A1','A3','A4','A6','A9','A15','A16'],axis='columns')

#concate df with new dummy variables
merge=pd.concat([df,A1_dummy,A3_dummy,A4_dummy,A6_dummy,A9_dummy,A15_dummy],axis='columns')

#train medel
X=merge
Y=A16_dummy.Success
X_train, X_test, y_train, y_test = train_test_split(X,Y)

#classify=svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
classify=RandomForestClassifier(n_estimators=1000, bootstrap = True,max_features = 'sqrt')
classify.fit(X_train,y_train)

print(classify.score(X_test,y_test))

###########################################################################################
#############     PRE PROCESS TEST DATA     ###############################################

df2 = pd.read_csv("testdata.csv")

#replace '?' in A2&A14 with value 0
df2['A2'].replace({'?':0}, inplace=True)      
df2['A14'].replace({'?':0}, inplace=True)

#convert column A2 & A14 to numeric
df2.A2=pd.to_numeric(df2.A2)  
df2.A14=pd.to_numeric(df2.A14)

#convert boolean to int
df2.A8 = df2.A8.astype(int)	
df2.A11 = df2.A11.astype(int)	
df2.A13 = df2.A13.astype(int)

#create dummy variables
A1_dummy2=pd.get_dummies(df2.A1).drop(['?'],axis='columns')
A3_dummy2=pd.get_dummies(df2.A3)
A4_dummy2=pd.get_dummies(df2.A4)
A6_dummy2=pd.get_dummies(df2.A6).drop(['?'],axis='columns')
A9_dummy2=pd.get_dummies(df2.A9)
A15_dummy2=pd.get_dummies(df2.A15).drop(['p'],axis='columns')

#drop categorical variables
df2=df2.drop(['A1','A3','A4','A6','A9','A15'],axis='columns')

#concate df with new dummy variables
Test=pd.concat([df2,A1_dummy2,A3_dummy2,A4_dummy2,A6_dummy2,A9_dummy2,A15_dummy2],axis='columns')
out=classify.predict(Test)
print(out)
out=np.where(out==1,'Success',out)
out=np.where(out=='0','Failure',out)
print(out)      
np.savetxt("outsvm.csv",out,fmt='%s', delimiter=",")





