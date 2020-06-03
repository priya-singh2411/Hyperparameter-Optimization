#Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#Loading dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
#print(dataset)

#Dividing dataset to dependent(x) and independent(y) features
x=dataset.iloc[ : ,[2,3]].values
y=dataset.iloc[:,4].values

#Split the dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=5)

#Scaling down the dependent data (x) to a standard scale
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#Creating SVM model
svm = SVC(kernel='linear',random_state=0)
svm.fit(x_train, y_train)
pred=svm.predict(x_test)

#Checking the accuracy of the model developed
acc= accuracy_score(y_test,pred)
print('The accuracy of model generated is : {} '.format(acc))


#Applying hyperparameter optimization- GridSearchCV 
svm_param=[{'C':[10,100,200,300,500],'kernel':['linear']},
           {'C':[10,100,200,300,500],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.6,0.8,0.7]}]
Gcv = GridSearchCV(estimator=svm,param_grid=svm_param,
                   scoring='accuracy',
                   cv=10,
                   n_jobs=-1)
Gcv=Gcv.fit(x_train,y_train)

#Find the best score and parameters giving best score in GridSearchCV
accuracy=Gcv.best_score_
best_parameter= Gcv.best_params_
print('Grid Search accuracy is : {} and given by parameters : {}'
      .format(np.round(accuracy,4),best_parameter))
#print(best_parameter)

#Training the model again on the best parameters value obtained in GridSearchCV
svm = SVC(kernel='rbf',C=10,gamma=0.3)
svm.fit(x_train, y_train)
pred=svm.predict(x_test)
acc_new= accuracy_score(y_test,pred)
print('New accuracy score of SVM model after applying best params from Grid Search CV {}'.format(acc_new))
