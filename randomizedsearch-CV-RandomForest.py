#Randomized Search CV - Random Forest 

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv('Social_Network_Ads.csv')
#dataset

#Split data into dependent and independent features.
x= dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#Split dataset into train and test data
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=50)

#Scaling down all the features to a standard scale
scale = StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.fit_transform(x_test)

#Creating initial model without any hyperparameter optimization
model = RandomForestClassifier(n_estimators=10, 
                               criterion='entropy',random_state=50)
model.fit(x_train,y_train)

#creating a model again without any parameters.
est = RandomForestClassifier(n_jobs=-1)

#creating a value dictionary to perform Hyperparameter optimization on
val={'n_estimators':[100,200,300,400,500],
     'criterion':['gini','entropy'],
     'max_depth':[3,5,10,None],
     'max_features':randint(1,3),
     'bootstrap':[True,False],
     'min_samples_leaf':randint(1,4)
     }

#Hyperparameter optimization function to find best paramaters and score
def hypertuning_model(est,val,x,y,iter_no):
    rcv=RandomizedSearchCV(est,
                                param_distributions=val,
                                n_jobs=-1,
                                 n_iter=iter_no,
                                 cv=10)
    rcv.fit(x,y)
    best_p=rcv.best_params_
    best_s=rcv.best_score_
    return best_p, best_s

#best_parameter, best_scores = hypertuning_model(est, val, x, y, 45)
#print('Best parameters : {}'.format(best_parameter))
#print('Best score : {}'.format(best_scores))


#Creating final model on the parameters given by Hyperparameter optimization
model1 = RandomForestClassifier(bootstrap= True, criterion='entropy',
                                   max_depth= 10, max_features= 2,
                                   min_samples_leaf= 3, n_estimators=100)
model1.fit(x_train,y_train)
pred=model1.predict(x_test)

#Accuracy of model
acc=accuracy_score(y_test, pred)
cm=confusion_matrix(y_test, pred)
c_score=cross_val_score(model1, x,y,cv=10,scoring='accuracy')
print('Accuracy score',acc)
print('Confusion Matrix',cm)
print('Cross Validation score',c_score.mean())