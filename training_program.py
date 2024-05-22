import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv("C://Users/Public/MachineLearning-1/sonar data.csv", header=None)

sonar_data.head

sonar_data.shape

sonar_data.describe()

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

X= sonar_data.drop(columns=60, axis=1)
Y= sonar_data[60]

#print (X)
#print(Y)


#to train and test the data
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)
#print(X.shape, X_train.shape, X_test.shape)

#using logistic regression for model training since we are training binaries
model=LogisticRegression()
model.fit(X_train,Y_train)

#model evaluation to find the accuracy on the training model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#show the training modelaccuracy score in percentage
training_accuracy_percentage= training_data_accuracy*100
#print("The accuracy on the training data is", training_accuracy_percentage, "percent")

#model evaluation to find the accuracy on the test model
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

#show the testing model accuracy score in percentage
testing_accuracy_percentage= testing_data_accuracy*100
#print("The accuracy on the training data is", testing_accuracy_percentage, "percent")

