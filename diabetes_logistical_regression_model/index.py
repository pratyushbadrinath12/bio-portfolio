import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_csv("diabetes.csv")

Pregnancies= list(data["Pregnancies"])
Glucose= list(data["Glucose"])
BloodPressure = list(data["BloodPressure"])
SkinThickness= list(data["SkinThickness"])
Insulin= list(data["Insulin"])
BMI= list(data["BMI"])
DiabetesPedigreeFunction= list(data["DiabetesPedigreeFunction"])
Age= list(data["Age"])
Outcome = list(data["Outcome"])

predict = "Outcome"

X = list(zip(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age))
Y = list(Outcome)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.1, random_state = 42)

model = linear_model.LogisticRegression(max_iter=500)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)