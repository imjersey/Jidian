from sklearn import svm
from sklearn.model_selection import GridSearchCV 
import numpy as np
from numpy import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
sample_len = 200
sample_height = 6

#parameters = {'kernel':('linear', 'rbf'), 'C': [8 ], 'gamma': [0.5]}

parameters = {'kernel':('linear', 'rbf'), 'C': [2**q for q in range(-4, 4) ], 'gamma': [2**q for q in range(-4, 3) ]}

svr = svm.SVC()
model = GridSearchCV(svr, parameters)

base_path = MLexperiments.config.parameters.DATA_PATH
original_path = os.path.join(base_path, "original")
plots_path = os.path.join(base_path, "plots")
csv_path = os.path.join(base_path, "csv")
interfer_path = os.path.join(base_path, "interfer")
label_path = os.path.join(base_path, "label")
CNNinput_path = os.path.join(base_path, "CNNinput")


save_name = os.path.join(CNNinput_path, "CNNinput.npy")

np_input = np.load(save_name)

np_input = np_input.reshape((int(np_input.size / (sample_len * sample_height)), sample_height * sample_len))
np_input.shape

save_name = os.path.join(label_path, "label.npy")
labelData = np.load(save_name)

print("read succesful")
X, X2, y, y2 = train_test_split(np_input, labelData, test_size=0.33, random_state=42)

print("fitting")
model.fit(X, y)
#Predict Output
print("predicting")
predicted= model.predict(X2)

count = 0
for i in range(0, y2.shape[0]):
	if y2[i] == predicted[i]:
		count +=1

count / y2.shape[0]

cnf_matrix = confusion_matrix(y2, predicted)
print(cnf_matrix)

print(model.best_score_)
print(model.best_estimator_)

