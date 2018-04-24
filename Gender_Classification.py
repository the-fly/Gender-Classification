import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#reading .csv file
dat = pd.read_csv("Voice/MyVoice.csv")
predictThis = pd.read_csv("Voice/TestSample.csv")

#plotting histograms
male = dat.loc[dat['label']=='male']
female = dat.loc[dat['label']=='female']
fig, axes = plt.subplots(5, 2, figsize=(5,10))
attributes = axes.ravel()
for i in range(10):
    attributes[i].hist(male.ix[:,i], bins=20, color='#123b56', alpha=.9)
    attributes[i].hist(female.ix[:, i], bins=20, color="#d52923", alpha=.9)
    attributes[i].set_title(list(male)[i])
    attributes[i].set_yticks(())    
attributes[0].set_xlabel("Magnitude")
attributes[0].set_ylabel("Frequency")
attributes[0].legend(["Male", "Female"], loc="best")
fig.tight_layout()

#Changing label from male/female to 0/1
dat.loc[:,'label'][dat['label']=="male"] = 0
dat.loc[:,'label'][dat['label']=="female"] = 1
predictThis.loc[:,'label'][predictThis['label']=="male"] = 0
predictThis.loc[:,'label'][predictThis['label']=="female"] = 1

#Division of data into train and test data
dat_train, dat_test = train_test_split(dat, random_state=0, test_size=.2)
scaler = StandardScaler()
scaler.fit(dat_train.ix[:,0:10])

#Seperaing labels from features
X_train = scaler.transform(dat_train.ix[:,0:10])
X_test = scaler.transform(dat_test.ix[:,0:10])
X_predict = scaler.transform(predictThis.ix[:,0:10])
y_train = list(dat_train['label'].values)
y_test = list(dat_test['label'].values)
y_predict = list(predictThis['label'].values)

#Training MLP Classifier
mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
print("\n\n\n\n\n")
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))

#Predctiong on owr data
i = 1
print("\n\n")
for ex in mlp.predict(X_predict):
	print("Prediction for sample "+str(i) + " is:")
	if(ex):
		print("Female")
	else:
		print("Male")
	i+=1
    
plt.show()