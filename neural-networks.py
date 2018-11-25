
# coding: utf-8

# ## Using Keras to Build and Train Neural Networks

# In this exercise we will use a neural network to predict diabetes using the Pima Diabetes Dataset.  We will start by training a Random Forest to get a performance baseline.  Then we will use the Keras package to quickly build and train a neural network and compare the performance.  We will see how different network structures affect the performance, training time, and level of overfitting (or underfitting).
# 
# ## UCI Pima Diabetes Dataset
# 
# * UCI ML Repositiory (http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
# 
# 
# ### Attributes: (all numeric-valued)
#    1. Number of times pregnant
#    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#    3. Diastolic blood pressure (mm Hg)
#    4. Triceps skin fold thickness (mm)
#    5. 2-Hour serum insulin (mu U/ml)
#    6. Body mass index (weight in kg/(height in m)^2)
#    7. Diabetes pedigree function
#    8. Age (years)
#    9. Class variable (0 or 1)

# The UCI Pima Diabetes Dataset which has 8 numerical predictors and a binary outcome.

# In[ ]:


#Preliminaries

#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from __future__ import absolute_import, division, print_function  # Python 2/3 compatibility

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[ ]:


## Import Keras objects for Deep Learning

from keras.models  import Sequential, K
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop


# In[ ]:


## Load in the data set (Internet Access needed)
import requests
import io

       
#url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
url = "pima-indians-diabetes.csv"
names = ["times_pregnant", "glucose_tolerance_test", "blood_pressure", "skin_thickness", "insulin", 
         "bmi", "pedigree_function", "age", "has_diabetes"]
diabetes_df = pd.read_csv(url, names=names)


# In[ ]:


# Take a peek at the data -- if there are lots of "NaN" you may have internet connectivity issues
print(diabetes_df.shape)
diabetes_df.sample(5)


# In[ ]:


X = diabetes_df.iloc[:, :-1].values
y = diabetes_df["has_diabetes"].values

print(len(X))
print(len(y))
X = X[9:]
y = y[9:]


# In[ ]:


# Split the data to Train, and Test (75%, 25%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11111)


# In[ ]:


np.mean(y), np.mean(1-y)


# Above, we see that about 35% of the patients in this dataset have diabetes, while 65% do not.  This means we can get an accuracy of 65% without any model - just declare that no one has diabetes. We will calculate the ROC-AUC score to evaluate performance of our model, and also look at the accuracy as well to see if we improved upon the 65% accuracy.
# ## Exercise: Get a baseline performance using Random Forest
# To begin, and get a baseline for classifier performance:
# 1. Train a Random Forest model with 200 trees on the training data.
# 2. Calculate the accuracy and roc_auc_score of the predictions.
# 
# __Note:__ AUROC is a figure for comparing **false positive rate** to **true positive rate**. To know more about how to calculate AUROC refer to __[here](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it)__.
# 

# In[ ]:


## Train the RF Model
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train, y_train)


# In[ ]:


# Make predictions on the test set - both "hard" predictions, and the scores (percent of trees voting yes)

# predict(X)
# Predict class for X.
# predict_proba(X)
# Predict class probabilities for X.

y_pred_class_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)

# Below two results should be equal for big data sets.
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_rf)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_rf[:,1])))


# Below code will generate AUROC plot for the random forest prediction result.<br>
# The **Accuracy** in AUROC is measured by the area under the ROC curve. An area of 1 represents a perfect test; an area of .5 represents a worthless. <br>
# The x axis is the **False positive rate (FPR)** and the y axis is the **True positive rate (TPR)**. Check this __[Ref](http://gim.unmc.edu/dxtests/roc3.htm)__ for more details.

# In[ ]:


def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])


plot_roc(y_test, y_pred_prob_rf[:, 1], 'RF')


# ## Build a Single Hidden Layer Neural Network
# 
# We will use the Sequential model to quickly build a neural network.  Our first network will be a single layer network.  We have 8 variables, so we set the input shape to 8.  Let's start by having a single hidden layer with 12 nodes.

# In[ ]:


## First let's normalize the data
## This aids the training of neural nets by providing numerical stability
## Random Forest does not need this as it finds a split only, as opposed to performing matrix multiplications

## The StandardScaler assumes your data is normally distributed within each feature and will scale them such 
## that the distribution is now centred around 0, with a standard deviation of 1.


normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)


# In[ ]:


# Define the Model 
# Input size is 8-dimensional
# 1 hidden layer, 12 hidden nodes, sigmoid activation
# Final layer has just one node with a sigmoid activation (standard for binary classification)

model_1 = Sequential([
    Dense(12, input_shape=(8,), activation="relu"),
    Dense(1, activation="sigmoid")
])


# In[ ]:


#  This is a nice tool to view the model you have created and count the parameters

model_1.summary()


# ### Comprehension question:
# Why do we have 121 parameters?  Does that make sense?
# 
# 
# Let's fit our model for 200 epochs.

# In[ ]:


# Fit(Train) the Model

# Compile the model with Optimizer, Loss Function and Metrics
# Roc-Auc is not available in Keras as an off the shelf metric yet, so we will skip it here.

model_1.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])
run_hist_1 = model_1.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=200)
# the fit function returns the run history. 
# It is very convenient, as it contains information about the model fit, iterations etc.


# In[ ]:


## Like we did for the Random Forest, we generate two kinds of predictions
#  One is a hard decision, the other is a probabilitistic score.

y_pred_class_nn_1 = model_1.predict_classes(X_test_norm)
y_pred_prob_nn_1 = model_1.predict(X_test_norm)


# In[ ]:


# Let's check out the outputs to get a feel for how keras apis work.
y_pred_class_nn_1[:10]


# In[ ]:


y_pred_prob_nn_1[:10]


# In[ ]:


# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_1)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_1)))

plot_roc(y_test, y_pred_prob_nn_1, 'NN')


# There may be some variation in exact numbers due to randomness, but you should get results in the same ballpark as the Random Forest - between 75% and 85% accuracy, between .8 and .9 for AUC.

# Let's look at the `run_hist_1` object that was created, specifically its `history` attribute.

# In[ ]:


run_hist_1.history.keys()


# Let's plot the training loss and the validation loss over the different epochs and see how it looks.

# In[ ]:


fig, ax = plt.subplots()
ax.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()


# Looks like the losses are still going down on both the training set and the validation set.  This suggests that the model might benefit from further training.  Let's train the model a little more and see what happens. Note that it will pick up from where it left off. Train for 1000 more epochs.

# In[ ]:


## Note that when we call "fit" again, it picks up where it left off
run_hist_1b = model_1.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1000)


# In[ ]:


n = len(run_hist_1.history["loss"])
m = len(run_hist_1b.history['loss'])
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(range(n), run_hist_1.history["loss"],'r', marker='.', label="Train Loss - Run 1")
ax.plot(range(n, n+m), run_hist_1b.history["loss"], 'hotpink', marker='.', label="Train Loss - Run 2")

ax.plot(range(n), run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss - Run 1")
ax.plot(range(n, n+m), run_hist_1b.history["val_loss"], 'LightSkyBlue', marker='.',  label="Validation Loss - Run 2")

ax.legend()


# Note that this graph begins where the other left off.  While the training loss is still going down, it looks like the validation loss has stabilized (or even gotten worse!).  This suggests that our network will not benefit from further training.  What is the appropriate number of epochs?

# ## Exercise
# Now it's your turn.  Do the following in the cells below:
# - Build a model with two hidden layers, each with 6 nodes
# - Use the "relu" activation function for the hidden layers, and "sigmoid" for the final layer
# - Use a learning rate of .003 and train for 1500 epochs
# - Graph the trajectory of the loss functions, accuracy on both train and test set
# - Plot the roc curve for the predictions
# 
# Experiment with different learning rates, numbers of epochs, and network structures

# ## Your Answer

# In[ ]:


#Write your code here. 
model_2 = Sequential([
    Dense(6, input_shape=(8,), activation="relu"),
    Dense(6, input_shape=(8,), activation="relu"),
    Dense(1, activation="sigmoid")
])

model_2.summary()

model_2.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])
run_hist_2 = model_2.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1500)


# In[ ]:


n = len(run_hist_2.history["loss"])

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
ax.plot(range(n), (run_hist_2.history["loss"]),'r.', label="Train Loss")
ax.plot(range(n), (run_hist_2.history["val_loss"]),'b.', label="Validation Loss")
ax.legend()
ax.set_title('Loss over iterations')

ax = fig.add_subplot(1, 2, 2)
ax.plot(range(n), (run_hist_2.history["acc"]),'r.', label="Train Acc")
ax.plot(range(n), (run_hist_2.history["val_acc"]),'b.', label="Validation Acc")
ax.legend(loc='lower right')
ax.set_title('Accuracy over iterations')


# In[ ]:


y_pred_class_nn_2 = model_2.predict_classes(X_test_norm)
y_pred_prob_nn_2 = model_2.predict(X_test_norm)
print('')
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_2)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_2)))

plot_roc(y_test, y_pred_prob_nn_2, 'NN-2')

