
# coding: utf-8

# In[303]:

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import itertools
from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV


# In[304]:

path="/Users/guneevkaur/Desktop/GSE27562_series_matrix.txt"
output = pd.read_csv(path, sep="\t",skiprows=31,header=None,
                dtype = str,nrows=1).transpose()


# In[305]:

output.head()


# In[306]:

output[0]


# In[307]:

path="/Users/guneevkaur/Desktop/GSE27562_series_matrix.txt"
f= (pd.read_csv(path, sep="\t",skiprows=69,
                dtype = str)).transpose()


# In[308]:

f.head()


# In[309]:

y =output[0:][1:].values


# In[310]:

y


# In[311]:

f


# In[312]:

x=np.array(f)


# In[313]:

x= x[1:]


# In[314]:

x.shape


# In[315]:

x=np.delete(x, 54675, 1)


# In[316]:

x= x.astype(np.float)


# In[317]:

np.argwhere(np.isnan(x))


# In[318]:

y= y.astype(np.str)


# In[319]:

a ="PBMC_normal_training_" 
string=[a+str(i) for i in range(1,11)]
normal = map(str,string)

for i in range(0,10):
    y= np.core.defchararray.replace(y,normal[i],'Normal',count=None)

    
# np.char.strip(c, 'A')


# In[320]:

a ="PBMC_malignant_training_" 
string=[a+str(i) for i in range(1,11)]
malignant = map(str,string)

for i in range(0,10):
    y= np.core.defchararray.replace(y,malignant[i],'Malignant',count=None)


# In[321]:

a ="PBMC_benign_training_" 
string=[a+str(i) for i in range(1,11)]
benign = map(str,string)

for i in range(0,10):
    y=np.core.defchararray.replace(y,benign[i],'Benign',count=None)


# In[322]:

a ="PBMC_ectopic_validation_" 
string=[a+str(i) for i in range(1,23)]
benign = map(str,string)

for i in range(0,22):
    y= np.core.defchararray.replace(y,benign[i],'Benign',count=None)


# In[323]:

a ="PBMC_normal_validation_" 
string=[a+str(i) for i in range(1,22)]
normal_validation = map(str,string)

for i in range(0,21):
    y=np.core.defchararray.replace(y,normal_validation[i],'Normal',count=None)


# In[324]:

a ="PBMC_malignant_validation_" 
string=[a+str(i) for i in range(1,42)]
malignant_validation=map(str,string)

for i in range(0,41):
    y=np.core.defchararray.replace(y,malignant_validation[i],'Malignant',count=None)


# In[325]:

a ="PBMC_benign_validation_" 
string=[a+str(i) for i in range(1,27)]
benign_validation=map(str,string)

for i in range(0,26):
    y=np.core.defchararray.replace(y,benign_validation[i],'Benign',count=None)


# In[326]:

a ="PBMC_post-surgery_validation_" 
string=[a+str(i) for i in range(1,15)]
post_surgery=map(str,string)

for i in range(0,14):
    y=np.core.defchararray.replace(y,post_surgery[i],'Post_Surgery',count=None)


# In[327]:

#Removing extra characters

for i in range(0,10) :
    y=np.char.strip(y, str(i))


# In[291]:

y.shape


# In[399]:

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2,random_state=70)


# In[400]:

x_train.shape


# In[401]:

x_test.shape


# In[402]:

x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.2,random_state=70)


# In[403]:

x_train.shape


# In[404]:

x_val.shape


# In[405]:

x_test.shape


# In[406]:

y_train = y_train.ravel()
y_test = y_test.ravel()

neighbor = KNeighborsClassifier(n_neighbors = 5,weights='distance',metric='euclidean',algorithm='auto',p=2)


# In[407]:

neighbor.fit(x_train, y_train) 
y_pred = neighbor.predict(x_test)
print "Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",


# In[408]:

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

# generate with  labels
labels = []
cm = confusion_matrix(y_test,y_pred, labels)

# confusion matrix
print_cm(cm, labels)


# In[409]:

# this will print confusion matrix plot without normalization

def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Normal','Malignant','Benign','Post_Surgery'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Normal','Malignant','Benign','Post_Surgery'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[410]:

# learning_curve module
#It will visualize how well the model is performing based on number of samples we're training on
#Generate plot of the test and traning learning curve

def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure() 
    plt.title(title) 
    if ylim is not None: 
        plt.ylim(*ylim) 
    plt.xlabel("Training examples") 
    plt.ylabel("Score") 
    train_sizes, train_scores, test_scores = learning_curve( estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1) 
    train_scores_std = np.std(train_scores, axis=1) 
    test_scores_mean = np.mean(test_scores, axis=1) 
    test_scores_std = np.std(test_scores, axis=1) 
    plt.grid() 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r") 
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g") 
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score") 
    plt.legend(loc="best") 
    return plt 


# In[411]:

#Cross Validation Set

# split data 80:20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[412]:

#Choose Estimator as KNN
estimator = KNeighborsClassifier(n_neighbors=2,weights='distance',algorithm='auto',metric='euclidean') 


# In[413]:

# shufflesplit as cross-validation generator 
cv = ShuffleSplit(x_train.shape[0], n_iter=10, test_size=0.2) 


# In[414]:

#applying the cv on training data using GridSearchcv which uses F1 score for tuning data 
classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(n_neighbors=[2,3,4,5]), scoring='accuracy')


# In[415]:

#x_train=x_train.ravel()
y_train=y_train.ravel()


#fit the training data 
classifier.fit(x_train, y_train)



# In[416]:

#learning curve
title = 'Learning Curves (kNN, $\n_neighbors=%.6f$)' %classifier.best_estimator_.n_neighbors 
estimator = KNeighborsClassifier(n_neighbors=classifier.best_estimator_.n_neighbors) 
plot_learning_curve(estimator, title, x_train, y_train, cv=cv) 
plt.show() 


# In[417]:

y_pred = classifier.predict(x_test) 


# In[418]:

print "Final Classification Report" 
print "Generalization Accuracy:", accuracy_score(y_test,y_pred)*100


# In[ ]:



