#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, KernelPCA

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[5]:


df = pd.read_csv('train_data.csv')

X = df.drop(['class4', 'class2'], axis=1)
X = X.loc[:, X.columns[range(2, X.shape[1], 2)]]

X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)

y_class2 = df['class2']
y_class4 = df['class4']


# In[6]:


classifiers = [
    ('logistic', LogisticRegression()),
    ('kNeighbour', KNeighborsClassifier(3)),
    ('svcLinear', SVC(kernel="linear", C=0.025, probability=True)),
    ('svc', SVC(gamma=2, C=1, probability=True)),
    ('gaussian', GaussianProcessClassifier(1.0 * RBF(1.0))),
    ('decissionTree', DecisionTreeClassifier(max_depth=5)),
    ('rfc', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ('mlp', MLPClassifier(alpha=1, max_iter=1000)),
    ('ada', AdaBoostClassifier()),
    ('gaussianNB', GaussianNB()),
    ('qda', QuadraticDiscriminantAnalysis())]


# In[7]:


p = 0.7
variance = p * (1 - p)


# In[8]:


dimension_reductions_y2 = [
    ('iso', Isomap(n_components=30)),
    ('lle', LocallyLinearEmbedding(n_components=10)), 
    ('llemodified', LocallyLinearEmbedding(n_components=10, method='modified', n_neighbors=90)),
    ('svd', TruncatedSVD(n_components=10)),
    ('lda', LinearDiscriminantAnalysis(n_components=1)),
    ('pca', PCA()),
    ('kpca', KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=1)),
    ('sel', VarianceThreshold(threshold=variance)),
    ('kbest', SelectKBest(f_classif, k=10)), 
    ('kbestmutual', SelectKBest(mutual_info_classif, k=10)),
    ('select', SelectFromModel(LinearSVC(penalty="l2"))),
    ('selecttree', SelectFromModel(ExtraTreesClassifier(n_estimators=20))),
    ('rfe', RFE(estimator=DecisionTreeClassifier(), n_features_to_select=20))]

dimension_reductions_y4 = [
    ('iso', Isomap(n_components=10)),
    ('lle', LocallyLinearEmbedding(n_components=10)), 
    ('llemodified', LocallyLinearEmbedding(n_components=10, method='modified', n_neighbors=90)),
    ('svd', TruncatedSVD(n_components=5)),
    ('lda', LinearDiscriminantAnalysis(n_components=2)),
    ('pca', PCA()),
    ('kpca', KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=1)),
    ('sel', VarianceThreshold(threshold=variance)),
    ('kbest', SelectKBest(f_classif, k=10)), 
    ('kbestmutual', SelectKBest(mutual_info_classif, k=10)),
    ('select', SelectFromModel(LinearSVC(penalty="l2"))),
    ('selecttree', SelectFromModel(ExtraTreesClassifier(n_estimators=10))),
    ('rfe', RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10))]


# In[11]:


def k_fold_cross_validation(ml_pipeline, X, y, n=5, k=10, score='accuracy'):
   
    cv = RepeatedStratifiedKFold(n_splits = n, 
                                 n_repeats = k, 
                                 random_state = 1)
    n_scores = cross_val_score(ml_pipeline, X, y, 
                               scoring = score, cv = cv, 
                               n_jobs = -1)
    
    return(np.array([np.mean(n_scores), np.std(n_scores)]))


# In[12]:


columns = ['accuracy_mean', 'accuracy_std', 
           'accuracy_scaled_mean', 'accuracy_scaled_std']
statistics_y2 = pd.DataFrame(index = columns)
statistics_y4 = pd.DataFrame(index = columns)


# In[14]:


y = y_class2

for model_used in classifiers:
    model = Pipeline([model_used])

    not_scaled = k_fold_cross_validation(model, X, y)
    scaled = k_fold_cross_validation(model, X_scaled, y)

    data = np.concatenate((not_scaled, scaled))
    statistics_y2[ model_used[0] ] = data
    break
    for feature_selection in dimension_reductions_y2:
        model = Pipeline([feature_selection, model_used])

        not_scaled = k_fold_cross_validation(model, X, y)
        scaled = k_fold_cross_validation(model, X_scaled, y)

        column = model_used[0] + '_' + feature_selection[0]
        data = np.concatenate((not_scaled, scaled))
        statistics_y2[ column ] = data
        break


# In[15]:


statistics_transpose_y2 = statistics_y2.transpose(copy=True)
statistics_transpose_y2


# In[16]:


statistics_transpose_y2.describe()


# In[21]:


y = y_class4

for model_used in classifiers:
    model = Pipeline([model_used])

    not_scaled = k_fold_cross_validation(model, X, y)
    scaled = k_fold_cross_validation(model, X_scaled, y)

    data = np.concatenate((not_scaled, scaled))
    statistics_y4[ model_used[0] ] = data
    break

    for feature_selection in dimension_reductions_y4:
        model = Pipeline([feature_selection, model_used])

        not_scaled = k_fold_cross_validation(model, X, y)
        scaled = k_fold_cross_validation(model, X_scaled, y)

        column = model_used[0] + '_' + feature_selection[0]
        data = np.concatenate((not_scaled, scaled))
        statistics_y4[ column ] = data
        break


# In[24]:


statistics_transpose_y4 = statistics_y4.transpose(copy=True)
statistics_transpose_y4


# In[25]:


statistics_transpose_y4.describe()


# In[ ]:




