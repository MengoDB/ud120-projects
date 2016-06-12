Enron Email project
==============


### Overview

This project is to find out people of fraud interest from Enron Email and financial data. I used several machine learning models to identify people of interest and compared with the people who has committed fraud, improving the accuracy and capibility of the prediction models. Throught this project, I got to understand the process of investigating a topic by exploring data and building machine learning models.


### Data Exploration

## Number of data points

146

## Features

The dataset includes two categories of feature, namely financial features and email features.

Financial features:  'salary', 'deferral\_payments', 'total\_payments', 'loan\_advances', 'bonus',
                       'restricted\_stock\_deferred', 'deferred\_income', 'total\_stock\_value', 'expenses',
                       'exercised\_stock\_options', 'other', 'long\_term\_incentive', 'restricted\_stock', 'director\_fees'.

Email features: 'to\_messages', 'from\_poi\_to\_this\_person', 'from\_messages', 'from\_this\_person\_to\_poi',
                       'shared\_receipt\_with\_poi', 'fraction\_to\_poi', 'fraction\_from\_poi', 'text\_learn\_pred'.

## POI

POI is the lable in the dataset represents people who has committed fraud. There are 18 data points with POI=1.

## Outliers

As I checked the dataset, I removed two outliers.

One is 'TOTAL'. Because this data point is the total number of all the other data points. The other one is 'THE TRAVEL AGENCY IN THE PARK'. I removed it because this is a data point about an agency while other data points are for investigated people.


### Feature Selection

DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
	Accuracy: 0.81940	Precision: 0.31508	Recall: 0.30200	F1: 0.30840	F2: 0.30453
	Total predictions: 15000	True positives:  604	False positives: 1313	False negatives: 1396	True negatives: 11687

[ 0.          0.          0.03571429  0.          0.          0.          0.
  0.03021978  0.12368584  0.35716797  0.          0.04761905  0.08081004
  0.          0.04761905  0.          0.0860119   0.          0.06428571
  0.12686637  0.        ]


Using Decision Tree as a sample model to select features, we could found that 8 features in the model with all features are important (importance > 0).  They are: 'salary', 'total\_stock\_value', 'expenses', 'exercised\_stock\_options', 'restricted\_stock', 'from\_messages','shared\_receipt\_with\_poi','fraction\_to\_poi'.

Among the 8 features, fraction\_to\_poi' is derived from 'from\_messages' by fraction computation. I selected 'fraction\_to\_poi' and removed 'from\_messages' as 'fraction\_to\_poi' would be better interpreted.

As a result, I selected 7 features to build my machine learning models. They are: 'salary', 'total\_stock\_value', 'expenses', 'exercised\_stock\_options', 'restricted\_stock','shared\_receipt\_with\_poi','fraction\_to\_poi'.


## Algorithm

### Decision Tree

```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
```

DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
	Accuracy: 0.82943	Precision: 0.38010	Recall: 0.30750	F1: 0.33997	F2: 0.31971
	Total predictions: 14000	True positives:  615	False positives: 1003	False negatives: 1385	True negatives: 10997

[ 0.          0.          0.19949142  0.41495529  0.01587334  0.17674825
  0.1929317 ]

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
```

GaussianNB()
	Accuracy: 0.84743	Precision: 0.44753	Recall: 0.29000	F1: 0.35194	F2: 0.31196
	Total predictions: 14000	True positives:  580	False positives:  716	False negatives: 1420	True negatives: 11284

### K-Nearest Neighbors

```python
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

knn = KNeighborsClassifier()
estimators = [('scale', StandardScaler()), ('knn', knn)]
clf = Pipeline(estimators)
```
Pipeline(steps=[('scale', StandardScaler(copy=True, with\_mean=True, with\_std=True)), ('knn', KNeighborsClassifier(algorithm='auto', leaf\_size=30, metric='minkowski',
           metric\_params=None, n\_neighbors=5, p=2, weights='uniform'))])
	Accuracy: 0.85143	Precision: 0.38095	Recall: 0.06400	F1: 0.10959	F2: 0.07678
	Total predictions: 14000	True positives:  128	False positives:  208	False negatives: 1872	True negatives: 11792

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
```
DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
	Accuracy: 0.85136	Precision: 0.44490	Recall: 0.16350	F1: 0.23912	F2: 0.18718
	Total predictions: 14000	True positives:  327	False positives:  408	False negatives: 1673	True negatives: 11592



## Algorithm Tuning

I used grid search to automatically tune the KNN model and Decision Tree model.

### Decision Tree Tuning

```python
from sklearn.grid_search import GridSearchCV
from sklearn import tree

tree_clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ('gini', 'entropy'),
              'splitter': ('best', 'random')}
clf = GridSearchCV(tree\_clf, parameters, scoring='recall')
```



GridSearchCV(cv=None,
       estimator=DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best'),
       fit\_params={}, iid=True, loss\_func=None, n\_jobs=1,
       param\_grid={'splitter': ('best', 'random'), 'criterion': ('gini', 'entropy')},
       pre\_dispatch='2*n\_jobs', refit=True, score\_func=None,
       scoring='recall', verbose=0)
	Accuracy: 0.81914	Precision: 0.35206	Recall: 0.31650	F1: 0.33333	F2: 0.32303
	Total predictions: 14000	True positives:  633	False positives: 1165	False negatives: 1367	True negatives: 10835

best\_params
{'splitter': 'best', 'criterion': 'entropy'}
DecisionTreeClassifier(compute\_importances=None, criterion='entropy',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
	Accuracy: 0.82936	Precision: 0.38930	Recall: 0.34200	F1: 0.36412	F2: 0.35052
	Total predictions: 14000	True positives:  684	False positives: 1073	False negatives: 1316	True negatives: 10927



### KNN Tuning

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

knn = KNeighborsClassifier()

estimators = [('scale', StandardScaler()), ('knn', knn)]
pipeline = Pipeline(estimators)
parameters = {'knn__n_neighbors': [1, 8],
              'knn__algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto')}
clf = GridSearchCV(pipeline, parameters, scoring='recall')

```


GridSearchCV(cv=None,
       estimator=Pipeline(steps=[('scale', StandardScaler(copy=True, with\_mean=True, with\_std=True)), ('knn', KNeighborsClassifier(algorithm='auto', leaf\_size=30, metric='minkowski',
           metric\_params=None, n\_neighbors=5, p=2, weights='uniform'))]),
       fit\_params={}, iid=True, loss\_func=None, n\_jobs=1,
       param\_grid={'knn\_\_algorithm': ('ball\_tree', 'kd\_tree', 'brute', 'auto'), 'knn\_\_n\_neighbors': [1, 8]},
       pre\_dispatch='2*n\_jobs', refit=True, score\_func=None,
       scoring='recall', verbose=0)
	Accuracy: 0.82971	Precision: 0.36885	Recall: 0.27000	F1: 0.31178	F2: 0.28529
	Total predictions: 14000	True positives:  540	False positives:  924	False negatives: 1460	True negatives: 11076

best\_params
{'knn\_\_algorithm': 'ball\_tree', 'knn\_\_n\_neighbors': 1}
RandomForestClassifier(bootstrap=True, compute\_importances=None,
            criterion='gini', max\_depth=None, max\_features='auto',
            max\_leaf\_nodes=None, min\_density=None, min\_samples\_leaf=1,
            min\_samples\_split=2, n\_estimators=10, n\_jobs=1,
            oob\_score=False, random\_state=None, verbose=0)
	Accuracy: 0.82971	Precision: 0.36885	Recall: 0.27000	F1: 0.31178	F2: 0.28529
	Total predictions: 14000	True positives:  540	False positives:  924	False negatives: 1460	True negatives: 11076


## Result


Algorithm | Precision | recall
--- | --- | ---
Decision Tree (Tuned) | 0.39 | 0.34
KNN (Tuned) | 0.42 | 0.16
Gaussian Naive Bayes | 0.45 | 0.29
Random Forest | 0.44 | 0.16

From this result, I select Decision Tree as the final algorithm considering performance both on precision and recall.

The specific algorithm shows below.

DecisionTreeClassifier(compute\_importances=None, criterion='entropy',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
[ 0.          0.          0.46685463  0.          0.03981759  0.21697698
  0.2763508 ]



## Validation

For the chosen algorithm, we need to validate it to see how well the algorithm generalizes beyond the training dataset. A classic mistake we might make is to use same dataset for training and testing.

The whole dataset we have includes only 146 data points, which is very small. So I chose stratified shuffle split cross validation to validate the selected algorithm.

## Evaluation

### Precision

0.39

Precision, referred as positive predictive value, here indicated that 39% of people who are predicted as poi are truly people of interests.

### Recall

0.34

Recall, referred as true positive rate or sensitivity, here indicated that among people of interests, 34% of them are correctly predicted via our final algorithm.

















