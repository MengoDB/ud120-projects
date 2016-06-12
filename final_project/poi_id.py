#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier



def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    fraction = 0.
    if poi_messages != 'NaN':
        fraction = float(poi_messages) / float(all_messages)

    return fraction

def AddFeature(dataset):
    ### the function to add fraction features to the feature list.
    submit_dict = {}
    for name in dataset:
        data_point = dataset[name]
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
        data_point["fraction_from_poi"] = fraction_from_poi

        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
        submit_dict[name] = {"fraction_from_poi": fraction_from_poi,
                             "fraction_to_poi": fraction_to_poi}
        data_point["fraction_to_poi"] = fraction_to_poi
    return dataset


def decisionTree(feature_list, dataset):
    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    test_classifier(clf, dataset, feature_list)
    print clf.feature_importances_
    return clf

def KNN(feature_list, dataset):
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    knn = KNeighborsClassifier()
    estimators = [('scale', StandardScaler()), ('knn', knn)]
    clf = Pipeline(estimators)
    test_classifier(clf, dataset, feature_list)
    return clf

def GaussianNB(feature_list, dataset):
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    test_classifier(clf, dataset, feature_list)
    #score = clf.
    return clf


def RandomForest(feature_list,dataset):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    test_classifier(clf,dataset,feature_list)
    return clf

def tuneDT(feature_list, dataset):


    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn import tree

    tree_clf = tree.DecisionTreeClassifier()
    parameters = {'criterion': ('gini', 'entropy'),
                  'splitter': ('best', 'random')}
    clf = GridSearchCV(tree_clf, parameters, scoring='recall')
    test_classifier(clf, dataset, feature_list)
    print '###best_params'
    print clf.best_params_
    test_classifier(clf.best_estimator_,dataset,feature_list)
    return clf.best_estimator_

def tuneKNN(feature_list, dataset):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.grid_search import GridSearchCV

    knn = KNeighborsClassifier()
    # feature scale
    estimators = [('scale', StandardScaler()), ('knn', knn)]
    pipeline = Pipeline(estimators)
    parameters = {'knn__n_neighbors': [1, 8],
                  'knn__algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto')}
    clf = GridSearchCV(pipeline, parameters, scoring='recall')
    test_classifier(clf, dataset, feature_list)
    print '###best_params'
    print clf.best_params_
    test_classifier(clf.best_estimator_, dataset, feature_list)
    return clf.best_estimator_










### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'restricted_stock','shared_receipt_with_poi','fraction_to_poi'
]  # You will need to use more features

features_list_all = [  # financial features
                       'poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                       'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                       'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                       # email features
                       'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                       'shared_receipt_with_poi',
                       #added features
                       'fraction_to_poi', 'fraction_from_poi'
]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

    ### Task 2: Remove outliers
    data_dict.pop('TOTAL',0)
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict
    my_dataset = AddFeature(my_dataset)

    ### Extract all features and labels from dataset for local testing
    data2=featureFormat(my_dataset, features_list_all, sort_keys=True)
    labels_all, features_all=targetFeatureSplit(data2)

    ### Extract selected features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)




### Task 4: Try a varity of classifiers
if __name__ == '__main__':
    clf0 = decisionTree(features_list_all,my_dataset)
    clf1 = decisionTree(features_list,my_dataset)
    clf1_1 = tuneDT(features_list,my_dataset)
    clf2 = GaussianNB(features_list,my_dataset)
    clf3 = KNN(features_list,my_dataset)
    clf4 = RandomForest(features_list,my_dataset)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    clf3_1 = tuneKNN(features_list,my_dataset)
    #clf = Kmeans(features_list,my_dataset)
    #clf = tuneKmeans(features_list,my_dataset)

### Task 6: Choose final classifier model and Dump classifier, dataset, and features_list so anyone can
### check my results.

    print '###final algorithm is:'
    final_clf = clf1_1
    print final_clf
    print final_clf.feature_importances_
    dump_classifier_and_data(final_clf, my_dataset, features_list)







