#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".





features_list = ['poi', 'salary','deferral_payments','fraction_to_poi','fraction_from_poi']  # You will need to use more features

features_list_all = [  # financial features
                       'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                       'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                       'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                       # email features
                       'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                       'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi', 'text_learn_pred'
]




def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if poi_messages != 'NaN':
        fraction = float(poi_messages) / float(all_messages)

    return fraction

def AddFeature(dataset):
    submit_dict = {}
    for name in dataset:
        data_point = dataset[name]

        # print
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
        # print fraction_from_poi
        data_point["fraction_from_poi"] = fraction_from_poi

        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
        #print fraction_to_poi
        submit_dict[name] = {"fraction_from_poi": fraction_from_poi,
                             "fraction_to_poi": fraction_to_poi}
        data_point["fraction_to_poi"] = fraction_to_poi
    return dataset


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
    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
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

def Kmeans(feature_list,dataset):
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=2, tol=0.001)
    test_classifier(clf,dataset,feature_list)
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
    return clf.best_estimator_

def tuneKmeans(feature_list,dataset):
    from sklearn.cluster import KMeans
    from sklearn.grid_search import GridSearchCV
    km_clf = KMeans(n_clusters=2, tol=0.001)

    parameters = {'n_clusters': (2,10)}
    clf = GridSearchCV(km_clf, parameters, scoring='accuracy')
    test_classifier(clf, dataset, feature_list)
    print '###best_params'
    print clf.best_params_
    return clf.best_estimator_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


    from sklearn.cross_validation import train_test_split

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
if __name__ == '__main__':
    clf = decisionTree(features_list,my_dataset)
    clf = tuneDT(features_list,my_dataset)
    clf = GaussianNB(features_list,my_dataset)
    clf = KNN(features_list,my_dataset)
    clf = tuneKNN(features_list,my_dataset)
    #clf = Kmeans(features_list,my_dataset)
    #clf = tuneKmeans(features_list,my_dataset)
    clf = RandomForest(features_list,my_dataset)
    #dump_classifier_and_data(clf, my_dataset, features_list)







