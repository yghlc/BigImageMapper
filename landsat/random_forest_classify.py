#!/usr/bin/env python
# Filename: random_forest_classify 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 28 June, 2019
"""

import sys, os
from optparse import OptionParser

HOME = os.path.expanduser('~')
# Landuse_DL
codes_dir = HOME + '/codes/PycharmProjects/Landuse_DL'
sys.path.insert(0, codes_dir)
# sys.path.insert(0, os.path.join(codes_dir, 'datasets'))
sys.path.insert(0, os.path.join(codes_dir, 'planetScripts'))        # for import function in planet_svm_classify.py
from planetScripts.planet_svm_classify import classify_pix_operation
from planetScripts.planet_svm_classify import get_output_name
from planetScripts.planet_svm_classify import read_whole_x_pixels

# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.basic as basic
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib  # save and load model
from sklearn import model_selection

model_saved_path = "sk_rf_trained.pkl"
scaler_saved_path = "scaler_saved.pkl"

def example_rf():

    # Pandas is used for data manipulation
    import pandas as pd
    # Read in data and display first 5 rows
    features = pd.read_csv('temps.csv')
    # print(features.head(5))
    print('The shape of our features is:', features.shape)
    # print(features.describe())

    # One-hot encode the data using pandas get_dummies
    features = pd.get_dummies(features)

    # Display the first 5 rows of the last 12 columns
    # print(features.iloc[:, 5:].head(5))

    # Labels are the values we want to predict
    labels = np.array(features['actual'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('actual', axis=1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)

    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)


    ########  imputing missing values, or converting temporal variables into cyclical representations  ########

    # The baseline predictions are the historical averages
    baseline_preds = test_features[:, feature_list.index('average')]
    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')



    ###### Visualizing a Single Decision Tree #########
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz

    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
    # Use dot file to create a graph
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')


    # Limit depth of tree to 3 levels
    rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
    rf_small.fit(train_features, train_labels)
    # Extract the small tree
    tree_small = rf_small.estimators_[5]
    # Save the tree as a png image
    export_graphviz(tree_small, out_file='small_tree.dot', feature_names=feature_list, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file('small_tree.dot')
    graph.write_png('small_tree.png')


    ######## Variable Importances  ############
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    ######## Visualizations ################
    import matplotlib.pyplot as plt
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()

    # Use datetime for creating date objects for plotting
    import datetime
    # Dates of training values
    months = features[:, feature_list.index('month')]
    days = features[:, feature_list.index('day')]
    years = features[:, feature_list.index('year')]
    # List and then convert to datetime object
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    # Dataframe with true values and dates
    true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
    # Dates of predictions
    months = test_features[:, feature_list.index('month')]
    days = test_features[:, feature_list.index('day')]
    years = test_features[:, feature_list.index('year')]
    # Column of dates
    test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
                  zip(years, months, days)]
    # Convert to datetime objects
    test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
    # Dataframe with predictions and dates
    predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})
    # Plot the actual values
    plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
    # Plot the predicted values
    plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
    plt.xticks(rotation='60')
    plt.legend()
    # Graph labels
    plt.xlabel('Date')
    plt.ylabel('Maximum Temperature (F)')
    plt.title('Actual and Predicted Values')
    plt.show()

    # Make the data accessible for plotting
    true_data['temp_1'] = features[:, feature_list.index('temp_1')]
    true_data['average'] = features[:, feature_list.index('average')]
    true_data['friend'] = features[:, feature_list.index('friend')]
    # Plot all the data as lines
    plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual', alpha=1.0)
    plt.plot(true_data['date'], true_data['temp_1'], 'y-', label='temp_1', alpha=1.0)
    plt.plot(true_data['date'], true_data['average'], 'k-', label='average', alpha=0.8)
    plt.plot(true_data['date'], true_data['friend'], 'r-', label='friend', alpha=0.3)
    # Formatting plot
    plt.legend()
    plt.xticks(rotation='60')
    # Lables and title
    plt.xlabel('Date')
    plt.ylabel('Maximum Temperature (F)')
    plt.title('Actual Max Temp and Variables')
    plt.show()

    pass


class classify_pix_operation_rf(classify_pix_operation):

    """perform classify operation on raster images using random forest"""

    def __init__(self):
        super(classify_pix_operation, self).__init__()


    def __del__(self):
        # super(classify_pix_operation, self).__del__()   #  Feel free not to call
        pass


    def train_rf_classifier(self, training_X, training_y):
        '''
        train random forest classifer
        :param training_X: X array, an array of size [n_records, n_features(fields)]
        :param training_y: y array, an array of size [n_records, 1 (class)]
        :return: True if successful, Flase otherwise
        '''

        if self.__classifier is None:
            self.__classifier = RandomForestClassifier(n_estimators=25)
        else:
            basic.outputlogMessage('warning, classifier already exist, this operation will replace the old one')
            self.__classifier = RandomForestClassifier(n_estimators=25)   # LinearSVC()  #SVC()

        if os.path.isfile(scaler_saved_path) and self.__scaler is None:
            self.__scaler = joblib.load(scaler_saved_path)
            result = self.__scaler.transform(training_X)
            X = result.tolist()
        elif self.__scaler is not None:
            result = self.__scaler.transform(training_X)
            X = result.tolist()
        else:
            X = training_X
            basic.outputlogMessage('warning, no pre-processing of data before training')

        y = training_y

        basic.outputlogMessage('Training data set nsample: %d, nfeature: %d' % (len(X), len(X[0])))

        # X_train = X
        # y_train = y
        # # for test by hlc
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.95, random_state=0)
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

        # print('Parameters currently in use:\n')
        # print(self.__classifier.get_params())

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the serach grid
        search_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        basic.outputlogMessage(str(search_grid))

        clf = model_selection.GridSearchCV(RandomForestClassifier() , search_grid, cv=5,
                                           scoring='f1_macro', n_jobs=-1, verbose=3)

        clf.fit(X_train, y_train)

        basic.outputlogMessage("Best parameters set found on development set:" + str(clf.best_params_))
        basic.outputlogMessage("Grid scores on development set:\n")

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            basic.outputlogMessage("%0.3f (+/-%0.03f) for %r"
                                   % (mean, std * 2, params))

        # fit_model = self.__classifier.fit(X,y)
        # basic.outputlogMessage(str(fit_model))

        # save the classification model
        joblib.dump(clf, model_saved_path)




        pass



def main(options, args):

    # example_rf()

    basic.outputlogMessage('Is_preprocessing:' + str(options.ispreprocess))
    basic.outputlogMessage('Is_training:' + str(options.istraining))

    classify_obj = classify_pix_operation_rf()

    if options.ispreprocess:
        # preprocessing
        input_tif = args[0]
        if os.path.isfile(scaler_saved_path) is False:
            # #read whole data set for pre-processing
            X, _, _ = read_whole_x_pixels(input_tif)
            classify_obj.pre_processing(X)
        else:
            basic.outputlogMessage('warning, scaled model already exist, skip pre-processing')

    elif options.istraining:
        # training
        # read training data (make sure 'subImages', 'subLabels' is under current folder)
        X, y = classify_obj.read_training_pixels_from_multi_images('subImages', 'subLabels')

        if os.path.isfile(model_saved_path) is False:
            classify_obj.train_rf_classifier(X, y)
        else:
            basic.outputlogMessage("warning, trained model already exist, skip training")

    else:
        # prediction
        input_tif = args[0]
        if options.output is not None:
            output = options.output
        else:
            output = get_output_name(input_tif)
        basic.outputlogMessage('staring prediction on image:' + str(input_tif))
        classify_obj.prediction_on_a_image(input_tif, output)




    pass

if __name__ == "__main__":
    usage = "usage: %prog [options]  input_image "
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: pixel-based image classification based on random forest classifier'

    parser.add_option("-p", "--ispreprocess",
                      action="store_true", dest="ispreprocess", default=False,
                      help="to indicate the script will perform pre-processing, if this set, istraining will be ignored")

    parser.add_option("-t", "--istraining",
                      action="store_true", dest="istraining", default=False,
                      help="to indicate the script will perform training process")

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    basic.setlogfile('RandomForest_log.txt')

    main(options, args)

