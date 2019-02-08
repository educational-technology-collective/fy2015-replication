"""
Copyright (c) 2018 The Regents of the University of Michigan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Train a model (LSTM, RNN, HMM, SVM, Logistic Regression) to predict student's dropout behavior in MOOCs.
Or test a trained_model by providing predict labels and predict probabilities for given test data.
Take csv files with students' weekly features as input and return a trained LSTM model (Keras).

Each weekly csv is a list of users (rows), with columns corresponding to the features for that week.
Features come from table 1, pg 123, of the paper ("Temporal predication of dropouts in MOOCs:
Reaching the low hanging fruit through stacking generalization", Xing et al. 2016)

These models are generally a replication of the models in the paper ("Temporal
Models for Predicting Student Dropout in Massive Open Online Courses", Fei and Yeung 2015)

The output model is saved in HDF5 format and will contain the architecture, weights, training figuration
and states of the optimizer (allowing to resume training exactly where you left off), you can load it by:
    from keras.models import load_model
    model = load_model('my_model.h5')

Usage: python3 train_lstm.py \
-i /raw_data/path/to/feature_data.csv \
-l /raw_data/path/to/label_data.csv \
-f number of features per week \
-s hidden layer size \
-d definition of droupout, take in {1,2,3} \
-k an integer for number of folds in cross validation \
-o /output_file/path/to/my_output \
-m train, validation or test \
-t the name of the method you want to use (LSTM, RNN, SVM_RBF, SVM_LINEAR, LR) \
-a /trained_model/path/to/method.h5 (required only in test mode)

On Yuming's local environment
Train mode:
python train_lstm.py \
-i C:\\Users\\Umean\\Desktop\\MOOC\\vaccines_002_features.csv \
-l C:\\Users\\Umean\\Desktop\\MOOC\\vaccines_002_labels.csv \
-f 7 \
-o lstm.h5 \
-m train \
-t lstm

Test(prediction) mode:
python train_lstm.py \
-i C:\\Users\\Umean\\Desktop\\MOOC\\vaccines_002_features.csv \
-f 7 \
-a lstm.h5 \
-o lstm_predictions \
-m test \
-t lstm

Validation(CV) mode:
python train_lstm.py \
-i C:\\Users\\Umean\\Desktop\\MOOC\\vaccines_002_features.csv \
-l C:\\Users\\Umean\\Desktop\\MOOC\\vaccines_002_labels.csv \
-f 7 \
-k 5 \
-o auc_result \
-m validation \
-t lstm \
"""


import argparse
from keras import regularizers
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.models import load_model
import random
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_X_y
import pandas as pd
import numpy as np

EPOCHS = 100
BATCH_SIZE = 128
HIDDEN_SIZE = 20
NUM_FOLD = 5
RBF_SVM_MAX_N = 30000
LINEAR_SVM_MAX_N = 30000
CV_SUBSAMPLING_TYPE = 1
PARAMETER_SET_SIZE_THRESHOLD = 1500
KERAS_MODELS = ("LSTM", "LSTM_DROPOUT", "LSTM_L1REG", "RNN")
NON_KERAS_MODELS = ("LR", "SVM_RBF", "SVM_LINEAR")


def extract_XY(feature_data, label_data, n_feature, mode):
    """
    Extract well-shaped scaled feature and label tensors (X and y) from raw data in train and validation mode.
    Extract well-shaped scaled feature tensors and user_id arrays from raw data in test mode.
    Seperate features by weeks in raw data.
    :param feature_data: a pandas.Dataframe with columns for userID, features in different weeks.
    :param label_data: a pandas.Dataframe with columns for userID and label. None if in test mode.
    :param n_feature: an integer of the number of features per week.
    :param mode: a string indicates the purpose (train or test)
    :return: tuples of array (X, y), where X has shape ( , n_week, n_feature),
                                            y has shape ( ,1) and y are ids in test mode
    """
    N = feature_data.shape[0]
    if mode == "train" or mode == "validation":
        try:
            assert 'label_value' not in feature_data.columns, "Feature data shouldn't include labels in test mode"
        except AssertionError:
            print("[WARNING] Feature data shouldn't include labels in test mode")
            feature_data = feature_data.drop('label_value', 1)
        assert (feature_data.shape[1] - 1) % n_feature == 0, "Wrong input data shape"
        # check input data shape; if dimensions don't match, try filtering the feature data
        # (this exception occurs when the feature extraction pulls from non-clickstream data sources;
        # users without any clickstream entries are not counted in dropout labels)
        try:
            assert feature_data.shape[0] == label_data.shape[0], "userID doesn't match"
        except AssertionError:
            # filter feature data to only include users in label_data
            print("[WARNING] userID columns in feature and label data do not match; attempting to filter feature data")
            feature_data = feature_data[feature_data.userID.isin(label_data.userID)]
            N = feature_data.shape[0]
        n_week = int((feature_data.shape[1] - 1) / n_feature)
        X = np.array(feature_data.drop('userID', 1))
        X = scale(X)
        merged_data = pd.merge(feature_data, label_data, on='userID')
        y = np.array(merged_data["label_value"], dtype="int").reshape(N, 1)
    if mode == "test":
        try:
            assert 'label_value' not in feature_data.columns, "Feature data shouldn't include labels in test mode"
        except AssertionError:
            print("[WARNING] Feature data shouldn't include labels in test mode")
            feature_data = feature_data.drop('label_value', 1)
        assert (feature_data.shape[1] - 1) % n_feature == 0, "Wrong input data shape"
        n_week = int((feature_data.shape[1] - 1) / n_feature)
        X = np.array(feature_data.iloc[:, 1:])
        X = scale(X)
        y = feature_data.iloc[:, 0]
    X = X.reshape(N, n_week, n_feature)
    return X, y


def downsample_xy_to_n(X, y, n = 30000):
    """
    If X, y contain more than n samples, return a random subsample of size n.
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N,1)
    :param n: integer.
    :return: tuples of array (X, y), where X has shape ( , n_week, n_feature), y has shape ( ,1) and y are ids in test mode
    """
    assert X.shape[0] == y.shape[0], "feature and label vectors must be of same length"
    num_obs = X.shape[0]
    if num_obs > n:
        print("[INFO] downsampling data from size {} to size {}".format(num_obs, n))
        subsample_ix = random.sample(range(0, num_obs), n)
        X = X[subsample_ix,]
        y = y[subsample_ix,]
    return X, y


def droprate_lstm_train(X, y, hidden_size=HIDDEN_SIZE):
    """
    Construct a LSTM model to predict the type I dropout rate (See paper) from features in every week.
    Fit the model with train data.
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N,1)
    :param hidden_size: an integer of hidden layer size.
    :return: model: a fitted LSTM model as keras.models.Sequential
    """
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model


def droprate_lstm_dropout_train(X, y, hidden_size=HIDDEN_SIZE):
    """
    Construct a LSTM model with a single dropout layer after input later to predict the type I dropout rate (See paper) from features in every week.
    Fit the model with train data.
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N,1)
    :param hidden_size: an integer of hidden layer size.
    :return: model: a fitted LSTM model as keras.models.Sequential
    """
    model = Sequential()
    model.add(Droput(0.2, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(hidden_size))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model


def droprate_lstm_l1reg_train(X, y, hidden_size=HIDDEN_SIZE, l1_lambda=0.01):
    """
    Construct a LSTM model with a single dropout layer after input later to predict the type I dropout rate (See paper) from features in every week.
    Fit the model with train data.
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N,1)
    :param hidden_size: an integer of hidden layer size.
    :return: model: a fitted LSTM model as keras.models.Sequential
    """
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(X.shape[1], X.shape[2]),
                   kernel_regularizer=regularizers.l1(l1_lambda), activity_regularizer=regularizers.l1(l1_lambda)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model


def droprate_rnn_train(X, y, hidden_size=HIDDEN_SIZE):
    """
    Construct a RNN model to predict the type I dropout rate (See paper) from features in every week.
    Fit the model with train data.
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N,1)
    :param hidden_size: an integer of hidden layer size.
    :return: model: a fitted RNN model as keras.models.Sequential
    """
    model = Sequential()
    model.add(SimpleRNN(hidden_size, input_shape=(X.shape[1], X.shape[2],)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model


def droprate_lr_train(X, y, k=NUM_FOLD):
    """
    Construct a Logistic Regression model to predict the type I dropout rate (See paper) from features in every week.
    Fit the model with train data and use Cross Validation to choose best C.
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N,1)
    :param k: an interger of number of folds in cross validation for choosing C
    :return: logistic_classifier: a fitted Logistic Regression model as sklearn.LogisticRegressionCV
    """
    Cs = np.logspace(-1, 6, 8)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    y = np.ravel(y)
    print("[INFO] Select tuning parameters for LR")
    logistic_classifier = LogisticRegressionCV(Cs=Cs, cv=k, scoring="roc_auc")
    print("[INFO] Training logistic regression model")
    logistic_classifier.fit(X, y)
    print("[INFO] Best parameter: ", logistic_classifier.C_, " out of ", logistic_classifier.Cs_)
    print("[INFO] Accuracy:", logistic_classifier.score(X, y))
    return logistic_classifier


def droprate_svm_rbf_train(X, y, k=NUM_FOLD):
    """
    Construct a RBF kernel SVM classifier to predict the type I dropout rate (See paper) from features in every week.
    Fit the model with train data and use Cross Validation to choose best C and gamma.
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N,1)
    :param k: an interger of number of folds in cross validation for choosing C
    :return: svm_classifier: a fitted RBF kernel SVM classifier as sklearn.GridSearchCV
    """
    if X.shape[0] <= PARAMETER_SET_SIZE_THRESHOLD:
        C_RANGE = np.logspace(-1, 2, 4)
        GAMMA_RANGE = np.logspace(-1, 1, 3)
        print("[INFO] Large Parameters Set")
    else:
        C_RANGE = np.logspace(-1, 0, 2)
        GAMMA_RANGE = np.logspace(-2, -1, 2)
        print("[INFO] Small Parameters Set")
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = np.ravel(y)
    param_grid = dict(gamma=GAMMA_RANGE, C=C_RANGE)
    cv = StratifiedKFold(n_splits=k)
    print("[INFO] Select tuning parameters for RBF SVM")
    svm_classifier = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv, scoring="roc_auc")
    print("[INFO] Training RBF SVM model")
    svm_classifier.fit(X, y)
    print("[INFO] Best parameter: ", svm_classifier.best_params_)
    #print("Accuracy:", svm_classifier.score(X, y))
    return svm_classifier


def droprate_svm_linear_train(X, y, k=NUM_FOLD):
    """
    Construct a linear kernel SVM classifier to predict the type I dropout rate (See paper) from features in every week.
    Fit the model with train data and use Cross Validation to choose best C and gamma.
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N,1)
    :param k: an interger of number of folds in cross validation for choosing C
    :return: svm_classifier: a fitted linear kernel SVM classifier as sklearn.GridSearchCV
    """
    if X.shape[0] <= PARAMETER_SET_SIZE_THRESHOLD:
        C_RANGE = np.logspace(-1, 2, 4)
        print("[INFO] Large Parameters Set")
    else:
        C_RANGE = np.logspace(-1, 0, 2)
        print("[INFO] Small Parameters Set")
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = np.ravel(y)
    param_grid = dict(C=C_RANGE)
    cv = StratifiedKFold(n_splits=k)
    print("[INFO] Select tuning parameters for linear SVM")
    linear_svm_classifier = GridSearchCV(SVC(kernel='linear', probability=True),
                                         param_grid=param_grid, cv=cv, scoring="roc_auc")
    print("[INFO] Training linear SVM model")
    linear_svm_classifier.fit(X, y)
    print("[INFO] Best parameter: ", linear_svm_classifier.best_params_)
    # print("Accuracy:", svm_classifier.score(X, y))
    return linear_svm_classifier


def split_indices(y, num_fold):
    """
    Provide sets of train-test indices to split the raw data into several stratified folds
    :param y: labels of the raw_data, shape (N, 1)
    :param num_fold: an interger of the number of folds in Cross Validation.
    :return: a list of tuples (train_index, test_index) of length num_fold
                train_index: index of train data in each train-test set
                test_index: index of test data in each train-test set
    """
    skf = StratifiedKFold(n_splits=num_fold)
    N = y.shape[0]
    # np.set_printoptions(threshold=np.nan)
    indices = skf.split(np.zeros(N), y.flatten())
    return indices


def model_validation(train_func, model_type, X, y, num_fold):
    """
    Calculate the model's ROC_AUC score by Stratified K-Folds Cross Validation on whole input data.
    :param train_func: train function of certain type of model
    :param model_type: model type
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param y: a numpy array of labels, has shape (N, 1)
    :param num_fold: an interger of the number of folds in Cross Validation
    :return: a real value represents the AUC score
    """
    num_obs = X.shape[0]
    indices = split_indices(y, num_fold)
    cv_AUC = []
    cv_id = 1
    print("[INFO] Begin CV")
    for train_index, test_index in indices:
        print("[INFO] Fold %d" % cv_id)
        num_train_obs = len(train_index)
        if CV_SUBSAMPLING_TYPE == 2 and model_type == 'SVM' and num_train_obs > RBF_SVM_MAX_N:
            # Second type of downsampling in CV: (when num_train_obs > RBF_SVM_MAX_N)
            # In every fold process, randomly choose RBF_SVM_MAX_N samples to train SVM and predict on all the rest
            # Then repeat this process num_fold times and average the AUC
            train_index = random.sample(set(train_index), RBF_SVM_MAX_N)
            print("[INFO] downsampling data from size {} to size {}".format(num_train_obs, len(train_index)))
            test_index = list(set(range(0, num_obs)) - set(train_index))
        model = train_func(X[train_index], y[train_index])
        if model_type in ["LR", "SVM_RBF", 'SVM_LINEAR']:
            X_test = X[test_index]
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
            y_pred = model.predict_proba(X_test)[:, 1]
        if model_type in ["LSTM", "RNN"]:
            y_pred = model.predict_proba(X[test_index])
        cv_AUC.append(roc_auc_score(y[test_index], y_pred))
        cv_id += 1
    scores = np.mean(np.array(cv_AUC))
    print("[INFO] AUC: %.2f%%" % (scores * 100))
    return scores


def model_evaluate(model_type, raw_data, n_feature, num_fold, hidden_size=20):
    """
    Evaluate the LSTM or rnn model by Stratified K-Folds Cross Validation.
    :param model_type: model type
    :param raw_data: a pandas.Dataframe with columns for userID, features in different weeks and label
    :param n_feature: an integer of the number of features per week
    :param num_fold: an interger of the number of folds in Cross Validation
    :param hidden_size: an integer of hidden layer size.
    :return: a real value represents the accuracy.
    """
    X, y = extract_XY(raw_data, n_feature, "train")
    model_unfitted = Sequential()
    if model_type == "RNN":
        model_unfitted.add(SimpleRNN(hidden_size, input_shape=(X.shape[1], X.shape[2],)))
    if model_type == "LSTM":
        model_unfitted.add(LSTM(hidden_size, input_shape=(X.shape[1], X.shape[2],)))
    model_unfitted.add(Dense(1, activation='sigmoid'))
    model_unfitted.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    indices = split_indices(y, num_fold)
    cv_AUC = []
    for train_index, test_index in indices:
        model = clone_model(model_unfitted)
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.fit(X[train_index], y[train_index], epochs=EPOCHS, batch_size=BATCH_SIZE)
        y_pred = model.predict_proba(X[test_index])
        cv_AUC.append(roc_auc_score(y[test_index], y_pred))
    scores = np.mean(np.array(cv_AUC))
    print("AUC: %.2f%%" % (scores * 100))
    return scores


def lr_svm_predict(model, X, user_id):
    """
    Do predictions based on the given trained LR or SVM model and the data features.
    :param model: the trained sklearn model
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param user_id: an array of the user_ids
    :return: a pandas DataFrame of user_ids, predict probabilities and predict labels.
    """
    N = X.shape[0]
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    print("[INFO] Predicting")
    y_prob = model.predict_proba(X)[:, 1].reshape(N)
    y_pred = model.predict(X).reshape(N)
    predictions = pd.DataFrame({'userID': user_id, 'prob': y_prob, 'pred': y_pred})
    predictions = predictions[['userID', 'prob', 'pred']]
    return predictions


def lstm_rnn_predict(model, X, user_id):
    """
    Do predictions based on the given trained LSTM or RNN model and the data features.
    :param model: the trained Keras model
    :param X: a numpy array of features, has shape ( , n_week, n_feature)
    :param user_id: an array of the user_ids
    :return: a pandas DataFrame of user_ids, predict probabilities and predict labels.
    """
    N = X.shape[0]
    print("[INFO] Predicting")
    y_prob = model.predict_proba(X).reshape(N)
    y_pred = model.predict_classes(X).reshape(N)
    predictions = pd.DataFrame({'userID': user_id, 'prob': y_prob, 'pred': y_pred})
    predictions = predictions[['userID', 'prob', 'pred']]
    return predictions


def create_dummy_model(X, y):
    """
    Creates a dummy model; used when training of other models fails (usually due to CV folds or trainin sets which contain only a single outcome class).
    :param X:
    :param y:
    :return:
    """
    # see https://github.com/scikit-learn/scikit-learn/issues/10786 for description of why this is necessary; bug in sklearn
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    X_converted, y_converted = check_X_y(X, y)
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X=X_converted, y=y_converted) #note that this will trigger a DataConversionWarning, that is ok
    return dummy_model


def dummy_predict(model, X, user_id):
    """
    Do predictions with a dummy model trained using create_dummy_model().
    :param model:
    :param X:
    :param user_id:
    :return:
    """
    N = X.shape[0]
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    print("[INFO] Predicting")
    y_pred = model.predict(X).reshape(N)
    if model.classes_[0] == 1:
        y_prob = np.ones(N)
    elif model.classes_[0] == 0:
        y_prob = np.zeros(N)
    else:
        y_prob = np.full(N, np.nan)
    predictions = pd.DataFrame({'userID': user_id, 'prob': y_prob, 'pred': y_pred})
    predictions = predictions[['userID', 'prob', 'pred']]
    return predictions


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--feature_data", help="Where we read the raw feature data",
                           type=str, default=".", required=True)
    argparser.add_argument("-l", "--label_data", help="Where we read the raw label data",
                           type=str, default=".", required=False)
    argparser.add_argument("-f", "--n_feature", help="Number of features per week",
                           type=int, default=7, required=True)
    argparser.add_argument("-s", "--hidden_size", help="Size of hidden layer",
                           type=int, default=20, required=False)
    argparser.add_argument("-d", "--drop_definition", help="Choice of dropout definition",
                           type=int, default=1, required=False)
    argparser.add_argument("-k", "--num_fold", help="Number of folds in cross validation",
                           type=int, default=5, required=False)
    argparser.add_argument("-o", "--output_file", help="Where we output the model or predictions",
                           type=str, default=".", required=False)
    argparser.add_argument("-m", "--mode", help="Train or test",
                           type=str, default=".", required=True)
    argparser.add_argument("-t", "--model_type", help="The method you want to use",
                           type=str, default=".", required=True)
    argparser.add_argument("-a", "--trained_model", help="Where we read the trained model",
                           type=str, default=".", required=False)
    args = argparser.parse_args()

    input_feature = args.feature_data
    input_label = args.label_data
    n_feature = args.n_feature
    drop_definition = args.drop_definition
    hidden_size = args.hidden_size  # default 20, same as the paper
    num_fold = args.num_fold
    output_file = args.output_file
    mode = args.mode.lower()
    model_type = args.model_type.upper()
    if model_type == 'SVM':
        print("[INFO] default SVM is SVM_RBF")
        model_type = 'SVM_RBF'
    trained_model = args.trained_model

    assert drop_definition in [1, 2, 3], "definition of droupout should be 1, 2 or 3"
    assert mode in ["train", "test", "validation"], "need to choose a mode: train, validation or test"
    assert model_type in ['LSTM', 'RNN', 'LR', 'SVM_RBF', 'SVM_LINEAR'], "wrong model type"

    feature_data = pd.read_csv(input_feature, header=0, sep=',')
    if mode == "train" or mode == "validation":
        assert input_label != ".", "need label data in test mode"
        label_data = pd.read_csv(input_label, header=0, sep=',')
    else:
        label_data = None
    X, y = extract_XY(feature_data, label_data, n_feature, mode)
    if mode == "train":
        assert output_file != ".", "the output model directory is required in train mode"
        try:
            if model_type == "LSTM":
                model = droprate_lstm_train(X, y, hidden_size)
            elif model_type == "RNN":
                model = droprate_rnn_train(X, y, hidden_size)
            elif model_type == "LSTM_DROPOUT":
                model = droprate_lstm_dropout_train(X, y, hidden_size)
            elif model_type == "LSTM_L1REG":
                model = droprate_lstm_l1reg_train(X, y, hidden_size)
            elif model_type == "LR":
                model = droprate_lr_train(X, y, num_fold)
            elif model_type == "SVM_RBF":
                X, y = downsample_xy_to_n(X, y, n=RBF_SVM_MAX_N)
                model = droprate_svm_rbf_train(X, y, num_fold)
            elif model_type == "SVM_LINEAR":
                X, y = downsample_xy_to_n(X, y, n=LINEAR_SVM_MAX_N)
                model = droprate_svm_linear_train(X, y, num_fold)
        except ValueError as ve:
            print("[ERROR/WARNING]: {}",format(ve))
            print("DEFAULTING TO MAJORITY CLASS PREDICTION FOR MODEL TYPE {}".format(model_type))
            model = create_dummy_model(X, y)
        if model_type in KERAS_MODELS:
            model.save(output_file)
        elif model_type in NON_KERAS_MODELS:
            joblib.dump(model, output_file)
    if mode == "validation":
        assert output_file != ".", "the output model directory is required in validation mode"
        if model_type == "LSTM":
            auc_score = model_validation(droprate_lstm_train, model_type, X, y, num_fold)
        elif model_type == "RNN":
            auc_score = model_validation(droprate_rnn_train, model_type, X, y, num_fold)
        elif model_type == "LSTM_DROPOUT":
            auc_score = model_validation(droprate_lstm_dropout_train, model_type, X, y, num_fold)
        elif model_type == "LSTM_L1REG":
            auc_score = model_validation(droprate_lstm_l1reg_train, model_type, X, y, num_fold)
        elif model_type == "LR":
            auc_score = model_validation(droprate_lr_train, model_type, X, y, num_fold)
        elif model_type == "SVM_RBF":
            if CV_SUBSAMPLING_TYPE == 1:
                # first type of downsampling in CV: (if num_obs > RBF_SVM_MAX_N*num_fold/(num_fold-1))
                # randomly choose RBF_SVM_MAX_N*num_fold/(num_fold-1) samples
                # to ensure the number of train obs in CV is no more than RBF_SVM_MAX_N
                # and put this subset of samples into CV
                X, y = downsample_xy_to_n(X, y, n=int(RBF_SVM_MAX_N*num_fold/(num_fold-1)))
            auc_score = model_validation(droprate_svm_rbf_train, model_type, X, y, num_fold)
        elif model_type == "SVM_LINEAR":
            if CV_SUBSAMPLING_TYPE == 1:
                X, y = downsample_xy_to_n(X, y, n=int(LINEAR_SVM_MAX_N*num_fold/(num_fold-1)))
            auc_score = model_validation(droprate_svm_linear_train, model_type, X, y, num_fold)
        with open(output_file, 'w') as f:
            f.write(str(auc_score))
    if mode == "test":
        assert trained_model != ".", "a trained model is required in test mode"
        assert output_file != ".", "the output file directory is required in test mode"
        if model_type in KERAS_MODELS:
            try:
                model = load_model(trained_model)
                predictions = lstm_rnn_predict(model, X, y)
            except OSError: # occurs when dummy model was used
                model = joblib.load(trained_model)
                if isinstance(model, DummyClassifier):
                    predictions = dummy_predict(model, X, y)
            predictions.to_csv(output_file, index=False)
        elif model_type in NON_KERAS_MODELS:
            model = joblib.load(trained_model)
            if isinstance(model, DummyClassifier):
                predictions = dummy_predict(model, X, y)
            else:
                predictions = lr_svm_predict(model, X, y)
            predictions.to_csv(output_file, index=False)
    # For developer test
    # droprate_lstm_auc = model_evaluate("LSTM", feature_data, n_feature, 5)
    # droprate_rnn_auc = model_evaluate("RNN", feature_data, n_feature, 5)
    # print("LSTM AUC: %.2f \t RNN AUC: %.2f \t " % (droprate_lstm_auc, droprate_rnn_auc))
    # return 0


if __name__ == '__main__':
    main()
