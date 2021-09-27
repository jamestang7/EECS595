# %pip install sklearn

# For Programming Problem 2, we will implement a naive Bayesian sentiment classifier which learns to classify movie reviews as positive or negative.
#
# Please implement the following functions below: train(), predict(), evaluate(). Feel free to use any Python libraries such as sklearn, numpy, etc. 
# DO NOT modify any function definitions or return types, as we will use these to grade your work. However, feel free to add new functions to the file to avoid redundant code (e.g., for preprocessing data).
#
# *** Don't forget to additionally submit a README_2 file as described in the assignment. ***
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score


# Description: Trains the naive Bayes classifier.
# Inputs: String for the file location of the training data (the "training" directory).
# Outputs: An object representing the trained model.
path = '/Users/jamestang1/Dropbox (University of Michigan)/UM James/um/EECS 595'

def train(training_path):
    # TODO: implement data loading, preprocessing, and model training
    # Set training and testing path
    training_path = path + '/HW1_data/training'
    testing_path = path + '/HW1_data/testing'
    for file in os.listdir(training_path):
        if file == 'pos':
            pos_train_file = [txts for txts in os.listdir(training_path+ '/'+file)]
            
        else:
            neg_train_file = [txts for txts in os.listdir(training_path+ '/'+file)]
    for file in os.listdir(testing_path):
        if file == 'pos':
            pos_test_file = [txts for txts in os.listdir(testing_path+ '/'+file)]
            
        else:
            neg_test_file = [txts for txts in os.listdir(testing_path+ '/'+file)]

    def df(path):
        """
        

        Parameters
        ----------
        path : str
            path of the training txt

        Returns
        -------
        df_train : pd.DataFrame
            Dataframe of training data

        """
        txt_lst_pos = []
        os.chdir(path+'/'+'pos')
        for txt_name in pos_train_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_pos.append(data)
        os.chdir(path+'/'+'neg')
        txt_lst_neg = []
        for txt_name in neg_train_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_neg.append(data)
        text_lst = txt_lst_pos + txt_lst_neg
        bool_lst = np.concatenate((np.ones(len(txt_lst_pos)),np.zeros(len(txt_lst_neg))))
        df_train = pd.DataFrame()
        df_train['word'] = text_lst
        df_train['target']=bool_lst
        return df_train
    def df_test(path):
        """
        

        Parameters
        ----------
        path : str
            path of the training txt

        Returns
        -------
        df_train : pd.DataFrame
            Dataframe of testing data

        """
        txt_lst_pos = []
        os.chdir(path+'/'+'pos')
        for txt_name in pos_test_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_pos.append(data)
        os.chdir(path+'/'+'neg')
        txt_lst_neg = []
        for txt_name in neg_test_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_neg.append(data)
        text_lst = txt_lst_pos + txt_lst_neg
        bool_lst = np.concatenate((np.ones(len(txt_lst_pos)),np.zeros(len(txt_lst_neg))))
        df_train = pd.DataFrame()
        df_train['word'] = text_lst
        df_train['target']=bool_lst
        return df_train
    X_train = df(training_path)
    X_test = df_test(testing_path)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['word'])

    #FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf.toarray(), X_train['target'])  
    
    return nb_tfidf

# Description: Runs prediction of the trained naive Bayes classifier on the test set, and returns these predictions.
# Inputs: An object representing the trained model (whatever is returned by the above function), and a string for the file location of the test data (the "testing" directory).
# Outputs: An object representing the predictions of the trained model on the testing data, and an object representing the ground truth labels of the testing data.
def predict(trained_model, testing_path):
    # TODO: implement data loading, preprocessing, and model prediction
    testing_path = path + '/HW1_data/testing'
    for file in os.listdir(TRAINING_PATH):
        if file == 'pos':
            pos_train_file = [txts for txts in os.listdir(TRAINING_PATH+ '/'+file)]
            
        else:
            neg_train_file = [txts for txts in os.listdir(TRAINING_PATH+ '/'+file)]
    for file in os.listdir(testing_path):
        if file == 'pos':
            pos_test_file = [txts for txts in os.listdir(testing_path+ '/'+file)]
            
        else:
            neg_test_file = [txts for txts in os.listdir(testing_path+ '/'+file)]
    
    def df(path):
        """
        

        Parameters
        ----------
        path : str
            path of the training txt

        Returns
        -------
        df_train : pd.DataFrame
            Dataframe of training data

        """
        txt_lst_pos = []
        os.chdir(path+'/'+'pos')
        for txt_name in pos_train_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_pos.append(data)
        os.chdir(path+'/'+'neg')
        txt_lst_neg = []
        for txt_name in neg_train_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_neg.append(data)
        text_lst = txt_lst_pos + txt_lst_neg
        bool_lst = np.concatenate((np.ones(len(txt_lst_pos)),np.zeros(len(txt_lst_neg))))
        df_train = pd.DataFrame()
        df_train['word'] = text_lst
        df_train['target']=bool_lst
        return df_train
    def df_test(path):
        """
        

        Parameters
        ----------
        path : str
            path of the training txt

        Returns
        -------
        df_train : pd.DataFrame
            Dataframe of testing data

        """
        txt_lst_pos = []
        os.chdir(path+'/'+'pos')
        for txt_name in pos_test_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_pos.append(data)
        os.chdir(path+'/'+'neg')
        txt_lst_neg = []
        for txt_name in neg_test_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_neg.append(data)
        text_lst = txt_lst_pos + txt_lst_neg
        bool_lst = np.concatenate((np.ones(len(txt_lst_pos)),np.zeros(len(txt_lst_neg))))
        df_train = pd.DataFrame()
        df_train['word'] = text_lst
        df_train['target']=bool_lst
        return df_train
        """
        

        Parameters
        ----------
        path : str
            path of the training txt

        Returns
        -------
        df_train : pd.DataFrame
            Dataframe of testing data

        """
        txt_lst_pos = []
        os.chdir(path+'/'+'pos')
        for txt_name in pos_test_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_pos.append(data)
        os.chdir(path+'/'+'neg')
        txt_lst_neg = []
        for txt_name in neg_test_file:
            data = ''
            with open(txt_name,encoding = 'windows-1252') as infile:
                data += infile.read()
            txt_lst_neg.append(data)
        text_lst = txt_lst_pos + txt_lst_neg
        bool_lst = np.concatenate((np.ones(len(txt_lst_pos)),np.zeros(len(txt_lst_neg))))
        df_train = pd.DataFrame()
        df_train['word'] = text_lst
        df_train['target']=bool_lst
        return df_train
    
    X_train = df(TRAINING_PATH)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['word'])
    trained_model.fit(X_train_tfidf.toarray(), X_train['target']) 
    X_test = df_test(testing_path)
    X_test_tfidf = tfidf_vectorizer.transform(X_test['word'])
    y_predict = trained_model.predict(X_test_tfidf.toarray())
    model_predictions = pd.Series(y_predict)
    ground_truth = X_test['target']
    return model_predictions, ground_truth


# Description: Evaluates the accuracy of model predictions using the ground truth labels.
# Inputs: An object representing the predictions of the trained model, and an object representing the ground truth labels for the testing data.
# Outputs: Floating-point accuracy of the trained model on the test set.
def evaluate(model_predictions, ground_truth):
    # TODO: implement evaluation metrics for the predictions
    print(classification_report(ground_truth,
                                model_predictions,
                                target_names = ['neg','pos']))
    cm = confusion_matrix(ground_truth,model_predictions)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn)/(sum([tn, fp, fn, tp]))
    return accuracy


# GRADING: We will be using lines like these to run your functions (from a separate file). You can run the file naivebayes.py in the command line (e.g., "python naivebayes.py") to verify that your code works as expected for grading.
TRAINING_PATH= path +'/HW1_data/training' # TODO: replace with your path
TESTING_PATH= path +'/HW1_data/testing' # TODO: replace with your path

trained_model = train(TRAINING_PATH)
model_predictions, ground_truth = predict(trained_model, TESTING_PATH)
accuracy = evaluate(model_predictions, ground_truth)
print('Accuracy: %s' % str(accuracy))






