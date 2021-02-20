import numpy as np
import pandas as PD
import pprint as pp
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
# from sklearn import cross_validation
from sklearn import metrics
# https://towardsdatascience.com/data-preprocessing-with-scikit-learn-missing-values-8dff2c266db
from sklearn.impute import SimpleImputer

# https://www.kaggle.com/scratchpad/notebooke53322c554/edit

def openCSV(filename):
    #rootpath to the folder on the computer
    rootPath = r"C:\Users\Turing\Desktop\ML Project\Code"
    filepath = rootPath + r"\\" + filename
    # read in file as dataframe via Pandas
    df = PD.read_csv(filepath)
    # PP.pprint(df)
    # return the dataframe
    return df



def fixMissingData(df):
    columns = df.columns
    # iterate through all the columns and modify the missing values
    for column in columns:
        # convert column to float values from strings
        df[column] = PD.to_numeric(df[column], errors='coerce')
    return df

def preprocess(data):
    # split the CSV by columns
    predictor = data.iloc[:, 1:7]  # all rows, all the features and no labels
    criteron  = data.iloc[:, 9:-1]  # all rows, label only
    criteron = fixMissingData(criteron)

    # replace missing values 
    predictor.replace('',0)
    criteron.replace('',0)
    criteron = criteron.fillna(0)
    # PD.to_numeric(criteron, errors='coerce')
    # convert the dataframe to float 
    criteron = criteron.astype(float)

    # the predictor values matrix needs to be massaged to fit the ML later on
    # transposing and summing the values creates the correct format
    predictor = predictor.transpose().sum()

    # https://stackoverflow.com/questions/41925157/logisticregression-unknown-label-type-continuous-using-sklearn-in-python
    predictor = predictor.astype(int)

    return predictor, criteron

    # return X, y

def splitData(df):
    # split data into predictor and criterion dataframes
    predictor, criteron = preprocess(df)
    # Split the data into training and test sets
    predictor_train, predictor_test, criteron_train, criteron_test = train_test_split(predictor, criteron)
    return predictor_train, predictor_test, criteron_train, criteron_test 

def clfModel(predictor_train, criteron_train, predictor_test, criteron_test):
    # using CLF as the predictor method
    clf = svm.SVC()
    print(type(predictor_train))
    # fit the CLF model to our training data
    clf.fit(X=criteron_train,y=predictor_train) 
    # store the results of the scoring from the CLF model
    results = clf.score(criteron_test, predictor_test)
    print(results)

if __name__ == "__main__":
    print("Hello World")
    # name of the CSV to open
    df = openCSV("Training.csv")
    predictor_train, predictor_test, criteron_train, criteron_test = splitData(df)
    print("predictor Train: " + str(predictor_train.shape))
    print("predictor Test: " + str(predictor_test.shape))
    print("Criterion Train: " + str(criteron_train.shape))
    print("Criterion Test: " + str(criteron_test.shape))
    pp.pprint(criteron_train)    
    pp.pprint(predictor_train)
    print(criteron_test.dtypes)
    print("----------")
    print(predictor_train.dtypes)
    clfModel(predictor_train, criteron_train,predictor_test, criteron_test)

    # predictor.str.replace('','0').astype(float)
    # df.str.replace(r'\$-','0').astype(float)
    # preprocessor = SimpleImputer(missing_values='', strategy='constant', verbose='integer')
    # preprocessor.fit(predictor)
    # pp.pprint(predictor)
        
    # pp.pprint(predictors)
    # print(predictor.columns)
    # pp.pprint(criteron.columns)