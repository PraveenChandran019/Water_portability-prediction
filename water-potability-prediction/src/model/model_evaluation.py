import numpy as np
import pandas as pd

import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def load_data(filepath):
    test_data = pd.read_csv(filepath)
    return test_data
    
def  data_tranform(test_data):
    x_test = test_data.iloc[:,0:-1].values
    y_test = test_data.iloc[:,-1].values
    return x_test,y_test

def model_predict(x_test):
    model = pickle.load(open("model.pkl","rb"))
    y_pred = model.predict(x_test)
    return y_pred

def metrics(y_test,y_pred):
    acc = accuracy_score(y_test,y_pred)
    pre = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    metrics_dict = {
    "acc" : acc,
    "precision" : pre,
    "recall" : recall,
    "f1-score" : f1
    }
    return metrics_dict

def save_metrics(metrics_dict):
    with open("metrics.json","w") as file:
        metrics_file = json.dump(metrics_dict,file,indent = 4)
    return metrics_file

def main():
    data = load_data("./data/processed/test_processed.csv")
    x_test,y_test = data_tranform(data)
    y_pred = model_predict(x_test)
    metrics_dict = metrics(y_test,y_pred)
    metrics_save = save_metrics(metrics_dict)
    
if __name__ == "__main__" :
    main()
