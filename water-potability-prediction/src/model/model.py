import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

def load_data(filepath):
    train_data = pd.read_csv(filepath)
    return train_data

def load_paramas(filepath):
    n_estimators = yaml.safe_load(open(filepath))['model']['n_estimators']
    return n_estimators

def data_transform(train_data):
    x_train = train_data.drop(columns=['Potability'],axis = 1)
    y_train = train_data['Potability']
    return x_train,y_train

def model_building(n_estimators,x_train,y_train):
    clf = RandomForestClassifier(n_estimators= n_estimators)
    clf.fit(x_train,y_train)
    return clf

def save_model(clf):
    model_file = pickle.dump(clf,open("model.pkl","wb"))
    return model_file

def main():
    data = load_data("./data/processed/train_processed.csv")
    params = load_paramas("params.yaml")
    x_train, y_train = data_transform(data)
    model = model_building(params,x_train, y_train)
    model_saved = save_model(model)
    
if __name__ == "__main__":
    main()
    
    
    


