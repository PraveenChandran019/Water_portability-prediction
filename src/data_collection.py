import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise Exception(f"error {e} from data filepath {filepath}")

def load_params(filepath:str) -> float:
    try:
        with open(filepath,"r") as file:
            test_size = yaml.safe_load(file)['data_collection']['test_size']
        return test_size
    except Exception as e:
        raise Exception(f"error {e} from yaml file path {filepath}")

def split_data(data : pd.DataFrame, test_size:float) -> pd.DataFrame :
    try:
        train_data, test_data = train_test_split(data, test_size= test_size, random_state=42)
        return train_data, test_data
    except Exception as e:
        raise Exception(f"error {e} , It is from splitting data")

        
def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        file = df.to_csv(filepath, index = False)
        return file
    except Exception as e:
        raise Exception(f"Error saving data : {e}")

def main():
        data_path = r"C:\Users\Naveena\OneDrive\Desktop\MLOPS\water_potability.csv"
        params_path = "params.yaml" 
        raw_data_path = os.path.join("data", "raw")
        
        try:
            data = load_data(data_path)
            test_size = load_params(params_path)
            train_data, test_data = split_data(data,test_size)
            
            os.makedirs(raw_data_path, exist_ok=True)
            
            save_data(train_data,os.path.join(raw_data_path, "train.csv"))
            save_data(test_data,os.path.join(raw_data_path, "test.csv"))
            
        except Exception as e:
            raise Exception(f"error {e}")


if __name__ == "__main__":
    main()
