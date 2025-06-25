import pandas as pd
import numpy as np
import os


def fill_missing_with_mean(df):
    try:
        for col in df.columns:
            if df[col].isnull().any():
                mean_value = df[col].mean()
                df[col].fillna(mean_value,inplace = True)
        return df
    except Exception as e:
        raise Exception(f"ERROR {e}")
    
def main():
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        train_processed =  fill_missing_with_mean(train_data)
        test_data = pd.read_csv("./data/raw/test.csv")
        test_processed =  fill_missing_with_mean(test_data)
        data_path = os.path.join("data","processed")
        os.makedirs(data_path)
        train_processed.to_csv(os.path.join(data_path,"train_processed_mean.csv"),index = False) 
        test_processed.to_csv(os.path.join(data_path,"test_processed_mean.csv"),index = False)
        
    except Exception as e:
        raise Exception(f"ERROR{e}")

if __name__ == "__main__":
    main()


