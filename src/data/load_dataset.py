import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = '../../data/processed'


def load_dataset(db_number):
    if db_number < 10:
        data_filepath = f'{DATA_DIR}/imgs_data0{db_number}.csv'
    else:
        data_filepath = f'{DATA_DIR}/imgs_data{db_number}.csv'
    df = pd.read_csv(data_filepath)
    return df
