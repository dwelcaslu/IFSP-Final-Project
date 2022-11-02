import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = '../../data/processed'


def load_dataset(site_number):
    if site_number < 10:
        data_filepath = f'{DATA_DIR}/site-0{site_number}.csv'
    else:
        data_filepath = f'{DATA_DIR}/site-{site_number}.csv'
    df = pd.read_csv(data_filepath)
    return df
