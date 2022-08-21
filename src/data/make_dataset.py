import os
from pathlib import Path
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image
import pydicom as dicom


def main(data_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    raw_data_dir = os.path.join(data_dir, 'raw')
    interim_data_dir = os.path.join(data_dir, 'interim')
    processed_data_dir = os.path.join(data_dir, 'processed')

    print('\nMaking interim dataset from raw data...')
    make_interim_dataset(raw_data_dir, interim_data_dir)
    print('done!')

    print('\nMaking processed dataset from interim data...')
    make_processed_dataset(interim_data_dir, processed_data_dir)
    print('done!')


def make_interim_dataset(raw_data_dir, interim_data_dir):
    """
    Transforming the raw data in the interim version:
    - loading data from dicom files and saving as jpg
    - creating csv files for cataloging each database
    - setting the target arrays
    """
    if os.path.isdir(interim_data_dir):
        shutil.rmtree(interim_data_dir)
    os.mkdir(interim_data_dir)

    # Cataloging database 01:
    base01_dir = os.path.join(raw_data_dir, 'base01_chest_xray')
    base01_interim_dir = os.path.join(interim_data_dir, 'base01_chest_xray')
    data01 = []
    for stage in os.listdir(base01_dir):
        database_stage_path = os.path.join(base01_dir, stage)
        for outcome in os.listdir(database_stage_path):
            folderpath = os.path.join(database_stage_path, outcome)
            for filename in os.listdir(folderpath):
                filepath = os.path.join(folderpath, filename)
                data01.append({"img_filepath": filepath.replace("\\", '/'), "original_stage": stage, "target": outcome})
    data01 = pd.DataFrame(data01)
    data01["target"] = data01["target"].replace('NORMAL', 0).replace('PNEUMONIA', 1)
    # Saving base 01 into interim folder:
    os.mkdir(base01_interim_dir)
    img_filepath_new = []
    for filepath in data01['img_filepath']:
        new_filepath = os.path.join(base01_interim_dir, filepath.split('/')[-1])
        image = Image.open(filepath)
        image.save(new_filepath)
        img_filepath_new.append(new_filepath)
    data01['img_filepath'] = img_filepath_new
    base01_csv_filepath = os.path.join(interim_data_dir, 'base01_chest_xray.csv')
    data01.to_csv(base01_csv_filepath, index=False)
    print(f'Saved {data01.shape[0]} images in {base01_interim_dir}')

    # Cataloging database 02:
    base02_dir = os.path.join(raw_data_dir, 'base02_rsna-pneumonia-detection-challenge')
    base02_interim_dir = os.path.join(interim_data_dir, 'base02_rsna-pneumonia-detection-challenge')
    data02 = pd.read_csv(os.path.join(base02_dir, 'stage_2_train_labels.csv'))
    train_folder = os.path.join(base02_dir, 'stage_2_train_images')
    data02['img_filepath'] = data02['patientId'].apply(lambda x: os.path.join(train_folder, x + '.dcm').replace("\\", '/'))
    data02 = data02.drop_duplicates('patientId').reset_index(drop=True)
    data02['target'] = data02['Target']
    data02 = data02.drop(columns='Target')
    # Saving base 02 into interim folder:
    os.mkdir(base02_interim_dir)
    img_filepath_new = []
    for filepath in data02['img_filepath']:
        new_filepath = os.path.join(base02_interim_dir, filepath.split('/')[-1]).replace('.dcm', '.jpeg')
        ds = dicom.dcmread(filepath)
        image = Image.fromarray(ds.pixel_array)
        image.save(new_filepath)
        img_filepath_new.append(new_filepath)
    data02['img_filepath'] = img_filepath_new
    base02_csv_filepath = os.path.join(interim_data_dir, 'base02_rsna-pneumonia-detection-challenge.csv')
    data02.to_csv(base02_csv_filepath, index=False)
    print(f'Saved {data02.shape[0]} images in {base02_interim_dir}')


def make_processed_dataset(interim_data_dir, processed_data_dir, 
                           n_samples=5000, img_size=(256,256), test_size=0.2):
    """
    Transforming the interim data in the final version:
    - splitting the databases into different hospital folders
    - resizing the images
    - splitting the databaes into train and test
    """
    if os.path.isdir(processed_data_dir):
        shutil.rmtree(processed_data_dir)
    os.mkdir(processed_data_dir)

    # database_csv_list = ['base01_chest_xray.csv', 'base02_rsna-pneumonia-detection-challenge.csv']
    database_csv_list = ['base02_rsna-pneumonia-detection-challenge.csv']

    split_number = 0
    for dtbase in database_csv_list:
        dtbase_csv_path = os.path.join(interim_data_dir, dtbase)
        df = pd.read_csv(dtbase_csv_path)
        n_splits = int(df.shape[0]/n_samples)
        n_samples_per_split = int(df.shape[0]/n_splits)

        for _ in range(n_splits):
            split_number += 1
            if split_number < 10:
                dtbase_csv_path = os.path.join(processed_data_dir, f'imgs_data0{split_number}')
            else:
                dtbase_csv_path = os.path.join(processed_data_dir, f'imgs_data{split_number}')
            os.mkdir(dtbase_csv_path)
            df_split = df.sample(n=n_samples_per_split, replace=False, random_state=42)
            df = df.drop(index=df_split.index)

            img_filepath_new = []
            for filepath in df_split['img_filepath']:
                new_filepath = os.path.join(dtbase_csv_path, filepath.split('\\')[-1])
                image = Image.open(filepath)
                image = image.resize(size=img_size)
                image.save(new_filepath)
                img_filepath_new.append(new_filepath)
            df_split['img_filepath'] = img_filepath_new
            df_split_train, df_split_test = train_test_split(df_split, test_size=test_size, random_state=42)
            df_split_train['split'] = 'train'
            df_split_test['split'] = 'test'
            df_split = pd.concat([df_split_train, df_split_test])[['img_filepath', 'split', 'target']]
            df_split.to_csv(f'{dtbase_csv_path}.csv', index=False)
            print(f'Saved {df_split.shape[0]} images in {dtbase_csv_path}')


def reset_random_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # make some random data
    reset_random_seeds()

    PROJECT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = os.path.join (PROJECT_DIR, 'data')
    main(DATA_DIR)
