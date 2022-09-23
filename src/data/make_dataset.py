import os
import random
import shutil
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


# In our project we’ll be working with medical images from different sources, emulating a
# scenario with multiple hospitals. Many datasets are currently available on the internet,
# and websites like Kaggle are a great source of data and other resources.
# For this project, we found datasets from different sources, available on Kaggle:
# Chest X-Ray Images (Pneumonia): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# RSNA Pneumonia Detection Challenge: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge
# NIH Chest X-rays: https://www.kaggle.com/datasets/nih-chest-xrays/data
# The datasets are composed of x-ray images of the patient's lungs and are labeled as normal or
# with pneumonia. The different datasets can be used as data from different sources (hospitals)
# but have the same goal: finding the patient's health condition (normal or with pneumonia) from x-ray images.


BASE_NAMES = ['nih-chest-xrays']
IMG_SIZE_PROCESSED = (224, 224)
N_SPLITS = 5
SPLITS_NAMES = 'imgs_data'
TEST_SIZE = 0.2
VALID_SIZE = 0.25
RANDOM_STATE_SEED = 42
N_ROUNDS = 5
ROUNDS_DIST = [0.4, 0.15, 0.15, 0.15, 0.15]
CLASSES_LABELS = ['No Finding', 'Sick']
TARGET_DICT = {CLASSES_LABELS[i]: i for i in range(len(CLASSES_LABELS))}


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

    # Cataloging database 03:
    base03_name = 'nih-chest-xrays'
    if base03_name in BASE_NAMES:
        base03_dir = os.path.join(raw_data_dir, base03_name)
        base03_interim_dir = os.path.join(interim_data_dir, base03_name)
        data03 = pd.read_csv(os.path.join(base03_dir, 'Data_Entry_2017.csv'))
        img_filepath = [y for x in os.walk(base03_dir) for y in glob(os.path.join(x[0], '*.png'))]
        df = pd.DataFrame({'img_filepath': [path.replace('\\', '/') for path in img_filepath]})
        df['Image Index'] = df['img_filepath'].apply(lambda x: x.split('/')[-1])
        df = df.sort_values(by='Image Index')
        data03 = data03.sort_values(by='Image Index')
        data03['img_filepath'] = df['img_filepath']
        data03['ID'] = data03['Patient ID']
        data03['target'] = data03['Finding Labels'].apply(lambda x: TARGET_DICT['No Finding'] if x == 'No Finding' else TARGET_DICT['Sick'])
        data03 = data03.drop(columns=['Finding Labels', 'Patient ID'])
        base03_csv_filepath = os.path.join(interim_data_dir, f'{base03_name}.csv')
        data03.to_csv(base03_csv_filepath, index=False)
        print(f'Saved {data03.shape[0]} images in {base03_interim_dir}')


def make_processed_dataset(interim_data_dir, processed_data_dir, n_splits=N_SPLITS,
                           test_size=TEST_SIZE, valid_size=VALID_SIZE,
                           splits_names = SPLITS_NAMES,
                           n_rounds=N_ROUNDS, rounds_dist=ROUNDS_DIST):
    """
    Transforming the interim data in the final version:
    - splitting the databases into different hospital folders
    - resizing the images
    - splitting the databaes into train and test
    """
    if os.path.isdir(processed_data_dir):
        shutil.rmtree(processed_data_dir)
    os.mkdir(processed_data_dir)

    database_csv_list = [f'{base_name}.csv' for base_name in BASE_NAMES]
    split_number = 0
    for dtbase in database_csv_list:
        dtbase_csv_path = os.path.join(interim_data_dir, dtbase)
        df = pd.read_csv(dtbase_csv_path)
        unique_ids = pd.DataFrame({"ID": df['ID'].unique()})
        n_ids_per_split = int(unique_ids.shape[0]/n_splits)
        print(dtbase, 'nº of splits:', n_splits, '\nunique IDs:', unique_ids.shape[0], '\nIDs per split:', n_ids_per_split, '\n')

        for _ in range(n_splits):
            split_number += 1
            if split_number < 10:
                dtbase_csv_path = os.path.join(processed_data_dir, f'{splits_names}0{split_number}')
            else:
                dtbase_csv_path = os.path.join(processed_data_dir, f'{splits_names}{split_number}')
            os.mkdir(dtbase_csv_path)
            ids_selected = unique_ids.sample(n=n_ids_per_split, replace=False, random_state=RANDOM_STATE_SEED)
            df_split = df[df['ID'].isin(ids_selected['ID'])]
            unique_ids = unique_ids[~unique_ids['ID'].isin(ids_selected['ID'])]

            img_filepath_new = []
            for filepath in df_split['img_filepath']:
                new_filepath = os.path.join(dtbase_csv_path, filepath.split('/')[-1])
                image = Image.open(filepath)
                image = img_transform(image)
                image.save(new_filepath)
                img_filepath_new.append(new_filepath.replace('\\', '/'))
            df_split.loc[:, 'img_filepath'] = img_filepath_new
            # Splitting data into train and test:
            samplelist = df_split["ID"].unique()
            training_samp, test_samp = train_test_split(samplelist, test_size=test_size, random_state=RANDOM_STATE_SEED)
            training_samp, valid_samp = train_test_split(training_samp, test_size=valid_size, random_state=RANDOM_STATE_SEED)
            df_split_train = df_split[df_split['ID'].isin(training_samp)]
            df_split_train.loc[:, 'split'] = 'train'
            df_split_valid = df_split[df_split['ID'].isin(valid_samp)]
            df_split_valid.loc[:, 'split'] = 'valid'
            df_split_test = df_split[df_split['ID'].isin(test_samp)]
            df_split_test.loc[:, 'split'] = 'test'
            df_split = pd.concat([df_split_train, df_split_valid, df_split_test])
            df_split = df_split.sort_values(by=['ID', 'Follow-up #']).reset_index(drop=True)
            # Setting the Federated round number:
            df_split_count = df_split.groupby('ID').count()
            round_number = []
            for f_ups in df_split_count['Follow-up #'].values:
                round_number_per_id = np.sort(np.random.choice([i+1 for i in range(n_rounds)],
                                                               f_ups, p=rounds_dist))
                round_number = np.concatenate((round_number, round_number_per_id))
            df_split['round_number'] = round_number.astype(int)

            df_split = df_split[['ID', 'img_filepath', 'split', 'round_number', 'target']]
            df_split.to_csv(f'{dtbase_csv_path}.csv', index=False)
            print(f'Saved {df_split.shape[0]} images in {dtbase_csv_path}')


def img_transform(img, img_size=IMG_SIZE_PROCESSED):
    img_transf = img.resize(size=img_size)
    return img_transf


def reset_random_seeds(seed=RANDOM_STATE_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # make some random data
    reset_random_seeds()

    PROJECT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = os.path.join (PROJECT_DIR, 'data')
    main(DATA_DIR)
