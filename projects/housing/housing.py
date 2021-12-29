import os, tarfile, joblib
import numpy as np
import pandas as pd
import urllib
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

os.chdir("..")
BASE_PATH = os.path.abspath(os.curdir)
BASE_NAME = os.path.basename(__file__)

class Housing:

    def __init__(self):
        np.random.seed(42) # to set random seed
        self.df = None # actual data
        self.columns = [] # columns names
        self.num_columns = [] # numerical columns names
        self.cat_columns = [] # categorical columns names
        self.train = None # train data
        self.test = None # test data
        self.pipeline = None # pipeline

    # fetch data
    def fetch_data(self, data_url, data_path, tgz_filename):
        os.makedirs(data_path, exist_ok=True)
        tgz_path = os.path.join(data_path, tgz_filename)
        urllib.request.urlretrieve(data_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=data_path)
        housing_tgz.close()

    # load data
    def load_data(self, data_path, csv_filename):
        csv_path = os.path.join(data_path, csv_filename)
        self.df = pd.read_csv(csv_path)
        return self.df


    # update columns
    def update_columns(self, cols=None, num_cols=None, cat_cols=None):
        if cols: self.columns = cols
        if num_cols: self.num_columns = num_cols
        if cat_cols: self.cat_columns = cat_cols

    # train test split
    def split_train_test(self, df, test_size):
        shuffled_indices = np.random.permutation(len(df))
        test_set_size = int(len(df)*test_size)
        train_indices = shuffled_indices[test_set_size:]
        test_indices = shuffled_indices[:test_set_size]
        self.train = df.iloc[train_indices] 
        self.test = df.iloc[test_indices] 
        return self.train, self.test 

    def test_set_check(self, identifier, test_ratio):
        return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32
    
    # train test split using checksum
    def split_train_test_by_id(self, data, test_ratio, id_column):
        """
        This ensures the same train test split on each updated dataset
        """
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio))
        self.train = data.loc[~in_test_set]
        self.test = data.loc[in_test_set]
        return self.train, self.test 

    # split train-test using sklearn
    def sk_split_train_test(self, df, test_size=0.2, random_state=42):
        self.train, self.test = train_test_split(df, test_size=test_size, random_state=random_state)
        return self.train, self.test

    # stratified split using sklearn
    def sk_stratified_split(self, df, cat_, n_splits=1, test_size=0.2, random_state=42):
        split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        for train_index, test_index in split.split(df, df[cat_]):
            self.train = df.loc[train_index]
            self.test = df.loc[test_index]
        
        return self.train, self.test

    # sklearn pipeline
    def sk_pipeline(self, df, pipeline_list):
        self.pipeline = Pipeline(pipeline_list)
    
    # Handling the cat and numerical data at a time
    def sk_column_transformer(self, df, pipeline_list):
        self.pipeline = ColumnTransformer(pipeline_list)

    # display score of corss-validation and etc..
    def display_score(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())
    
    # saving model
    def save_model(self, model, dir_name, model_name, extension=".pkl"):
        save_path = os.path.join(BASE_PATH, "models/", dir_name)
        joblib.dump(model, save_path + model_name + extension)

    # load model
    def load_model(self, model, dir_name, model_name, extension=".pkl"):
        save_path = os.path.join(BASE_PATH, "models/", dir_name)
        joblib.load(model, save_path + model_name + extension)