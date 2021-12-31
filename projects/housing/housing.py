import os, tarfile, joblib
import numpy as np
import pandas as pd
import urllib
from zlib import crc32

os.chdir("..")
BASE_PATH = os.path.abspath(os.curdir)
DIR_NAME = os.path.basename(__file__).split(".")[0]

class Housing:

    def __init__(self):
        np.random.seed(42) # to set random seed
        self.df = None # actual data
        self.train = None # train data
        self.test = None # test data

    # fetch data
    def fetch_data(self, data_url, tgz_filename):
        data_path = os.path.join(BASE_PATH, "datasets", DIR_NAME)
        os.makedirs(data_path, exist_ok=True)
        tgz_path = os.path.join(data_path, tgz_filename)
        urllib.request.urlretrieve(data_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=data_path)
        housing_tgz.close()

    # load data
    def load_data(self, csv_filename):
        data_path = os.path.join(BASE_PATH, "datasets", DIR_NAME)
        csv_path = os.path.join(data_path, csv_filename)
        self.df = pd.read_csv(csv_path)
        return self.df

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

    # saving model
    def save_model(self, model, model_name, extension=".pkl"):
        save_path = os.path.join(BASE_PATH, "models", DIR_NAME)
        joblib.dump(model, save_path + model_name + extension)

    # load model
    def load_model(self, model, model_name, extension=".pkl"):
        save_path = os.path.join(BASE_PATH, "models", DIR_NAME)
        joblib.load(model, save_path + model_name + extension)
