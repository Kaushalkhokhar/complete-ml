import os
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage import io
import tensorflow as tf
from tensorflow import keras

print("Number of available GPU's: ", len(tf.config.experimental.list_physical_devices("GPU")))

# if "CNN" in os.path.abspath(os.curdir): os.chdir("..")
# BASE_DIR = os.path.abspath(os.curdir)
BASE_DIR = "D:\\TheCompleteML\\projects"
data_dir = os.path.join(BASE_DIR, "datasets", "classification", "flowers")
data_dirs = [os.path.join(data_dir, dir_) for dir_ in os.listdir(data_dir) if "processed" not in dir_]

data_augmentation = tf.keras.Sequential([keras.layers.RandomFlip("horizontal_and_vertical"), 
                                            keras.layers.RandomRotation(0.2),
                                            keras.layers.RandomContrast(0.5),
                                            keras.layers.RandomZoom((-0.3, 0.3), (-0.3, 0.3))
                                            ])

dims = (150, 150)
channels = 3
n_features = dims[0] * dims[1] * channels

def preprocess(line):
    defs = [tf.constant([], dtype = tf.float32)] * (n_features + 1)
    xy = tf.io.decode_csv(line, record_defaults=defs)
    X = tf.stack(xy[:-1])
    y = tf.stack(xy[-1:])
    
    # prcessing steps
    X = tf.divide(X, 255)
    X = tf.reshape(X, [dims[0], dims[1], channels])
    X = data_augmentation(X)
    X = tf.image.rot90(X)
    X = tf.image.random_brightness(X, 0.2)
    
    return X, y


def preprocess_test(X):
    # prcessing steps
    
    X = tf.image.resize_with_crop_or_pad(X, 150 ,150)
    X = data_augmentation(X)
    X = tf.image.rot90(X)
    X = tf.image.random_brightness(X, 0.2)
    X = tf.divide(X, 255)
    
    return X

def read_csv_pipeline(paths, n_readers, n_repeat, shuffle_buffer_size, n_read_threds, n_parse_threads, batch_size):
    filepaths = tf.data.Dataset.list_files(paths, seed=42)
    dataset = filepaths.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1), cycle_length=n_readers)
    dataset = dataset.shuffle(shuffle_buffer_size).repeat(n_repeat)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    return dataset.batch(batch_size).prefetch(1)

class ResidualLayer(keras.layers.Layer):
    
    def __init__(self, fm, strides=1, ksize=3, padding="same", activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.fm = fm
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.activation = keras.activations.get(activation)
        self.normalization = keras.layers.BatchNormalization()
        self.mainc_layers = [keras.layers.Conv2D(self.fm, 
                                                kernel_size=self.ksize, 
                                                strides=self.strides, 
                                                padding=self.padding, 
                                                use_bias=False),
                            self.normalization, 
                            self.activation,
                            keras.layers.Conv2D(self.fm, 
                                                kernel_size=self.ksize, 
                                                strides=1, 
                                                padding=self.padding, 
                                                use_bias=False),
                            self.normalization]
        self.skipc_layers = []
        if strides > 1:
            self.skipc_layers = [keras.layers.Conv2D(self.fm, 
                                                    kernel_size=1, 
                                                    strides=self.strides, 
                                                    padding=self.padding,
                                                    use_bias=False),
                                self.normalization]
    def get_config(self):
        config = super().get_config()
        config.update({"fm": self.fm,
                        "ksize": self.ksize,
                        "strides": self.strides,
                        "padding": self.padding,
                        "activation": self.activation
                        })
        return config
    
    def call(self, inputs):
        z = inputs
        for layer in self.mainc_layers:
            z = layer(z)
        skip_z = inputs
        for layer in self.skipc_layers:
            skip_z = layer(skip_z)
        return self.activation(z+skip_z)

def mini_resnet_9cl():
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64,
                                kernel_size=5, 
                                strides=2, 
                                padding="same", 
                                use_bias=False, 
                                input_shape=[dims[0], dims[1], channels]))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
    pfm = 64
    for fm in [64, 128, 256, 512]:
        strides = 1 if pfm == fm else 2
        model.add(ResidualLayer(fm=fm, strides=strides))
        pfm = fm
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(5, activation="softmax"))
    
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, 
                optimizer="nadam", 
                metrics=keras.metrics.sparse_categorical_accuracy)
    
    model_target = os.path.join(BASE_DIR, "models", "mini_resnet_9cl.h5")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_target, save_best_only=True)
    early_stop_cb = keras.callbacks.EarlyStopping(patience=10)
    callbacks = [checkpoint_cb, early_stop_cb]
    
    return model, callbacks

class loading_and_splitting:
    
    def __init__(self, data_dirs, dims, channels=3, target_dir=data_dir):
        self.total_images = 0
        self.minh = np.inf
        self.minw = np.inf
        self.dims = dims
        self.channels = channels
        self.target_dir = target_dir
        self.data_dirs = data_dirs
        self.class_map = {k:v.split("\\")[-1] for k, v in enumerate(data_dirs)}
        
        self.header_list = [f"x{i}" for i in range(self.dims[0]*self.dims[1]*self.channels)] + ["label"]
        self.sample_list = [random.sample(range(len(os.listdir(path))), 
                                          len(os.listdir(path))) for path in data_dirs]
        for item in self.sample_list:
            self.total_images += len(item)
        self.generate_samples()
        
    
    def generate_csvs(self):
        header_list = [f"x{i}" for i in range(self.dims[0]*self.dims[1]*self.channels)] + ["label"]
        for set_ in ["train", "valid", "test"]:
            with open(os.path.join(self.target_dir, f"{set_}.csv"), "w") as f:
                df = pd.DataFrame(list(), columns=header_list)
                df.to_csv(f, index=False)
                
    def generate_samples(self):
        self.sample_seq = random.sample(range(self.total_images), self.total_images)
        self.train_seq = self.sample_seq[:int(len(self.sample_seq)*0.8)]
        self.valid_seq = self.sample_seq[int(len(self.sample_seq)*0.8):int(len(self.sample_seq)*0.9)]
        self.test_seq = self.sample_seq[int(len(self.sample_seq)*0.9):]
    
    def crop_image(self, image):
        h, w, d = image.shape
        if h >= self.minh and w >= self.minw:
            image = image[int(h/2)-64:int(h/2)+64, 
                          int(w/2)-64:int(w/2)+64, 
                          :]
            return image
    
    def crop_or_pad(self, image):
        image = tf.image.resize_with_crop_or_pad(image, self.dims[0], self.dims[0])
        return image.numpy()
    
    def shuffle_and_save(self):
        empty = []
        train = np.zeros((1, self.dims[0]*self.dims[1]*self.channels + 1))
        valid = np.zeros((1, self.dims[0]*self.dims[1]*self.channels + 1))
        test = np.zeros((1, self.dims[0]*self.dims[1]*self.channels + 1))
        count = 0
        while len(empty) != len(self.data_dirs):
            sel_dir = np.random.randint(0, len(self.data_dirs))
            if sel_dir in empty: continue
            dir_ = self.data_dirs[sel_dir]
            if not self.sample_list[sel_dir]:
                empty.append(sel_dir)
            else:
                count += 1
                print(f"Processing: {count}")
                sel_image = self.sample_list[sel_dir].pop()
                image = io.imread(os.path.join(dir_, os.listdir(dir_)[sel_image]))
                
                h, w, d = image.shape
                if h < self.minh: self.minh = h
                if w < self.minw: self.minw = w
                if self.minh < self.dims[0]: self.minh = self.dims[0]
                if self.minw < self.dims[1]: self.minw = self.dims[1]
                
                # image = self.crop_image(image)
                image = self.crop_or_pad(image)
                
                if not isinstance(image, np.ndarray): continue
                if sel_image in self.train_seq: 
                    train = np.append(train, np.append(image.flatten(), sel_dir).reshape(1,-1), axis=0)
                elif sel_image in self.valid_seq: 
                    valid = np.append(valid, np.append(image.flatten(), sel_dir).reshape(1,-1), axis=0)
                elif sel_image in self.test_seq: 
                    test = np.append(test, np.append(image.flatten(), sel_dir).reshape(1,-1), axis=0)
        
        train = train[1:, :]
        valid = valid[1:, :]
        test = test[1:, :]
        
        for prefix, arr in zip(["train", "valid", "test"], [train, valid, test]):
            self.split_and_save(arr, os.path.join(self.target_dir, "processed", prefix), prefix)
        
    def split_and_save(self, arr, target_dir, prefix, split_count=10):
        os.makedirs(target_dir, exist_ok=True)
        for i in range(split_count):
            df = pd.DataFrame(arr[i*int(arr.shape[0]//split_count):(i+1)*int(arr.shape[0]//split_count), :], 
                             columns=self.header_list)
            df.to_csv(os.path.join(target_dir, "{}_{}.csv".format(prefix, i+1)), index=False)

def run_model():
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    ls = loading_and_splitting(data_dirs=data_dirs, 
                                dims=dims, 
                                channels=channels, 
                                target_dir=data_dir)
    class_map = ls.class_map
    # ls.shuffle_and_save()

    set_dir = os.path.join(data_dir, "processed")
    train_paths = [f"{os.path.join(set_dir, 'train')}\\{item}" for item in os.listdir(os.path.join(set_dir, "train"))]
    valid_paths = [f"{os.path.join(set_dir, 'valid')}\\{item}" for item in os.listdir(os.path.join(set_dir, "valid"))]
    test_paths = [f"{os.path.join(set_dir, 'test')}\\{item}" for item in os.listdir(os.path.join(set_dir, "test"))]

    n_readers = 5
    n_repeat = 8
    shuffle_buffer_size = 400
    n_read_threads = None
    n_parse_threads = 5
    batch_size = 32

    train_set = read_csv_pipeline(train_paths, n_readers, 
                                n_repeat, shuffle_buffer_size, 
                                n_read_threads, n_parse_threads, 
                                batch_size)

    valid_set = read_csv_pipeline(valid_paths, n_readers, 
                                n_repeat, shuffle_buffer_size, 
                                n_read_threads, n_parse_threads, 
                                batch_size)

    test_set = read_csv_pipeline(test_paths, n_readers, 
                                n_repeat, shuffle_buffer_size, 
                                n_read_threads, n_parse_threads, 
                                batch_size)
    
    model, callbacks = mini_resnet_9cl()

    history = model.fit(train_set, epochs=25, validation_data=valid_set, callbacks=callbacks)

if __name__ == "__main__":
    run_model()