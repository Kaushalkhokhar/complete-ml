import os
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage import io
import tensorflow as tf
from tensorflow import keras
import ktrain

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

K = keras.backend

class OneCycleSchedulerNadam(keras.callbacks.Callback):
    def __init__(self, iterations, 
                 max_lrate, 
                 start_lrate=None,
                 last_iterations=None, 
                 last_lrate=None,
                 max_b1rate=0.95,
                 min_b1rate=0.85,
                 max_b2rate=0.9995,
                 min_b2rate=0.9985):
        
        self.iterations = iterations
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        
        self.max_lrate = max_lrate
        self.start_lrate = start_lrate or max_lrate / 10
        self.last_lrate = last_lrate or self.start_lrate / 1000
        
        self.max_b1rate = max_b1rate
        self.min_b1rate = min_b1rate
        self.last_b1rate = max_b1rate
        
        self.max_b2rate = max_b2rate
        self.min_b2rate = min_b2rate
        self.last_b2rate = max_b2rate

        self.iteration = 0
        
        self.rate = []
        self.b1 = []
        self.b2 = []

        self.loss = []
        self.val_loss = []
        self.accuracy = []
        self.val_accuracy = []
    
    def _interpolate(self, iter1, iter2, lrate1, lrate2):
        return ((lrate2 - lrate1) * (self.iteration - iter1)
                / (iter2 - iter1) + lrate1)
    
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_lrate, self.max_lrate)
            b1 = self._interpolate(0, self.half_iteration, self.max_b1rate, self.min_b1rate)
            b2 = self._interpolate(0, self.half_iteration, self.max_b2rate, self.min_b2rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration, self.max_lrate, self.start_lrate)
            b1 = self._interpolate(self.half_iteration, 2 * self.half_iteration, self.min_b1rate, self.max_b1rate)
            b2 = self._interpolate(self.half_iteration, 2 * self.half_iteration, self.min_b2rate, self.max_b2rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations, self.start_lrate, self.last_lrate)
            b1 = self.last_b1rate
            b2 = self.last_b2rate
            
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, rate)
        K.set_value(self.model.optimizer.beta_1, b1)
        K.set_value(self.model.optimizer.beta_2, b2)
        
    def on_batch_end(self, batch, logs):
        self.rate.append(K.get_value(self.model.optimizer.learning_rate))
        self.b1.append(K.get_value(self.model.optimizer.beta_1))
        self.b2.append(K.get_value(self.model.optimizer.beta_2))

        self.loss.append(logs["loss"])
        self.accuracy.append(logs["sparse_categorical_accuracy"])

class OneCycleSchedulerSGD(keras.callbacks.Callback):
    def __init__(self, iterations, 
                 max_lrate, 
                 start_lrate=None,
                 last_iterations=None, 
                 last_lrate=None,
                 max_momentum=0.95,
                 min_momentum=0.85):
        
        self.iterations = iterations
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        
        self.max_lrate = max_lrate
        self.start_lrate = start_lrate or max_lrate / 10
        self.last_lrate = last_lrate or self.start_lrate / 1000
        
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        self.last_momentum = max_momentum
        
        self.iteration = 0
        
        self.rate = []
        self.momentum = []

        self.loss = []
        self.val_loss = []
        self.accuracy = []
        self.val_accuracy = []
    
    def _interpolate(self, iter1, iter2, lrate1, lrate2):
        return ((lrate2 - lrate1) * (self.iteration - iter1)
                / (iter2 - iter1) + lrate1)
    
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_lrate, self.max_lrate)
            momentum = self._interpolate(0, self.half_iteration, self.max_momentum, self.min_momentum)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration, self.max_lrate, self.start_lrate)
            momentum = self._interpolate(self.half_iteration, 2 * self.half_iteration, self.min_momentum, self.max_momentum)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations, self.start_lrate, self.last_lrate)
            momentum = self.last_momentum
            
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, rate)
        K.set_value(self.model.optimizer.beta_1, momentum)
        
    def on_batch_end(self, batch, logs):
        self.rate.append(K.get_value(self.model.optimizer.learning_rate))
        self.momentum.append(K.get_value(self.model.optimizer.beta_1))

        self.loss.append(logs["loss"])
        self.accuracy.append(logs["sparse_categorical_accuracy"])


class InceptionLayer(keras.layers.Layer):
    
    def __init__(self, fms, **kwargs):
        super().__init__(**kwargs)
        self.bn1_cnn = keras.layers.Conv2D(filters=fms[0],
                                           kernel_size=1, 
                                           strides=1, 
                                           padding="same", 
                                           use_bias=False,
                                           activation="relu")
        self.bn2_cnn = keras.layers.Conv2D(filters=fms[3],
                                           kernel_size=1, 
                                           strides=1, 
                                           padding="same", 
                                           use_bias=False,
                                           activation="relu")
        self.bn3_cnn = keras.layers.Conv2D(filters=fms[4],
                                           kernel_size=1, 
                                           strides=1, 
                                           padding="same", 
                                           use_bias=False,
                                           activation="relu")
        self.bn4_cnn = keras.layers.Conv2D(filters=fms[5],
                                           kernel_size=1, 
                                           strides=1, 
                                           padding="same", 
                                           use_bias=False,
                                           activation="relu")
        self.cnn1 = keras.layers.Conv2D(filters=fms[1],
                                        kernel_size=3, 
                                        strides=1, 
                                        padding="same", 
                                        use_bias=False,
                                        activation="relu")
        self.cnn2 = keras.layers.Conv2D(filters=fms[0],
                                        kernel_size=5, 
                                        strides=1, 
                                        padding="same", 
                                        use_bias=False,
                                        activation="relu")
        self.mp = keras.layers.MaxPool2D(pool_size=3, 
                                         strides=1, 
                                         padding="same")
    
    def get_config(self):
        config = super().get_config()
        config.update({"fms": self.fms})
        return config

    def call(self, inputs):
        z1 = self.bn1_cnn(inputs)
        z2 = self.cnn1(self.bn3_cnn(inputs))
        z3 = self.cnn2(self.bn4_cnn(inputs))
        z4 = self.bn2_cnn(self.mp(inputs))
        
        return tf.concat([z1, z2, z3, z4], axis=3)

def googlenet_mini():
    """
    name ecodes the model architecture
    architecture:
        first            CNN ---> MP ---> LRN ---> Repeated twice
        second           IL ---> IL ---> MP ---> IL ---> IL
        third            GAP
        fourth            Dense ---> including dropout
    """

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, 
                                  kernel_size=5, 
                                  strides=2,  
                                  use_bias=False,
                                  padding="same", 
                                  input_shape=[dims[0], dims[0], channels], 
                                  ))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, 
                                     strides=2,
                                     padding="same"))
    
    model.add(keras.layers.Lambda(lambda inputs: tf.nn.local_response_normalization(inputs)))
    
    model.add(keras.layers.Conv2D(64, 
                                  kernel_size=1, 
                                  strides=1,  
                                  use_bias=False,
                                  padding="same", 
                                  ))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Conv2D(192, 
                                  kernel_size=3, 
                                  strides=1,  
                                  use_bias=False,
                                  padding="same", 
                                  ))
    model.add(keras.layers.Activation("relu"))
    
    model.add(keras.layers.Lambda(lambda inputs: tf.nn.local_response_normalization(inputs)))

    model.add(keras.layers.MaxPool2D(pool_size=3, 
                                     strides=2,
                                     padding="same"))


    model.add(InceptionLayer([64, 128, 32, 32, 96, 16]))
    model.add(InceptionLayer([128, 192, 96, 94, 128, 32]))
    model.add(keras.layers.MaxPool2D(pool_size=3, 
                                     strides=2,
                                     padding="same"))
    model.add(InceptionLayer([192, 208, 48, 94, 96, 16]))
    model.add(InceptionLayer([160, 224, 64, 64, 112, 24]))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(0.1))
    
    model.add(keras.layers.Dense(5, activation="softmax"))
    
    return model


if __name__ == "__main__":
    run_model()