import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from utility import resampling


class PreprocessData():
  def __init__(self, raw_data, batch_size, window_width, pred_points,  resampling_rate=None, output_index=None):
    self.raw_data = raw_data
    self.batch_size = batch_size
    self.window_width = window_width
    self.pred_points = pred_points
    self.output_index = output_index

    if resampling_rate:
      self.raw_data = resampling(self.raw_data, resampling_rate)

    self.num_train_samples = int(0.5 * len(self.raw_data)//self.batch_size * self.batch_size) + self.window_width +self.pred_points
    self.num_val_samples = int(0.25 * len(self.raw_data)//self.batch_size * self.batch_size) + self.window_width +self.pred_points
    self.num_test_samples = (len(self.raw_data) - self.num_train_samples - self.num_val_samples)//self.batch_size * self.batch_size - self.batch_size + self.window_width +self.pred_points

    self.normalization()

    

    self.train_data = self.raw_data[: self.num_train_samples]
    self.val_data = self.raw_data[self.num_train_samples : self.num_train_samples + self.num_val_samples]
    self.test_data = self.raw_data[self.num_train_samples + self.num_val_samples : self.num_train_samples + self.num_val_samples + self.num_test_samples]

  def normalization(self):
    #正規化
    mean = self.raw_data[:self.num_train_samples].mean(axis=0)
    self.raw_data -= mean
    std = self.raw_data[:self.num_train_samples].std(axis=0)
    self.raw_data /= std

  def splitdata_for_transformer(self, emb_data):
    enc = np.array([emb_data[i:i+self.window_width] for i in range(len(emb_data)-self.window_width-self.pred_points)])
    dec = np.array([emb_data[i+self.window_width-1:i+self.window_width-1+self.pred_points] for i in range(len(emb_data)-self.window_width-self.pred_points)])
    
    if self.output_index:
      y = np.array([emb_data[i+self.window_width:i+self.window_width+self.pred_points, self.output_index] for i in range(len(emb_data)-self.window_width-self.pred_points)])
    else:
      y = np.array([emb_data[i+self.window_width:i+self.window_width+self.pred_points] for i in range(len(emb_data)-self.window_width-self.pred_points)])

    return enc, dec, y


  def splitdata_for_NN(self, emb_data):
      inp = np.array([emb_data[i:i+self.window_width] for i in range(len(emb_data)-self.window_width-self.pred_points)])
      
      if self.output_index:
        y = np.array([emb_data[i+self.window_width+self.pred_points, self.output_index] for i in range(len(emb_data)-self.window_width-self.pred_points)])
      else:
        y = np.array([emb_data[i+self.window_width+self.pred_points] for i in range(len(emb_data)-self.window_width-self.pred_points)])
      return inp, y

  def make_data_for_transformer(self):
    enc_train, dec_train, y_train = self.splitdata_for_transformer(self.train_data)
    enc_val, dec_val, y_val = self.splitdata_for_transformer(self.val_data)
    enc_test, dec_test, y_test = self.splitdata_for_transformer(self.test_data)


    enc_train = tf.constant(enc_train)
    enc_val = tf.constant(enc_val)
    enc_test = tf.constant(enc_test)
    dec_train = tf.constant(dec_train)
    dec_val = tf.constant(dec_val)
    dec_test = tf.constant(dec_test)
    y_train = tf.constant(y_train)
    y_val = tf.constant(y_val)
    y_test = tf.constant(y_test)

    print('encinput_train ', enc_train.shape)
    print('decinput_train ', dec_train.shape)
    print('y_train ', y_train.shape)

    print('encinput_val ', enc_val.shape)
    print('decinput_val ', dec_val.shape)
    print('y_val ', y_val.shape)

    print('encinput_test ', enc_test.shape)
    print('decinput_test ', dec_test.shape)
    print('y_test ', y_test.shape)

    return enc_train, dec_train, y_train, enc_val, dec_val, y_val, enc_test, dec_test, y_test

  def make_data_for_NN(self):
    x_train, y_train = self.splitdata_for_NN(self.train_data)
    x_val, y_val = self.splitdata_for_NN(self.val_data)
    x_test, y_test = self.splitdata_for_NN(self.test_data)


    x_train = tf.constant(x_train)
    x_val = tf.constant(x_val)
    x_test = tf.constant(x_test)
    y_train = tf.constant(y_train)
    y_val = tf.constant(y_val)
    y_test = tf.constant(y_test)

    print('input_train ', x_train.shape)
    print('y_train ', y_train.shape)

    print('input_val ', x_val.shape)
    print('y_val ', y_val.shape)

    print('input_test ', x_test.shape)
    print('y_test ', y_test.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test



def textbook_method(raw_data, temperature, sampling_rate, sequence_length, delay, batch_size):
  num_train_samples = int(0.5 * len(raw_data))
  num_val_samples = int(0.25 * len(raw_data))
  num_test_samples = len(raw_data) - num_train_samples - num_val_samples
  print("num_train_samples:", num_train_samples)
  print("num_val_samples:", num_val_samples)
  print("num_test_samples:", num_test_samples)

  mean = raw_data[:num_train_samples].mean(axis=0)
  raw_data -= mean
  std = raw_data[:num_train_samples].std(axis=0)
  raw_data /= std

  train_dataset = keras.utils.timeseries_dataset_from_array(
      raw_data[:-delay],
      targets=temperature[delay:],
      sampling_rate=sampling_rate,
      sequence_length=sequence_length,
      shuffle=True,
      batch_size=batch_size,
      start_index=0,
      end_index=num_train_samples)

  val_dataset = keras.utils.timeseries_dataset_from_array(
      raw_data[:-delay],
      targets=temperature[delay:],
      sampling_rate=sampling_rate,
      sequence_length=sequence_length,
      shuffle=True,
      batch_size=batch_size,
      start_index=num_train_samples,
      end_index=num_train_samples + num_val_samples)

  test_dataset = keras.utils.timeseries_dataset_from_array(
      raw_data[:-delay],
      targets=temperature[delay:],
      sampling_rate=sampling_rate,
      sequence_length=sequence_length,
      shuffle=True,
      batch_size=batch_size,
      start_index=num_train_samples + num_val_samples)
  return train_dataset, val_dataset, test_dataset


def textbook_method_multistep(raw_data, temperature, sampling_rate, sequence_length, delay, pred_points, batch_size):
  input_length = sequence_length + pred_points -1
  
  num_train_samples = int(0.5 * len(raw_data))
  num_val_samples = int(0.25 * len(raw_data))
  num_test_samples = len(raw_data) - num_train_samples - num_val_samples
  print("num_train_samples:", num_train_samples)
  print("num_val_samples:", num_val_samples)
  print("num_test_samples:", num_test_samples)

  sequence_length *= sampling_rate

  mean = raw_data[:num_train_samples].mean(axis=0)
  raw_data -= mean
  std = raw_data[:num_train_samples].std(axis=0)
  raw_data /= std

  #tar_temp = np.array([temperature[i+delay : i+delay+pred_points] for i in range(len(temperature[delay:-pred_points]))])
  tar_temp = []
  for i in range(len(temperature[:-delay-pred_points])):
    if i < pred_points:
      tar_temp.append(None)
    else:
      tar_temp.append(temperature[i+1-pred_points : i+1])
  tar_temp = np.array(tar_temp)

  train_dataset = TimeseriesGenerator(raw_data[:-delay-pred_points], 
                                      tar_temp, 
                                      length=input_length, 
                                      sampling_rate=sampling_rate, 
                                      shuffle=True,
                                      batch_size=batch_size,
                                      start_index=0,
                                      end_index=num_train_samples)

  
  val_dataset = TimeseriesGenerator(raw_data[:-delay-pred_points], 
                                      tar_temp, 
                                      length=input_length, 
                                      sampling_rate=sampling_rate, 
                                      shuffle=True,
                                      batch_size=batch_size,
                                      start_index=num_train_samples,
                                      end_index=num_train_samples + num_val_samples)

  
  test_dataset = TimeseriesGenerator(raw_data[:-delay-pred_points], 
                                      tar_temp, 
                                      length=input_length, 
                                      sampling_rate=sampling_rate, 
                                      shuffle=False,
                                      batch_size=batch_size,
                                      start_index=num_train_samples + num_val_samples)
  
  return train_dataset, val_dataset, test_dataset