import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from utility import resampling


class PreprocessData():
  """学習用データセット生成クラス。以下を実行する。
    ・train, validation, testデータへの分割
    ・説明変数データの正規化
    ・1ステップ予測用データセット生成
    ・seq2seq用データセット生成
  """
  def __init__(self, 
              raw_data, 
              temperature, 
              sequence_length : int, 
              delay : int, 
              batch_size: int, 
              sampling_rate : int=None):
    """
    Args:
        raw_data (_type_): 説明変数の配列
        temperature (_type_): 目的変数の配列
        sequence_length (int): 説明変数時系列の時系列長
        delay (int): 説明変数の先頭インデックスに対する、目的変数の先頭インデックスの時間遅れ
        batch_size (int): ミニバッチの説明変数と目的変数の組み合わせの個数。
        sampling_rate (int, optional): データリサンプリングの間隔. Defaults to None.
    """
    self.raw_data = raw_data
    self.targets = temperature
    self.sequence_length = sequence_length
    self.delay = delay
    self.batch_size = batch_size
    self.sampling_rate = sampling_rate

    self.num_train_samples = int(0.5 * len(raw_data))
    self.num_val_samples = int(0.25 * len(raw_data))
    self.num_test_samples = len(raw_data) - self.num_train_samples - self.num_val_samples
    print("num_train_samples:", self.num_train_samples)
    print("num_val_samples:", self.num_val_samples)
    print("num_test_samples:", self.num_test_samples)
    
    self.mean, self.std = self.normalization()

  def normalization(self):
    #正規化
    mean = self.raw_data[:self.num_train_samples].mean(axis=0)
    self.raw_data -= mean
    std = self.raw_data[:self.num_train_samples].std(axis=0)
    self.raw_data /= std
    return mean, std

  def generate_dataset_single(self):
    train_dataset = keras.utils.timeseries_dataset_from_array(
        self.raw_data[:-self.delay],
        targets=self.targets[self.delay:],
        sampling_rate=self.sampling_rate,
        sequence_length=self.sequence_length,
        shuffle=True,
        batch_size=self.batch_size,
        start_index=0,
        end_index=self.num_train_samples)

    val_dataset = keras.utils.timeseries_dataset_from_array(
        self.raw_data[:-self.delay],
        targets=self.targets[self.delay:],
        sampling_rate=self.sampling_rate,
        sequence_length=self.sequence_length,
        shuffle=True,
        batch_size=self.batch_size,
        start_index=self.num_train_samples,
        end_index=self.num_train_samples + self.num_val_samples)

    test_dataset = keras.utils.timeseries_dataset_from_array(
        self.raw_data[:-self.delay],
        targets=self.targets[self.delay:],
        sampling_rate=self.sampling_rate,
        sequence_length=self.sequence_length,
        shuffle=True,
        batch_size=self.batch_size,
        start_index=self.num_train_samples + self.num_val_samples)
    return train_dataset, val_dataset, test_dataset

  def generate_dataset_multistep(self, pred_points : int):
    """
    seq2seq用データセットを生成する。tensorflowには、以下のように時系列と時系列が組となるseq2seq用データセット生成メソッドがない。
    [enc_seq, dec_seq] => target_seq
    したがって、時系列と単一ステップのデータの組を生成するTimeseriesGeneratorを活用する。

    例えば、
    raw_data = [0, 1, 2, 3, 4, 5, 6, ...]
    から、
    len(enc_seq), len(dec_seq), len(target_seq) = 3, 2, 2
    となるような以下のデータセットを生成したい場合を考える。
    [0, 1, 2, 3] -> [3, 4]
    [1, 2, 3, 4] -> [4, 5]
    [2, 3, 4, 5] -> [5, 6]
              .
              .
              .

    TimeseriesGeneratorの目的変数には以下のようなデータを与えることで実現できる。
    tar_temp = [None, None, None, None, [3, 4], [4, 5], [5, 6]]
    TimeseriesGeneratorでは、入力系列[enc_seq, dec_seq]の長さ分進んだインデックスから目的変数を取得する。
    したがって、上記配列はその分ずらすために入力系列長分だけNoneを先頭に与えている。
    ただし、本対象はエンコーダ入力系列末尾のインデックスとデコーダ入力系列先頭のインデックスを一致させているため、
    入力系列長は、len(enc_seq) + len(dec_seq) -1となることに注意。

    Args:
        pred_points (int): _description_

    Returns:
        _type_: _description_
    """
    input_length = self.sequence_length + pred_points - self.sampling_rate
    sequence_length = self.sampling_rate * self.sequence_length

    #tar_temp生成
    resampled_targets = []
    for i in range(0, len(self.targets), self.sampling_rate):
      resampled_targets.append(self.targets[i])
    resampled_targets = np.array(resampled_targets)

    tar_temp = []
    for i in range(len(resampled_targets[:-self.delay//self.sampling_rate-pred_points//self.sampling_rate])):
      if i < pred_points//self.sampling_rate:
        tar_temp.append(None)
      else:
        tar_temp.append(resampled_targets[i+1-pred_points//self.sampling_rate : i+1])
    tar_temp = np.array(tar_temp)

    train_dataset = TimeseriesGenerator(self.raw_data[:-self.delay-pred_points], 
                                        tar_temp, 
                                        length=input_length, 
                                        sampling_rate=self.sampling_rate, 
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        start_index=0,
                                        end_index=self.num_train_samples)

    
    val_dataset = TimeseriesGenerator(self.raw_data[:-self.delay-pred_points], 
                                        tar_temp, 
                                        length=input_length, 
                                        sampling_rate=self.sampling_rate, 
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        start_index=self.num_train_samples,
                                        end_index=self.num_train_samples + self.num_val_samples)

    
    test_dataset = TimeseriesGenerator(self.raw_data[:-self.delay-pred_points], 
                                        tar_temp, 
                                        length=input_length, 
                                        sampling_rate=self.sampling_rate, 
                                        shuffle=False,
                                        batch_size=self.batch_size,
                                        start_index=self.num_train_samples + self.num_val_samples)
    
    return train_dataset, val_dataset, test_dataset


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