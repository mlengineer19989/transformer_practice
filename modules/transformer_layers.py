import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_angles(pos, i, d_model):
  """_summary_

  Args:
      pos (ndarray): 0, 1,..., 定数　を順に値にもつ縦ベクトル。最大値の定数はtransformerクラスのインスタンス時に定義される。
      i (ndarray): 0, 1, ..., d_modelを順に値にもつ横ベクトル。
      d_model (int): モデル内での特徴量の次元。多変量時系列データ(time✖️d_origin)の場合、(time✖️d_model)に変換することになる。

  Returns:
      _type_: _description_
  """
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # 配列中の偶数インデックスにはsinを適用; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # 配列中の奇数インデックスにはcosを適用; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerEncoder(layers.Layer):
  """transformerのEncoderのMulti Head Atttentionやその周辺の繰り返されるlayerをまとめて実装する。
  """
  def __init__(self, embed_dim : int, dense_dim : int, num_heads : int, rate : float, **kwargs):
    """_summary_

    Args:
        embed_dim (int): モデル内での特徴量の次元。多変量時系列データ(time✖️d_origin)の場合、(time✖️d_model)に変換することになる。
        dense_dim (int): 全結合層のユニット数。
        num_heads (int): MuliHeadAtteintionでのheadの個数
        rate (float): dropoutのパラメータ
    """
    super().__init__(**kwargs)
    self.embed_dim = embed_dim    #入力トークンベクトルのサイズ
    self.dense_dim = dense_dim     #内側のDense層の次元
    self.num_heads = num_heads    #Attentionのヘッドの個数
    self.attention = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim
    )
    self.dense_proj = keras.Sequential(
        [layers.Dense(dense_dim, activation="relu"), 
         layers.Dense(embed_dim),]
    )
    self.layernorm_1 = layers.LayerNormalization()
    self.layernorm_2 = layers.LayerNormalization()

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  #計算はcall()で行う
  def call(self, inputs, training, mask=None):
    #Embedding層が生成するマスクは2次元だが、Attention層には
    #3次元または4次元のマスクを渡す必要があるため、階数を増やす
    if mask is not None:
      mask = mask[:, tf.newaxis, :]

    attention_output = self.attention(inputs, inputs, attention_mask = mask)
    attention_output = self.dropout1(attention_output, training=training)
    out1 = self.layernorm_1(inputs + attention_output)

    proj_output = self.dense_proj(out1)
    proj_output = self.dropout2(proj_output, training=training)
    out2 = self.layernorm_2(out1 + proj_output)

    return out2

  def get_config(self):
    config = super().get_config()
    config.update({"embed_dim": self.embed_dim, 
                   "num_heads": self.num_heads, 
                   "dense_dim": self.dense_dim, 
                   })
    return config



#参考になったサイト
#Embedding layer(https://agirobots.com/word2vec-and-embeddinglayer/)
#位置符号化(https://cvml-expertguide.net/terms/dl/seq2seq-translation/transformer/positional-encoding/)


class Encoder(tf.keras.layers.Layer):
  """transformerのEncoderを実装するLayerクラス
  """
  def __init__(self, num_layers : int, d_model : int, num_heads : int, dff : int,
               maximum_position_encoding : int, rate=0.1):
    """
    Args:
        num_layers (int): TransformerEncoder（MuliHeadAtteintionLayerとその周辺部）を繰り返す回数
        d_model (int): モデル内での特徴量の次元。多変量時系列データ(time✖️d_origin)の場合、(time✖️d_model)に変換することになる。
        num_heads (int): MuliHeadAtteintionでのheadの個数
        dff (int): TransformerEncoderで実装される全結合層のユニット数。
        maximum_position_encoding (int): 入力系列の位置エンコーディングで用いる値。transformer_layers.positional_encodingの説明参照。
        rate (float, optional): dropoutのパラメータ. Defaults to 0.1.
    """
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.Input = layers.Dense(self.d_model)
    
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)


    self.enc_layers = [TransformerEncoder(d_model, dff, num_heads, rate) 
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    # xのサイズは(batch_size, input_seq_len)

    seq_len = tf.shape(x)[1]

    if len(x.shape)==2:
      x = tf.expand_dims(x, -1)

    x = self.Input(x)
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)



class TransformerDecoder(tf.keras.layers.Layer):
  """transformerのDecoderのMulti Head Atttentionやその周辺の繰り返されるlayerをまとめて実装する。
  """
  def __init__(self, embed_dim : int, dense_dim : int, num_heads : int, rate : float, **kwargs):
    """_summary_

    Args:
        embed_dim (int): モデル内での特徴量の次元。多変量時系列データ(time✖️d_origin)の場合、(time✖️d_model)に変換することになる。
        dense_dim (int): 全結合層のユニット数。
        num_heads (int): MuliHeadAtteintionでのheadの個数
        rate (float): dropoutのパラメータ
    """
    super().__init__(**kwargs)
    self.embed_dim = embed_dim
    self.dense_dim = dense_dim
    self.num_heads = num_heads

    self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    self.dense_proj = keras.Sequential(
        [layers.Dense(dense_dim, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(embed_dim),  # (batch_size, seq_len, d_model)
      ])

    self.layernorm_1 = layers.LayerNormalization()
    self.layernorm_2 = layers.LayerNormalization()
    self.layernorm_3 = layers.LayerNormalization()

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

    self.supports_masking = True

  def get_config(self):
    config = super().get_config()
    config.update({"embed_dim": self.embed_dim, 
                   "num_heads": self.num_heads, 
                   "dense_dim": self.dense_dim})
    return config

  def get_causal_attention_mask(self, inputs):
    input_shape = tf.shape(inputs)
    batch_size ,sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)

    #教科書のiとjの向きが逆っぽいので修正
    mask = tf.cast(i < j, dtype="int32")

    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))

    mult = tf.concat([tf.expand_dims(batch_size, -1), 
                      tf.constant([1, 1], dtype=tf.int32)], axis=0)
    return tf.tile(mask, mult)


  def call(self, inputs, encoder_outputs, training, mask=None):
    causal_mask = self.get_causal_attention_mask(inputs)

    if mask is not None:
      padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
      #2つのマスクをマージ
      padding_mask = tf.minimum(padding_mask, causal_mask)
    else:
      padding_mask = mask

    attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
    attention_output_1 = self.dropout1(attention_output_1, training=training)
    attention_output_1 = self.layernorm_1(inputs + attention_output_1)

    attention_output_2 = self.attention_1(query=attention_output_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask)
    attention_output_2 = self.dropout2(attention_output_2, training=training)
    attention_output_2 = self.layernorm_1(attention_output_1 + attention_output_2)

    proj_output = self.dense_proj(attention_output_2)
    proj_output = self.dropout3(proj_output, training=training)
    out = self.layernorm_3(attention_output_2 + proj_output)  # (batch_size, target_seq_len, d_model)
    return out


class Decoder(tf.keras.layers.Layer):
  """transformerのDecoderを実装するLayerクラス
  """
  def __init__(self, num_layers : int, d_model : int, num_heads : int, dff : int,
               maximum_position_encoding : int, rate=0.1):
    """
    Args:
        num_layers (int): TransformerEncoder（MuliHeadAtteintionLayerとその周辺部）を繰り返す回数
        d_model (int): モデル内での特徴量の次元。多変量時系列データ(time✖️d_origin)の場合、(time✖️d_model)に変換することになる。
        num_heads (int): MuliHeadAtteintionでのheadの個数
        dff (int): TransformerEncoderで実装される全結合層のユニット数。
        maximum_position_encoding (int): 入力系列の位置エンコーディングで用いる値。transformer_layers.positional_encodingの説明参照。
        rate (float, optional): dropoutのパラメータ. Defaults to 0.1.
    """
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.Input = layers.Dense(d_model)
    #self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [TransformerDecoder(d_model, dff, num_heads, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, padding_mask):

    seq_len = tf.shape(x)[1]
    #attention_weights = {}

    # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    if len(x.shape)==2:
      x = tf.expand_dims(x, -1)
    x = self.Input(x)
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.dec_layers[i](x, enc_output, training, padding_mask)

      # attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      # attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x