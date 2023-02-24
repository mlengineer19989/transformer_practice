import tensorflow as tf
import transformer_layers
from transformer_layers import Encoder, Decoder

#以下のモデルの参考サイト（https://qiita.com/tanaka_benkyo/items/00c5eb90101ccb5075d9）

class TransformerBaseTimeSeriesForecastModel(tf.keras.Model):
  """transformerによる時系列予測モデル用のLayerクラス
  """
  def __init__(self, num_layers : int, d_model : int, num_heads : int, dff : int, 
              pe_input : int, pe_target : int, window_width : int, rate=0.1):
    """
    Args:
        num_layers (int): TransformerEncoder（MuliHeadAtteintionLayerとその周辺部）を繰り返す回数
        d_model (int): モデル内での特徴量の次元。多変量時系列データ(time✖️d_origin)の場合、(time✖️d_model)に変換することになる。
        num_heads (int): MuliHeadAtteintionでのheadの個数
        dff (int): TransformerEncoderで実装される全結合層のユニット数。
        pe_input (int): 入力系列の位置エンコーディングで用いる値。transformer_layers.positional_encodingの説明参照。
        pe_target (int): 目標系列の位置エンコーディングで用いる値。transformer_layers.positional_encodingの説明参照。
        window_width (int): 時系列の窓幅。
        rate (float, optional): dropoutのパラメータ. Defaults to 0.1.
    """
    super().__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(1)

    self.window_width = window_width

  def call(self, inp, training, enc_padding_mask=None, dec_padding_mask=None):
    
    enc_input = inp[:,:self.window_width]
    dec_input = inp[:,self.window_width-1:]

    enc_output = self.encoder(enc_input, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output = self.decoder(dec_input, enc_output, training, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output



class EncoderWithMultiHeadAttentionBaseTimeSeriesForecastModel(tf.keras.layers.Layer):
  """transformerのEncoderによる時系列予測モデル用のLayerクラス
  """
  def __init__(self, num_layers : int, d_model : int, num_heads : int, dff : int,  
              pe_input : int, window_width : int, rate=0.1):
    """
    Args:
        num_layers (int): TransformerEncoder（MuliHeadAtteintionLayerとその周辺部）を繰り返す回数
        d_model (int): モデル内での特徴量の次元。多変量時系列データ(time✖️d_origin)の場合、(time✖️d_model)に変換することになる。
        num_heads (int): MuliHeadAtteintionでのheadの個数
        dff (int): TransformerEncoderで実装される全結合層のユニット数。
        pe_input (int): 入力系列の位置エンコーディングで用いる値。transformer_layers.positional_encodingの説明参照。
        window_width (int): 時系列の窓幅。
        rate (float, optional): dropoutのパラメータ. Defaults to 0.1.
    """
    super().__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

    self.final_layer = tf.keras.layers.Dense(1)

    self.window_width = window_width

  def call(self, inp, training, enc_padding_mask=None):
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output

