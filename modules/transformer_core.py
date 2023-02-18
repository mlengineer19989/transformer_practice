import tensorflow as tf
import transformer_layers
from transformer_layers import Encoder, Decoder

#以下のモデルの参考サイト（https://qiita.com/tanaka_benkyo/items/00c5eb90101ccb5075d9）

class TransformerBaseTimeSeriesForecastModel(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, window_width, rate=0.1):
    super().__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(1)

    self.window_width = window_width

  def call(self, inp, training, enc_padding_mask=None, dec_padding_mask=None):
    
    enc_input = inp[:,:self.window_width]
    dec_input = inp[:,self.window_width:]

    enc_output = self.encoder(enc_input, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output = self.decoder(dec_input, enc_output, training, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output



class EncoderWithMultiHeadAttentionBaseTimeSeriesForecastModel(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
              pe_input, window_width, rate=0.1):
    super().__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)

    self.final_layer = tf.keras.layers.Dense(1)

    self.window_width = window_width

  def call(self, inp, training, enc_padding_mask=None, dec_padding_mask=None):
    print(inp.shape)
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output

