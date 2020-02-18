import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding

# get the position encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    # according to order and input
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    """

    :param q:  shape==>(-1, seq_length_q, depth)
    :param k:  shape==>(-1, seq_length_k, depth)
    :param v:  shape==>(-1, seq_length_v, depth)
    :param mask:float tonser broadcast to (seq_length_q, seq_length_k)
    :return:
    """
    # matmul_qk ==> (seq_length_q, seq_length_k)

    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # print('matmul_qk', matmul_qk.shape)
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    attention_logits = matmul_qk/tf.sqrt(dk)

    # add the mask to the scaled tensor.
    # mask是padding或者后续的不需要的词设为1，故乘以一个无穷大，让softmax之后的prab为0，消除影响
    if mask is not None:
         # print(attention_logits.shape,  mask.shape)
         attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
     def __init__(self, num_heads, d_model):
         super(MultiHeadAttention, self).__init__()
         self.num_heads = num_heads
         self.d_model = d_model
         self.depth = self.d_model//self.num_heads
         self.Wq = Dense(d_model)
         self.Wk = Dense(d_model)
         self.Wv = Dense(d_model)

         self.fc = Dense(d_model)

     # convert to (batch_size, num_heads, seq_length, depth)
     def split_heads(self, batch_size, x):
         # print('x', x.shape)
         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
         x = tf.transpose(x, perm=[0, 2, 1, 3])
         return x
     def call(self, v, k, q, mask):
         batch_size = tf.shape(q)[0]
         # print('batch_size', batch_size)
         # print('k', k.shape)
         k = self.Wk(k)
         v = self.Wv(v)
         q = self.Wq(q)

         k =self.split_heads(batch_size, k)
         v =self.split_heads(batch_size, v)
         q =self.split_heads(batch_size, q)
         # output.shape => (batch_size, self.num_heads, self.seq_length_q. self.depth)
         # attention_weights.shape => (batch_size, self.num_heads, self.seq_length_q. self.seq_length_k)

         scale_attention, attention_weights= scaled_dot_product_attention(q, k, v, mask)
         scale_attention = tf.transpose(scale_attention, perm=[0, 2, 1, 3])

         concat = tf.reshape(scale_attention, (batch_size, -1, self.d_model))

         # output.shape  (batch_size, seq_length, d_model)
         output = self.fc(concat)

         return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.feed_forward = point_wise_feed_forward_network(d_model, dff)
        self.layerNorm1 = LayerNormalization()
        self.layerNorm2 = LayerNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    def call(self, x, training, mask):
         mha_out, _ = self.mha(x, x, x, mask)
         # print('mha_out', mha_out.shape)
         mha_out = self.dropout1(mha_out, training=training)
         x = self.layerNorm1(x + mha_out)

         feed_out = self.feed_forward(x)
         feed_out = self.dropout2(feed_out)
         out = self.layerNorm2(x+feed_out)

         return out

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads, d_model)
        self.mha2 = MultiHeadAttention(num_heads, d_model)

        self.feed_forward = point_wise_feed_forward_network(d_model, dff)

        self.normLayer1 = LayerNormalization()
        self.normLayer2 = LayerNormalization()
        self.normLayer3 = LayerNormalization()

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training, forward_mask, padding_mask):
        mha1_out, att_weigths_block1 = self.mha1(x, x, x, forward_mask)
        mha1_out = self.dropout1(mha1_out, training)
        out1 = self.normLayer1(mha1_out+x)

        mha2_out, att_weigths_block2= self.mha2(enc_output, enc_output, out1, padding_mask)
        mha2_out = self.dropout2(mha2_out, training)
        x = self.normLayer2(mha2_out + out1)

        feed_out = self.feed_forward(x)
        feed_out = self.dropout3(feed_out, training)
        out2 = self.normLayer3(x + feed_out)

        return out2, att_weigths_block1, att_weigths_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_encoder_layer, d_model, num_heads, diff, input_vocab_size, position, rate=0.1):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(position, self.d_model)
        self.enc_layers = [EncoderLayer(num_heads, d_model, diff, rate) for _ in range(num_encoder_layer)]
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, training, enc_pad_mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        # ????
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training, enc_pad_mask)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_dec_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_dec_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(num_heads, d_model , dff, rate)
                           for _ in range(num_dec_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        # print("x_x", x.shape)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            # print('x_22', x.shape)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # print('enc_output', enc_output.shape)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


def main():
    # sample_encoder = Decoder(2, 512, 8, 2048, 8500, 6000)
    #
    # temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    # enc_output = tf.random.uniform([64, 62, 512])
    # sample_decoder_output, att_weights = sample_encoder(temp_input, enc_output, training=False, look_ahead_mask=None,
    #                                                     padding_mask=None)
    #
    # print(sample_decoder_output.shape, att_weights['decoder_layer2_block2'].shape)  # (batch_size, input_seq_len, d_model)

    input = tf.random.uniform([32, 40])
    tar = tf.random.uniform([32, 40])
    mask = tf.random.uniform([32, 1, 1, 40])
    transformer = Transformer(2, 512, 8, 2048, 8500, 8500, 8500, 8500, 0.1)
    final_output, atten = transformer(input, tar, True, mask, mask , mask )
    print(final_output.shape)

if __name__ == '__main__':
    main()