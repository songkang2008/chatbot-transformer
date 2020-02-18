import tensorflow as tf
from tokenizer import get_tokenizer
import matplotlib.pyplot as plt
from attention_model import Transformer
import time


MAX_LENGTH = 40
BATCH_SIZE = 64

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

train_dataset, tokenizer_zh = get_tokenizer(MAX_LENGTH, BATCH_SIZE)
sentences_ask = "我觉得你可能在开玩笑"
sentences_ans = '但是我并不这么认为。'
encode_id =  tokenizer_zh.convert_tokens_to_ids(sentences_ask)
print('encode_id', encode_id)
print(tokenizer_zh.convert_ids_to_tokens(encode_id))

encode_id_2 = tokenizer_zh.convert_tokens_to_ids(sentences_ans)
print('encode_id', encode_id_2)
print(tokenizer_zh.convert_ids_to_tokens(encode_id_2))

def mask_padding_token(seq):
    seq = tf.cast(tf.equal(seq, 0), dtype=tf.float32)
    # padding to the (batch, 1, 1, seq_length)
    return seq[:, tf.newaxis, tf.newaxis, :]
# The look-ahead mask is used to mask the future tokens in a sequence.
# In other words, the mask indicates which entries should not be used.
#
# This means that to predict the third word, only the first and second word will be used.
# Similarly to predict the fourth word, only the first, second and the third word will be used and so on.
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = 21128
target_vocab_size = 21130
dropout_rate = 0.1
epochs =30

print('target_vocab_size-sk', target_vocab_size)
# define the learning_rate
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate)
#
# temp_learning_rate_schedule=CustomSchedule(d_model)
# plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.xlabel("train_step")
# plt.ylabel('learining_rate')
# plt.show()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pre):
    mask = tf.logical_not(tf.equal(real, 0))
    loss = loss_object(real, pre)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)
def create_masks(inp, tar):
    # encode padding mask
    enc_padding_mask = mask_padding_token(inp)
    # mask the output of encoder
    dec_padding_mask = mask_padding_token(inp)

    dec_look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = mask_padding_token(tar)
    combine_mask = tf.maximum(dec_look_ahead_mask, dec_target_padding_mask)

    return enc_padding_mask, combine_mask, dec_padding_mask

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

# create the checkpoint path
checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

for epoch in range(epochs):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    for (step, (inp, tar)) in enumerate(train_dataset):
        tar_inp = tar[:, :-1]
        # print("tar_inp", tar_inp)
        tar_real = tar[:, 1:]
        # print('tar_teal', tar_real)

        enc_padding_mask, combine_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            output, attention_weights = transformer(inp, tar_inp, True, enc_padding_mask, combine_mask,
                                                    dec_padding_mask)
            # print(tar_real, output)
            loss = loss_function(tar_real, output)
        grad = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(grad, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, output)
        if step%50==0:
            print("loss", float(loss))
            print(('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format
                   (epoch + 1, step, train_loss.result(), train_accuracy.result())))
        if (epoch+1)%3==0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs/n'.format(time.time() - start))


def evaluate(inp_sentence):
    start_token = tokenizer_zh.convert_tokens_to_ids(['[CLS]'])
    print(start_token)
    end_token = tokenizer_zh.convert_tokens_to_ids(['[SEP]'])
    print(end_token)
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_zh.convert_tokens_to_ids(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [8250]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == 8251:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_zh.convert_tokens_to_ids(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [tokenizer_zh.convert_ids_to_tokens([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_zh.convert_ids_to_tokens([i]) for i in result
                            if i < 8225],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_zh.convert_ids_to_tokens([i for i in result
                                              if i < 8250])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


translate("我认为你是一个男孩。")
print("Real translation: this is a problem we have to solve .")

translate("我们不应该在这种时候浪费彼此的时间去做一些无所谓的事情。")
print("Real translation: this is a problem we have to solve .")


translate("我现在不应该随随便便的区委托一个律师干一件没有把我的事情。")

