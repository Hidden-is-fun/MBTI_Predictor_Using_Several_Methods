import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from pandas import DataFrame
from tensorflow.keras import layers
import bert
import pandas as pd
import re
import random
import math
import keras.backend as K
from keras.callbacks import LearningRateScheduler

is_predict = True  # Change the value to 'False' to train the model


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


def tokenize_text(text_input):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_input))


def lb_process(x):
    lb_str = ''
    lb_int = 0
    lb_str += 'E' if x[0] else 'I'
    lb_str += 'S' if x[1] else 'N'
    lb_str += 'T' if x[2] else 'F'
    lb_str += 'P' if x[3] else 'J'
    lb_int += 0 if x[0] else 8
    lb_int += 4 if x[1] else 0
    lb_int += 2 if x[2] else 0
    lb_int += 1 if x[3] else 0
    return lb_str, lb_int


def lb_generate(x):
    lb = 0
    lb += 8 if 'I' in x else 0
    lb += 4 if 'S' in x else 0
    lb += 2 if 'T' in x else 0
    lb += 1 if 'P' in x else 0
    return lb


def predict(data):
    predict_table = [[0 for i in range(16)] for j in range(16)]
    d = pd.read_csv('data/new_data.csv')
    d = d['type'][:1280].to_list()
    d = [lb_generate(i) for i in d]
    lb = []
    model = tf.keras.models.load_model(f'saved_models/bert_tokenize_15813')
    a = model.predict(data)
    '''
    for _ in range(len(a)):
        for i in range(16):
            predict_table[d[_]][i] += a[_][i]
    pd.DataFrame.to_csv(DataFrame(predict_table), '.csv', index=False, header=False)
    '''
    for _ in range(len(a)):
        res = 0
        val = -1
        for i in range(len(a[_])):
            if a[_][i] > val:
                val = a[_][i]
                res = i
        predict_table[res][d[_]] += 1
    pd.DataFrame.to_csv(DataFrame(predict_table), '15813.csv', index=False, header=False)


if __name__ == '__main__':

    # hyper parameters
    BATCH_SIZE = 32
    EMB_DIM = 300
    CNN_FILTERS = 100
    DNN_UNITS = 256
    OUTPUT_CLASSES = 16
    DROPOUT_RATE = 0
    NB_EPOCHS = 1000
    max_len = 2048

    # raw data
    data_set = pd.read_csv("data/new_data.csv")
    y = [lb_generate(x) for x in data_set['type']]
    text = []
    personality_type = ['IE', 'NS', 'FT', 'JP']
    for _i in range(len(data_set)):
        _text = data_set["posts"][_i]
        _text = _text[1:-1]
        _text = re.sub(r'https?:\\/\\/.*?[\s+]', ' ', _text)
        _text = re.sub(r'http?:\\/\\/.*?[\s+]', ' ', _text)
        _text = _text.replace('...|||', '.')
        _text = _text.replace('|||', ' ')
        text.append(_text)

    # Creating a BERT Tokenizer
    BertTokenizer = bert.bert_tokenization.FullTokenizer

    # bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    bert_layer = hub.KerasLayer("models/BERT_Tokenizer", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    # Tokenize all the text
    tokenized_text = [tokenize_text(i) for i in text]

    # Prerparing Data For Training
    text_with_len = [[text, y[i], len(text)]
                     for i, text in enumerate(tokenized_text)]
    # text_with_len.sort(key=lambda x: x[2])
    # sorted_text_labels = [(text_lab[0], text_lab[1]) for text_lab in text_with_len]
    sorted_text_labels = [(text_lab[0][:max_len], text_lab[1]) for text_lab in text_with_len]
    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_text_labels,
                                                       output_types=(tf.int32, tf.int32))

    # batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
    batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((max_len,), ()))

    TOTAL_BATCHES = math.ceil(len(sorted_text_labels) / BATCH_SIZE)
    TEST_BATCHES = TOTAL_BATCHES * 3 // 20
    # batched_dataset.shuffle(TOTAL_BATCHES, seed=1, reshuffle_each_iteration=False)
    test_data = batched_dataset.take(TEST_BATCHES)
    train_data = batched_dataset.skip(TEST_BATCHES)

    if is_predict:
        predict(test_data)
        exit(0)

    VOCAB_LENGTH = len(tokenizer.vocab)

    text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                            embedding_dimensions=EMB_DIM,
                            cnn_filters=CNN_FILTERS,
                            dnn_units=DNN_UNITS,
                            model_output_classes=OUTPUT_CLASSES,
                            dropout_rate=DROPOUT_RATE)

    # text_model = tf.keras.models.load_model(f'saved_models/bert_tokenize_94317')

    if OUTPUT_CLASSES == 2:
        text_model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])
    else:
        text_model.compile(loss="sparse_categorical_crossentropy",
                           optimizer="adam",
                           metrics=["sparse_categorical_accuracy"])

    earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4)


    def scheduler(epoch):
        return 1e-4 / math.pow(5, epoch // 10)


    reduce_lr = LearningRateScheduler(scheduler)
    text_model.fit(train_data,
                   epochs=NB_EPOCHS,
                   validation_data=test_data,
                   callbacks=[reduce_lr, earlystop_callback]
                   )
    results = text_model.evaluate(test_data)
    print(results)
    r = random.randint(10000, 99999)
    text_model.save(f'saved_models/bert_tokenize_{r}')
    print(f'Saved with code {r}')
