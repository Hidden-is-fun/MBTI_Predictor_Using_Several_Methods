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

# Count of each type in test_set -> [342, 0, 344, 3, 0, 0, 0, 0, 23, 0, 0, 568, 0, 0, 0, 0]
# After process -> [219, 268, 162, 195, 24, 40, 30, 50, 28, 100, 33, 101, 6, 7, 5, 12]
# As you see, Many types of posts are not contain in the test_set
# So we should process the dataset to generate a new dataset first.
# Run the code below if you run it for the FIRST TIME!
'''
def encode(string):
    res = 0
    res += 0 if 'I' in string else 8
    res += 0 if 'N' in string else 4
    res += 0 if 'F' in string else 2
    res += 0 if 'J' in string else 1
    return res


def decode(integer):
    res = []
    res_str = ''
    for _ in range(4):
        if integer % 2 == 0:
            res.append(0)
        else:
            integer -= 1
            res.append(1)
        integer /= 2
    res = list(res.__reversed__())
    tags = ['IE', 'NS', 'FT', 'JP']
    for i in range(4):
        res_str += tags[i][1] if res[i] else tags[i][0]
    return res_str


data_set = pd.read_csv("data/mbti.csv")
posts = data_set['posts']
lb = data_set['type'].to_list()
lb_encoded = [encode(i) for i in lb]
picked = [False] * len(data_set)
data_pack = [[picked[i], lb_encoded[i], posts[i]] for i in range(len(data_set))]
random.shuffle(data_pack)
type_count = [0] * 16
for i in lb_encoded:
    type_count[i] += 1
type_count = [i * 15 // 100 for i in type_count]
new_pack = []
for _ in range(16):
    count = 0
    for __ in range(len(data_pack)):
        if data_pack[__][1] == _:
            data_pack[__][0] = True
            new_pack.append([data_pack[__][1], data_pack[__][2]])
            count += 1
        if count >= type_count[_]:
            break
random.shuffle(new_pack)
for _ in range(len(data_pack)):
    if not data_pack[_][0]:
        new_pack.append([data_pack[_][1], data_pack[_][2]])
new_lb = [decode(i[0]) for i in new_pack]
new_posts = [i[1] for i in new_pack]
pd.DataFrame.to_csv(DataFrame({'type': new_lb, 'posts': new_posts}), 'data/new_data.csv', index=False, encoding='utf-8')
exit(0)
'''

is_predict = False  # Change the value to 'False' to train the model


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
    lb = [[] for i in range(1280)]
    bin_type = ['IE', 'NS', 'FT', 'JP']
    for i in bin_type:
        model = tf.keras.models.load_model(f'saved_models/{i}_bert_tokenize')
        a = model.predict(data)
        for _ in range(len(a)):
            # print(a[_][0])
            lb[_].append(0 if a[_][0] < .5 else 1)
    lb = [lb_process(_)[1] for _ in lb]
    for _ in range(len(d)):
        predict_table[lb[_]][d[_]] += 1
    pd.DataFrame.to_csv(DataFrame(predict_table), '.csv', index=False, header=False)


if __name__ == '__main__':

    # hyper parameters
    BATCH_SIZE = 32
    EMB_DIM = 300
    CNN_FILTERS = 100
    DNN_UNITS = 256
    OUTPUT_CLASSES = 2
    DROPOUT_RATE = 0.2
    NB_EPOCHS = 1000
    max_len = 2048

    # raw data
    data_set = pd.read_csv("data/new_data.csv")
    y_4axis = [[], [], [], []]
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
        for _ in range(4):
            y_4axis[_].append(0 if data_set["type"][_i][_] == personality_type[_][0] else 1)

    # Creating a BERT Tokenizer
    BertTokenizer = bert.bert_tokenization.FullTokenizer

    # bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    bert_layer = hub.KerasLayer("models/BERT_Tokenizer", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    # Tokenize all the text
    tokenized_text = [tokenize_text(i) for i in text]

    test_data_labels = []
    for _i in range(4):
        # Prerparing Data For Training
        text_with_len = [[text, y_4axis[_i][i], len(text)]
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

        lb = []
        for _ in test_data:
            batched_lb = np.array(_[1])
            for i in batched_lb:
                lb.append(i)
        test_data_labels.append(lb)

        VOCAB_LENGTH = len(tokenizer.vocab)
        text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                                embedding_dimensions=EMB_DIM,
                                cnn_filters=CNN_FILTERS,
                                dnn_units=DNN_UNITS,
                                model_output_classes=OUTPUT_CLASSES,
                                dropout_rate=DROPOUT_RATE)

        if OUTPUT_CLASSES == 2:
            text_model.compile(loss="binary_crossentropy",
                               optimizer="adam",
                               metrics=["accuracy"])
        else:
            text_model.compile(loss="sparse_categorical_crossentropy",
                               optimizer="adam",
                               metrics=["sparse_categorical_accuracy"])

        earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2)


        def scheduler(epoch):
            return 1e-4 / math.pow(2, epoch // 4)


        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')

        reduce_lr = LearningRateScheduler(scheduler)
        text_model.fit(train_data,
                       epochs=NB_EPOCHS,
                       validation_data=test_data,
                       callbacks=[earlystop_callback, reduce_lr])
        results = text_model.evaluate(test_data)
        print(results)
        print(f'{personality_type[_i][0]}/{personality_type[_i][1]} Trained Successfully!\n Accuracy: {results[1] * 100}%')
        text_model.save(f'saved_models/{personality_type[_i][0]}{personality_type[_i][1]}_bert_tokenize')
        # print(f'model {personality_type[_i][0]}/{personality_type[_i][1]} saved.\n\n')
    '''
    _ = [0] * 16
    __ = [8 * test_data_labels[0][i] +
          4 * test_data_labels[1][i] +
          2 * test_data_labels[2][i] +
          1 * test_data_labels[3][i] for i in range(len(test_data_labels[0]))]
    print(__)
    for _i in __:
        _[_i] += 1
    print(_)
    '''
