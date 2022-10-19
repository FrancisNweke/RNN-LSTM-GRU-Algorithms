import pandas as pd
from tensorflow.python.keras.layers import LSTM, Embedding, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import Input
from keras.losses import BinaryCrossentropy
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
import re
import string
import nltk

nltk.download('stopwords')

# region Load train and test datasets
train_dataset = pd.read_csv('data/train.csv')
test_dataset = pd.read_csv('data/test.csv')

print(f'Train Data Shape: {train_dataset.shape}')
print(f'Test Data Shape: {test_dataset.shape}\n')

print(train_dataset.head())  # visualize train data
print(test_dataset.head())  # visualize test data


# endregion

# region Data Preprocessing
def RemoveURL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


# https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022
def RemovePunct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


print(f'\nString Punctuations: {string.punctuation}\n')

pattern = re.compile(r"https?://(\S+|www)\.\S+")
for t in test_dataset['text']:
    matches = pattern.findall(t)
    for match in matches:
        print(t)
        print(match)
        print(pattern.sub(r"", t))
    if len(matches) > 0:
        break

# Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine
# has been programmed to ignore, both when indexing entries for searching and when retrieving them
# as the result of a search query.
stop = set(stopwords.words("english"))


# https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
def RemoveStopword(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


print(f'\nEnglish Stop Words: {stop}')

train_dataset['text'] = train_dataset['text'].map(RemoveURL)
train_dataset['text'] = train_dataset['text'].map(RemovePunct)
train_dataset['text'] = train_dataset['text'].map(RemoveStopword)

test_dataset['text'] = test_dataset['text'].map(RemoveURL)
test_dataset['text'] = test_dataset['text'].map(RemovePunct)
test_dataset['text'] = test_dataset['text'].map(RemoveStopword)

print(train_dataset['text'])
print(test_dataset['text'])


# Count unique words
def WordCounter(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


train_counter = WordCounter(train_dataset['text'])
test_counter = WordCounter(test_dataset['text'])

print(f'\nMost common words in train dataset: {train_counter.most_common(5)}')
print(f'Most common words in test dataset: {test_counter.most_common(5)}')

num_unique_words_train = len(train_counter)
num_unique_words_test = len(test_counter)

train_X, eval_X, train_y, eval_y = train_test_split(train_dataset['text'], train_dataset['target'], train_size=0.8)

print(train_X.shape)

# vectorize a text corpus by turning each text into a sequence of integers
tokenizer = Tokenizer(num_words=num_unique_words_train)
tokenizer.fit_on_texts(train_X)  # fit only to training

# each word has unique index
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_X)
eval_sequences = tokenizer.texts_to_sequences(eval_X)
test_sequences = tokenizer.texts_to_sequences(test_dataset['text'])

# Max number of words in a sequence
max_length = 20

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
eval_padded = pad_sequences(eval_sequences, maxlen=max_length, padding="post", truncating="post")

# Check reversing the indices

# flip (key, value)
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


decoded_text = decode(train_sequences[10])
print(f'{train_sequences[10]} => {decoded_text}')
# endregion

# region Train LSTM Model
# Embedding: https://www.tensorflow.org/tutorials/text/word_embeddings
# Turns positive integers (indexes) into dense vectors of fixed size. (other approach could be one-hot-encoding)

# Word embeddings give us a way to use an efficient, dense representation in which similar words have
# a similar encoding. Importantly, you do not have to specify this encoding by hand. An embedding is a
# dense vector of floating point values (the length of the vector is a parameter you specify).

input_layer = Input(shape=max_length)
embed_layer = Embedding(num_unique_words_train, 32, input_length=max_length)(input_layer)

# The layer will take as input an integer matrix of size (batch, input_length),
# and the largest integer (i.e. word index) in the input should be no larger than num_words (vocabulary size).
# Now model.output_shape is (None, input_length, 32), where `None` is the batch dimension.


lstm_layer = LSTM(64, dropout=0.1)(embed_layer)
fc_layer = Dense(1, activation='sigmoid')(lstm_layer)

model = Model(input_layer, fc_layer)

model.summary()

# optim = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss=BinaryCrossentropy(from_logits=False), optimizer='Adam', metrics=['accuracy'])

model.fit(train_padded, train_y, epochs=20, validation_data=(eval_padded, eval_y), verbose=2)
# endregion

# region Run test
tokenizer_test = Tokenizer(num_words=num_unique_words_test)
tokenizer_test.fit_on_texts(test_dataset['text'])  # fit only to testing

# each word has unique index
word_index_test = tokenizer.word_index

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

test_decoded = decode(test_sequences[11])
print(f'{test_sequences[11]} => {test_decoded}')

predictions = model.predict(test_padded)
predictions = [1 if p > 0.5 else 0 for p in predictions]

print(predictions[0:20])
# endregion
