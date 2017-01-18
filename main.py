from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import sys

# path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
# text = open(path).read().lower()
text = open('./textdatasets/tinyshakespeare.txt').read().lower()

# path = 'miron'
# text = open(path).read().lower()
# text = ''.join(ch for ch in text if ch in 'йцукенгшщзхъфывапролджэячсмитьбюё- \n')
print('Corpus length:', len(text))

chars = sorted(list(set(text)))
print('Total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen + 1, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + 1:i + 1 + maxlen])

print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

for i, sentence in enumerate(next_chars):
    for t, char in enumerate(sentence):
        y[i, t, char_indices[char]] = 1

print('Vectorization completed')

# 2 stacked LSTM
print('Creating model...')
model = Sequential()
model.add(LSTM(512, input_dim=len(chars), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.4))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')  # can try out Adam

print('Model ready')
print(model.summary())


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


for iteration in range(1, 51):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(X, y, batch_size=128, nb_epoch=1)

    # saving models at some iterations
    # if iteration==1 or iteration==3 or iteration==5 or iteration==10 or iteration==30 or iteration==50:
    #    model.save_weights('LSTM_weights_'+str(iteration)+'.h5', overwrite=True)

    print('Loss is ', history.history['loss'][0])

    for diversity in [0.2, 0.5, 0.7, 1.0, 1.1, 1.2, 2.0]:
        print()
        print('----- diversity:', diversity)

        seed_string = "hello it is me "
        print("seed string -->", seed_string)
        print('The generated text is')
        sys.stdout.write(seed_string)

        for i in range(320):
            x = np.zeros((1, len(seed_string), len(chars)))
            for t, char in enumerate(seed_string):
                x[0, t, char_indices[char]] = 1.
            preds = model.predict(x)[0]
            next_index = sample(preds[len(seed_string) - 1], diversity)  # np.argmax(preds[len(seed_string) - 1])

            next_char = indices_char[next_index]
            seed_string = seed_string + next_char

            sys.stdout.write(next_char)

        sys.stdout.flush()