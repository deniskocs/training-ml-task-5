import tensorflow as tf
import zipfile
from lxml import etree
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU


print("TensorFlow version:", tf.__version__)

class FB2Parser:
    def __init__(self, text):
        self.root = etree.fromstring(text)

    def get_content(self):
        # Найти элемент 'body' в дереве XML
        ns = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
        body = self.root.find('fb:body', ns)
        if body is None:
            raise ValueError("No 'body' in FB2 text")
        # Использовать 'itertext()' для извлечения всего текста из 'section' без тегов
        for section in body:
            content = '\n'.join(section.itertext())
        return content

def read_fb2_from_zip(zip_path, fb2_file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(fb2_file_name) as f:
            text = f.read().decode('windows-1251')
    parser = FB2Parser(text.encode("windows-1251"))
    return parser.get_content()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    book_text = read_fb2_from_zip("/Users/denischilik/repos/trainings/ml/training-ml-task-5/res/lazarevich-knyaz-tmy.fb2.zip", "lazarevich-knyaz-tmy.fb2/1.fb2")
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts([book_text])

    # Преобразование всех строк в токены сразу
    token_sequence = tokenizer.texts_to_sequences([book_text])[0]

    # Создание подпоследовательностей для каждой последовательности токенов
    sequences = []
    window_size = 3
    for i in range(1, len(token_sequence) - window_size):
        sequence = token_sequence[i:i + window_size]
        sequences.append(sequence)

    max_length = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    vocab_size = len(tokenizer.word_index) + 1

    # Создание модели
    model = Sequential()
    model.add(Embedding(vocab_size, 20, input_length=max_length - 1))
    model.add(GRU(20))
    model.add(Dense(vocab_size, activation='softmax'))

    # Компиляция модели
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # Обучение модели
    model.fit(X, y, epochs=1000, batch_size=128)

    # Предсказание следующего слова
    input_text = "Спутник марса неожиданно остановился, движение продолжилось спустя несколько минут"
    for i in range(200):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]

        encoded_text = pad_sequences([encoded_text], maxlen=window_size - 1, padding='pre')
        y_pred = model.predict(encoded_text, verbose=0)
        predicted_class = np.argmax(y_pred, axis=-1)

        for word, index in tokenizer.word_index.items():
            if index == predicted_class:
                input_text += " " + word
                print(input_text)
                break


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
