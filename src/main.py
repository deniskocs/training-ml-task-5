import tensorflow as tf
import zipfile
from lxml import etree
from tensorflow import keras

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
        # Использовать 'itertext()' для извлечения всего текста из 'body' без тегов
        content = ''.join(body.itertext())
        return content

def read_fb2_from_zip(zip_path, fb2_file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        print("we here")
        with z.open(fb2_file_name) as f:
            text = f.read().decode('windows-1251')
    parser = FB2Parser(text.encode("windows-1251"))
    return parser.get_content()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(read_fb2_from_zip("/Users/denischilik/repos/trainings/ml/training-ml-task-5/res/lazarevich-knyaz-tmy.fb2.zip", "lazarevich-knyaz-tmy.fb2/1.fb2"))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
