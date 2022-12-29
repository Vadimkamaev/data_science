from servise_ds import okno
import cat_encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tkinter import messagebox
import pandas as pd

class Сobrabotka:

    def __init__(self):
        self.progalist = []

    # диалоговое окно
    def quest(self, s):
        return messagebox.askyesno("Вопрос", s)

    # формируем текст программы
    def proga(self, s):
        self.progalist.append(s)
        print(s)

    def print_proga(self):
        for i in self.progalist:
            print(i)

    # формируем раздел импорта
    def importmodul(self):
        self.proga('')
        self.proga('import pandas as pd')
        self.proga('import cat_encoder')
        self.proga('')

    # завершающий код программы
    def finis_proga(self, name_y):
        self.proga('')
        self.proga('def no_optim()')
        self.proga('    train, test = read_file()')
        self.proga('    train = start(train)')
        self.proga('    test = start(test)')
        self.proga('    train, test = encoding(train, name_y, test)')
        self.proga(f"    у = train.pop('{name_y}')")


    # если в y есть null
    def null_in_y(self, name_y):
        sumnul = self.train[name_y].isnull().sum()
        if sumnul != 0:
            print(f'В колонке y усть {sumnul} null. Решите эту проблему до запуска')
            return

    # определяем и удаляем дубликаты
    def dubl_del(self):
        dubli = self.train.duplicated().sum()
        if dubli > 0:
            if self.quest(f"Есть {dubli} дубликатов. Удалить?"):
                self.train.drop_duplicates()
                self.proga('    df.drop_duplicates()')

    # ищем колонку содержащую Id
    def find_id(self):
        sp=self.train.columns
        for col in sp: # цикл по названиям колонок
            if self.train[col].nunique() == self.lendf:
                typ = self.train[col].dtypes
                s = col.lower()  # переводим в нижний регистр
                if s.find('id') > -1: # если в названии колонки есть подстрока id
                    if self.quest(f"Колонка {col} типа {typ} имеет уникальные значения. "
                            f"Сделать её индексом?"):
                        self.train.set_index(col, drop=True, inplace=True)
                        self.proga(f"    df.set_index('{col}', drop=True, inplace=True) # делаем индекс из колонки {col}")
                        return

    # рассматриваем возможность преобразовать колонки типа float в int
    def float_int(self):
        df_float = self.datatypes[(self.datatypes.str[0:5] =='float')
                                  | (self.datatypes.str[0:5] =='Float')] # колонки типа float
        for name, row in df_float.items(): # цикл по колонкам типа float
            ser = self.train[name]
            ser_int = ser.round() # серия округленная до целых
            sl = ((ser_int - ser).abs() < 0.00000000001) # если число близко к целому, то sl = Тrue
            kol_int = sl.sum() # количество близких к целому значений
            proc_int = round(kol_int/self.train[name].notna().sum()*100)
            if proc_int > 70:
                if self.quest(f"В колонке {name} типа {row} {proc_int}% чисел, из тех которые != null, близки к целым. "
                              f"Сделать все числа в этом столбце целыми и преобразовать к целому типу?"):
                    self.train[name] = ser_int
                    self.proga(f"    df['{name}'] = df['{name}'].round() # меняем вещественный тип на целые")
                    pass


    # кодируем категориальные данные и удаляем не нужные колонки
    def encoding(self, name_y):
        drop_list = [] # список колонок для удаления
        encod_list = [] # список колонок для кодирования
        df_kat = self.datatypes[(self.datatypes == 'object') | (self.datatypes == 'category')
                               | (self.datatypes.str[0:3] =='str')]
        for name, row in df_kat.items(): # цикл по колонкам категориального типа
            ser = self.train[name]
            uniq = ser.nunique() # количество уникальных значений в колонке
            unproc = uniq / self.train[name].notna().sum()*100 # процент уникальных значений
            if unproc < 5:
                encod_list.append(name)
            elif unproc < 50:
                if self.quest(f"В колонке {name} типа {row} есть {uniq} уникальных значений. Это "
                              f"{round(unproc)}% от количества значений тех которые != null. "
                              f"Закодировать эти значения?"):
                    encod_list.append(name)
            else:
                if self.quest(f"В колонке {name} типа {row} есть {uniq} уникальных значений. Это "
                              f"{round(unproc)}% от количества значений тех которые != null. "
                              f"Такие колонки надо обрабатывать заранее или удалять. Удалить колонку?"):
                    drop_list.append(name)
        self.train.drop(columns = drop_list, inplace=True)
        self.test.drop(columns=drop_list, inplace=True)
        self.proga(f"    train.drop(columns = '{drop_list}', inplace=True) # удаляем ненужные колонки")
        self.proga(f"    test.drop(columns = '{drop_list}', inplace=True) # удаляем ненужные колонки")
        self.train = self.train.convert_dtypes()
        self.test = self.test.convert_dtypes()
        self.proga('    train = train.convert_dtypes() # переводим колонки в наилучшие возможные типы dtypes')
        self.proga('    test = test.convert_dtypes() # переводим колонки в наилучшие возможные типы dtypes')
        self.proga('    return df')
        self.coding(encod_list, name_y)

    def coding(self, encod_list, name_y):
        self.proga('')
        self.proga('def encoding(train, name_y, test)')
        for name in encod_list:
            self.train, self.test = cat_encoder.coder(self.train, self.test, name, name_y)
            self.proga(f"    train, test = cat_encoder.coder(train, test, '{name}', '{name_y}')"
                       f" # кодируем числами колонку {name}")
        self.proga('    return train, test')


    def init(self):
        self.datatypes = self.train.dtypes  # серия с типами
        self.datatypes = self.datatypes.astype(str) # серия с типами в строковом формате
        self.lendf = len(self.train) # количество строк в датафрейме

    def main_func(self, train, name_y, test):
        self.train = train
        self.test = test
        self.init()
        self.null_in_y(name_y) # если в y есть null
        self.dubl_del() # определяем и удаляем дубликаты
        self.float_int()  # рассматриваем возможность преобразовать столбцы типа float в int
        self.find_id() # ищем колонку содержащую Id
        self.encoding(name_y) # перекодировка

        self.finis_proga(name_y)
        #self.print_proga()

obrabotka = Сobrabotka()

# вызываемая процедура
def start_df(train, name_y, test): # датафрейм содержащий X и y. Имени поля y
    obrabotka.importmodul()
    obrabotka.proga(f"def start(df):")
    obrabotka.main_func(train, name_y, test)

def start_cvs(train_cvs, name_y, test_csv, separator = ','): # имя файла с датафреймом содержащий X и y. Имени поля y
    train = pd.read_csv(train_cvs, sep=separator)
    test = pd.read_csv(test_csv, sep=separator)
    obrabotka.importmodul()
    obrabotka.proga('def read_file(): # загрузка датафреймов из файлов')
    obrabotka.proga(f'    train = pd.read_csv(train_cvs, sep="{separator}")')
    obrabotka.proga(f'    test = pd.read_csv(test_cvs, sep="{separator}")')
    obrabotka.proga('    return train, test')
    obrabotka.proga('')
    obrabotka.proga(f"def start(df):")
    obrabotka.main_func(train, name_y, test)

start_cvs("C:\\kaggle\\Титаник\\train.csv", 'Survived', "C:\\kaggle\\Титаник\\test.csv")



