from tkinter import *
from tkinter import font
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

class Okno:

    def printinfo(self):
        print(self.df.info())

    def printhead(self):
        print(self.df.head())

    def printdescribe(self):
        print(self.df.describe())

    def printdtypes(self):
        print(self.df.dtypes.value_counts())

    def printcolumns(self):
        print(self.df.columns.tolist())

    def printnool(self):
        vser = self.df.isnull().sum()
        vser = vser[vser !=0]
        l = len(self.df)
        vdf = pd.DataFrame({'Кол null' : vser, '% null' : vser/l*100, 'Не null' : vser-l})
        print(vdf)
        print(f'Всего строк в DataFrame {l}')

    # колонки целого типа
    def printintcolumn(self):
        l = len(self.df)
        datatypes = self.df.dtypes
        datatypes = datatypes.astype(str)
        datatypes = datatypes[(datatypes.str[0:3] =='int') | (datatypes.str[0:3] =='Int')]
        vdf = self.df[datatypes.index]
        datamax = vdf.max()
        datamin = vdf.min()
        datanunique = vdf.nunique()
        datanull = vdf.isnull().sum()
        vdf = pd.DataFrame({'min': datamin, 'max': datamax, 'nunique' : datanunique, 'not null': l-datanull,
                            'null' : datanull})
        print(f'Колонки целого типа. Всего строк в DataFrame {l}')
        print(vdf)

    # колонки типа object с категориальными признаками
    def printobjectcolumn(self):
        l = len(self.df)
        datatypes = self.df.dtypes
        datatypes = datatypes.astype(str)
        datatypes = datatypes[(datatypes == 'object') | (datatypes == 'category')
                              | (datatypes.str[0:3] =='str')]
        vdf = self.df[datatypes.index]
        datanunique = vdf.nunique()
        datanull = vdf.isnull().sum()
        vdf = pd.DataFrame({'nunique': datanunique, 'not null': l-datanull,'null': datanull})
        print(f'Колонки типа object. Всего строк в DataFrame {l}')
        print(vdf)

    # колонки вещественного типа
    def printfloatcolumn(self):
        l = len(self.df)
        datatypes = self.df.dtypes
        datatypes = datatypes.astype(str)
        datatypes = datatypes[(datatypes.str[0:5] =='float') | (datatypes.str[0:5] =='Float')]
        vdf = self.df[datatypes.index]
        datamax = vdf.max()
        datamin = vdf.min()
        datanunique = vdf.nunique()
        datanull = vdf.isnull().sum()
        rdf = pd.DataFrame({'min': datamin, 'max': datamax, 'nunique': datanunique, 'not null': l-datanull,
                            'null': datanull, '~int' : 0})
        for label, ser in vdf.items():
            s = ser.round()
            sl = ((s-ser).abs() > 0.00000000001)
            ll = sl.value_counts()[False]
            rdf.at[label, '~int'] = ll
        print(f'Колонки вещественного типа. Всего строк в DataFrame {l}')
        print(rdf)

    # начальная инфомация на экране
    def infostart(self):
        try:
            sum_dup = self.df.duplicated().sum()
        except:
            sum_dup = 0
        Label(text=f"Размер shape = {self.df.shape}").pack()
        Label(text=f"Дубликаты duplicated().sum() = {sum_dup}").pack()
        Label(text=f"Пустоты isnull().sum().sum() = {self.df.isnull().sum().sum()}").pack()
        Label(text=f"Поле № 1. Ось x").pack()
        self.kolonka1 = Entry(width=45)
        self.kolonka1.pack()
        Label(text=f"Поле № 2. Ось y").pack()
        self.kolonka2 = Entry(width=45)
        self.kolonka2.pack()

    # гистограмма поля № 1
    def hist_ogramma(self):
        nam = self.kolonka1.get()
        if nam not in self.df.columns:
            self.printcolumns()
            print('В поле ввода "Поле №1" введена строка не идентичная имени колонки')
        else:
            self.df[nam].hist()
            plt.show()

    # Зависимость поля № 1 от индекса. Линейный график plot
    def plot_ogramma(self):
        nam = self.kolonka1.get()
        if nam not in self.df.columns:
            self.printcolumns()
            print('В поле ввода "Поле №1" введена строка не идентичная имени колонки')
        else:
            self.df[nam].plot()
            plt.show()

    # математическая диаграмма, изображающая значения двух переменных в виде точек на плоскости
    def scatter_plot(self):
        nam1 = self.kolonka1.get()
        if nam1 not in self.df.columns:
            self.printcolumns()
            print('В поле ввода "Поле № 1" введена строка не идентичная имени колонки')
        else:
            nam2 = self.kolonka2.get()
            if nam2 not in self.df.columns:
                self.printcolumns()
                print('В поле ввода "Поле № 2" введена строка не идентичная имени колонки')
            else:
                self.df.plot.scatter(x = nam1, y = nam2)
                plt.show()

    # математическая диаграмма, изображающая значения двух переменных в виде точек на плоскости
    def scatter_plot_groupby_x(self):
        nam1 = self.kolonka1.get()
        if nam1 not in self.df.columns:
            self.printcolumns()
            print('В поле ввода "Поле № 1" введена строка не идентичная имени колонки')
        else:
            nam2 = self.kolonka2.get()
            if nam2 not in self.df.columns:
                self.printcolumns()
                print('В поле ввода "Поле № 2" введена строка не идентичная имени колонки')
            else:
                df_groupby = self.df[[nam1, nam2]].groupby(nam1, as_index=False).mean()
                df_groupby.plot.scatter(x = nam1, y = nam2)
                plt.show()

    # математическая диаграмма, изображающая значения двух переменных в виде точек на плоскости
    def scatter_plot_groupby_y(self):
        nam1 = self.kolonka1.get()
        if nam1 not in self.df.columns:
            self.printcolumns()
            print('В поле ввода "Поле № 1" введена строка не идентичная имени колонки')
        else:
            nam2 = self.kolonka2.get()
            if nam2 not in self.df.columns:
                self.printcolumns()
                print('В поле ввода "Поле № 2" введена строка не идентичная имени колонки')
            else:
                df_groupby = self.df[[nam1, nam2]].groupby(nam2, as_index=False).mean()
                df_groupby.plot.scatter(x = nam1, y = nam2)
                plt.show()

    # диаграмма, изображающая значения двух переменных в виде линии на плоскости
    def line_plot(self):
        nam1 = self.kolonka1.get()
        if nam1 not in self.df.columns:
            self.printcolumns()
            print('В поле ввода "Поле № 1" введена строка не идентичная имени колонки')
        else:
            nam2 = self.kolonka2.get()
            if nam2 not in self.df.columns:
                self.printcolumns()
                print('В поле ввода "Поле № 2" введена строка не идентичная имени колонки')
            else:
                self.df.plot(x = nam1, y = nam2)
                plt.show()

    # под меню grafik_menu
    def pod_grafik_menu(self, grafik_menu):
        grafik_menu.add_command(label="Гистограмма поля № 1 hist", command=self.hist_ogramma)
        grafik_menu.add_command(label="Зависимость поля № 1 от индекса plot", command=self.plot_ogramma)
        grafik_menu.add_command(label="Зависимость 2-х полей. plot.skatter", command=self.scatter_plot)
        grafik_menu.add_command(label="Зависимость 2-х полей. groupby по x", command=self.scatter_plot_groupby_x)
        grafik_menu.add_command(label="Зависимость 2-х полей. groupby по y", command=self.scatter_plot_groupby_y)
        grafik_menu.add_command(label="Зависимость 2-х полей. plot.line", command=self.line_plot)

    # под меню otchet_menu
    def pod_otchet_menu(self, otchet_menu):
        otchet_menu.add_command(label="Все колонки .columns.tolist", command=self.printcolumns)
        otchet_menu.add_command(label="null", command=self.printnool)
        otchet_menu.add_separator()
        otchet_menu.add_command(label="Типы столбцов .dtypes.value_count", command=self.printdtypes)
        otchet_menu.add_command(label="Колонки целого типа", command=self.printintcolumn)
        otchet_menu.add_command(label="Колонки строкового типа", command=self.printobjectcolumn)
        otchet_menu.add_command(label="Колонки вещественного типа", command=self.printfloatcolumn)
        otchet_menu.add_separator()
        #otchet_menu.add_command(label="Стандартные методы DataFrame")
        otchet_menu.add_command(label="DataFrame.info", command=self.printinfo)
        otchet_menu.add_command(label="Голова .head", command=self.printhead)
        otchet_menu.add_command(label="Описать .describe", command=self.printdescribe)

    # создаем главное меню
    def create_menu(self):
        self.mainmenu = Menu(self.root, font="Arial 10")
        otchet_menu = Menu(self.mainmenu, tearoff=0, font="Arial 10")
        grafik_menu = Menu(self.mainmenu, tearoff=0, font="Arial 10")
        self.root.config(menu=self.mainmenu)
        self.mainmenu.add_cascade(label='Операции с df', menu=otchet_menu)
        self.mainmenu.add_cascade(label='Графики', menu=grafik_menu)
        self.pod_otchet_menu(otchet_menu)
        self.pod_grafik_menu(grafik_menu)

    # запускаем главное окно программы
    def start_okno(self):
        pd.options.display.width = 0
        self.root = Tk()  # окно
        self.sw = self.root.winfo_screenwidth()  # считывание размеров экрана
        self.sh = self.root.winfo_screenheight()
        self.root.geometry('%dx%d+%d+%d' % (self.sw//4, self.sh//3, 0, 0))  # размеры окна установить
        #Text(self.root, font="Arial 12")
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        self.root.title("DataFrame")
        self.create_menu()
        self.infostart()
        self.root.mainloop()

    def csv(self, s, separator = ','): # просмотр датафрейма из файла с переданным в параметрах именем
        self.df = pd.read_csv(s, sep =separator)
        self.start_okno()

    def vewdf(self, df): # просмотр передаваемого в параметрах датафрейма
        self.df = df
        self.start_okno()

okno = Okno()
if __name__ == "__main__":
    #okno.csv("C:\\kaggle\\Титаник\\train.csv")
    #okno.csv("C:\\kaggle\\Титаник\\test.csv")
    #okno.csv("C:\\kaggle\\белки\\test1.csv")
    #okno.csv("C:\\kaggle\\белки\\train_updates_20220929.csv")
    #okno.csv("C:\\kaggle\\белки\\supertext.csv")
    okno.csv("C:\\kaggle\\МикроБизнес\\revealed_test.csv")


