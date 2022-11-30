import pandas as pd
import titanik
import tit_test
from ml_sklearn import bin

# разделение датафрейма на части, на основании значений столбца
class Cdelimdf:
# параметр - датафрейм с 2-я столбцами, делим 1-й столбец на основании 2-го по признаку
# максимального различия средних значений во втором столбце в смежных отрезках
# параметр kol_int - количество интервалов на которое делим
# параметр left минимальный отступ с начала, от минимума
# minval - минимальное количество элементов во внутреннем отрезке
    def spisok_interval(self, df, kol_int, minval):
        col = df.columns # индекс с именами колонок
        name1 = col[0] # имя первого столбца
        name2 = col[1] # имя второго столбца
        #df.sort_values(by=[name1], inplace=True) # сортируем по первому столбцу
        max = df[name1].max() # максимальное значение во втором столбце
        min = df[name1].min() # минимальное значение во втором столбце
        interval = (max-min)//kol_int # размер интервала при равномерном разделении датафрейма
        self.rez =[i*interval//2 + (min+max)//4 for i in range(0, kol_int+1)] # результат разбиения до оптимизации
        self.rez[kol_int] = max
        self.rez[0] = min-1
        self.kol_elem = kol_int * [0] # отладочное, количество элементов в отрезке
        self.mean = kol_int * [0]
        for z in range(0, 50):
            izm = 0  # количество изменений при оптимизации
            # цикл от 1 до предпоследнего элемента self.rez с оптимизацией каждого элемента;
            # нулевой и последний это границы интервала и не оптимизируются
            for i in range(1, kol_int):
                rez = self.optimiz(df, i, name1, name2, minval) # оптимизируем i-ю точку разбиения
                if self.rez[i] != rez: # i-я точка при оптимизации изменилась
                    self.rez[i] = rez
                    izm += 1
            if izm == 0:
                #print(pd.DataFrame({'интерв от':self.rez[:-1], 'до':self.rez[1:], 'кол. элем': self.kol_elem, 'средняя вел.': self.mean}))
                #print(self.mean)
                return self.rez
        #print(self.kol_elem)
        #print(self.mean)
        return self.rez # возвращаем список с интервалами для деления 1-го столбца
        #df['num_n1'] = pd.cut(df['name1'], bins=self.rez, labels=False)

    # оптимизируем i-ю точку разбиения
    def optimiz(self, df, i, name1, name2, minval):
        maxrazn = 0
        maxj = self.rez[i]
        j = self.rez[i - 1]
        while j < self.rez[i + 1] - 1:
            j += 1
            # признак нахождения элемента в интервале от i - 1 до j элемента
            maska1 = (df[name1] > self.rez[i - 1]) & (df[name1] <= j)
            kol1 = maska1.sum() # количество элементов в интервале
            if kol1 < minval: # если количество меньше минимально допустимого
                continue
            if (kol1 < 5 * minval) & (i > 1):  # если количество меньше минимально допустимого *5
                maska1 = (df[name1] > self.rez[i - 2]) & (df[name1] <= j)
                if (maska1.sum() < 5 * minval) & (i > 2):  # если количество меньше минимально допустимого *5
                    maska1 = (df[name1] > self.rez[i - 3]) & (df[name1] <= j)
                    if (maska1.sum() < 5 * minval) & (i > 3):  # если количество меньше минимально допустимого *5
                        maska1 = (df[name1] > self.rez[i - 4]) & (df[name1] <= j)

            #признак нахождения элемента в интервале от j до i+1 элемента
            maska2 = (df[name1] > j) & (df[name1] <= self.rez[i + 1])
            kol2 = maska2.sum() # количество элементов в интервале
            if kol2 < minval: # если количество меньше минимально допустимого
                continue
            if (kol2 < 5 * minval) & (i < len(self.rez)-2) :  # если количество меньше минимально допустимого *5
                maska2 = (df[name1] > j) & (df[name1] <= self.rez[i + 2])
                if (maska2.sum() < 5 * minval) & (i < len(self.rez)-3):  # если количество меньше минимально допустимого
                    maska2 = (df[name1] > j) & (df[name1] <= self.rez[i + 3])
                    if (maska2.sum() < 5 * minval) & (i < len(self.rez) - 4):  # если количество меньше минимально допустимого
                        maska2 = (df[name1] > j) & (df[name1] <= self.rez[i + 4])
            mean1 = df[maska1][name2].mean()
            mean2 = df[maska2][name2].mean()
            razn = abs(mean1 - mean2) # разность между средним значением предыдущего и последующего отрезков
            if razn > maxrazn:
                maxrazn = razn
                maxj = j
                self.kol_elem[i-1] = kol1
                self.kol_elem[i] = kol2
                self.mean[i-1] = mean1
                self.mean[i] = mean2
        return maxj

    def optimizac(self):
        dfmain = pd.read_csv("C:\\kaggle\\Титаник\\train.csv")
        dfmain.loc[(dfmain['Age'] < 1), 'Age'] = 0
        dfmain = dfmain[pd.notnull(dfmain['Age'])]
        kol_str = len(dfmain)
        pd.options.display.width = 0  # вывод данных во всю ширину окна
        kol_rez = 15 # количество результатов
        ps = ['0']*kol_rez
        rezdf = pd.DataFrame({'max': [0.0]*kol_rez, 'кол инт': ps, 'мин кол': ps, 'rez' : ps})
        for i in range(10,26):
            j=10
            while j*i < kol_str*0.7:
                j+=1
                df = dfmain.copy()
                rez = delimdf.spisok_interval(df[['Age', 'Survived']], i, j)
                y = df.pop('Survived')
                df = titanik.predobrabotka.opt(df, rez)  # предобработка данных
                gbdt, m = bin.gbdt_mod(df, y, 1)
                maska = (rezdf['max'] == 0)
                if maska.sum() !=0:
                    for z in range(0,kol_rez):
                        if rezdf.iloc[z,0] ==0:
                            rezdf.iloc[z, 0] = m
                            rezdf.iloc[z, 1] = i
                            rezdf.iloc[z, 2] = j
                            s = ', '.join(map(str, rez))
                            rezdf.iloc[z, 3] = s
                            break
                else:
                    mal = rezdf['max'].min()
                    if mal < m:
                        for z in range(0, kol_rez):
                            if rezdf.iloc[z, 0] < m:
                                rezdf.iloc[z, 0] = m
                                rezdf.iloc[z, 1] = i
                                rezdf.iloc[z, 2] = j
                                s = ', '.join(map(str, rez))
                                rezdf.iloc[z, 3] = s
                                break
            print(i)
            rezdf.sort_values(by=['max'], inplace=True)  # сортируем по первому столбцу
            print(rezdf)
        for z in range(0,kol_rez):
            print('accuracy=',rezdf.iloc[z, 0], 'кол. интервалов=', rezdf.iloc[z, 1], 'мин. разм. инт.', rezdf.iloc[z, 2])
            print(rezdf.iloc[z, 3])


delimdf = Cdelimdf()
if __name__ == "__main__":
    # delimdf.optimizac()

    df = pd.read_csv("C:\\kaggle\\Титаник\\train.csv")
    df.loc[(df['Age'] < 1), 'Age'] = 0
    df = df[pd.notnull(df['Age'])]
    df = tit_test.analiz.ot_klassa(df)
    rez = delimdf.spisok_interval(df[['Age','Survived']], 12, 20)
    print(rez)
    print('Кол. строк', len(df))
