import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder # перекодировка категориальных признаков
from sklearn.metrics import accuracy_score, roc_auc_score#, precision_score, recall_score
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from ml_sklearn import bin

# предобработка данных
class Cpredobrabotka:

    # создаем поле - количество чел по 1 билету
    def bilet(self, df):
        ser_kol_tik = df['Ticket'].value_counts()
        x = ser_kol_tik[df['Ticket']]
        l = x.tolist()
        df['Koltik']= l
        return df

    # делаем объединенное поле
    def obedinit(self, data):
        data['Embarked'] = data['Embarked'].fillna('S')
        le = LabelEncoder() # 0.730 - 0.954
        le.fit(data['Embarked'])
        data['Embarked'] = le.transform(data['Embarked'])
        return data

    # цыфруем кабину
    def kabina_cif(self, data):
        data.loc[pd.notna(data['Cabin']), 'Cabin'] = data['Cabin'].str[0]
        data['Cabin'] = data['Cabin'].fillna('T')
        le = LabelEncoder()
        le.fit(data['Cabin'])
        data['Cabin'] = le.transform(data['Cabin'])
        #del data['Cabin']
        return data

    def prise_bilet(self, data):
        ser_kol_tik = data['Ticket'].value_counts()
        x = ser_kol_tik[data['Ticket']]
        l = x.tolist()
        data['Koltik']= l
        maska1 = (data['Pclass'] == 1) & (data['Fare'] > data['Koltik'] * 39.9/2)
        data.loc[maska1, 'Fare'] = data['Fare'] / data['Koltik']
        maska2 = (data['Pclass'] == 2) & (data['Fare'] > data['Koltik'] * 14.21/2)
        data.loc[maska2, 'Fare'] = data['Fare'] / data['Koltik']
        maska3 = (data['Pclass'] == 3) & (data['Fare'] > data['Koltik'] * 8.45/2)
        data.loc[maska3, 'Fare'] = data['Fare'] / data['Koltik']
        del data['Koltik'] #0.827141 - 0.830511

        # разность между ценой и средней ценой билета (вроде не волияет вообще)
        # data.loc[data['Pclass'] == 1, 'Fare'] = data['Fare'] - 39.9  #0.8293882646691635
        # data.loc[data['Pclass'] == 2, 'Fare'] = data['Fare'] - 14.21
        # data.loc[data['Pclass'] == 3, 'Fare'] = data['Fare'] - 8.45

        # то, что выше правильнее, но хуже для результата

        # maska1 = (data['Pclass'] == 1) & (data['Fare'] > 1.6 * 39.9) # 0.88121
        # data.loc[maska1, 'Fare'] = data['Fare'] / (data['SibSp'] + data['Parch'] + 1)
        # maska2 = (data['Pclass'] == 2) & (data['Fare'] > 1.6 * 14.21)
        # data.loc[maska2, 'Fare'] = data['Fare'] / (data['SibSp'] + data['Parch'] + 1)
        # maska3 = (data['Pclass'] == 3) & (data['Fare'] > 1.6 * 8.45)
        # data.loc[maska3, 'Fare'] = data['Fare'] / (data['SibSp'] + data['Parch'] + 1)
        # ????
        #data['Fare'] = data['Fare'] / (data['SibSp'] + data['Parch'] + 1)
        return data

    def vozrast_segment(self, data, x1, x2, x3, x4):
        data.loc[data['Age'] <= x1, 'Age'] = 0
        data.loc[(data['Age'] > x1) & (data['Age'] <= x2), 'Age'] = 1
        data.loc[(data['Age'] > x2) & (data['Age'] <= x3), 'Age'] = 2
        data.loc[(data['Age'] > x3) & (data['Age'] <= x4), 'Age'] = 3
        data.loc[(data['Age'] > x4), 'Age'] = 4
        return data

    def vozrtast(self, data):
        # kaggle 0.7679 № 1  .... 0.74401
        #spis = [-1.0, 5.0, 9.0, 15.0, 31.0, 36.0, 38.0, 40.0, 44.0, 47.0, 55.0, 60.0, 80.0]#0.829388
        # kaggle 0.7488
        spis = [-1.0, 5.0, 13.0, 16.0, 18.0, 30.0, 35.0, 36.0, 39.0, 42.0, 45.0, 47.0, 50.0, 54.0, 60.0, 80.0]


        data['Age'] = pd.cut(data['Age'], bins=spis, labels=False)

        #data = self.vozrast_segment(data, 1, 10, 48, 60) #0.836167290886392

        #data.loc[(data['Age'] < 1), 'Age'] = 1 #0.876; 0.719 - 0.958
        return data

    def vozr_iz_imeni(self, df):
        #df["Len_name"] = df["Name"].str.len()
        mask_nan = pd.isna(df['Age'])

        maska1 = df['Name'].str.find('Mrs.') > 0
        df.loc[maska1, 'Name'] = '4'
        df.loc[maska1 & mask_nan,'Age'] = 35

        maska2 = df['Name'].str.find('Miss.') > 0
        df.loc[maska2, 'Name'] = '3'
        df.loc[maska2 & mask_nan, 'Age'] = 21

        maska3 = df['Name'].str.find('Master.') > 0
        df.loc[maska3, 'Name'] = '2'
        df.loc[maska3 & mask_nan, 'Age'] = 5

        maska4 = df['Name'].str.find('Mr.') > 0
        df.loc[maska4, 'Name'] = '0'
        df.loc[maska4 & mask_nan, 'Age'] = 45

        maska5 = ~maska1 & ~maska2 & ~maska3 & ~maska4

        masmen = maska5 & (df['Sex'] == 'male')
        df.loc[masmen, 'Name'] = '1'
        df.loc[masmen & mask_nan, 'Age'] = 45
        maswum = maska5 & (df['Sex'] == 'female')
        df.loc[maswum, 'Name'] = '5'
        df.loc[maswum & mask_nan, 'Age'] = 32  #0.8874571025115587
        df['Name'].astype(int)
        return df


    # 2-я версия
    def versia2(self, data):
        data = self.obedinit(data)             #0.8770 - 0.87719
        data = self.kabina_cif(data)           #0.886590068
        data = self.prise_bilet(data) # делим цену билета на кол. чел. в семье

        #data.loc[pd.notna(data['Cabin']), 'Cabin'] = data['Cabin'].str[0]
        data = data.drop(columns=['PassengerId', 'Ticket', 'Name'])
        data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
        data = self.vozrtast(data)
        return data

    # для оптимизации
    def opt(self, data, spis):
        data = self.obedinit(data)             #0.8770 - 0.87719
        data = self.kabina_cif(data)           #0.886590068
        data = self.prise_bilet(data) # делим цену билета на кол. чел. в семье

        #data.loc[pd.notna(data['Cabin']), 'Cabin'] = data['Cabin'].str[0]
        data = data.drop(columns=['PassengerId', 'Ticket', 'Name'])
        data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
        data['Age'] = pd.cut(data['Age'], bins=spis, labels=False)
        return data


predobrabotka = Cpredobrabotka()


# предобработка данных
def predobradotka_data(data):
    data = predobrabotka.vozr_iz_imeni(data)
    data = predobrabotka.versia2(data)
    return data
    #bin.gbdt_mod(data, y)

if __name__ == "__main__":
    pd.options.display.width = 0 # вывод данных во всю ширину окна
    data = pd.read_csv("C:\\kaggle\\Титаник\\train.csv")
    #analiz.ot_vozrasta(data)

    # y = data['Survived']
    # data = data.drop(columns=['Survived'])
    y = data.pop('Survived')
    data = predobradotka_data(data) # предобработка данных
    gbdt, m = bin.gbdt_mod(data, y)

    #########CV_gbdt = bin.gbdt_optimizm(data, y)
    #считываем тестовые данные

    data_test = pd.read_csv('C:\\kaggle\\Титаник\\test.csv')
    data_test.loc[data_test['Fare'].isnull(), 'Fare'] = 8
    passenger_id = data_test['PassengerId'] # id пассажиров
    data_test = predobradotka_data(data_test) # предобработка данных
    y_pred_gbdt = gbdt.predict(data_test)
    y_pred_gbdt = pd.DataFrame(y_pred_gbdt, columns=['Survived'])
    y_pred_gbdt['PassengerId'] = passenger_id
    y_pred_gbdt = y_pred_gbdt[['PassengerId', 'Survived']]
    y_pred_gbdt.to_csv('C:\\kaggle\\Титаник\\submission_gbdt.csv', index=None)

