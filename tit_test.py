from servise_ds import okno

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

# исследование данных
class Canaliz:
    def bilet(self, data):
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
        del data['Koltik']

        # средняя цена билета
        # mas = (df['Koltik'] == 1)
        # s1 = df[(df['Pclass'] == 1) & mas]['Fare'].mean()
        # s2 = df[(df['Pclass'] == 2) & mas]['Fare'].mean()
        # s3 = df[(df['Pclass'] == 3) & mas]['Fare'].mean()
        # print (s1, s2, s3)

        #df = df.sort_values(by=['Koltik','Ticket'])
        # rez = df[['Koltik', 'Survived']].groupby(['Koltik']).mean()
        # rez2 = df[['Koltik', 'Survived']].groupby(['Koltik']).count()
        # rez2.rename(columns={'Survived': 'Всего'}, inplace=True)
        # rez = rez.join(rez2)
        # rez = rez.sort_values(by='Survived')
        # print(rez)

    def ot_vozrasta(self, df):
        #df = df[df['Pclass']==3]
        rez = df[['Parch', 'Sex', 'Survived']].groupby(['Parch', 'Sex']).mean()
        rez2 = df[['Parch', 'Sex', 'Survived']].groupby(['Parch', 'Sex']).sum()
        rez2.rename(columns={'Survived': 'Всего'}, inplace=True)
        rez = rez.join(rez2)
        rez = rez.sort_values(by='Survived')
        print(rez)

    def ot_klassa(self, df):
        df['sp'] = df['Sex'].astype(str)+df['Pclass'].astype(str)
        rez = df[['sp', 'Survived']].groupby(['sp']).mean()
        rez2 = df[['sp', 'Survived']].groupby(['sp']).value_counts().groupby(['sp']).sum()
        rez2 = pd.DataFrame({'Всего':rez2})
        rez = rez2.join(rez)
        rez = rez.sort_values(by='Survived')
        print(rez)
        for i in range(0, len(rez)):
            ind = rez.index[i]
            sur = rez['Survived'].iloc[i]
            df.loc[df['sp'] == ind,'Survived'] = df['Survived'] - sur
        return df

    def kabina(self, df):
        df.loc[pd.notna(df['Cabin']), 'Cabin'] = df['Cabin'].str[0]
        data['Cabin'] = data['Cabin'].fillna('T')
        rez = df[['Cabin', 'Survived']].groupby(['Cabin']).mean()
        rez2 = df[['Cabin', 'Pclass']].groupby(['Cabin']).mean()
        rez3 = df[['Cabin','Name']].groupby(['Cabin']).count()
        rez = rez.join(rez2)
        rez = rez.join(rez3)
        rez = rez.sort_values(by='Survived')
        print(rez)

    # средняя цена билета от класса
    def cena_bileta(self, df):
        mas = ((df['SibSp'] + df['Parch']) == 0)
        s1 = df[(df['Pclass'] == 1) & mas].mean()
        s2 = df[(df['Pclass'] == 2) & mas].mean()
        s3 = df[(df['Pclass'] == 3) & mas].mean()
        print (s1, s2, s3)

    def analitika_vozr_iz_imeni(self, df):
        vdf = df[['Name','Sex', 'Age']]
        vdf = vdf[vdf['Sex']=='female']
        maska = pd.notna(vdf['Age'])
        vdf = vdf[maska]
        maska = vdf['Name'].str.find('Mrs.') > 0
        mrdf = vdf[maska]
        notmrdf = vdf[~maska]
        maska = notmrdf['Name'].str.find('Miss.') > 0
        masterdf = notmrdf[~maska]
        print(masterdf.describe())
        print(notmrdf.head(30))
        #Выводы для женщин
        # для 'Mrs.' Age: mean 35, min 14, max 63
        # для 'Miss.' Age: mean 21, min 1, max 63
        # для не 'Mrs.' и не 'Miss.' mean 32, min 24, max 49
        # Выводы для мужчин:
        # для 'Master.' Age: mean 4.574167, max 12
        # для не 'Master.' и не 'Mr.' Age: mean 45, min 23, max 70
        # для 'Mr.' Age: mean 32, min 11, max 80

    # ищем nan
    def nan_ischem(self, data):
        #rez = data[pd.isna(data['Embarked'])]
        rez = data[(data['Pclass'] == 1) & (data['Cabin'].str[0] == 'B')]      # билет 113572
        rez = rez.sort_values(by='Cabin')
        pd.set_option('display.max_columns', 100)
        print(rez.head(30))

analiz = Canaliz()
if __name__ == "__main__":
    data = pd.read_csv("C:\\kaggle\\Титаник\\train.csv")
    analiz.ot_klassa(data)

    y = data['Survived']
    data = data.drop(columns=['Survived'])


    #считываем тестовые данные
    data_test = pd.read_csv('C:\\kaggle\\Титаник\\test.csv')
    data_test.loc[data_test['Fare'].isnull(), 'Fare'] = 8

