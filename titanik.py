import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

# def analitika_vozr_iz_imeni():
#     vdf = df[['Name','Sex', 'Age']]
#     vdf = vdf[vdf['Sex']=='female']
#     maska = pd.notna(vdf['Age'])
#     vdf = vdf[maska]
#     maska = vdf['Name'].str.find('Mrs.') > 0
#     mrdf = vdf[maska]
#     notmrdf = vdf[~maska]
#     maska = notmrdf['Name'].str.find('Miss.') > 0
#     masterdf = notmrdf[~maska]
#     print(masterdf.describe())
#     print(notmrdf.head(30))
    #Выводы для женщин
    # для 'Mrs.' Age: mean 35, min 14, max 63
    # для 'Miss.' Age: mean 21, min 1, max 63
    # для не 'Mrs.' и не 'Miss.' mean 32, min 24, max 49
    # Выводы для мужчин:
    # для 'Master.' Age: mean 4.574167, max 12
    # для не 'Master.' и не 'Mr.' Age: mean 45, min 23, max 70
    # для 'Mr.' Age: mean 32, min 11, max 80

def vozr_iz_imeni(df):
    #data["Len_name"] = data["Name"].str.len()
    mask_nan = pd.isna(df['Age'])

    maska1 = df['Name'].str.find('Mrs.') > 0
    df.loc[maska1, 'Name'] ='Mrs'
    df.loc[maska1 & mask_nan,'Age'] = 35

    maska2 = df['Name'].str.find('Miss.') > 0
    df.loc[maska2, 'Name'] = 'Miss'
    df.loc[maska2 & mask_nan, 'Age'] = 21

    maska3 = df['Name'].str.find('Master.') > 0
    df.loc[maska3, 'Name'] = 'Master'
    df.loc[maska3 & mask_nan, 'Age'] = 5

    maska4 = df['Name'].str.find('Mr.') > 0
    df.loc[maska4, 'Name'] = 'Mr'
    df.loc[maska4 & mask_nan, 'Age'] = 45

    maska5 = ~maska1 & ~maska2 & ~maska3 & ~maska4
    df.loc[maska5, 'Name'] = 'NM'
    df.loc[maska5 & mask_nan, 'Age'] = 40
    return data
#    print(df[['Name','Sex', 'Age']].head(30))

# 1-я версия
def versia1(data):

    # это не правильно
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    data.loc[data['Sex'] == 'female', 'Sex'] = 0
    data.loc[data['Sex'] == 'male', 'Sex'] = 1

    # m = pd.get_dummies(data['Embarked'])
    # data = data.drop(columns=['Embarked'])
    # data.join(m)  # 0.8550448103598696, но сошлось

    data['Embarked'] = data['Embarked'].fillna('S')
    le = LabelEncoder()
    le.fit(data['Embarked'])
    data['Embarked'] = le.transform(data['Embarked'])
    return data


# 2-я версия
def versia2(data):
    data['Fare'] = data['Fare'] / (data['SibSp'] + data['Parch'] +1)# без этого 'Fare' 0.00437163
    # это не правильно
    data = data.drop(columns=['PassengerId', 'Ticket', 'Cabin'])
    data.loc[data['Sex'] == 'female', 'Sex'] = 0
    data.loc[data['Sex'] == 'male', 'Sex'] = 1

    data.loc[data['Age'] <=7, 'Age'] = 0
    data.loc[(data['Age'] > 7) & (data['Age'] <= 35), 'Age'] = 1
    data.loc[(data['Age'] > 35) & (data['Age'] <= 100), 'Age'] = 2

    m = pd.get_dummies(data[['Embarked', 'Name']])
    data = data.drop(columns=['Embarked', 'Name'])
    data = data.join(m)  #0.8754966485996073

    # data = data.drop(columns=['Fare', 'Embarked_Q', 'Embarked_C']) # ОПТИМАЛЬНО ПРИ ЛИНЕЙНОЙ МОДЕЛИ

    # data['Embarked'] = data['Embarked'].fillna('Ru')
    # le = LabelEncoder()
    # le.fit(data['Embarked'])
    # data['Embarked'] = le.transform(data['Embarked']) #0.8571480940710041
    return data

data = pd.read_csv("C:\\kaggle титаник\\train.csv")
y = data['Survived']
data = data.drop(columns=['Survived'])
data = vozr_iz_imeni(data)
data = versia2(data)
#обучение линейной модели
# lr = LogisticRegression()
#
# lr.fit(data, y)
# lr_preds = lr.predict(data)
# lr_preds_proba = lr.predict_proba(data)[:, 1]
#
# print('accuracy', accuracy_score(y, lr_preds))
# print('roc_auc', roc_auc_score(y, lr_preds_proba))
#
# cross_val_scores = cross_val_score(lr, data, y, cv=5, scoring='roc_auc')
# pd.set_option('display.max_columns', 50)
# print(data.head(1))
# print(cross_val_scores)
# print(np.mean(cross_val_scores))
# print(lr.coef_) #0.8754966485996073

gbdt = GradientBoostingClassifier()

gbdt.fit(data, y)
gbdt_preds = gbdt.predict(data)
gbdt_preds_proba = gbdt.predict_proba(data)[:, 1]

print('accuracy', accuracy_score(y, gbdt_preds))
print('roc_auc', roc_auc_score(y, gbdt_preds_proba))

cross_val_scores = cross_val_score(gbdt, data, y, cv=5, scoring='roc_auc')
print(cross_val_scores)
print(np.mean(cross_val_scores)) # 0.8829318016505656




# pd.set_option('display.max_rows', 100)

#
#print(data.info())
#
#print(df.head(10))