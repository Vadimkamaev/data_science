import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from catboost import Pool, CatBoostClassifier
import random
from imblearn.under_sampling import RandomUnderSampler

def balance_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    y_pred / np.sum(y_pred, axis=1)[:, None]
    nc = np.bincount(y_true)
    w0, w1 = 1 / (nc[0] / y_true.shape[0]), 1 / (nc[1] / y_true.shape[0])
    logloss = (-w0 / nc[0] * (np.sum(np.where(y_true == 0, 1, 0) * np.log(y_pred[:, 0]))) - w1 / nc[1] * (
        np.sum(np.where(y_true != 0, 1, 0) * np.log(y_pred[:, 1])))) / (w0 + w1)
    return logloss

def load_data():
    global Train, features#, Test, sample_submission #, Train0, greeks
    # Train0 = pd.read_csv('C:\\kaggle\\Возраст\\train.csv')
    Train = pd.read_csv('C:\\kaggle\\Возраст\\train_folds.csv')
    Test = pd.read_csv('C:\\kaggle\\Возраст\\test.csv')
    # greeksdf = pd.read_csv('C:\\kaggle\\Возраст\\greeks.csv')
    # sample_submission = pd.read_csv('C:\\kaggle\\Возраст\\sample_submission.csv')
    Train['EJ'] = Train['EJ'].replace({'A': 0, 'B': 1})
    Test['EJ']  = Test['EJ'].replace({'A': 0, 'B': 1})
    features = Train.columns[1:-2] # список названий колонок трайна

# final_valid_predictions = {}
# final_test_predictions = []  # Это для формирования окончательного результата
def make_kfold():
    global Train
    y = Train['Class']
    kol1 = y.value_counts()[1] # 108
    kol0 = y.value_counts()[0] # 509
    min_kfold = kol0//kol1 # 4
    kol_kfold_big = kol0 - min_kfold * kol1 # 77
    # Создаем kol_kfold_big = 77 штук kfold - ов с количеством элементов min_kfold+1 = 5 и
    # kol1 - kol_kfold_big = 31 штук kfold - ов с количеством элементов min_kfold = 4
    l_kfold = [i//5 for i in range(kol_kfold_big*5)]
    lk = [kol_kfold_big + i//4 for i in range(0, (kol1-kol_kfold_big)*4)]
    l_kfold1 = l_kfold + lk
    Train.reset_index(inplace=True)
    Train.loc[Train['Class']==0, 'kfold'] = l_kfold1
    l_kfold = [i for i in range(kol1)]
    Train.loc[Train['Class'] == 1, 'kfold'] = l_kfold


def one_kfold(train, val, kfold):
    global Train_Next, param, shag

    X_train = train[features]
    y_train = train['Class']

    eval = val  # pd.concat([val, train0.loc[0:nachalo], train0.loc[konec:kol0]])

    train_dataset = Pool(data=X_train, label=y_train, cat_features=["EJ"])
    eval_dataset = Pool(data=eval[features], label=eval['Class'], cat_features=["EJ"])
    if shag == 1:
        stop = 50  # 50 лучше чем 100 и 30 приблизительно так же как 40
    elif shag == 2:
        stop = param  # 3 # 3 может лучше чем 4 может лучше чем 5 лучше чем 10 лучше чем 15
    params = {
        # "iterations": 10000,
        "iterations": 50000,
        "verbose": False,
        "learning_rate": 0.12,
        # "learning_rate": 0.15,
        "depth": 4,
        'auto_class_weights': 'Balanced',
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass:use_weights=False',
        'early_stopping_rounds': stop
    }

    model = CatBoostClassifier(**params)
    model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True)
    # model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True, verbose_eval=100)

    preds_valid = model.predict_proba(val[features])
    preds_valid1 = model.predict(val[features])

    # Этот фрамент для формирования окончательного результата
    # preds_test  = model.predict_proba(Test[features])
    # final_test_predictions.append(preds_test)

    pred = preds_valid[:, 0]
    # pred0 = preds_valid1

    Train_Next.loc[Train['kfold'] == kfold, shag] = list(pred)
    preds_valid[:, 0] = pred
    preds_valid[:, 1] = 1 - pred

    # preds_valid[:,1]=pred
    # preds_valid[:,0]=1-pred

    logloss = log_loss(val['Class'], preds_valid)
    blogloss = balance_logloss(val['Class'], preds_valid)

    return logloss, blogloss

def one_prohod():
    global Train, features, shag, param, rezult
    # списки оцнеки ошибок
    list_logloss  = [] # logloss
    list_blogloss = [] # blogloss
    final_test_pred =[]
    kfold=0
    while True:
        print('Fold:' + str(kfold) + '=>', end='')
        # train = Train[Train['kfold'] != kfold].reset_index(drop=True)
        # val = Train[Train['kfold'] == kfold].reset_index(drop=True)  # валидационный датасет
        train = Train[Train['kfold'] != kfold]
        val = Train[Train['kfold'] == kfold] # валидационный датасет
        if len(val.index) == 0:
            break
        logloss, blogloss = one_kfold(train, val, kfold)
        list_logloss.append(logloss)
        list_blogloss.append(blogloss)
        if kfold % 10 == 0:
            print()
            print('Log loss mean:', np.mean(list_logloss), 'Balance Log loss mean:', np.mean(list_blogloss))
        kfold+=1

#     print('Log loss:', list_logloss)
#     print('Balance Log loss:', list_blogloss)
    print()
    print('ИТОГО: ====>  Log loss mean:', np.mean(list_logloss), 'Balance Log loss mean:', np.mean(list_blogloss), '------')
    rezult.loc[len(rezult.index)] = [shag, param, np.mean(list_logloss), np.mean(list_blogloss)]

from datetime import datetime
def greek_times():
    global greeksdf, Train, features
    # Greeks содержат информацию о времени, которую мы можем использовать, нам просто нужно разобрать ее на int/nan.
    greeksdf = pd.read_csv('C:\\kaggle\\Возраст\\greeks.csv')
    times = greeksdf.Epsilon.copy()
    times[greeksdf.Epsilon != 'Unknown'] = greeksdf.Epsilon[greeksdf.Epsilon != 'Unknown'].map(lambda x: datetime.strptime(x,'%m/%d/%Y').toordinal())
    times[greeksdf.Epsilon == 'Unknown'] = np.nan
    Train = pd.concat((Train, times), axis=1)
    features = list(features)
    features.append('Epsilon')


random.seed(1)
load_data()
greek_times()
make_kfold()
rezult = pd.DataFrame(columns=['shag', 'param', 'Log_loss', 'Balance'])
# Train0 = Train.copy()
features = list(features)
# features0 = features.copy()
param = 0
shag = 1
print('------------------------ param =', param, '--------------------------!!!!!!!!!!!!!!')
Train_Next = Train.copy()
Train_Next[1] = 0
one_prohod()
features.append(shag)
Train = Train_Next.copy()
Train0 = Train.copy()
features0 = features.copy()
shag = 2
for param in range(50, 51, 1):
    print('------------------------ param =', param, '--------------------------!!!!!!!!!!!!!!')
    Train_Next = Train0.copy()
    features = features0.copy()
    Train_Next[2] = 0
    one_prohod()

rezult.sort_values(by='Log_loss', inplace=True)
print(rezult[rezult['shag'] == 1].head(40))
print('2-й шаг')
rezult.sort_values(by='Balance', inplace=True)
print(rezult[rezult['shag'] == 2].head(40))