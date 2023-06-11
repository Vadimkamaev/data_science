import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from catboost import Pool, CatBoostClassifier
import random

import warnings
warnings.filterwarnings('ignore')

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
    global Train, param
    # valid_ids = val.Id.values.tolist()  # список ID валидационного датасета
    # kol1 = y.value_counts()[1]
    # kol0 = y.value_counts()[0]

    X_train = train[features]
    y_train = train['Class']

    eval = val #pd.concat([val, train0.loc[0:nachalo], train0.loc[konec:kol0]])

    train_dataset = Pool(data=X_train, label=y_train, cat_features=["EJ"])
    eval_dataset = Pool(data=eval[features], label=eval['Class'], cat_features=["EJ"])

    params = {
        # "iterations": 10000,
        "iterations": 50000,
        "verbose": False,
        "learning_rate": 0.16,
        "depth": 4,
        'auto_class_weights': 'Balanced',
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass:use_weights=False',
        'early_stopping_rounds': 50,
    }
    model = CatBoostClassifier(**params)
    model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True)
    # model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True, verbose_eval=100)

    preds_valid = model.predict_proba(val[features])
    preds_valid1 = model.predict(val[features])

    # Этот фрамент для формирования окончательного результата
    # preds_test  = model.predict_proba(Test[features])
    # final_test_predictions.append(preds_test)

    pred = preds_valid[:,0]
    # pred0 = preds_valid1

    Train.loc[Train['kfold'] == kfold, 'pred'] = list(pred)
    preds_valid[:,0]=pred
    preds_valid[:,1]=1-pred


    # preds_valid[:,1]=pred
    # preds_valid[:,0]=1-pred

    logloss = log_loss(val['Class'], preds_valid)
    blogloss = balance_logloss(val['Class'], preds_valid)

    return logloss, blogloss

def one_prohod():
    global Train, features, shag, param
    # списки оцнеки ошибок
    list_logloss  = [] # logloss
    list_blogloss = [] # blogloss
    kfold=0
    while True:
        print('Fold: ' + str(kfold) + '==>', end='')
        # train = Train[Train['kfold'] != kfold].reset_index(drop=True)
        # val = Train[Train['kfold'] == kfold].reset_index(drop=True)  # валидационный датасет
        train = Train[Train['kfold'] != kfold]
        val = Train[Train['kfold'] == kfold] # валидационный датасет
        if len(val.index) == 0:
            break
        logloss, blogloss = one_kfold(train, val, kfold)
        list_logloss.append(logloss)
        list_blogloss.append(blogloss)
        print(kfold, logloss, blogloss)
        kfold+=1
        # if kfold == 3:
        #     break

    print('Log loss:', list_logloss)
    print('Balance Log loss:', list_blogloss)
    print('Log loss mean:', np.mean(list_logloss), 'Balance Log loss mean:', np.mean(list_blogloss))
    rezult.loc[len(rezult.index)] = [1, param, np.mean(list_logloss), np.mean(list_blogloss)]

from datetime import datetime
def greek_times():
    global greeksdf, Train, features
    # Greeks содержат информацию о времени, которую мы можем использовать, нам просто нужно разобрать ее на int/nan.
    greeksdf = pd.read_csv('C:\\kaggle\\Возраст\\greeks.csv')
    times = greeksdf.Epsilon.copy()
    times[greeksdf.Epsilon != 'Unknown'] = greeksdf.Epsilon[greeksdf.Epsilon != 'Unknown'].map(lambda x: datetime.strptime(x,'%m/%d/%Y').toordinal())
    times[greeksdf.Epsilon == 'Unknown'] = np.nan
    Train = pd.concat((Train, times), 1)
    features = list(features)
    features.append('Epsilon')

random.seed(1)
load_data()
# preds_valid = np.zeros((len(Train.index), 2))
# rezult = pd.DataFrame(columns=['pred', 'Log_loss', 'Balance'])
# for pred in range(17000,18000,1): # 0.17504 - идеальная середина
#     pred = pred/100000
#     preds_valid[:,0]=pred
#     preds_valid[:,1]=1-pred
#     logloss = log_loss(Train['Class'], preds_valid)
#     blogloss = balance_logloss(Train['Class'], preds_valid)
#     rezult.loc[len(rezult.index)] = [pred, logloss, np.mean(blogloss)]
# rezult.sort_values(by='Balance', inplace=True)
# print(rezult.head(40))


greek_times()
make_kfold()
rezult = pd.DataFrame(columns=['shag', 'param', 'Log_loss', 'Balance'])
Train['pred'] = 0
Train0 = Train.copy()
features = list(features)
sum_preds_valid = []
for param in range(1,2,1):
    # random.shuffle(features)
    print('------------------------ param =', param, '--------------------------!!!!!!!!!!!!!!')
    one_prohod()
rezult.sort_values(by='Log_loss', inplace=True)
print(rezult[rezult['shag']==1].head(40))

preds_valid = np.zeros((len(Train.index), 2))
print('!!!!---------------- ИЩЕМ ОПТИМАЛЬНЫЙ ПОРОГ СНИЗУ -----------------!!!!!!!!!!!!!!')
rezult = pd.DataFrame(columns=['param', 'Log_loss', 'Balance'])
for param in range(0,1005,5):
    param = param/1000
    preds = Train['pred'].copy()
    preds[preds < param]=0
    # preds_valid[:, 1] = preds
    # preds_valid[:, 0] = 1 - preds

    preds_valid[:, 1] = 1 - preds
    preds_valid[:, 0] = preds
    # tr_class = Train['Class']
    logloss = log_loss(Train['Class'], preds_valid)
    blogloss = balance_logloss(Train['Class'], preds_valid)
    rezult.loc[len(rezult.index)] = [param, logloss, np.mean(blogloss)]
rezult.sort_values(by='Log_loss', inplace=True)
print(rezult.head(60))

print('!!!!---------------- ИЩЕМ ОПТИМАЛЬНЫЙ ПОРОГ СВЕРХУ -----------------!!!!!!!!!!!!!!')
rezult = pd.DataFrame(columns=['param', 'Log_loss', 'Balance'])
for param in range(0,1005,5):
    param = param/1000
    preds = Train['pred'].copy()
    preds[preds > 1 - param]=1
    # preds_valid[:, 1] = preds
    # preds_valid[:, 0] = 1 - preds

    preds_valid[:, 1] = 1 - preds
    preds_valid[:, 0] = preds
    # tr_class = Train['Class']
    logloss = log_loss(Train['Class'], preds_valid)
    blogloss = balance_logloss(Train['Class'], preds_valid)
    rezult.loc[len(rezult.index)] = [param, logloss, np.mean(blogloss)]
rezult.sort_values(by='Log_loss', inplace=True)
print(rezult.head(60))

rezult = pd.DataFrame(columns=['param', 'Log_loss', 'Balance'])
preds0 = Train['pred'].copy()
preds0[preds0 > 1 - 0.205] = 1 # ПОРОГ СВЕРХУ
preds0[preds0 < 0.17] = 0 # ПОРОГ СНИЗУ
preds_valid[:, 1] = 1 - preds0
preds_valid[:, 0] = preds0
# tr_class = Train['Class']
logloss = log_loss(Train['Class'], preds_valid)
blogloss = balance_logloss(Train['Class'], preds_valid)
print('при порогах снизу 0.17 и сверху 0.205')
print('logloss=', logloss, 'blogloss=', blogloss)

print('!!!!---------------- ИЩЕМ ОПТИМАЛЬНЫЙ СЕРЕДИННЫЙ ПОРОГ -----------------!!!!!!!!!!!!!!')
preds = preds0.copy()
preds[(preds > 0) & (preds < 1)] =0.17504
preds_valid[:, 1] = 1 - preds
preds_valid[:, 0] = preds
# tr_class = Train['Class']
logloss = log_loss(Train['Class'], preds_valid)
blogloss = balance_logloss(Train['Class'], preds_valid)
print('при порогах снизу 0.17 и сверху 0.205 со всеми другими значениями = 0.17504')
print('logloss=', logloss, 'blogloss=', blogloss)

# print('!!!!---------------- ИЩЕМ ОПТИМАЛЬНЫЙ СЕРЕДИННЫЙ ПОРОГ -----------------!!!!!!!!!!!!!!')
# rezult = pd.DataFrame(columns=['param', 'Log_loss', 'Balance'])
# for param in range(100,-40,-5):
#     param = param/100
#     preds = preds0.copy()
#     preds[(preds > 0)&(preds < 1)] = preds[(preds > 0)&(preds < 1)] * (1-param) + 0.17504 * param
#     preds[preds < 0] = 0
#     preds[preds > 1] = 1
#     preds_valid[:, 1] = 1 - preds
#     preds_valid[:, 0] = preds
#     # tr_class = Train['Class']
#     logloss = log_loss(Train['Class'], preds_valid)
#     blogloss = balance_logloss(Train['Class'], preds_valid)
#     rezult.loc[len(rezult.index)] = [param, logloss, np.mean(blogloss)]
# rezult.sort_values(by='Balance', inplace=True)
# print(rezult.head(60))

# print('!!!!---------------- ИЩЕМ ОПТИМАЛЬНЫЙ СЕРЕДИННЫЙ ПОРОГ СВЕРХУ -----------------!!!!!!!!!!!!!!')
# rezult = pd.DataFrame(columns=['param', 'Log_loss', 'Balance'])
# for param in range(0,1005,5):
#     param = param/1000
#     preds = Train['pred'].copy()
#     preds[(preds < param)&(preds > 0.17504)]=0.17504
#     preds_valid[:, 1] = 1 - preds
#     preds_valid[:, 0] = preds
#     # tr_class = Train['Class']
#     logloss = log_loss(Train['Class'], preds_valid)
#     blogloss = balance_logloss(Train['Class'], preds_valid)
#     rezult.loc[len(rezult.index)] = [param, logloss, np.mean(blogloss)]
# rezult.sort_values(by='Log_loss', inplace=True)
# print(rezult.head(20))