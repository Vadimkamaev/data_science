import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from catboost import Pool, CatBoostClassifier

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
    global Train, features, Test#, sample_submission #, Train0, greeks
    # Train0            = pd.read_csv('C:\\kaggle\\Возраст\\train.csv')
    Train = pd.read_csv('C:\\kaggle\\Возраст\\train_folds.csv')
    Test = pd.read_csv('C:\\kaggle\\Возраст\\test.csv')
    # greeks            = pd.read_csv('C:\\kaggle\\Возраст\\greeks.csv')
    # sample_submission = pd.read_csv('C:\\kaggle\\Возраст\\sample_submission.csv')
    Train['EJ'] = Train['EJ'].replace({'A': 0, 'B': 1})
    Test['EJ']  = Test['EJ'].replace({'A': 0, 'B': 1})
    features = Train.columns[1:-2] # список названий колонок трайна

# final_valid_predictions = {}
# final_test_predictions = []  # Это для формирования окончательного результата

def one_prohod():
    global Train, Train_Next, features, shag, param, final_test_predictions, Test
    # списки оцнеки ошибок
    s  = [] # logloss
    bs = [] # blogloss
    final_test_pred = []
    for k in range(5):
        print('Fold: '+str(k)+'==>',  end='')
        train = Train[Train['kfold'] !=k].reset_index(drop=True)
        val = Train[Train['kfold'] ==k].reset_index(drop=True) # валидационный датасет
        valid_ids = val.Id.values.tolist() # список ID валидационного датасета

        train_dataset = Pool(data=train[features], label=train['Class'], cat_features=["EJ"] )
        eval_dataset  = Pool(data=val[features], label=val['Class'], cat_features=["EJ"])
        if shag == 1:
            stop = 50 # 50 лучше чем 100 и 30 приблизительно так же как 40
        elif shag == 2:
            stop = 3 # 3 может лучше чем 4 может лучше чем 5 лучше чем 10 лучше чем 15
        params = {
            #"iterations": 10000,
            "iterations": 50000,
            "verbose": False,
            "learning_rate": 0.12,
            # "learning_rate": 0.15,
            "depth": 4,
            'auto_class_weights':'Balanced',
            'loss_function':'MultiClass',
            'eval_metric':'MultiClass:use_weights=False',
            'early_stopping_rounds': stop
        }
        model = CatBoostClassifier(**params)
        model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True)

        preds_valid = model.predict_proba(val[features])

        maska = Train_Next['Id'].isin(valid_ids)
        Train_Next.loc[maska, shag] = preds_valid[:,0]

        # Этот фрамент для формирования окончательного результата
        preds_test  = model.predict_proba(Test[features])
        final_test_pred.append(preds_test)

        logloss  = log_loss(val['Class'], preds_valid)
        blogloss = balance_logloss(val['Class'], preds_valid)

        s.append(logloss)
        bs.append(blogloss)

        print(k, logloss, blogloss)

    final_test_predictions = (final_test_pred[0] + final_test_pred[1] + final_test_pred[2] +
                              final_test_pred[3] + final_test_pred[4]) / 5
    Test[shag]=final_test_predictions[:,0]
    print('Log loss:', s)
    print('Balance Log loss:', bs)
    print('Log loss mean:', np.mean(s), 'Balance Log loss mean:', np.mean(bs))
    rezult.loc[len(rezult.index)] = [shag, param, np.mean(s), np.mean(bs)]

load_data()
rezult = pd.DataFrame(columns=['shag', 'param', 'Log_loss', 'Balance'])
Train0 = Train.copy()
features0 = features.copy()
for param in range(1,2,1):
    print('------------------------ param =', param, '--------------------------!!!!!!!!!!!!!!')
    Train_Next = Train.copy()
    for shag in range(1,3,1):
        Train_Next[shag] = 0
        one_prohod()
        features = np.append(features, shag)
        Train = Train_Next.copy()

    # Train = Train0.copy()

    features = features0.copy()
rezult.sort_values(by='Log_loss', inplace=True)
print(rezult[rezult['shag']==1].head(40))
print('2-й шаг')
rezult.sort_values(by='Balance', inplace=True)
print(rezult[rezult['shag']==2].head(40))
# print('3-й шаг')
# rezult.sort_values(by='Balance', inplace=True)
# print(rezult[rezult['shag']==3].head(40))

preds_valid = np.zeros((len(Train.index), 2))
print('!!!!---------------- ИЩЕМ ОПТИМАЛЬНЫЙ ПОРОГ СНИЗУ -----------------!!!!!!!!!!!!!!')
rezult = pd.DataFrame(columns=['param', 'Log_loss', 'Balance'])
for param in range(0,200,5):
    param = param/1000
    preds = Train[1].copy()
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
print(rezult.head(40))

print('!!!!---------------- ИЩЕМ ОПТИМАЛЬНЫЙ ПОРОГ СВЕРХУ -----------------!!!!!!!!!!!!!!')
rezult = pd.DataFrame(columns=['param', 'Log_loss', 'Balance'])
for param in range(0,200,5):
    param = param/1000
    preds = Train[1].copy()
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
print(rezult.head(40))


