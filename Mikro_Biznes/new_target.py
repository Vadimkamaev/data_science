import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm # прогресс бар
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from servise_ds import okno
import obrabotka_filtr

# определение среднего размера ошибки
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * np.mean(smap)

# получение вектора ошибоk
def vsmape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * smap


# Валидация
def validacia(raw, start_val, stop_val, rezult, blac_cfips, max_cfips, lastactive, param=1):
    # маска при которой была валидация и по которой сверяем ошибки
    # maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & \
    #         (raw.cfips <= max_cfips) & (raw.lastactive > lastactive)  & (~raw['cfips'].isin(blac_cfips))
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val)
    target = raw.loc[maska, 'microbusiness_density']
    ypred = raw.loc[maska, 'ypred']
    err_mod = smape(target, ypred)
    print('Предсказано иссдедуемой моделью. Ошибка SMAPE:',err_mod)

    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)
    print('Равенство последнему значению. Ошибка SMAPE:', err_last)

    ypred_target = raw.loc[maska, 'ypred_target']
    tar_get = raw.loc[maska, 'target']
    err_target = smape(tar_get, ypred_target)
    print('Ошибка target:', err_target)

    raw.loc[maska, 'error']=vsmape(target, ypred)
    raw.loc[maska, 'error_last']=smape(target, mbd_lag1)
    # dfnew = raw[(raw["dcount"] == 38) | (raw["dcount"] == 37)]
    # dfnew = dfnew[['row_id', 'cfips', 'microbusiness_density', 'target', 'ypred_target', 'ypred', 'error', 'error_last']]
    dif_err = err_last - err_mod # положительная - хорошо
    rezult.loc[len(rezult.index)] = [lastactive, param, err_mod, dif_err, err_target]
    return rezult

# Моульт для следующего месяца
def mult1(raw, mes_val, param, param2):
    kol_mes = 12
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    maska = raw[(raw.first_day_of_month >= '2022-09-01') & (raw.first_day_of_month <= '2022-11-01')]
    mult_column_to_mult = {f'smape_{mult}': mult for mult in [1.00, 1.0025, 1.005]}
    mult_to_priority = {1: 1, 1.0025: 0.4, 1.005: 0.2}
    train_data = raw[maska].copy()

    y_true = train_data['microbusiness_density']
    for mult_column, mult in mult_column_to_mult.items():
        train_data['y_pred'] = train_data['mbd_lag1'] + mult
        train_data[mult_column] = vsmape(y_true, train_data['y_pred']) * mult_to_priority[mult]

    df_agg = train_data.groupby('cfips')[list(mult_column_to_mult.keys())].mean()
    df_agg['best_mult'] = df_agg.idxmin(axis=1).map(mult_column_to_mult)
    df_agg= df_agg['best_mult']
    raw = raw.join(df_agg, on='cfips')
    #
    maska = raw.dcount == mes_val
    raw.loc[maska, 'ypred'] = raw.loc[maska, 'mbd_lag1'] + raw.loc[maska, 'best_mult']
    raw.loc[maska, 'multi'] = raw.loc[maska, 'best_mult']
    raw.drop('best_mult', axis=1, inplace = True)


    raw['ypred'].fillna(0, inplace = True)
    return raw

def posle_sglashivfnia(raw, rezult, lastactive = 0):
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    blac_cfips = []
    max_cfips = raw['cfips'].max()  # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    #raw = new_cfips(raw, 10, max_cfips)

    raw['ypred_target']=0

    # возможные поля 'Population', 'proc_covdeat', 'pct_bb', 'pct_foreign_born', 'pct_it_workers', 'unemploy'
    start_val = 6  # первый месяц валидации с которого проверяем модель
    stop_val = 38  # последний месяц валидации до которого проверяем модель

    # цикл по оптимизируемому параметру модели
    for param in range(2,12,1):
        # создаем новый таргет
        raw = new_target(raw, param/10000)
        raw, train_col = build_lag(raw, param)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
        # здесь должен начинаться цикл перебирающий все комбинации из списка полей
        znachenie = pd.DataFrame({'columns': train_col, 'importance': 0})
        # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
        start_time = datetime.now()  # время начала работы модели
        for mes_val in tqdm(range(start_val, stop_val + 1)):  # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
            # здесь должен начинаться цикл по перебору номера первого месяца для трайна
            mes_1 = 2#4
            raw, znachenie = vsia_model1(raw, mes_1, mes_val, train_col, znachenie, param)

        df = raw.sort_values(by = 'ypred_target')

        blac_cfips = mace_blac_list(raw, start_val, mes_val, blac_cfips)
        print('Обучение + предсказание + обработка заняло', datetime.now() - start_time)
        # валидация по результату обработки всех месяцев валидации в цикле
        rezult = validacia(raw, start_val, stop_val, rezult, blac_cfips, max_cfips, lastactive, param)
    # здесь заканчиваются циклы оптимизации
    rezult.to_csv("C:\\kaggle\\МикроБизнес\\rez_optim1.csv", index=False)
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_vmeste_forest.csv", index=False)
    print('Значимость колонок трайна в модели')
    znachenie.sort_values(by='importance', ascending=False, inplace=True)
    print(znachenie)
    rezult.sort_values(by='dif_err', ascending=False, inplace=True)
    print(rezult.head(40))


# МОДЕЛЬ.
def glavnaia(raw, rezult):
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['median_hh_inc'] = raw.groupby('cfips')['median_hh_inc'].ffill()
    raw['ypred'] = 0
    max_cfips = raw['cfips'].max() # максимальная реальна 'cfips', больше неё фиктивные 'cfips'

    # raw = raw[raw['cfips'] <= max_cfips]
    #raw = new_cfips(raw, 10, max_cfips)

    posle_sglashivfnia(raw, rezult)

if __name__ == "__main__":
    # train, test = obrabotka_filtr.start()
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)
    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0.csv")
    rezult = pd.DataFrame(columns=['model','param', 'error', 'dif_err', 'err_target'])
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv")
    glavnaia(raw, rezult)
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_otdelno.csv")
    # raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    # raw['ypred'] = 0
    # posle_sglashivfnia(raw, rezult)
