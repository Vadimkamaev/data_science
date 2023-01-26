# модель на основе длины трендов
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from servise_ds import okno
import obrabotka_filtr

# создание лагов
def build_features(raw, target='microbusiness_density', lags=1):
    feats = []  # список имен лагов и сумм окон
    # создаем лаги target='microbusiness_density' от 1 до lags
    for lag in range(1, lags):   #XGB. Ошибка SMAPE: 1.07924633283354
        # shift - сдвиг на определеное кол. позиций
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)
        # из значения 'active' следующей через лаг строки вычитается значение
        # текущей строки. diff - разность элемента df с элементом через lag
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')

# обработка прохода 1-го символа в тренде
def len_trend_1(znak_trend, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend):
    if finish_trend:
        if tip_fin_trend == '0':
            tip_fin_trend = znak_trend
            l_fin_trend += 1
        elif tip_fin_trend != znak_trend:
            finish_trend = False
            # тренд закончился
            return True, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend
        else:
            l_fin_trend += 1
    else:
        if tip_trend == znak_trend:
            l_trend += 1
        else:
            if tip_trend != '0':
                trend.loc[len(trend.index)] = (l_trend, tip_trend)
                tip_trend = znak_trend
                l_trend = 1
                # тренд закончился
                return True, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend
            else:
                tip_trend = znak_trend
                l_trend = 1
    # тренд не закончился
    return False, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend

# Подсчет средней длины тренда и длины последнего тренда
def len_trend(loc_train):
    l = len(loc_train)
    tip_fin_trend = '0'  # тип последнего тренда - символ '0' означает, что пока не известно
    l_fin_trend = 0  # длина последнего тренда
    finish_trend = True
    trend = pd.DataFrame(columns=['l', 'tip'])  # список длин трендов и направлений
    i = l-1
    j = l-1
    tip_trend = '0'  # тип текущего тренда
    l_trend = 0  # длина текущего тренда
    while i >=0: # цикл по трендам
        while j >= 0: # цикл по отдельно взятому тренду
            i_loc_train = loc_train['microbusiness_density'].iloc[j]
            i_1 = loc_train['microbusiness_density'].iloc[j-1]
            j -= 1
            if abs(i_loc_train - i_1)  < 0.000000001: # тренд равенства
                br, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend = \
                    len_trend_1(0, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend)
                if br :
                    break
            if i_loc_train > i_1: # тренд восходящий
                br, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend = \
                    len_trend_1(1, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend)
                if br :
                    break
            if i_loc_train < i_1: # тренд вниз
                br, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend = \
                    len_trend_1(-1, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend)
                if br :
                    break
        i = j
    l0 = trend[trend['tip']==0]['l'].mean() # средняя длина горизонтального тренда
    if pd.isna(l0):
        l0 = 0
        print('NAN 0')
    l_ap = trend[trend['tip'] == 1]['l'].mean()  # средняя длина возрастающего тренда
    if pd.isna(l_ap):
        l_ap = 0
        print('NAN ap')
    l_down = trend[trend['tip'] == -1]['l'].mean() # средняя длина нисходящего тренда
    if pd.isna(l_down):
        l_down = 0
        print('NAN down')
    oll_l_mean = trend['l'].mean() # средняя длина всех трендов
    if pd.isna(oll_l_mean):
        oll_l_mean = 0
        print('NAN oll')
    return l0, l_ap, l_down, oll_l_mean, l_fin_trend, tip_fin_trend

# Подсчет длин трендов и прогноз на основе этой модели
def len_trend_do_month(train, n): # n - номер месяца, который мы предсказываем
    unic_cfips = train['cfips'].unique() # уникальные 'cfips'
    for cfips in unic_cfips:
        loc_train = train[train['cfips'] == cfips]
        l0, l_ap, l_down, oll_l_mean, l_fin_trend, tip_fin_trend = len_trend(loc_train.iloc[:n])
        row_id = loc_train.iloc[n]['row_id'] # записываем в n элемент результат анализа с 0 по n-1
        maska = (train['row_id']==row_id)
        #maska = (train['cfips']==cfips) & (raw['dcount']==n-1)
        train.loc[maska, 'l0'] = l0
        train.loc[maska, 'l_ap'] = l_ap
        train.loc[maska, 'l_down'] = l_down
        train.loc[maska, 'oll_l_mean'] = oll_l_mean
        train.loc[maska, 'l_fin_trend'] = l_fin_trend
        train.loc[maska, 'tip_fin_trend'] = tip_fin_trend
        smotri = train.sort_values(by='l_fin_trend')
        pass
        # real = loc_train['microbusiness_density'].iloc[n] - loc_train['microbusiness_density'].iloc[n-1] # реальный тренд, который мы пытаемся предсказать
    return train

# заполняем train данными о длине и знаке последнего тренда и средней длине восходящих и нисходящих трендов
def glavnaia(train, min_m = 15, max_m = 39):
    train[['l0', 'l_ap', 'l_down', 'oll_l_mean', 'l_fin_trend', 'tip_fin_trend']]=0
    i = min_m # номер месяца с которого начинаем предсказания
    while i < max_m: # цикл по месяцам
        train = len_trend_do_month(train, i)
        i +=1
    return train

# создание таргета - тренда со значениями -1,0,1
def build_target(raw):
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(1) # лаг 1
    raw['target'] = raw['microbusiness_density'] - raw['target']
    raw['target'] = np.sign(raw['target'])
    return raw

# создание лагов
def build_lag(raw, lags=1):
    feats = []  # список полей используемых в таргете
    # создаем лаги 'target' от 1 до lags
    for lag in range(1, lags+1):
        # shift - сдвиг на определеное кол. позиций
        raw[f'target_lag_{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        feats.append(f'target_lag_{lag}')
    return raw, feats

# получение трайна и 'y' (игрик) для модели
def train_and_y(raw, mes_1, mes_val, train_col):
    # маска тренировочной выборки
    maska_train = (raw.istest == 0) & (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    train = raw.loc[maska_train, train_col]
    y = raw.loc[maska_train, 'target']
    return train, y

# получение х_тест и y_тест
def x_and_y_test(raw, mes_val, train_col):
    # маска валидационной выборки. Валидация по 1 месяцу
    maska_val = (raw.istest == 0) & (raw.dcount == mes_val)
    X_test = raw.loc[maska_val, train_col]
    y_test = raw.loc[maska_val, 'target']
    return X_test, y_test

# валидация
def validacia(y_pred, y_test, rezult):
    # здесь должен быть цикл по размеру границы
    gran = 0.1
    y_pred_new=np.zeros(3135)
    y_pred_new[y_pred < -gran] = -1
    y_pred_new[y_pred > gran] = 1
    dfrez = pd.DataFrame({'istina':y_test, 'pred': y_pred_new})
    dfrez1 = dfrez[(dfrez['pred'] == 1)]
    vsego = len(dfrez1.index)
    verno = len(dfrez1[(dfrez1['istina'] == 1)].index)
    if vsego > 0.5:
        proc_plus = verno/vsego*100
    else:
        proc_plus = 0
    print ('Процент верных из предсказанного + тренда:', proc_plus)

    dfrez1 = dfrez[(dfrez['pred'] == -1)]
    vsego = len(dfrez1.index)
    verno = len(dfrez1[(dfrez1['istina'] == -1)].index)
    if vsego > 0.5:
        proc_minus = verno / vsego * 100
    else:
        proc_minus = 0
    print ('Процент верных из предсказанного - тренда:', proc_minus)
    rezult.loc[len(rezult.index)] = [proc_plus, proc_minus]
    return rezult


if __name__ == "__main__":
    # train, test = obrabotka_filtr.start()
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw1.csv")
    # raw = glavnaia(raw)
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)
    rezult = pd.DataFrame(columns=['proc_plus','proc_minus'])
    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw2.csv")
    raw = build_target(raw)
    # здесь должен начинаться цикл по количеству лагов
    lags = 0
    raw, train_col = build_lag(raw, lags)

    raw['sonapr'] = raw['l0']
    raw.loc[raw['tip_fin_trend'] == 1, 'sonapr'] = raw.loc[raw['tip_fin_trend'] == 1, 'l_ap']
    raw.loc[raw['tip_fin_trend'] == -1, 'sonapr'] = raw.loc[raw['tip_fin_trend'] == -1, 'l_down']

    raw['napr'] = 0
    raw.loc[raw['tip_fin_trend'] == 1, 'napr'] = raw.loc[raw['tip_fin_trend'] == 1, 'l_ap']
    raw.loc[raw['tip_fin_trend'] == -1, 'napr'] = - raw.loc[raw['tip_fin_trend'] == -1, 'l_down']


    raw['fin_trend'] = raw['l_fin_trend'] * raw['l_fin_trend']

    raw['l_fin_trend'] = raw['l_fin_trend'] - raw['sonapr']
    # здесь должен начинаться цикл перебирающий все комбинации из списка полей
    train_col += ['l_fin_trend', 'sonapr', 'fin_trend', 'napr']
    # бесполезны 'tip_fin_trend', ? 'oll_l_mean','l_ap', 'l_down',
    # proc_plus    54.529186
    # proc_minus   36.643998 с 'oll_l_mean'

    mes_val = 17 # здесь должен начинаться цикл по перебору номера валидационного месяца
    while mes_val <= 37:
        # здесь должен начинаться цикл по перебору номера первого месяца для трайна
        mes_1 = mes_val - 1
        # proc_plus
        # 49.723300
        # proc_minus
        # 38.882726
        X_train, y_train = train_and_y(raw, mes_1, mes_val, train_col) # получение трайна и 'y' (игрик) для модели
        X_test, y_test = x_and_y_test(raw, mes_val, train_col) # получение х_тест и y_тест
        # Создаем модель
        rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=322)
        # Обучаем
        rf.fit(X_train, y_train)
        # Предсказываем
        y_pred = rf.predict(X_test)
        # валидация
        rezult = validacia(y_pred, y_test, rezult)
        # Упорядычиваем наши фичи по значениям весов, от самой полезной к самой бесполезной
        df_importances = sorted(list(zip(X_train.columns, rf.feature_importances_.ravel())), key=lambda tpl: tpl[1],
                                reverse=True)
        # Создаем табличку, в которой будет показан признак и его вес
        df_importances = pd.DataFrame(df_importances, columns=['feature', 'importance'])
        # Нумируем колонки, чтобы не путать их
        df_importances = df_importances.set_index('feature')
        # Выводим табличку
        print(df_importances.head(10))
        mes_val += 1
    print(rezult.mean())
    print(rezult.head(22))



    # for i, y_pred_row in y_pred.iterrows():
    #     if y_pred_row < -gran:

# Подсчет длин трендов и прогноз на основе этой модели
# def oll_len_trend(train):
#     unic_cfips = train['cfips'].unique()
#     df_unic_cfips = pd.DataFrame({'l_mean':0, 'oll_l_mean':0, 'l_finish':0}, index=unic_cfips)
#     for cfips in unic_cfips:
#         # Подсчет средней длины тренда и длины последнего тренда
#         l_mean, oll_l_mean, l_finish, tip_fin_trend = len_trend(train[train['cfips'] == cfips])
#         df_unic_cfips.loc[cfips] = (l_mean, oll_l_mean, l_finish)
#     return df_unic_cfips