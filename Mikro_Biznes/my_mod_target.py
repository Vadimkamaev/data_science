import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm # прогресс бар
from sklearn.ensemble import RandomForestRegressor
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

# УДАЛЕНИЕ ВЫБРОСОВ. Применется изменение всех значений до выброса
# две точки оптимизации l_vibr. Первая от 0 до 3. Вторая от 9 до 11
def del_outliers(raw, l_vibr=0.018, verh=0.0144, niz = 0.0046): # l_vibr, verh, niz - параметры для оптимизации
    # для verh и niz 0.0045 лучше чем 0.003 и чем 0.006
    raw['mbd_gladkaya']=0 # в результате функции создается новая колонка - сглаженный 'microbusiness_density'
    raw['mbd_gladkaya_dif'] = 0  # изменение сглаженного 'microbusiness_density'
    outliers = []  # список выбросов
    cnt = 0  # счетчик выбросов
    vsego = 0
    # цикл по уникальным cfips
    print('Удаление выбросов')
    for cfips in tqdm(raw.cfips.unique()):  # tqdm - прогресс бар
        maska_cfips = (raw['cfips'] == cfips)  # маска отфильтрованная по cfips
        tmp = raw.loc[maska_cfips].copy().reset_index(drop=True)  # df фильтрован по cfips
        # массив значений microbusiness_density при cfips из цикла
        mb_d = tmp.microbusiness_density.values.copy()
        # цикл назад от предпоследнего до 2-го элемента
        for i in range(38, 2, -1): # у умного чувака было с 37 по 2(не вкл.) элемента вниз
            # среднее значение microbusiness_density с 0-го по i-й элемент * l_vibr
            thr = l_vibr * np.mean(mb_d[:i])
            difa = mb_d[i] - mb_d[i - 1]
            if (difa >= thr) or (difa <= -thr):  # если microbusiness_density изменился больше чем на l_vibr%
                if difa > 0:
                    mb_d[:i] += difa - verh # verh=0.0144 error 1.791025, dif_err -0.401444
                else:
                    mb_d[:i] += difa + niz # 0.0046 error 1.791025, dif_err -0.401444
                outliers.append(cfips)  # добавляем cfips в список выбросов
                cnt += 1  # счетчик выбросов
            vsego += 1
        # mb_d[0] = mb_d[1] * 0.99  # у умного чувака была эта фигня
        raw.loc[maska_cfips, 'mbd_gladkaya'] = mb_d
        raw.loc[maska_cfips, 'mbd_gladkaya_dif'] = raw.loc[maska_cfips, 'mbd_gladkaya'].diff(1).fillna(0)
    outliers = np.unique(outliers)
    print('Кол. cfips с разрывами =', len(outliers), 'Кол. разрывов =', cnt, '% разрывов',cnt/vsego*100)

    # raw['smeschenie'] = 0
    # raw['target'] = 0
    # for cfips in tqdm(raw.cfips.unique()):
    #     maska_cfips = (raw['cfips'] == cfips)  # маска отфильтрованная по cfips
    #     maska = maska_cfips & (raw['dcount'] == 0)
    #     x = raw.loc[maska, 'mbd_gladkaya']
    #     x.reset_index(drop=True, inplace=True)
    #     raw.loc[maska_cfips, 'smeschenie'] = x[0]
    # raw['target'] = raw['microbusiness_density']-raw['smeschenie']
    # 'target' ='microbusiness_density' следующего месяца при том же 'cfips'
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    raw['target'] = raw['microbusiness_density'] / raw['target']  - 1
    raw['target'].fillna(0.01, inplace = True)
    # в этих 'cfips' значение 'active' аномально маленькие
    raw.loc[raw['cfips'] == 28055, 'target'] = 0.0
    raw.loc[raw['cfips'] == 48269, 'target'] = 0.0
    pass
    return raw

# приведение начальной точки таргета к 0
def make_target(raw):
    raw['smeschenie'] = 0
    raw['target'] = 0
    for cfips in tqdm(raw.cfips.unique()):
        maska_cfips = (raw['cfips'] == cfips)  # маска отфильтрованная по cfips
        raw.loc[maska_cfips, 'smeschenie'] = raw[maska_cfips & raw.dcount == 0, 'mbd_gladkaya']
    raw.loc['target'] = raw['microbusiness_density']-raw['smeschenie']
    pass

# скользящая средняя
def weighted_moving_average(raw, column, n):
    nam = 'EMA_' + str(n)
    EMA = pd.Series(raw[column].ewm(span=n, adjust=False, min_periods=n).mean(), name = nam)
    raw = raw.join(EMA)
    return raw, nam

# создание лагов
def build_lag(raw, lags=3, target = 'target'):
    # создаем 1 лаг по колонке 'microbusiness_density'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw['mbd_lag1'].fillna(0.01, inplace=True)
    raw['target_lag1'] = raw.groupby('cfips')['mbd_lag1'].shift(1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    raw['target_lag1'] = raw['mbd_lag1'] / raw['target_lag1']  - 1
    raw['target_lag1'].fillna(0.01, inplace = True)
    train_col = ['target_lag1']

    # создаем лаги target с учетом сглаживания от 1 до lags
    for lag in range(1, lags):
        # shift - сдвиг на определеное кол. позиций
        raw[f'mbd_lag{lag+1}'] = raw[f'mbd_lag{lag}'] - raw.groupby('cfips')['mbd_gladkaya_dif'].shift(lag)
        raw[f'mbd_lag{lag+1}'].fillna(0.01, inplace=True)
        raw[f'target_lag{lag+1}'] = raw.groupby('cfips')[f'mbd_lag{lag+1}'].shift(1)
        # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
        raw[f'target_lag{lag+1}'] = raw[f'mbd_lag{lag+1}'] / raw[f'target_lag{lag+1}']  - 1
        raw[f'target_lag{lag+1}'].fillna(0.01, inplace = True)
        train_col = [f'target_lag{lag+1}']

    # создаем скользящую среднюю. Вывод: Со всеми скользящими средними хуже чем без них
    # лучшая скользящая средняя - 3 : error 1.839445, dif_err -0.449865, следю ск. ср. - 8, 6
    nam = 'EMA_3' # лучшая скользящая средняя - 3 : error 1.791025, dif_err -0.401444
    EMA = pd.Series(raw['target_lag1'].ewm(span=3, adjust=False, min_periods=3).mean(), name = nam)
    raw[nam] = EMA
    train_col += [nam]

    nam = 'EMA_7' # скол. ср. 10+7+3 error 1.791025, dif_err -0.401444
    EMA = pd.Series(raw['target_lag1'].ewm(span=7, adjust=False, min_periods=7).mean(), name=nam)
    raw[nam] = EMA
    train_col += [nam]

    nam = 'EMA_10' # скол. ср. 10+7+3 error 1.791025, dif_err -0.401444
    EMA = pd.Series(raw['target_lag1'].ewm(span=10, adjust=False, min_periods=10).mean(), name = nam)
    raw[nam] = EMA
    train_col += [nam]

    df = raw[raw.isnull()]
    pass

    # создаем сумму окна. Вывод: Со всеми скользящими средними хуже чем без них
    # лучшая сумма окна - 7: error 1.844054, dif_err -0.454473/ след. сум. ок. 3, 6
    # nam = 'mbd_rollmea'
    # raw[nam] = raw.groupby('cfips')['target_lag1'].transform(lambda s: s.rolling(ema, min_periods=1).sum())
    # train_col += [nam]

    # for lag in range(1, lags+1):
    #     # shift - сдвиг на определеное кол. позиций
    #     raw[f'mbdgd_lag_{lag}'] = raw.groupby('cfips')['mbd_gladkaya_dif'].shift(lag)
    #     train_col.append(f'mbdgd_lag_{lag}')
    # error 1.550974  dif_err -0.339597

    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['active_lag1']
  #  df = raw[['row_id', target, 'mbd_gladkaya', 'mbd_gladkaya_dif', nam]+train_col]
    return raw, train_col


# получение трайна и 'y' (игрик) для модели, mes_1 - первый месяц с которого используем трайн для модели
def train_and_y(raw, mes_1, mes_val, train_col, target): # train_col - список используемых полей трейна
    # маска тренировочной выборки
    maska_train = (raw.istest == 0) & (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    train = raw.loc[maska_train, train_col]
    y = raw.loc[maska_train, target]
    return train, y

# Получение х_тест и y_тест. mes_val - месяц по которому проверяем модель
def x_and_y_test(raw, mes_val, train_col, target): # train_col - список используемых полей трейна
    # маска валидационной выборки. Валидация по 1 месяцу
    maska_val = (raw.istest == 0) & (raw.dcount == mes_val)
    X_test = raw.loc[maska_val, train_col]
    y_test = raw.loc[maska_val, target]
    return X_test, y_test

# Сохраняем результат.
def sbor_rezult(raw, y_pred, y_test, mes_val):
    vsmape(y_test, y_pred)
    #Почему то у умного чувака mes_val + 1 ! Вроде должно быть mes_val
    raw.loc[raw.dcount == mes_val, 'ypred'] = y_pred
    return raw

# создание блек листа 'cfips'
def mace_blac_list(raw, start_val, stop_val):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    # создаем датафрейм со столбцами error и error_last
    dt = raw.loc[(raw.dcount >= 30) & (raw.dcount <= 38)].groupby(['cfips', 'dcount'])[['error', 'error_last']].last()
    # преобразуем dt в серию булевых значений 'miss'
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    ser_err = dt.groupby('cfips')['error'].mean()
    ser_error_last = dt.groupby('cfips')['error_last'].mean()
    df_dt = pd.DataFrame({'cfips': seria_dt.index, 'miss': seria_dt, 'hit':(ser_error_last-ser_err)})
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() > 0.5
    print('количство cfips предсказанных хуже чем моделью =', len(seria_dt))
    # if not_blec == 1: # 1 - без блек листа
    #     df_dt.to_csv("C:\\kaggle\\МикроБизнес\\blec_list.csv", index = False)

# Валидация
def validacia(raw, start_val, stop_val, rezult, model=1, param=1):
    # ПРЕОБРАЗОВАНИЕ ТАРГЕТА К 'microbusiness_density'. В 'ypred' мы нашли таргет.
    raw['ypred'] = (raw['ypred']+1)*raw['mbd_lag1']

 #   raw['ypred'] = raw['smeschenie'] + raw['ypred'] # Теперь в 'ypred' - 'microbusiness_density' error - 1.822742
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val)
    err_mod = smape(raw.loc[maska, 'microbusiness_density'], raw.loc[maska, 'ypred'])
    print('Предсказано иссдедуемой моделью. Ошибка SMAPE:',err_mod)
    err_last = smape(raw.loc[maska, 'microbusiness_density'], raw.loc[maska, 'mbd_lag1'])
    print('Равенство последнему значению. Ошибка SMAPE:', err_last)
    dif_err = err_last - err_mod # положительная - хорошо
    mace_blac_list(raw, start_val, stop_val)
    rezult.loc[len(rezult.index)] = [model, param, err_mod, dif_err]
    return rezult

def vsia_model(raw, mes_1, mes_val, train_col, znachenie, param_mod = 500):
    target = 'target'
    # получение трайна и 'y' (игрик) для модели
    X_train, y_train = train_and_y(raw, mes_1, mes_val, train_col, target)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col, target)
    # Создаем модель
    rf = RandomForestRegressor(n_estimators=100, min_samples_split=250, n_jobs=2, random_state=322)
    # лучшее n_estimators=900 лучше чем 500, лучше чем 100,  разница ошибки между 900 и 100 - 0.5%
    # 100 деревьев в 5 раз быстрее 900 деревьев
    rf.fit(X_train, y_train)
    # Предсказываем
    y_pred = rf.predict(X_test)
    # сохраняем результат обработки одного цикла
    raw = sbor_rezult(raw, y_pred, y_test, mes_val)

    # # Упорядычиваем наши фичи по значениям весов, от самой полезной к самой бесполезной
    # df_importances = sorted(list(zip(X_train.columns, rf.feature_importances_.ravel())), key=lambda tpl: tpl[1],
    #                         reverse=True)
    # # Создаем табличку, в которой будет показан признак и его вес
    # df_importances = pd.DataFrame(df_importances, columns=['feature', 'importance'])
    # df_importances = df_importances.set_index('feature')  # Нумеруем колонки, чтобы не путать их
    # print(df_importances.head(12))  # Выводим табличку

    # прибавляем значимость столбцов новой модели к значениям предыдущих
    znachenie['importance'] = znachenie['importance'] + rf.feature_importances_.ravel()
    return raw, znachenie

def glavnaia(raw, rezult):
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['median_hh_inc'] = raw.groupby('cfips')['median_hh_inc'].ffill()
    raw['ypred'] = 0
    #okno.vewdf(raw)
    #df = raw[raw['median_hh_inc'].isnull()]
    # здесь должен начинаться цикл по оптимизации сглаживания
    for vibr in range(18,19): # при 10 - error - 1.801009
        raw = del_outliers(raw, l_vibr=0.018, verh=0.0144, niz=0.0046) # сглаживание, убираем выбросы
        # здесь должен начинаться цикл по количеству лагов
        lags = 3
        raw, train_col = build_lag(raw, lags) # создаем лаги
        # здесь должен начинаться цикл перебирающий все комбинации из списка полей
        train_col += ['state_i', 'proc_covill', 'pct_college', 'median_hh_inc', 'sp500', 'month']
        znachenie = pd.DataFrame({'columns': train_col, 'importance': 0})
        # возможные поля 'Population', 'proc_covdeat', 'pct_bb', 'pct_foreign_born', 'pct_it_workers', 'unemploy'
        start_val = 30 # первый месяц валидации с которого проверяем модель
        start_val = 38 # error
        stop_val = 38 # последний месяц валидации до которого проверяем модель
        # цикл по оптимизируемому параметру модели
        # for param_mod in tqdm(range(100,1200,400)):
        # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
        for mes_val in range(start_val, stop_val+1): # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
            start_time = datetime.now()    # время начала работы модели
            # здесь должен начинаться цикл по перебору номера первого месяца для трайна
            mes_1 = 10
            raw, znachenie = vsia_model(raw, mes_1, mes_val, train_col, znachenie)
            print('Обучение + предсказание + обработка заняло', datetime.now() - start_time)
        # валидация по результату обработки всех месяцев валидации в цикле

        rezult = validacia(raw, start_val, stop_val, rezult, vibr)
        rezult.to_csv("C:\\kaggle\\МикроБизнес\\rez_optim.csv", index=False)

    # здесь заканчиваются циклы оптимизации
    print('Значимость колонок трайна в модели')
    znachenie.sort_values(by='importance', ascending=False, inplace=True)
    print(znachenie)
    rezult.sort_values(by='dif_err', inplace=True)
    #print(rezult.mean())
    print(rezult.head(22))


if __name__ == "__main__":
    # train, test = obrabotka_filtr.start()
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)
    rezult = pd.DataFrame(columns=['model','param', 'error', 'dif_err'])
    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv")
    glavnaia(raw, rezult)
