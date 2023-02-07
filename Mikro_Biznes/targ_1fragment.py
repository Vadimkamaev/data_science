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

# создание нового таргета
def new_target(raw, param=0.0054):
    # 'target' ='microbusiness_density' предыдущего месяца при том же 'cfips'
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    # 'target' ='microbusiness_density' текущего месяца делить на предыдущего месяца - 1
    raw['target'] = raw['microbusiness_density'] / raw['target'] - 1
    raw.loc[(raw['microbusiness_density'] == 0)|(raw['target'] > 10),'target'] = 0
    raw['target'].fillna(0, inplace=True)

    raw.loc[(raw['target'] < - 0.0054), 'target'] = - 0.0054
    raw.loc[(raw['target'] > 0.0054), 'target'] = 0.0054 # 2.18317 -0.026207

    # raw.loc[(raw['target'] > param), 'target'] = param
    # raw.loc[(raw['target'] < -param), 'target'] = -param

    # в этих 'cfips' значение 'active' аномально маленькие
    raw.loc[raw['cfips'] == 28055, 'target'] = 0.0
    raw.loc[raw['cfips'] == 48269, 'target'] = 0.0
    return raw

# создание лагов  error 1.379034  0.010546  146.796656
def build_lag(raw, param): #
    train_col = []  # список полей используемых в трайне
    #создаем лаги 'target'
    for lag in range(1, 17): #17 -  1.409514  0.009615   144.354724
        raw[f'target_lag{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        train_col.append(f'target_lag{lag}')
    #создаем 1 лаг 'microbusiness_density'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw['mbd_lag1'].fillna(method='bfill', inplace=True)
    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['mbd_lag1', 'active_lag1', 'median_hh_inc', 'month'] #  1.403340  0.015789   144.360897
    return raw, train_col

# получение трайна и 'y' (игрик) для модели, mes_1 - первый месяц с которого используем трайн для модели
def train_and_y(raw, mes_1, mes_val, train_col): # train_col - список используемых полей трейна
    # маска тренировочной выборки 1.408914 -0.097942
    maska_train = (raw.istest == 0) & (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    train = raw.loc[maska_train, train_col]
    y = raw.loc[maska_train, 'target']
    return train, y

# Получение х_тест и y_тест. mes_val - месяц по которому проверяем модель
def x_and_y_test(raw, mes_val, train_col): # train_col - список используемых полей трейна
    # маска валидационной выборки. Валидация по 1 месяцу
    maska_val = (raw.istest == 0) & (raw.dcount == mes_val)
    X_test = raw.loc[maska_val, train_col]
    y_test = raw.loc[maska_val, 'target']
    return X_test, y_test

# создание блек листа 'cfips' для теста, может делать это моделью
def mace_blac_list(raw, mes_1, mes_val, blac_cfips, param=-1):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    raw['error_otdelno'] = vsmape(raw['microbusiness_density'], raw['ypred_otdelno'])
    # создаем датафрейм со столбцами error и error_last и индексом 'cfips' + 'dcount'
    dt = raw.loc[(raw.dcount >= mes_1) & (raw.dcount <= mes_val)].groupby(['cfips', 'dcount'])[['error', 'error_last','lastactive']].last()
    # добавляем в dt столбец булевых значений 'miss' - количество ошибок > или <
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    ser_err = dt.groupby('cfips')['error'].mean()
    ser_lastactive = dt.groupby('cfips')['lastactive'].mean()
    ser_error_last = dt.groupby('cfips')['error_last'].mean()
    df_dt = pd.DataFrame({'cfips': seria_dt.index, 'lastactive':ser_lastactive, 'miss': seria_dt, 'dif_err':ser_error_last-ser_err})
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() > 0.5
    print('количство cfips предсказанных хуже чем моделью =', len(seria_dt))
    # Признак отправления в блек лист
    blec_maska = (df_dt['miss'] >= 0.5) # от 0.5 (большой блек) до 1 (малый блек) Проблема на 1-м месяце исключит всех
    df_dt.sort_values(by='lastactive', inplace=True)
    shad = 40
    for lastactive in range(30, 800, shad):
        maska = (df_dt['lastactive']>lastactive)&(df_dt['lastactive']<=lastactive+shad)
        horoshie = df_dt[~blec_maska&maska].count()['lastactive']
        vsego = df_dt[maska].count()['lastactive']
        print(f'От {lastactive} до {lastactive+shad} :хор.{horoshie}, всего {vsego}, % хор={horoshie/vsego*100}')
    lastactive = lastactive+shad
    shad = 1000000000
    maska = (df_dt['lastactive'] > lastactive) & (df_dt['lastactive'] <= lastactive + shad)
    horoshie = df_dt[~blec_maska & maska].count()['lastactive']
    vsego = df_dt[maska].count()['lastactive']
    print(f'От {lastactive} до {lastactive + shad} :хор.{horoshie}, всего {vsego}, % хор={horoshie / vsego * 100}')
    return blac_cfips

# вывод на печать информацию по качеству работы модели в каждом штате
def print_state(raw, maska):
    df_state = pd.DataFrame({'state_i':0, 'state': 0, 'err_mod': 0, 'err_last': 0, 'diff_err': 0, 'lastactive': 0}, index=[0])
    for state in raw.state.unique():
        maska2 = (raw['state'] == state) & maska
        state_i = raw.loc[maska2, 'state_i']
        state_i = state_i.iloc[0]
        target_m = raw.loc[maska2, 'microbusiness_density']
        ypred_m = raw.loc[maska2, 'ypred']
        err_mod1 = smape(target_m, ypred_m)
        mbd_lag11 = raw.loc[maska2, 'mbd_lag1']
        err_last1 = smape(target_m, mbd_lag11)
        st_l_active = raw[maska2]['lastactive'].mean()
        df_state.loc[len(df_state.index)] = (state_i, state, err_mod1, err_last1, err_last1 - err_mod1, st_l_active)
    df_state.sort_values(by='diff_err', inplace=True)
    print(df_state.head(53))

# вывод на печать информацию по качеству работы модели в каждом месяце
def print_month(raw, maska):
    df_state = pd.DataFrame({'month': 0, 'err_mod': 0, 'err_last': 0, 'diff_err': 0, 'lastactive': 0}, index=[0])
    for dcount in raw[maska].dcount.unique():
        maska2 = (raw['dcount'] == dcount) & maska
        target_m = raw.loc[maska2, 'microbusiness_density']
        ypred_m = raw.loc[maska2, 'ypred']
        err_mod1 = smape(target_m, ypred_m)
        mbd_lag11 = raw.loc[maska2, 'mbd_lag1']
        err_last1 = smape(target_m, mbd_lag11)
        st_l_active = raw[maska2]['lastactive'].mean()
        df_state.loc[len(df_state.index)] = (dcount, err_mod1, err_last1, err_last1 - err_mod1, st_l_active)
    #df_state.sort_values(by='diff_err', inplace=True)
    print(df_state.head(53))

# Валидация
def validacia(raw, start_val, stop_val, rezult, blac_test_cfips, max_cfips, param1=1, param2=1, param3=1):
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & \
            (raw.cfips <= max_cfips)  & (~raw['cfips'].isin(blac_test_cfips))
    target = raw.loc[maska, 'microbusiness_density']
    ypred = raw.loc[maska, 'ypred']
    err_mod = smape(target, ypred)
    print('Предсказано иссдедуемой моделью. Ошибка SMAPE:',err_mod)
    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)
    print('Равенство последнему значению. Ошибка SMAPE:', err_last)

    y_no_blac = raw.loc[maska, 'y_no_blac']
    err_y_no_blac = smape(target, y_no_blac)
    print('Без блек листа. Ошибка SMAPE:', err_y_no_blac)

    dif_err = err_last - err_mod # положительная - хорошо
    dif_no_blac = err_y_no_blac - err_mod # положительная - хорошо
    rezult.loc[len(rezult.index)] = [param1, param2, param3, err_mod, dif_err, dif_no_blac]
    print_month(raw, maska)
    return rezult

def vsia_model1(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0):
    # получение трайна и 'y' (игрик) для модели
    df = raw[~raw['cfips'].isin(blac_cfips)]
    X_train, y_train = train_and_y(df, mes_1, mes_val, train_col)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col)
    # Создаем модель
    model = xgb.XGBRegressor(
        tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
        n_estimators=850,
        learning_rate=0.0071, #важная штука
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    # Предсказываем
    y_pred = model.predict(X_test)

    y_pred = y_pred + 1
    y_pred = raw.loc[raw.dcount == mes_val, 'mbd_lag1'] * y_pred

    # сохраняем результат обработки одного цикла
    raw.loc[raw.dcount == mes_val, 'ypred'] = y_pred
    # прибавляем значимость столбцов новой модели к значениям предыдущих
    znachenie['importance'] = znachenie['importance'] + model.feature_importances_.ravel()
    return raw, znachenie

def modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips):
    start_time = datetime.now()  # время начала работы модели
    znachenie = pd.DataFrame({'columns': train_col, 'importance': 0})
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
    for mes_val in tqdm(range(start_val, stop_val + 1)):  # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
        # здесь должен начинаться цикл по перебору номера первого месяца для трайна
        mes_1 = 2 #
        raw, znachenie = vsia_model1(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0)
    #blac_cfips = mace_blac_list(raw, start_val, stop_val, blac_cfips)
    print('Обучение + предсказание + обработка заняло', datetime.now() - start_time)
    # print('Значимость колонок трайна в модели')
    # znachenie.sort_values(by='importance', ascending=False, inplace=True)
    # print(znachenie)
    return raw, blac_cfips

# создание новых фиктивных cfips
def new_cfips(raw, lastactive, max_cfips, kol = 2):
    #kol - оптимально 2
    if max_cfips != raw['cfips'].max():
        print('Ошибка max_cfips')
    df = raw[raw['lastactive'] < lastactive].copy()
    df.sort_values('lastactive', ascending=False, inplace=True)
    if len(df.index) == 0:
        return raw
    list_cfips = df['cfips'].unique() #  массив уникальных cfips, которые группируются
    lec = len(list_cfips) # количество уникальных cfips, которые группируются
    le = lec + kol - lec%kol # длина матрицы - количество новых генерируемых cfips
    matrica = np.zeros((kol,le)) # матрица каждая строка которой - список cfips которые суммируются
    fin = le // kol  # номер элемента до которого все cfips использованы в 0-м ряду и больше их не применять
    for l in range(fin): # цикл по заполнению 0-го ряда
        for k in range(kol):
            x = list_cfips[l]
            matrica[0,l*kol+k] = x
    sled = fin # номер следующего cfips который вставляется в матрицу
    for k in range(1, kol): # цикл по заполнению рядов с 1 по последний
        for l in range(le-1,-1,-1): # цикл в котором заполняется один ряд матрицы
            if list_cfips[sled] == matrica[k-1, l]:
                sled+=1
            matrica[k, l] = list_cfips[sled]
            if sled == lec-1:
                sled = fin
            else:
                sled+=1
    # матрица заполнена, далее суммирование cfips
    # считываем стандартные колонки одинаковые у всех cfips
    cfips = list_cfips[0]
    dfcop = raw[raw['cfips']==cfips].copy()
    dfcop.reset_index(drop=True, inplace=True)
    dfcop[['row_id', 'cfips', 'county','state', 'first_day_of_month',
           'microbusiness_density', 'active',
           'county_i', 'pct_bb', 'pct_college', 'pct_foreign_born',
           'pct_it_workers', 'median_hh_inc', 'covill', 'covdeat', 'Population',
           'proc_covill', 'proc_covdeat', 'ypred',
           'error_otdelno', 'mbd_gladkaya', 'mbd_gladkaya_dif', 'mbd_lag1',
           'mbd_lag2', 'mbd_lag3', 'EMA_3', 'active_lag1', 'ypred_otdelno',
           'lastactive']] = 0
    dfcop['state_i']=100
    print('Последний цикл')
    for l in tqdm(range(le)):
        nas = 0
        gen_cfips = max_cfips+l+1 # номер нового сгенерированного cfips
        dfnew = dfcop.copy()
        dfnew['cfips'] = gen_cfips
        df =kol*[0]
        for i in range(kol):
            df[i] = raw[raw['cfips'] == matrica[i, l]].copy()
            df[i].reset_index(drop=True, inplace=True)
            dfnew['active'] = dfnew['active'] + df[i]['active']
            dfnew['lastactive'] = dfnew['lastactive'] + df[i]['lastactive']
            nas = nas + df[i]['Population'][0]
            # dfnew['proc_covill'] = dfnew['proc_covill'] + df[i]['proc_covill']*df[i]['Population'][0]
            # dfnew['pct_college'] = dfnew['pct_college'] + df[i]['pct_college'] * df[i]['Population'][0]
            # dfnew['median_hh_inc'] = dfnew['median_hh_inc'] + df[i]['median_hh_inc'] * df[i]['Population'][0]
        dfnew['microbusiness_density'] = dfnew['active']/nas
        # dfnew['proc_covill'] = dfnew['proc_covill']/nas
        # dfnew['pct_college'] = dfnew['pct_college']/nas
        # dfnew['median_hh_inc'] = dfnew['median_hh_inc'] /nas
        raw = pd.concat([raw,dfnew], ignore_index=True)
    return raw


def posle_sglashivfnia(raw, rezult):
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    # создаем блек-лист cfips которых не используем в тесте
    #min_int = 60; max_int = 100 #lastactive = 10142 error 2.125946 -0.019663     0.005498
    #min_int = 100; max_int = 140 #lastactive = 6000 error 1.853605 -0.006954     0.004485
    #min_int = 140; max_int = 190 #lastactive = 6690 error 1.744299 -0.000703     0.004817
    #min_int = 190; max_int = 230 #lastactive = 9690 error 1.394248  0.007670     0.009076
    #min_int = 230; max_int = 300 #lastactive = 16500 error 1.519239  0.010882     0.004581
    #min_int = 300; max_int = 500 #lastactive =17500  error 1.222168  0.007713     0.001428
    #min_int = 500; max_int = 1000 #lastactive =1500  error 1.065339  0.018434     0.003156
    #min_int = 1000; max_int = 2000 #lastactive = 2000 error 1.025943  0.027024     0.004030
    #min_int = 2000; max_int = 4000 #lastactive = 4500 error 0.909389  0.034093     0.004325
    #min_int = 4000; max_int = 8000 #lastactive = 9500 error 0.858602  0.054814     0.003971
    #min_int = 8000; max_int = 16000 #lastactive = 7200 error 0.701923  0.072343    0.005001
    #min_int = 16000; max_int = 32000 #lastactive = 15000 error 0.861022  0.081539     0.008299
    #min_int = 32000; max_int = 64000 #lastactive = 64000 error 0.751408  0.082716     0.006263
    #min_int = 22000; max_int = бескон. #lastactive = 20590
    min_int = 120
    max_int = 1000000000000
    blac_test_cfips = raw.loc[(raw['lastactive'] < min_int) | (raw['lastactive'] >= max_int), 'cfips']
    blac_test_cfips = blac_test_cfips.unique()
    max_cfips = raw['cfips'].max()  # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    for state_i in [45]: # при 0 и min_int = 140 удаление элементов негативно при min_int = 22000 тоже не очень
        for param in [1]: #1.403340  0.015789   144.360897
            for lastactive in range(120,121,90): # при 30 - 1.404981  0.014148     0.000272
                #max_active = 5730679879807890879767
                #param = 1
                #lastactive = -1
                # raw = raw[raw['cfips'] <= max_cfips]
                # создаем блек-лист cfips который не нужен в трайне
                # if lastactive >= max_int:
                #     maska = (raw['lastactive'] < min_int)|((raw['lastactive'] > max_int)&
                #                                            (raw['lastactive'] < lastactive))
                # elif lastactive <= min_int:
                #     maska = (raw['lastactive'] < lastactive)
                # else:
                #     print('Чо за ерунда?')

                maska = (raw['lastactive'] < lastactive)#&(raw['state_i'] == state_i)

                blac_cfips = raw.loc[maska, 'cfips'] # убираем из трайна
                blac_cfips = blac_cfips.unique()
                # создаем гибриды из cfips для трайна
                # raw = new_cfips(raw, lastactive, max_cfips)
                raw = new_target(raw,1)# param/10000)
                raw, train_col = build_lag(raw, param)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
                start_val = 6  # первый месяц валидации с которого проверяем модель
                stop_val = 38  # последний месяц валидации до которого проверяем модель

                #модель без блек листа
                # raw, blac_cfips = modeli_po_mesiacam(raw, start_val, stop_val, train_col, [])
                # raw['y_no_blac'] = raw['ypred']
                # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_no_blac.csv", index=False)

                # валидация по результату обработки всех месяцев валидации в цикле
                raw, blac_cfips = modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips)
                rezult = validacia(raw, start_val, stop_val, rezult, blac_test_cfips, max_cfips, lastactive, param, state_i)
                # print('Сортировка по dif_no_blac')
                # rezult.sort_values(by='dif_no_blac', inplace=True, ascending=False)
                # print(rezult.head(22))
                print('Сортировка по dif_err')
                rezult.sort_values(by='dif_err', inplace=True, ascending=False)
                print(rezult.head(22))


# МОДЕЛЬ.
def glavnaia(raw, rezult):
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['median_hh_inc'] = raw.groupby('cfips')['median_hh_inc'].ffill()
    raw['ypred'] = 0
    max_cfips = raw['cfips'].max() # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    lastactive = 40  # оптимально 205
    # for lastactive in range(46, 51, 1):
    #     for param in range(41, lastactive+1, 1):
    raw = raw[raw['cfips'] <= max_cfips]
    #raw = new_cfips(raw, param, max_cfips)
    posle_sglashivfnia(raw, rezult)

if __name__ == "__main__":
    # train, test = obrabotka_filtr.start()
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)
    rezult = pd.DataFrame(columns=['lastactive','param', 'kol', 'error', 'dif_err', 'dif_no_blac'])
    #raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv")
    # glavnaia(raw, rezult)

    #raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_otdelno.csv")

    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_no_blac.csv")

    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['ypred'] = 0
    #glavnaia(raw, rezult) # 1.379034  0.010546  146.796656
    posle_sglashivfnia(raw, rezult)
