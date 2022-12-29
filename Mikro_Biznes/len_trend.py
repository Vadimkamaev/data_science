# модель на основе длины трендов
import pandas as pd
from servise_ds import okno
import obrabotka_filtr

# обработка прохода 1-го символа в тренде
def len_trend_1(znak_trend, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend):
    if finish_trend:
        if tip_fin_trend == '0':
            tip_fin_trend = znak_trend
            l_fin_trend += 1
        elif tip_fin_trend != znak_trend:
            finish_trend = False
            return True, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend
        else:
            l_fin_trend += 1
    else:
        if tip_trend == znak_trend:
            l_trend += 1
        else:
            if tip_trend != '0':
                trend.loc[len(trend.index)] = (l_trend, znak_trend)
                tip_trend = znak_trend
                l_trend = 1
                return True, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend
            else:
                tip_trend = znak_trend
                l_trend = 1
    return False, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend

# Подсчет средней длины тренда
def len_trend(loc_train):
    l = len(loc_train)
    tip_fin_trend = '0'  # тип последнего тренда
    l_fin_trend = 0  # длина последнего тренда
    finish_trend = True
    trend = pd.DataFrame(columns=['l', 'tip'])  # список длин трендов того же направления, что и последний
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
                    len_trend_1('=', finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend)
                if br :
                    break
            if i_loc_train > i_1: # тренд восходящий
                br, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend = \
                    len_trend_1('+', finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend)
                if br :
                    break
            if i_loc_train < i_1: # тренд вниз
                br, finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend = \
                    len_trend_1('-', finish_trend, tip_fin_trend, l_fin_trend, tip_trend, l_trend, trend)
                if br :
                    break
        i = j
    l_mean = trend[trend['tip']==tip_fin_trend]['l'].mean()
    oll_l_mean = trend['l'].mean()
    if pd.isna(oll_l_mean):
        oll_l_mean = 1
        print('NAN')
    return l_mean, oll_l_mean, l_fin_trend, tip_fin_trend

# Подсчет длин трендов и прогноз на основе этой модели
def oll_len_trend(train):
    unic_cfips = train['cfips'].unique()
    df_unic_cfips = pd.DataFrame({'l_mean':0, 'oll_l_mean':0, 'l_finish':0}, index=unic_cfips)
    for cfips in unic_cfips:
        l_mean, oll_l_mean, l_finish, tip_fin_trend = len_trend(train[train['cfips'] == cfips])
        df_unic_cfips.loc[cfips] = (l_mean, oll_l_mean, l_finish)
    return df_unic_cfips

# Подсчет длин трендов и прогноз на основе этой модели
def len_trend_do_month(train, n): # n - номер месяца, который мы предсказываем
    pravilno_razvorot = 0
    kol_razvorot = 0
    err_preds = 0
    err_0 = 0
    pravilno_prodolshit = 0
    kol_prodolshit = 0
    unic_cfips = train['cfips'].unique()
    df_unic_cfips = pd.DataFrame({'l_mean':0, 'oll_l_mean':0, 'l_finish':0}, index=unic_cfips)
    for cfips in unic_cfips:
        loc_train = train[train['cfips'] == cfips]
        l_mean, oll_l_mean, l_finish, tip_fin_trend = len_trend(loc_train.iloc[:n])
        # предсказание
        preds = 0
        real = loc_train['microbusiness_density'].iloc[n] - loc_train['microbusiness_density'].iloc[n-1] # реальный тренд, который мы пытаемся предсказать
        # if l_finish == l_mean: # -2 - 28./-7.4 - 24; -3 - 29/0.97
            # тренд согласно предсказанию должен развернутся. Не работает. Вероятность сохранения тренда выше
            # всегда
        #if (l_finish - oll_l_mean > -1) & (l_finish - oll_l_mean < 1): # 0.74
        #if (l_finish - oll_l_mean > 0) & (l_finish - oll_l_mean < 1): # предсказание тренда = -1.8235890328796218
        if (l_finish - oll_l_mean > -0.5) & (l_finish - oll_l_mean < 1): #предсказание тренда = -1.8188457376270748
            # +2 rez = 15.52; +3  rez = 14.55
            # +1 rez = 11.95; 0 rez = 5.32; -1 rez = 3.39; -2 rez = 3.29; -3 rez = 3.48; -5 rez = 3.63
            # 5 - 12.; 2 - 14./18/14; 1 - 16./16; 0 - 15/14/6; +1 - 14/17/8; +2 - 13/26; +3 - 29/21;
            # тренд согласно предсказанию должен сохранится
            if tip_fin_trend == '-':
                if real > 0:
                    pravilno_prodolshit -= 1
                    kol_prodolshit += 1
                else:
                    pravilno_prodolshit += 1
                    kol_prodolshit += 1
            elif tip_fin_trend == '+':
                if real > 0:
                    pravilno_prodolshit += 1
                    kol_prodolshit += 1
                else:
                    pravilno_prodolshit -= 1
                    kol_prodolshit += 1
        # предсказание значения
        #k = 0.003 # Ошибка предсказания меньше ошибки модели = на 8.403000117876218
        #k = 0.004 Ошибка предсказания меньше ошибки модели = на 1.0330224599929352
        k = 0.003
        if l_finish - oll_l_mean + 2 < 0:
            trend = 1+k
        elif l_finish - oll_l_mean + 0 > 0: # на 6.36015339435653
        #elif l_finish - oll_l_mean + 1 > 0: # лучшее было это. Ошибка предсказания < ошибки модели на 8.403000117876218
            trend = 1
        else:
            trend = 1 + k / (l_finish - oll_l_mean + 3)**2
        # print('l_finish=',l_finish,'oll_l_mean=',oll_l_mean)
        # print('trend=',trend)
        if tip_fin_trend == '-':
            preds = loc_train['microbusiness_density'].iloc[n-1] / trend
        elif tip_fin_trend == '+':
            preds = loc_train['microbusiness_density'].iloc[n - 1] * trend
        else:
            preds = loc_train['microbusiness_density'].iloc[n-1]
        err_preds += abs(loc_train['microbusiness_density'].iloc[n]-preds)
        err_0 += abs(real)

        df_unic_cfips.loc[cfips] = (l_mean, oll_l_mean, l_finish)
        rez = pravilno_prodolshit / kol_prodolshit * 100 if kol_prodolshit > 0.001 else 0
    print ('pravilno_prodolshit =', rez, 'kol =', kol_prodolshit)
    print('err_preds =', err_preds, 'err_0 =', err_0)
    return rez, err_0, err_preds

def optim(train):
    sum = 0
    sum_err0 = 0
    sum_errpreds = 0
    i = 15
    while i < 39:
        rez, err_0, err_preds = len_trend_do_month(train, i)
        sum += rez
        sum_err0 += err_0
        sum_errpreds += err_preds
        print('sum_err0=',sum_err0,'sum_errpreds=',sum_errpreds)
        i +=1
    rez = sum / (39-15)
    best =  sum_err0 - sum_errpreds
    proc = best / sum_err0
    print('предсказание тренда =', rez, )
    print('Ошибка предсказания меньше ошибки модели = на', best, 'положительное число это хорошо')
    print('в % =', proc)


if __name__ == "__main__":
    train, test = obrabotka_filtr.start()
    optim(train)
    # df_unic_cfips.sort_values(by='oll_l_mean', inplace=True)
    # g = df_unic_cfips[['l_mean', 'l_finish']].groupby(['l_finish']).count()
    # g.rename(columns={'l_mean': 'count'})
    # z = df_unic_cfips.groupby(['l_finish']).mean()
 #   okno.vewdf(df_unic_cfips)