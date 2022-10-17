import pandas as pd

# получаем номер группы покупателя
def n_group(n):
    ng = 0
    while True:
        ng += n%10
        n = n//10
        if n == 0:
            return ng

# Функция, которая подсчитывает число покупателей, попадающих в каждую группу,
# если ID начинается с произвольного числа. На вход функция получает целые числа:
# n_customers (количество клиентов) и n_first_id (первый ID в последовательности).

def number_buyers(n_customers, n_first_id):
    l = len(str(n_customers+n_first_id))*9 # максимальное количество групп
    s_group = pd.Series([0]*l)
    for i in range(n_first_id, n_first_id + n_customers):
        ng = n_group(i)
        s_group[ng] = s_group[ng]+1
    s_group = s_group[s_group !=0]
    return s_group

# Функция, которая подсчитывает число покупателей, попадающих в каждую группу, если нумерация
# ID сквозная и начинается с 0. На вход функция получает целое число n_customers (кол. клиентов).
def number_buyers0(n_customers):
    return number_buyers(n_customers, 0)

if __name__ == '__main__':
    sg = number_buyers0(1456)
    print(sg)

# Протестирован этот вариант получения номера группы покупателя.
# Оказался в 2 с лишним раза медленнее.
# def n_group1(n):
#     return sum(list(map(int, list(str(n)))))

# Протестирован этот вариант.
# Оказался ~ на 10% медленнее при миллионе покупателей.
# def number_buyers(n_customers, n_first_id):
#     s_group = pd.Series(1, index = [n_group(n_first_id)])
#     for i in range(n_first_id +1, n_first_id + n_customers):
#         ng = n_group(i)
#         if ng in s_group.index:
#             s_group[ng] = s_group[ng]+1
#         else:
#             s_group = pd.concat([s_group, pd.Series(1, index=[ng])])
#     return s_group