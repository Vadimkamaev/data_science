import torch

# создаем набор случайных чисел в интервале от -10 до 10,
# rand - Возвращает тензор, заполненный случайными числами из равномерного распределения на интервале[0, 1)[ 0 ,1 )
# 100 - размер тензора, количество элементов
x_train = torch.rand(100)
x_train = x_train * 20.0 - 10.0

# создаем у для трайна - для обучающейся модели - синусоиду
y_train = torch.sin(x_train)

# создаем случайный шум. y_train.shape - размер тензора y_train
# randn - Возвращает тензор, заполненный случайными числами из нормального распределения
# со средним значением 0 и дисперсией 1 (также называемым стандартным нормальным распределением).
noise = torch.randn(y_train.shape) / 5.

# накладываем шум на функцию
y_train = y_train + noise

# перевораяиваем в вертикальный вид из горизонтального
x_train.unsqueeze_(1)
y_train.unsqueeze_(1);

# создаем валидационный датасет
# linspace - Создает одномерный тензор размера steps=100, значения которого равномерно распределены
# от start=-10 до end=10 включительно.
x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.sin(x_validation.data)

# перевораяиваем валидационный датасет в вертикальный вид из горизонтального
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1);

# СОЗДАЕМ МОДЕЛЬ
# создаем класс который наследуется от torch.nn.Module - нейросеть.
class SineNet(torch.nn.Module):
    # метод инициализации слоев. Предполагаем, что все слои одинакового размера
    def __init__(self, n_hidden_neurons): # n_hidden_neurons - количество скрытых нейронов в каждом слое
        super(SineNet, self).__init__() # инициализация родительского объекта
        # создание слоев:
        # 1-й слой назовем fc1 - fully connected layer - полносвязный слой - метод 'Linear'
        # 1 кол. входных нейронов, n_hidden_neurons - кол. выходных нейронов
        # если на входе не 1 - мерная функция, а n - мерная фигня, то на входе не 1, а n
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        # создаем функцию активации - берем сигмоиду. Здесь подойдет любая. Сигмоида самая простая
        self.act1 = torch.nn.Sigmoid()
        # создаем уще 1 полносвязный слой в котором 1 нейрон
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    # метод описывающий как слои нейронов последовательно применяются
    def forward(self, x):
        # сначала применяется слой fc1
        x = self.fc1(x)
        # то, что получилось в результате применения слоя fc1 передаем в функцию активации
        x = self.act1(x)
        # то что выдала функция активации передаем в слой fc2
        x = self.fc2(x)
        return x

sine_net = SineNet(50) # sine_net - нейронная сеть - объект который можно обучать

# оптимизатор - Adam один из вариантов градиентного спуска
# sine_net.parameters() - веса нейронной сети для оптимизации, lr=0.01 - шаг градиентного спуска
optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)

# функция потерь - Loss function
def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()