import torch

def target_function(x):
    return 2**x * torch.sin(2**-x)

class RegressionNet(torch.nn.Module):
    # метод инициализации слоев. Предполагаем, что все слои одинакового размера
    def __init__(self, n_hidden_neurons): # n_hidden_neurons - количество скрытых нейронов в каждом слое
        super(RegressionNet, self).__init__() # инициализация родительского объекта
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

net = RegressionNet(10)


# ------Dataset preparation start--------:
x_train =  torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Dataset preparation end--------:


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

def loss(pred, target):
    squares = torch.abs(pred - target)
    return squares.mean()


for epoch_index in range(500):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()

# Проверка осуществляется вызовом кода:
def metric(pred, target):
   return (pred - target).abs().mean()

print(metric(net.forward(x_validation), y_validation).item())
# (раскомментируйте, если решаете задание локально)