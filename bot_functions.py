# импорт библиотек
# работа с данными
import numpy as np
import pandas as pd

# графика
from matplotlib import pyplot as plt
from matplotlib import ticker

# метрика
from sklearn.metrics import mean_squared_error

# работа с нейронными сетями
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

# полоса загрузки
from tqdm import tqdm

# извлечение данных
def extract_data():
    '''Функция извлекает и обрабатывает данные. Возвращает обработанный датасет.'''
    # чтение csv
    df = pd.read_csv('fin.csv', sep=';')
    # удаление лишних данных
    df = df[['Дата операции', 'Сумма операции', 'Категория', 'Описание']]
    # переименование столбцов
    df.columns = ['dt', 'sum', 'category', 'description']
    # приведение типов
    df['dt'] = pd.to_datetime(df['dt'], dayfirst=True)
    df['sum'] = pd.to_numeric(df['sum'].str.replace(',', '.'))
    # Установка индекса
    df.index = df['dt']
    # сортировка по индексу
    df.sort_index(inplace=True)
    # добавление столбца баланса
    df['balance'] = df['sum'].cumsum()

    return df

# отображение баланса
def plot_balance(df, start_date, end_date):
    '''Функция принимает на вход 2 даты. Строит график баланса'''
    plt.figure(figsize=(12, 4))
    plt.plot(df.loc[start_date:end_date, 'balance'])
    plt.grid()
    plt.title('График баланса')
    plt.xlabel('Дата')
    plt.ylabel('Баланс')
    plt.savefig('balance.png')

# годовые изменения баланса
def plot_years(df):
    '''Функция строит график баланса за последние 3 года'''
    # определение последнего года в данных
    last_year = df.index.max().year

    # построение графиков баланса
    fig, ax = plt.subplots(nrows=3, figsize=(16, 10))
    fig.suptitle('Баланс за последние 3 года')
    fig.subplots_adjust(hspace=0.5)

    for i in range(3):
        year = str(last_year - i)
        df.loc[year + '-01-01':year +'-12-31', 'balance'].plot(ax=ax[i])
        
        xticks = ticker.MaxNLocator(12)
        yticks = ticker.MaxNLocator(10)
        ax[i].xaxis.set_major_locator(xticks)
        ax[i].yaxis.set_major_locator(yticks)

        ax[i].grid()
        ax[i].tick_params(labelrotation=0)
        ax[i].set_title(year + ' год')
        ax[i].set_xlabel('Дата')
        ax[i].set_ylabel('Баланс')

    plt.savefig('3_years.png')

# месячные траты по категориям
def plot_scats(df, month, year):
    '''Функция принимает на вход месяц и год. Выводит диаграмму трат по категориям.'''
    # формирование датафрейма месяца
    month_data = df.loc[str(year) + '-' + str(month)]
    month_pivot = month_data.pivot_table(index='category', values='sum', aggfunc='sum')\
        .sort_values(by='sum', ascending=True)['sum']

    # построение графиков
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(month_pivot.index, 
            month_pivot)
    ax.grid()
    ax.set_xlabel('Сумма расходов')
    ax.set_ylabel('Категория')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title('Сумма расходов по категориям')

    fig.savefig('scats.png')

# месячные траты по частоте
def plot_rcats(df, month, year):
    '''Функция принимает на вход месяц и год. Выводит диаграмму трат по частоте.'''
    # формирование датафрейма месяца
    month_data = df.loc[str(year) + '-' + str(month)]
    month_pivot = month_data.pivot_table(index='category', values='sum', aggfunc='count')\
        .sort_values(by='sum', ascending=False)['sum']

    # построение графиков
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(month_pivot.index, 
            month_pivot,
            color='orange')
    ax.grid()
    ax.set_xlabel('Частота расходов')
    ax.set_ylabel('Категория')
    ax.invert_yaxis()
    ax.set_title('Частота расходов по категориям')
    
    fig.savefig('rcats.png')

# расчет времени опустошения бюджета
def balance_count(df):
    '''Функция возвращает количество месяцев, которое можно прожить на текущий баланс.'''
    # определение текущего баланса
    balance = np.abs(df.iloc[-1]['balance'])

    # определение среднемесячных трат
    avg_month = np.abs(df['2023-01-01': '2024-12-31']['sum'].resample('ME').sum().mean())

    # вывод количества месяцев
    eval = round(balance/avg_month, 2)
    return eval

# функция для создания дополнительных признаков
def make_features(dataframe, max_lag, rolling_mean_size):
    '''Функция принимает на вход набор данных, максимальную величину лага для признаков отставания,
    размер группировки для скользящей средней'''
    data = dataframe.copy()
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['balance'].shift(lag)

    data['rolling_mean'] = data['balance'].shift().rolling(rolling_mean_size).mean()
    
    return data

# определение класса датасета
class TabDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len_dataset = len(data)

    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, index):
        sample = self.data[index, :-1]
        target = self.data[index, -1:]
        return sample, target

# предсказание последних 30 дней
def predict(df):
    '''Функция строит график предсказания расходов'''
    # создание признаков
    features = make_features(df.drop(['dt', 'category', 'sum', 'description'], axis=1), 30, 7)
    features.dropna(inplace=True)

    # разделение выборки на тренировочную, валидационную и тестовую
    X_train = features[:round(len(features)*0.7)].drop('balance', axis=1)
    y_train = features[:round(len(features)*0.7)]['balance']
    X_val = features[round(len(features)*0.7): round(len(features)*0.8)].drop('balance', axis=1)
    y_val = features[round(len(features)*0.7): round(len(features)*0.8)]['balance']
    X_test = features[round(len(features)*0.8):].drop('balance', axis=1)
    y_test = features[round(len(features)*0.8):]['balance']

    # сборка выборок, переход к numpy
    train = np.hstack([X_train, y_train.values.reshape(len(y_train), 1)])
    val = np.hstack([X_val, y_val.values.reshape(len(y_val), 1)])
    test = np.hstack([X_test, y_test.values.reshape(len(y_test), 1)])

    # инициализация экземпеляров
    train_dataset = TabDataset(torch.FloatTensor(train))
    val_dataset = TabDataset(torch.FloatTensor(val))
    test_dataset = TabDataset(torch.FloatTensor(test))

    # определение загрузчиков
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # определение модели
    model = nn.Sequential()
    model.add_module('layer_1', nn.Linear(34, 32))
    model.add_module('relu', nn.ReLU())
    model.add_module('layer_2', nn.Linear(32, 16))
    model.add_module('relu', nn.ReLU())
    model.add_module('layer_3', nn.Linear(16, 1))

    # задание функции потерь и оптимизатора
    model_loss = nn.MSELoss()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # цикл обучения
    EPOCHS = 40
    train_loss = []
    train_mse = []
    val_loss = []
    val_mse = []

    for epoch in range(EPOCHS):
        
        # полоса загрузки
        train_loop = tqdm(train_loader, leave=False)

        # суммарное отклонение
        mse = []
        # обучение модели
        model.train()
        # расчет функций потерь
        running_train_loss = []
        for x, targets in train_loop:

            # прямой проход, расчет ошибки модели
            pred = model(x)
            loss = model_loss(pred, targets)

            # обратный проход
            # обнуление градиентов
            model_optimizer.zero_grad()
            # найти производную фукнции потерь
            loss.backward()
            # шаг оптимизации
            model_optimizer.step()
            # вывод данных
            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)
            train_loop.set_description(f'Epoch [{epoch+1}/{EPOCHS}], train_loss={mean_train_loss:.4f}')
            
            mse.append(mean_squared_error(pred.detach().numpy(), targets))

        # расчет значения метрики
        running_train_mse = sum(mse) / len(mse)
        # сохранение значений
        train_loss.append(mean_train_loss)
        train_mse.append(running_train_mse)

        # валидация
        model.eval()
        with torch.no_grad():
            # расчет функций потерь
            running_val_loss = []
            # суммарное отклонение
            mse = []
            for x, targets in val_loader:

                # прямой проход и расчёт ошибки
                pred = model(x)
                loss = model_loss(pred, targets)

                # вывод данных
                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss) / len(running_val_loss)

                mse.append(mean_squared_error(pred.detach().numpy(), targets))

            # расчет значения метрики
            running_val_mse = sum(mse) / len(mse)
            # сохранение значений
            val_loss.append(mean_val_loss)
            val_mse.append(running_val_mse)

        print(f'Epoch [{epoch+1}/{EPOCHS}], train_loss={mean_train_loss:.4f}, train_mse={running_train_mse:.4f}, val_loss={mean_val_loss:.4f}, val_mse={running_val_mse}')

    # Оценка лучшей модели на тестовой выборке
    mse = []
    test_preds = []

    model.eval()
    with torch.no_grad():
        for x, targets in test_loader:

            # прямой проход и расчёт ошибки
            pred = model(x)
            loss = model_loss(pred, targets)

            # расчёт
            mse.append(mean_squared_error(pred.detach().numpy(), targets))
            running_val_mse = sum(mse) / len(mse)
            # добавление предсказаний в список
            test_preds.extend(pred.detach().numpy().flatten())

    # расчет значения метрики
    print(f'Result MSE: {sum(mse) / len(mse)}')

    # график предсказаний
    fig, ax = plt.subplots()
    ax.plot(y_test.squeeze().tolist())
    ax.plot(test_preds)
    ax.grid()
    ax.set_title('Предсказание баланса')
    ax.set_xlabel('Дни')
    ax.set_ylabel('Баланс')

    plt.savefig('predict.png')