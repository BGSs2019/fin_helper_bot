# импорт библиотек
# работа с данными
import numpy as np
import pandas as pd

# графика
from matplotlib import pyplot as plt
from matplotlib import ticker

# работа с моделями машинного обучения
import torch
import torch.nn as nn
import torch.optim as optim
import copy

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

# предсказание последних 30 дней
def predict(df):
    '''Функция строит график предсказания расходов'''
    # создание признаков
    features = make_features(df.drop(['dt', 'category', 'sum', 'description'], axis=1), 30, 7)
    features.dropna(inplace=True)

    # разделение на тренировочную и тестовую выборки
    X_train = features[:-30].drop('balance', axis=1)
    y_train = features[:-30]['balance']
    X_test = features[-31:].drop('balance', axis=1)
    y_test = features[-31:]['balance']

    # переход к тензорам torch
    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    # определение модели нейросети
    model = nn.Sequential(
        nn.Linear(34, 48),
        nn.ReLU(),
        nn.Linear(48, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )

    # функция потерь и оптимизатор
    lossFunction = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # обучение нейросети
    epochs = 400
    batch_size = 20

    best_mse = np.inf
    best_weights = None
    history = []

    size = X_train.shape[0]

    for epoch in range(epochs):
        model.train()

        index = 0

        while index * batch_size <= size:
            
            X_batch = X_train[index:index + batch_size]
            y_batch = y_train[index:index + batch_size]

            y_pred = model(X_batch)

            loss = lossFunction(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            index += batch_size

        model.eval()
        y_pred = model(X_test)
        mse = lossFunction(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            print(f"best mse is {mse} on epoch {epoch}")
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # сохранение лучших весов модели
    model.load_state_dict(best_weights)

    # получение предсказаний на тестовой выборке
    test_preds = model(X_test).squeeze().tolist()

    # график предсказаний
    fig, ax = plt.subplots()
    ax.plot(y_test.squeeze().tolist())
    ax.plot(test_preds)
    ax.grid()
    ax.set_title('Предсказание баланса')
    ax.set_xlabel('Дни')
    ax.set_ylabel('Баланс')

    plt.savefig('predict.png')