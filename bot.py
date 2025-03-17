# импорт библиотек
import telebot
from datetime import datetime
import bot_functions

print('Hello world!')

# получение данных
df = bot_functions.extract_data()

start_time = datetime.now()

# получить api
def get_config():
    api_key = ""
    with open('config.ini', "r") as config_file:
        api_key = config_file.read()
    return api_key

# объявление экземпляра telebot
bot = telebot.TeleBot(get_config())

# действия при старте
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    if message.text == "/help" or message.text == "/start":
	    bot.send_message(message.chat.id, "Добрый день! Бот запущен в " + start_time.strftime("%d.%m.%Y %H:%M:%S"))

# получить сообщение
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "бот":
        # напечатать в консоли сообщение пользователя
        print(message.text)
        # ответить в чате
        bot.send_message(message.chat.id, "Добрый день! Бот работает с " + start_time.strftime("%d.%m.%Y %H:%M:%S"))
        
    if message.text.startswith("баланс"):
        # напечатать в консоли сообщение пользователя
        print(message.text)
        # получить начальную и конечную дату
        start_date, end_date = message.text.split()[1], message.text.split()[2]
        # построить график баланса пользователя
        bot_functions.plot_balance(df, start_date, end_date)
        bot.send_photo(message.chat.id, photo=open('balance.png', 'rb'))

    if message.text == "годовые":
        # напечатать в консоли сообщение пользователя
        print(message.text)
        # построить график годовых изменений
        bot_functions.plot_years(df)
        bot.send_photo(message.chat.id, photo=open('3_years.png', 'rb'))

    if message.text.startswith("сумма_по_категориям"):
        # напечатать в консоли сообщение пользователя
        print(message.text)
        # получить месяц и год
        month, year = message.text.split()[1], message.text.split()[2]
        # построить график сумм месячных трат по категориям
        bot_functions.plot_scats(df, month, year)
        bot.send_photo(message.chat.id, photo=open('scats.png', 'rb'))

    if message.text.startswith("частота_по_категориям"):
        # напечатать в консоли сообщение пользователя
        print(message.text)
        # получить месяц и год
        month, year = message.text.split()[1], message.text.split()[2]
        # построить график частот месячных трат по категориям
        bot_functions.plot_rcats(df, month, year)
        bot.send_photo(message.chat.id, photo=open('rcats.png', 'rb'))

    if message.text == "остаток":
        # напечатать в консоли сообщение пользователя
        print(message.text)
        # рассчитать остаток
        eval = bot_functions.balance_count(df)
        bot.send_message(message.chat.id, eval)

    if message.text == "предсказание":
        # напечатать в консоли сообщение пользователя
        print(message.text)
        # построить график предсказания
        bot_functions.predict(df)
        bot.send_photo(message.chat.id, photo=open('predict.png', 'rb'))


bot.polling(none_stop=True, interval=0)