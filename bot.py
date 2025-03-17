# импорт библиотек
import telebot
from datetime import datetime

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
        bot.send_message(message.chat.id, "Добрый день! Бот работает с " + start_time.strftime("%d.%m.%Y %H:%M:%S"))
    #write message text in the end of file
    print(message.text)

bot.polling(none_stop=True, interval=0)