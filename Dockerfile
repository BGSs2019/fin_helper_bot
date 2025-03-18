# образ шаблон
FROM python:3.12-slim

# копировать все файлы из директории компьютера в директорию контейнера
COPY . .

# установка библиотек 
RUN pip install -r requirements.txt

# запуск команд
CMD ["python", "bot.py"]