# Голосовой транскрибатор

Графическое приложение для записи и транскрибации речи с использованием Whisper и PyQt5.

## Возможности
- Выбор модели Whisper (tiny → large-v3)
- Переключение CPU/GPU
- Запись по горячей клавише
- Автоопределение языка (русский поддерживается)
- Фильтр тишины (VAD)
- Копирование результата в буфер

## Требования
- Python 3.12.8
- NVIDIA CUDA 13.0/13.1 (для GPU)
- Fedora 43 Linux

## Установка
```bash
git clone https://github.com/OlegNassikanov/voice-transcriber.git
cd voice-transcriber
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Запуск
```bash
python src/transcriber_faster_whisper.py
```
