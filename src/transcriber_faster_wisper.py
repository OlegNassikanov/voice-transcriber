#!/usr/bin/env python3
"""
Голосовой транскрибатор для Fedora (Faster-Whisper с GPU)
Требует: pip install PyQt5 pyaudio faster-whisper pyperclip numpy onnxruntime-gpu
"""

import sys
import pyaudio
import wave
import threading
import tempfile
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QComboBox,
                             QLabel, QRadioButton, QButtonGroup, QGroupBox)
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont
import pyperclip
from faster_whisper import WhisperModel

class TranscriptionSignals(QObject):
    """Сигналы для обновления GUI из другого потока"""
    text_ready = pyqtSignal(str)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

class VoiceTranscriber(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signals = TranscriptionSignals()
        self.is_recording = False
        self.audio_frames = []
        self.audio_thread = None
        self.model = None
        self.temp_file = None

        # Настройки аудио
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        self.check_cuda()
        self.init_ui()
        self.check_audio_devices()
        self.load_model()

        # Подключаем сигналы
        self.signals.text_ready.connect(self.update_transcription)
        self.signals.status_update.connect(self.update_status)
        self.signals.error_occurred.connect(self.show_error)

    def check_cuda(self):
        """Проверка доступности CUDA"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            self.cuda_available = 'CUDAExecutionProvider' in providers
            print(f"ONNX Runtime providers: {providers}")
            print(f"CUDA доступна: {self.cuda_available}")
        except Exception as e:
            print(f"Ошибка проверки CUDA: {e}")
            self.cuda_available = False

    def check_audio_devices(self):
        """Проверка доступных аудиоустройств"""
        try:
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            print(f"\n=== Найдено аудиоустройств: {device_count} ===")

            default_input = p.get_default_input_device_info()
            print(f"Устройство по умолчанию: {default_input['name']}")

            p.terminate()
        except Exception as e:
            print(f"Ошибка проверки устройств: {e}")

    def init_ui(self):
        self.setWindowTitle('Голосовой транскрибатор')
        self.setGeometry(100, 100, 700, 600)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Настройки модели
        settings_group = QGroupBox("Настройки")
        settings_layout = QVBoxLayout()

        # Выбор модели
        model_layout = QHBoxLayout()
        model_label = QLabel("Модель Whisper:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'])
        self.model_combo.setCurrentText('base')
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()

        # CPU/GPU переключатель
        device_layout = QHBoxLayout()
        device_label = QLabel("Устройство:")
        self.device_group = QButtonGroup()
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.gpu_radio)

        # Проверяем доступность CUDA
        if self.cuda_available:
            self.gpu_radio.setChecked(True)
            # Получаем имя GPU
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    gpu_name = result.stdout.strip().split('\n')[0]
                    device_label.setText(f"Устройство ({gpu_name}):")
            except:
                pass
        else:
            self.cpu_radio.setChecked(True)
            self.gpu_radio.setEnabled(False)
            self.gpu_radio.setToolTip("CUDA недоступна. Установите onnxruntime-gpu")

        self.cpu_radio.toggled.connect(self.on_device_changed)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.gpu_radio)
        device_layout.addStretch()

        settings_layout.addLayout(model_layout)
        settings_layout.addLayout(device_layout)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Кнопка записи
        self.record_button = QPushButton('🎤 Начать запись')
        self.record_button.setMinimumHeight(50)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)

        # Статус
        self.status_label = QLabel('Готов к работе')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: gray; font-size: 12px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Текстовое поле для транскрипции
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText('Здесь появится транскрибированный текст...')
        self.text_edit.setFont(QFont('Arial', 12))
        layout.addWidget(self.text_edit)

        # Кнопки внизу
        button_layout = QHBoxLayout()

        # Кнопка очистки
        self.clear_button = QPushButton('🗑️ Очистить')
        self.clear_button.setMinimumHeight(40)
        self.clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_button)

        # Кнопка копирования
        self.copy_button = QPushButton('📋 Копировать в буфер')
        self.copy_button.setMinimumHeight(40)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(self.copy_button)

        layout.addLayout(button_layout)

    def load_model(self):
        """Загрузка модели Whisper"""
        try:
            model_size = self.model_combo.currentText()
            device = "cuda" if (self.gpu_radio.isChecked() and self.cuda_available) else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            self.signals.status_update.emit(f'Загрузка модели {model_size} на {device.upper()}...')

            # Загружаем модель Faster-Whisper
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=os.path.expanduser("~/.cache/whisper")
            )

            self.signals.status_update.emit(f'✅ Модель {model_size} загружена ({device.upper()})')
            print(f"Модель загружена: {model_size} на {device}")

        except Exception as e:
            self.signals.error_occurred.emit(f'Ошибка загрузки модели: {str(e)}')
            print(f"Полная ошибка загрузки: {e}")
            import traceback
            traceback.print_exc()

    def on_model_changed(self):
        """Обработчик смены модели"""
        if not self.is_recording:
            self.load_model()

    def on_device_changed(self):
        """Обработчик смены устройства"""
        if not self.is_recording:
            self.load_model()

    def toggle_recording(self):
        """Переключение записи"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Начало записи"""
        self.is_recording = True
        self.audio_frames = []
        self.record_button.setText('⏹️ Остановить запись')
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.signals.status_update.emit('🔴 Идёт запись...')

        # Блокируем настройки во время записи
        self.model_combo.setEnabled(False)
        self.cpu_radio.setEnabled(False)
        self.gpu_radio.setEnabled(False)

        # Запуск записи в отдельном потоке
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

    def record_audio(self):
        """Запись аудио"""
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()

            # Получаем устройство по умолчанию
            default_device = p.get_default_input_device_info()
            device_index = default_device['index']

            print(f"Используется устройство: {default_device['name']}")

            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK,
            )

            while self.is_recording:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_frames.append(data)
                except Exception as e:
                    continue

        except Exception as e:
            self.signals.error_occurred.emit(f'Ошибка записи: {str(e)}')
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()

    def stop_recording(self):
        """Остановка записи"""
        self.is_recording = False
        self.record_button.setText('🎤 Начать запись')
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Ждём завершения потока записи
        if self.audio_thread:
            self.audio_thread.join()

        # Разблокируем настройки
        self.model_combo.setEnabled(True)
        self.cpu_radio.setEnabled(True)
        if self.cuda_available:
            self.gpu_radio.setEnabled(True)

        # Запуск транскрипции в отдельном потоке
        threading.Thread(target=self.transcribe_audio, daemon=True).start()

    def transcribe_audio(self):
        """Транскрибация аудио"""
        temp_file_path = None
        try:
            if not self.audio_frames:
                self.signals.error_occurred.emit('Нет записанных данных')
                return

            self.signals.status_update.emit('⏳ Транскрибация...')

            # Сохраняем аудио во временный файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_file_path = temp_wav.name

            wf = wave.open(temp_file_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()

            print(f"Аудио сохранено: {temp_file_path}")

            # Транскрибация с автоопределением языка
            segments, info = self.model.transcribe(
                temp_file_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Собираем текст
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text)
                print(f"Сегмент [{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")

            transcription = ' '.join(transcription_parts).strip()
            detected_lang = info.language if hasattr(info, 'language') else 'unknown'

            if transcription:
                self.signals.text_ready.emit(transcription)
                self.signals.status_update.emit(f'✅ Готово (язык: {detected_lang})')
                print(f"Итоговая транскрипция: {transcription}")
            else:
                self.signals.status_update.emit('⚠️ Речь не распознана')

        except Exception as e:
            self.signals.error_occurred.emit(f'Ошибка транскрибации: {str(e)}')
            print(f"Полная ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Удаляем временный файл
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

    def update_transcription(self, text):
        """Обновление текста транскрипции"""
        current_text = self.text_edit.toPlainText()
        if current_text:
            self.text_edit.append('\n' + text)
        else:
            self.text_edit.setText(text)

    def update_status(self, status):
        """Обновление статуса"""
        self.status_label.setText(status)
        self.status_label.setStyleSheet("color: gray; font-size: 12px; padding: 5px;")

    def show_error(self, error):
        """Показ ошибки"""
        self.status_label.setText(f'❌ {error}')
        self.status_label.setStyleSheet("color: red; font-size: 12px; padding: 5px;")

    def copy_to_clipboard(self):
        """Копирование в буфер обмена"""
        text = self.text_edit.toPlainText()
        if text:
            pyperclip.copy(text)
            self.signals.status_update.emit('📋 Скопировано в буфер обмена!')
        else:
            self.signals.status_update.emit('Нечего копировать')

    def clear_text(self):
        """Очистка текстового поля"""
        self.text_edit.clear()
        self.signals.status_update.emit('Текст очищен')

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = VoiceTranscriber()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
