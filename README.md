# Сравнение работы различных алгоритмов отслеживания объектов при смазывании.


## 📋 Описание
Проект позволяет замерить скорость работы трекеров и детекторов на видео с ручной разметкой. Поддерживаются:
- Классические трекеры: CSRT, Optical Flow, CamShift
- Детекторы: YOLOv8 (n, s, m, l, x)
- Гибридные подходы: YOLO + CSRT, YOLO + Optical Flow, YOLO + CamShift


## 🚀 Установка
1. Клонируйте репозиторий:
```
git clone <url-репозитория>
cd <название-папки>
```
2. Установите зависимости:
```
pip install -r requirements.txt
```


## 📁 Структура проекта
```
project/
├── README.md                   # Документация
├── requirements.txt            # Зависимости проекта
├── frame_extractor.py          # Зависимости проекта
├── manual_labeler.py           # Зависимости проекта
├── renamer.py                  # Зависимости проекта
├── converter.py                # Зависимости проекта
├── test_csrt.py                # Реализация трекера CSRT
├── test_optical_flow.py        # Реализация оптического потока
├── test_camshift.py            # Реализация CamShift
├── test_yolo.py                # Реализация YOLOv8
├── hybrid_yolo_csrt.py         # Гибрид YOLO + CSRT
├── hybrid_yolo_of.py           # Гибрид YOLO + Optical Flow
├── hybrid_yolo_camshift.py     # Гибрид YOLO + CamShift
├── analyzer.py                 # Анализ и визуализация результатов
├── time_statistics.py          # Основной скрипт для замера производительности
├── main.py                     # Консольный интерфейс для управления экспериментами
├── plots/                      # Графики для анализа
└── experiments/
    └── video_name/
        ├── my_frames/             # Кадры из видео
        ├── results/               # Результаты работы методов
        ├── manual_labels.json     # Ручная разметка
        └── benchmark_results.json # Результаты замеров
```


## 🎯 Использование
### Подготовка данных
1. Поместите последовательность кадров в папку experiments/название_видео/
2. Извлеките кадры в папку my_frames/
3. Создайте файл manual_labels.json с разметкой первого кадра в формате:
```
{
    "frame_000000.jpg": [x, y, width, height],
    "frame_000001.jpg": [x, y, width, height],
    ...
}
```
### Запуск модулей
```
python <название модуля> --video <название папки (видео) с кадрами и разметкой>
```


## Параметры
```
--video     # Название видео (папка в experiments/)               - обязательный параметр для тестирований методов
--model     # Конфигурация YOLOv8 (n, s, m, l, x)                 - для тестирования YOLO и гибридов
--class	    # Класс объекта (0=человек, 2=автомобиль, 3=мотоцикл) - для YOLO и гибридов
```


## Примеры запуска модулей
```
python test_yolo --video MotorRolling --model n --class 3   # Запуск YOLOv8n для отслеживания мотоцикла на видео MotorRolling
python test_csrt --video MotorRolling                       # Запуск CSRT на видео MotorRolling
python analyzer                                             # Запуск модуля оценки и визуализации результатов
```


## 🤝 Вклад
Если вы нашли ошибку или хотите улучшить проект, создайте Issue или Pull Request.
