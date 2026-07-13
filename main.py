import os
import subprocess
import sys

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def list_videos():
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        return []
    return [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]

def print_header():
    clear_console()
    print("=" * 60)
    print("   СИСТЕМА ОБНАРУЖЕНИЯ И ОТСЛЕЖИВАНИЯ ОБЪЕКТОВ")
    print("   (устойчивость к смазыванию изображения)")
    print("=" * 60)

def select_video():
    videos = list_videos()
    if not videos:
        print("Ошибка: папка experiments/ пуста или не найдена.")
        input("Нажмите Enter...")
        return None
    print("\nДоступные видео:")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v}")
    print("  0. Назад")
    try:
        choice = int(input("\nВыберите видео: "))
        if choice == 0:
            return None
        return videos[choice - 1]
    except (ValueError, IndexError):
        print("Неверный выбор.")
        return None

def select_tracker():
    print("\nВыберите трекер:")
    print("  1. CSRT")
    print("  2. Optical Flow")
    print("  3. CamShift")
    print("  4. Все трекеры")
    print("  0. Назад")
    try:
        choice = int(input("\nВыбор: "))
        if choice == 0:
            return None
        trackers = {
            1: ["test_csrt"],
            2: ["test_optical_flow"],
            3: ["test_camshift"],
            4: ["test_csrt", "test_optical_flow", "test_camshift"]
        }
        return trackers.get(choice, None)
    except (ValueError, IndexError):
        print("Неверный выбор.")
        return None

def select_hybrid():
    print("\nВыберите гибрид:")
    print("  1. YOLO+CSRT")
    print("  2. YOLO+Optical Flow")
    print("  3. YOLO+CamShift")
    print("  4. Все гибриды")
    print("  0. Назад")
    try:
        choice = int(input("\nВыбор: "))
        if choice == 0:
            return None
        hybrids = {
            1: ["hybrid_yolo_csrt"],
            2: ["hybrid_yolo_of"],
            3: ["hybrid_yolo_camshift"],
            4: ["hybrid_yolo_csrt", "hybrid_yolo_of", "hybrid_yolo_camshift"]
        }
        return hybrids.get(choice, None)
    except (ValueError, IndexError):
        print("Неверный выбор.")
        return None

def select_model():
    models = ['n', 's', 'm', 'l', 'x']
    print("\nВыберите модель YOLO:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. YOLOv8{m}")
    print("  6. Все модели")
    print("  0. Назад")
    try:
        choice = int(input("\nВыбор: "))
        if choice == 0:
            return None
        if choice == 6:
            return models
        return [models[choice - 1]]
    except (ValueError, IndexError):
        print("Неверный выбор.")
        return None

def select_model_for_hybrid():
    models = ['n', 's', 'm', 'l', 'x']
    print("\nВыберите модель для гибрида (YOLOv8n по умолчанию):")
    for i, m in enumerate(models, 1):
        print(f"  {i}. YOLOv8{m}")
    print("  6. Все модели")
    print("  0. Назад")
    try:
        choice = int(input("\nВыбор: "))
        if choice == 0:
            return None
        if choice == 6:
            return models
        return [models[choice - 1]]
    except (ValueError, IndexError):
        print("Неверный выбор. Будет использована модель n по умолчанию.")
        return ['n']

def select_class():
    print("\nВыберите класс объекта (человек по умолчанию):")
    print("  0. Человек")
    print("  1. Велосипед")
    print("  2. Автомобиль")
    print("  3. Мотоцикл")
    print("  9. Назад (отмена выбора видео)")
    try:
        choice = int(input("\nВыбор: "))
        if choice == 9:
            return None
        if choice in [0, 1, 2, 3]:
            return choice
        print("Неверный выбор. Будет использован класс 0 (человек).")
        return 0
    except ValueError:
        print("Неверный выбор. Будет использован класс 0 (человек).")
        return 0

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    while True:
        print_header()
        print("\nГлавное меню:")
        print("  1. Тестирование трекеров")
        print("  2. Тестирование YOLO")
        print("  3. Тестирование гибридных систем")
        print("  4. Все методы (трекеры + YOLO + гибриды)")
        print("  0. Выход")
        choice = input("\nВыберите действие: ")
        if choice == '0':
            print("До свидания!")
            sys.exit(0)
        video = select_video()
        if video is None:
            continue
        class_id = select_class()
        if class_id is None:
            continue
        if choice == '1':
            trackers = select_tracker()
            if trackers is None:
                continue
            for t in trackers:
                print(f"\nЗапуск {t}.py...")
                subprocess.run(["python", f"{t}.py", "--video", video])
        elif choice == '2':
            models = select_model()
            if models is None:
                continue
            for m in models:
                print(f"\nЗапуск test_yolo.py с моделью YOLOv8{m} и классом {class_id}...")
                subprocess.run(["python", "test_yolo.py", "--video", video, "--model", m, "--class", str(class_id)])
        elif choice == '3':
            hybrids = select_hybrid()
            if hybrids is None:
                continue
            models = select_model_for_hybrid()
            if models is None:
                continue
            for h in hybrids:
                for m in models:
                    print(f"\nЗапуск {h}.py с моделью YOLOv8{m} и классом {class_id}...")
                    subprocess.run(["python", f"{h}.py", "--video", video, "--model", m, "--class", str(class_id)])
        elif choice == '4':
            print("\nЗапуск всех тестов...")
            for t in ["test_csrt", "test_optical_flow", "test_camshift"]:
                subprocess.run(["python", f"{t}.py", "--video", video])
            for m in ['n', 's', 'm', 'l', 'x']:
                subprocess.run(["python", "test_yolo.py", "--video", video, "--model", m, "--class", str(class_id)])
            for h in ["hybrid_yolo_csrt", "hybrid_yolo_of", "hybrid_yolo_camshift"]:
                for m in ['n', 's', 'm', 'l', 'x']:
                    subprocess.run(["python", f"{h}.py", "--video", video, "--model", m, "--class", str(class_id)])
        else:
            print("Неверный выбор.")
        input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    main()