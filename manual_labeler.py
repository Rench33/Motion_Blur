import cv2
import os
import json

class ManualLabeler:
    def __init__(self, frames_dir="my_frames", output_file="manual_labels.json"):
        self.frames_dir = frames_dir
        self.output_file = output_file
        self.labels = {}  # Словарь для хранения разметки
        self.load_existing_labels()
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.current_frame = None
        self.current_filename = None
        self.display_frame = None  # Масштабированная версия для отображения

    def load_existing_labels(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                self.labels = json.load(f)
            print(f"Загружено {len(self.labels)} существующих меток")
        else:
            print("Файл разметки не найден, начинаем с чистого листа")

    def save_labels(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.labels, f, indent=4)
        print(f"Разметка сохранена в {self.output_file}")

    def mouse_callback(self, event, x, y, flags, param):
        # Масштабируем координаты обратно к оригинальному размеру
        scale_x = self.current_frame.shape[1] / self.display_frame.shape[1]
        scale_y = self.current_frame.shape[0] / self.display_frame.shape[0]
        x_orig = int(x * scale_x)
        y_orig = int(y * scale_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x_orig, y_orig
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_frame = self.display_frame.copy()
                self.draw_instructions(temp_frame)
                ix_disp = int(self.ix / scale_x) # Масштабируем для отображения
                iy_disp = int(self.iy / scale_y)
                x_disp = int(x_orig / scale_x)
                y_disp = int(y_orig / scale_y)
                cv2.rectangle(temp_frame, (ix_disp, iy_disp), (x_disp, y_disp), (0, 255, 0), 2)
                cv2.imshow('Manual Labeler', temp_frame)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x_orig), min(self.iy, y_orig)
            x2, y2 = max(self.ix, x_orig), max(self.iy, y_orig)
            width, height = x2 - x1, y2 - y1
            self.labels[self.current_filename] = [x1, y1, width, height]
            print(f"Разметка сохранена: {self.current_filename} -> [{x1}, {y1}, {width}, {height}]")

            self.save_labels()

            final_display = self.display_frame.copy()
            self.draw_instructions(final_display)
            x1_disp = int(x1 / scale_x)
            y1_disp = int(y1 / scale_y)
            x2_disp = int(x2 / scale_x)
            y2_disp = int(y2 / scale_y)
            cv2.rectangle(final_display, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 2)
            cv2.imshow('Manual Labeler', final_display)

    def draw_instructions(self, frame):
        cv2.putText(frame, f"Frame: {self.current_filename}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "DRAW: mouse | DELETE: D | NEXT: 6 | PREV: 4 | EXIT: Q",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.current_filename in self.labels: # Показываем существующую разметку синим цветом
            x, y, w, h = self.labels[self.current_filename]
            scale_x = frame.shape[1] / self.current_frame.shape[1]
            scale_y = frame.shape[0] / self.current_frame.shape[0]
            x_disp = int(x * scale_x)
            y_disp = int(y * scale_y)
            w_disp = int(w * scale_x)
            h_disp = int(h * scale_y)
            cv2.rectangle(frame, (x_disp, y_disp), (x_disp + w_disp, y_disp + h_disp), (255, 0, 0), 2)
            cv2.putText(frame, "CURRENT LABEL", (x_disp, y_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def label_frame(self, filename):
        self.current_filename = filename
        filepath = os.path.join(self.frames_dir, filename)
        if not os.path.exists(filepath):
            print(f"Файл {filepath} не найден!")
            return True
        self.current_frame = cv2.imread(filepath)
        if self.current_frame is None:
            print(f"Ошибка загрузки {filepath}")
            return True

        self.display_frame = self.resize_frame_for_display(self.current_frame) # Масштабируем кадр для отображения
        cv2.namedWindow('Manual Labeler', cv2.WINDOW_NORMAL) # Создаем окно и устанавливаем обработчик мыши
        cv2.setMouseCallback('Manual Labeler', self.mouse_callback)
        display_with_instructions = self.display_frame.copy()
        self.draw_instructions(display_with_instructions)
        cv2.imshow('Manual Labeler', display_with_instructions)

        while True: # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d') or key == 100:
                if self.current_filename in self.labels:
                    del self.labels[self.current_filename]
                    print(f"Разметка для {self.current_filename} удалена")
                    temp_frame = self.display_frame.copy()
                    self.draw_instructions(temp_frame)
                    cv2.putText(temp_frame, "LABEL DELETED", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Manual Labeler', temp_frame)
                    self.save_labels()
            elif key == 54 or key == 102:
                return "next"
            elif key == 52 or key == 97:
                return "prev"
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return "quit"
        cv2.destroyWindow('Manual Labeler')
        return "next"

    def resize_frame_for_display(self, frame, max_width=1200, max_height=800):
        height, width = frame.shape[:2]
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        else:
            return frame

    def run(self):
        frame_files = sorted([f for f in os.listdir(self.frames_dir)
                              if f.endswith('.jpg')])
        if not frame_files:
            print("В папке нет файлов кадров!")
            return
        print(f"Найдено {len(frame_files)} кадров для разметки")
        print("Управление:")
        print("  - Нарисуйте прямоугольник мышью")
        print("  - 'D' - удалить разметку")
        print("  - '6' - следующий кадр")
        print("  - '4' - предыдущий кадр")
        print("  - 'Q' - выход")
        print("  - На последнем кадре '6' вернет к первому")
        current_index = 0
        while True:
            filename = frame_files[current_index]
            print(f"\n--- Разметка {filename} ({current_index + 1}/{len(frame_files)}) ---")
            result = self.label_frame(filename)
            if result == "prev":
                current_index = (current_index - 1) % len(frame_files)
            elif result == "next":
                current_index = (current_index + 1) % len(frame_files)
            elif result == "quit":
                break
        self.save_labels()
        print(f"\n=== Разметка завершена ===")
        print(f"Всего размечено кадров: {len(self.labels)}")

if __name__ == "__main__":
    labeler = ManualLabeler(frames_dir="my_frames", output_file="manual_labels.json")
    labeler.run()