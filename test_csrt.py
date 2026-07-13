import cv2
import os
import json
import numpy as np
from glob import glob
import argparse

class CSRTTester:
    def __init__(self, video_name, frames_dir="my_frames", labels_file="manual_labels.json"):
        self.video_name = video_name
        self.base_dir = os.path.join("experiments", video_name)
        self.frames_dir = os.path.join(self.base_dir, frames_dir)
        self.labels_file = os.path.join(self.base_dir, labels_file)
        self.results_dir = os.path.join("experiments", video_name, "results", "csrt")
        if os.path.exists(self.results_dir):
            print(f"Очищаем папку результатов {self.results_dir}...")
            for filename in os.listdir(self.results_dir):
                file_path = os.path.join(self.results_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Ошибка удаления {file_path}: {e}")
        else:
            os.makedirs(self.results_dir)
        self.labels = self.load_labels()
        self.iou_scores = []
        self.frame_numbers = []

    def load_labels(self):
        with open(self.labels_file, 'r') as f:
            return json.load(f)

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, w1, h1 = box1 # Преобразуем в формат [x1, y1, x2, y2]
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        x_left = max(x1_1, x1_2) # Вычисляем координаты пересечения
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top) # Площадь пересечения
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area # Площадь объединения
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def calculate_cle(self, box1, box2):
        if box1 is None or box2 is None:
            return float('inf')
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return float('inf')
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        distance = np.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)
        return distance

    def draw_results(self, frame, ground_truth, tracker_box, iou_score, cle_score, status):
        result_frame = frame.copy()
        height, width = frame.shape[:2] # Масштабирование
        max_display_size = 1200
        if width > max_display_size or height > max_display_size:
            scale = max_display_size / max(width, height)
            new_w = int(width * scale)
            new_h = int(height * scale)
            result_frame = cv2.resize(result_frame, (new_w, new_h))
            scale_x, scale_y = scale, scale
        else:
            scale_x, scale_y = 1, 1
        x_gt, y_gt, w_gt, h_gt = ground_truth # Ручная разметка
        x_gt_disp = int(x_gt * scale_x)
        y_gt_disp = int(y_gt * scale_y)
        w_gt_disp = int(w_gt * scale_x)
        h_gt_disp = int(h_gt * scale_y)
        cv2.rectangle(result_frame, (x_gt_disp, y_gt_disp),
                      (x_gt_disp + w_gt_disp, y_gt_disp + h_gt_disp),
                      (0, 255, 0), 3)
        x_tr, y_tr, w_tr, h_tr = tracker_box # Предсказание трекера
        if w_tr > 1 and h_tr > 1:
            x_tr_disp = int(x_tr * scale_x)
            y_tr_disp = int(y_tr * scale_y)
            w_tr_disp = int(w_tr * scale_x)
            h_tr_disp = int(h_tr * scale_y)
            cv2.rectangle(result_frame, (x_tr_disp, y_tr_disp),
                          (x_tr_disp + w_tr_disp, y_tr_disp + h_tr_disp),
                          (0, 0, 255), 2)
        y_offset = 30
        cv2.putText(result_frame, f"IoU: {iou_score:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        if cle_score != float('inf'):
            cv2.putText(result_frame, f"CLE: {cle_score:.1f} px", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        cv2.putText(result_frame, f"Status: {status}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if iou_score > 0.7:
            color = (0, 255, 0)  # Зеленый
        elif iou_score > 0.3:
            color = (255, 255, 0)  # Желтый
        else:
            color = (0, 0, 255)  # Красный
        indicator_size = 20 # Индикатор в углу
        cv2.rectangle(result_frame,
                      (result_frame.shape[1] - indicator_size - 10, 10),
                      (result_frame.shape[1] - 10, 10 + indicator_size),
                      color, -1)
        return result_frame

    def resize_frame_for_display(self, frame, max_width=1200, max_height=800):
        height, width = frame.shape[:2]
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame

    def test_csrt(self):
        print("=== Тестирование CSRT трекера ===")
        print("=" * 50)
        labeled_frames = sorted(self.labels.keys())
        if not labeled_frames:
            print("Нет размеченных кадров!")
            return
        print(f"Найдено {len(labeled_frames)} размеченных кадров")
        print("Первый кадр для инициализации:", labeled_frames[0])
        first_frame_path = os.path.join(self.frames_dir, labeled_frames[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            print(f"Ошибка загрузки {first_frame_path}")
            return
        first_box = self.labels[labeled_frames[0]] # Получаем ручную разметку для первого кадра
        x, y, w, h = first_box
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
            print("✓ Трекер создан через cv2.legacy.TrackerCSRT_create()")
        except AttributeError as e:
            print(f"✗ cv2.legacy.TrackerCSRT_create не доступен: {e}")
            return
        success = tracker.init(first_frame, (x, y, w, h))
        if not success:
            print("Ошибка инициализации трекера!")
            return
        print(f"Трекер инициализирован на {first_frame_path}")
        print(f"Начальная позиция: x={x}, y={y}, w={w}, h={h}")
        print("\nЗапуск трекинга...")
        print("Управление: SPACE - пауза, Q - выход\n")
        self.iou_scores = []
        self.cle_scores = []
        self.frame_numbers = []
        self.tracking_status = []
        all_frames = sorted(glob(os.path.join(self.frames_dir, "*.jpg")))
        for i, frame_path in enumerate(all_frames):
            frame_name = os.path.basename(frame_path)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Ошибка загрузки {frame_path}")
                continue
            success, tracker_box = tracker.update(frame)
            if frame_name in self.labels:
                ground_truth = self.labels[frame_name]
                if success:
                    try: # Преобразуем tracker_box к правильному формату
                        if hasattr(tracker_box, '__len__') and len(tracker_box) == 4:
                            x_tr = int(tracker_box[0])
                            y_tr = int(tracker_box[1])
                            w_tr = int(tracker_box[2])
                            h_tr = int(tracker_box[3])
                            tracker_box_converted = [x_tr, y_tr, w_tr, h_tr]
                        else:
                            print(f"Ошибка: неверный формат tracker_box: {tracker_box}")
                            tracker_box_converted = [0, 0, 1, 1]
                            success = False
                    except Exception as e:
                        print(f"Ошибка конвертации: {e}")
                        tracker_box_converted = [0, 0, 1, 1]
                        success = False
                    if success:
                        iou = self.calculate_iou(ground_truth, tracker_box_converted)
                        cle = self.calculate_cle(ground_truth, tracker_box_converted)
                        status = "TRACKING"
                        if iou < 0.2:
                            status = "LOW IoU"
                        elif cle > 50:
                            status = "HIGH ERROR"
                        self.iou_scores.append(iou)
                        self.cle_scores.append(cle)
                        self.frame_numbers.append(i)
                        self.tracking_status.append(status)
                        print(f"Кадр {frame_name}: IoU={iou:.3f}, CLE={cle:.1f}px, Status={status}")
                        result_frame = frame.copy()
                        height, width = frame.shape[:2] # Масштабируем для отображения
                        max_display_size = 1200
                        if width > max_display_size or height > max_display_size:
                            scale = max_display_size / max(width, height)
                            new_w = int(width * scale)
                            new_h = int(height * scale)
                            result_frame = cv2.resize(result_frame, (new_w, new_h))
                            scale_x, scale_y = scale, scale
                        else:
                            scale_x, scale_y = 1, 1
                        x_gt, y_gt, w_gt, h_gt = ground_truth # Ручная разметка
                        x_gt_disp = int(x_gt * scale_x)
                        y_gt_disp = int(y_gt * scale_y)
                        w_gt_disp = int(w_gt * scale_x)
                        h_gt_disp = int(h_gt * scale_y)
                        cv2.rectangle(result_frame, (x_gt_disp, y_gt_disp),
                                      (x_gt_disp + w_gt_disp, y_gt_disp + h_gt_disp),
                                      (0, 255, 0), 3)
                        x_tr, y_tr, w_tr, h_tr = tracker_box_converted # Предсказание трекера
                        if w_tr > 1 and h_tr > 1:
                            x_tr_disp = int(x_tr * scale_x)
                            y_tr_disp = int(y_tr * scale_y)
                            w_tr_disp = int(w_tr * scale_x)
                            h_tr_disp = int(h_tr * scale_y)
                            cv2.rectangle(result_frame, (x_tr_disp, y_tr_disp),
                                          (x_tr_disp + w_tr_disp, y_tr_disp + h_tr_disp),
                                          (0, 0, 255), 2)
                        y_offset = 30 # Текст с информацией
                        cv2.putText(result_frame, f"IoU: {iou:.3f}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y_offset += 30
                        if cle != float('inf'):
                            cv2.putText(result_frame, f"CLE: {cle:.1f} px", (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            y_offset += 30

                        cv2.putText(result_frame, f"Status: {status}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        if iou > 0.7:
                            color = (0, 255, 0)  # Зеленый
                        elif iou > 0.3:
                            color = (255, 255, 0)  # Желтый
                        else:
                            color = (0, 0, 255)  # Красный
                        indicator_size = 20 # Индикатор в углу
                        cv2.rectangle(result_frame,
                                      (result_frame.shape[1] - indicator_size - 10, 10),
                                      (result_frame.shape[1] - 10, 10 + indicator_size),
                                      color, -1)
                    else:
                        iou = 0.0
                        cle = float('inf')
                        status = "ERROR"
                        self.iou_scores.append(iou)
                        self.cle_scores.append(cle)
                        self.frame_numbers.append(i)
                        self.tracking_status.append(status)
                        print(f"Кадр {frame_name}: IoU={iou:.3f}, CLE=---, Status={status}")
                        result_frame = frame.copy()
                        height, width = frame.shape[:2]
                        max_display_size = 1200
                        if width > max_display_size or height > max_display_size:
                            scale = max_display_size / max(width, height)
                            new_w = int(width * scale)
                            new_h = int(height * scale)
                            result_frame = cv2.resize(result_frame, (new_w, new_h))
                        cv2.putText(result_frame, "CONVERSION ERROR", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(result_frame, f"IoU: {iou:.3f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(result_frame, "Status: ERROR", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    iou = 0.0
                    cle = float('inf')
                    status = "LOST"
                    self.iou_scores.append(iou)
                    self.cle_scores.append(cle)
                    self.frame_numbers.append(i)
                    self.tracking_status.append(status)
                    print(f"Кадр {frame_name}: IoU={iou:.3f}, CLE=---, Status={status}")
                    result_frame = frame.copy()
                    height, width = frame.shape[:2] # Масштабируем для отображения
                    max_display_size = 1200
                    if width > max_display_size or height > max_display_size:
                        scale = max_display_size / max(width, height)
                        new_w = int(width * scale)
                        new_h = int(height * scale)
                        result_frame = cv2.resize(result_frame, (new_w, new_h))
                        scale_x, scale_y = scale, scale
                    else:
                        scale_x, scale_y = 1, 1
                    x_gt, y_gt, w_gt, h_gt = ground_truth
                    x_gt_disp = int(x_gt * scale_x)
                    y_gt_disp = int(y_gt * scale_y)
                    w_gt_disp = int(w_gt * scale_x)
                    h_gt_disp = int(h_gt * scale_y)
                    cv2.rectangle(result_frame, (x_gt_disp, y_gt_disp),
                                  (x_gt_disp + w_gt_disp, y_gt_disp + h_gt_disp),
                                  (0, 255, 0), 2)
                    h, w = result_frame.shape[:2] # Большой красный крест через весь кадр
                    cv2.line(result_frame, (0, 0), (w, h), (0, 0, 255), 5)
                    cv2.line(result_frame, (w, 0), (0, h), (0, 0, 255), 5)
                    cv2.putText(result_frame, "OBJECT LOST", (w // 4, h // 2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(result_frame, f"IoU: {iou:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(result_frame, "CLE: ---", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(result_frame, "Status: LOST", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                result_path = os.path.join(self.results_dir, f"result_{frame_name}")
                cv2.imwrite(result_path, result_frame)
                cv2.imshow('CSRT Tracking', result_frame)
                key = cv2.waitKey(500) & 0xFF  # 500 мс на кадр
                if key == ord('q'):
                    print("\nДосрочный выход по запросу пользователя")
                    break
                elif key == ord(' '):
                    print("Пауза. Нажмите любую клавишу для продолжения...")
                    cv2.waitKey(0)
        print("\nТестирование завершено.")
        cv2.destroyAllWindows()
        self.print_statistics()

    def print_statistics(self):
        if not self.iou_scores:
            print("\nНет данных для статистики!")
            return
        print("\n" + "=" * 50)
        print("СТАТИСТИКА")
        print("=" * 50)
        valid_indices = [i for i, status in enumerate(self.tracking_status)
                         if status == "TRACKING"]
        if valid_indices:
            valid_iou = [self.iou_scores[i] for i in valid_indices]
            valid_cle = [self.cle_scores[i] for i in valid_indices]
            print(f"Всего кадров: {len(self.iou_scores)}")
            print(f"Кадров с успешным трекингом: {len(valid_iou)}")
            print(f"Кадров с потерей объекта: {len(self.iou_scores) - len(valid_iou)}")
            print(f"Средний IoU (успешные кадры): {np.mean(valid_iou):.3f}")
            print(f"Средний CLE (успешные кадры): {np.mean(valid_cle):.1f} px")
            print(f"Медианный CLE: {np.median(valid_cle):.1f} px")
            print(f"Минимальный IoU: {np.min(valid_iou):.3f}")
            print(f"Максимальный IoU: {np.max(valid_iou):.3f}")
            successful = sum(1 for iou in self.iou_scores if iou > 0.5)
            success_rate = (successful / len(self.iou_scores)) * 100
            print(f"Успешных кадров (IoU > 0.5): {successful}/{len(self.iou_scores)} ({success_rate:.1f}%)")
            low_error = sum(1 for cle in valid_cle if cle < 20)
            low_error_rate = (low_error / len(valid_cle)) * 100 if valid_cle else 0
            print(f"Кадров с низкой ошибкой (CLE < 20px): {low_error}/{len(valid_cle)} ({low_error_rate:.1f}%)")
        else:
            print("Нет кадров с успешным трекингом!")
        stats = {
            "total_frames": len(self.iou_scores),
            "frames_with_tracking": len(valid_iou) if valid_indices else 0,
            "mean_iou": float(np.mean(valid_iou) if valid_indices else 0),
            "mean_cle": float(np.mean(valid_cle) if valid_indices else 0),
            "median_cle": float(np.median(valid_cle) if valid_indices else 0),
            "min_iou": float(np.min(valid_iou) if valid_indices else 0),
            "max_iou": float(np.max(valid_iou) if valid_indices else 0),
            "success_rate": float(success_rate),
            "low_error_rate": float(low_error_rate),
            "iou_scores": self.iou_scores,
            "cle_scores": self.cle_scores,
            "frame_numbers": self.frame_numbers,
            "tracking_status": self.tracking_status
        }
        with open(os.path.join(self.results_dir, "statistics.json"), 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"\nСтатистика сохранена в {self.results_dir}/statistics.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                        help="Название видео (папка в experiments/)")
    args = parser.parse_args()
    print(f"Запуск тестирования CSRT трекера на видео: {args.video}")
    print("=" * 50)
    tester = CSRTTester(video_name=args.video)
    tester.test_csrt()
    print("\n" + "=" * 50)
    print("Тестирование завершено!")

if __name__ == "__main__":
    main()