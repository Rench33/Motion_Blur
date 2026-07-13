import cv2
import os
import json
import numpy as np
from glob import glob
import argparse

class CamShiftTracker:
    def __init__(self):
        self.bbox = None
        self.roi_hist = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.first_update = True

    def init(self, frame, bbox):
        self.bbox = bbox
        x, y, w, h = bbox
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            print(f"Ошибка: bbox вне границ кадра!")
            return False
        roi = frame[y:y + h, x:x + w] # Выделяем ROI и переводим в HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Строим гистограмму по каналу Hue
        self.roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        return True

    def update(self, frame):
        if self.roi_hist is None:
            return False, [0, 0, 1, 1]
        if self.first_update:
            self.first_update = False
            return True, self.bbox
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Переводим кадр в HSV
        # Обратная проекция гистограммы на кадр
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        # Применяем CamShift
        ret, track_window = cv2.CamShift(dst, tuple(self.bbox), self.term_crit)
        if ret:
            pts = cv2.boxPoints(ret)
            pts = np.int32(pts)
            x, y, w, h = cv2.boundingRect(pts)
            self.bbox = [x, y, w, h]
            return True, self.bbox
        else:
            return False, [0, 0, 1, 1]

class CamShiftTester:
    def __init__(self, video_name, frames_dir="my_frames", labels_file="manual_labels.json"):
        self.video_name = video_name
        self.base_dir = os.path.join("experiments", video_name)
        self.frames_dir = os.path.join(self.base_dir, frames_dir)
        self.labels_file = os.path.join(self.base_dir, labels_file)
        self.results_dir = os.path.join("experiments", video_name, "results", "camshift")
        # Очищаем папку результатов
        if os.path.exists(self.results_dir):
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
        self.cle_scores = []
        self.frame_numbers = []
        self.tracking_status = []

    def load_labels(self):
        with open(self.labels_file, 'r') as f:
            return json.load(f)

    def calculate_iou(self, box1, box2):
        if box1 is None or box2 is None:
            return 0.0
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0

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
        return np.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)

    def test_camshift(self):
        print("=== Тестирование CamShift трекера ===")
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
        first_box = self.labels[labeled_frames[0]]
        tracker = CamShiftTracker()
        if not tracker.init(first_frame, first_box):
            print("Ошибка инициализации CamShift трекера!")
            return
        print(f"Трекер инициализирован")
        print(f"Начальная позиция: x={first_box[0]}, y={first_box[1]}, w={first_box[2]}, h={first_box[3]}")
        print("\nЗапуск трекинга...")
        print("Управление: SPACE - пауза, Q - выход\n")
        all_frames = sorted(glob(os.path.join(self.frames_dir, "*.jpg")))
        for i, frame_path in enumerate(all_frames):
            frame_name = os.path.basename(frame_path)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            success, tracker_box = tracker.update(frame)
            if frame_name in self.labels:
                ground_truth = self.labels[frame_name]
                if success:
                    iou = self.calculate_iou(ground_truth, tracker_box)
                    cle = self.calculate_cle(ground_truth, tracker_box)
                    status = "TRACKING"
                    if iou < 0.2:
                        status = "LOW IoU"
                    elif cle > 50:
                        status = "HIGH ERROR"
                else:
                    iou = 0.0
                    cle = float('inf')
                    status = "LOST"
                    tracker_box = [0, 0, 1, 1]
                self.iou_scores.append(iou)
                self.cle_scores.append(cle)
                self.frame_numbers.append(i)
                self.tracking_status.append(status)
                if status == "LOST":
                    print(f"Кадр {frame_name}: IoU={iou:.3f}, CLE=---, Status={status}")
                else:
                    print(f"Кадр {frame_name}: IoU={iou:.3f}, CLE={cle:.1f}px, Status={status}")
                # Визуализация
                vis_frame = self.draw_result(frame, ground_truth, tracker_box if success else None,
                                             iou, cle if success else None, status)
                result_path = os.path.join(self.results_dir, f"result_{frame_name}")
                cv2.imwrite(result_path, vis_frame)
                cv2.imshow('CamShift Tracking', vis_frame)
                key = cv2.waitKey(500) & 0xFF
                if key == ord('q'):
                    print("\nДосрочный выход")
                    break
                elif key == ord(' '):
                    print("Пауза. Нажмите любую клавишу...")
                    cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.print_statistics()

    def draw_result(self, frame, ground_truth, tracker_box, iou, cle, status):
        result_frame = frame.copy()
        height, width = frame.shape[:2]
        # Масштабирование для отображения
        max_display_size = 1200
        if width > max_display_size or height > max_display_size:
            scale = max_display_size / max(width, height)
            new_w = int(width * scale)
            new_h = int(height * scale)
            result_frame = cv2.resize(result_frame, (new_w, new_h))
            scale_x, scale_y = scale, scale
        else:
            scale_x, scale_y = 1, 1
        x_gt, y_gt, w_gt, h_gt = ground_truth # Рисуем ground truth (зеленый)
        x_gt_disp = int(x_gt * scale_x)
        y_gt_disp = int(y_gt * scale_y)
        w_gt_disp = int(w_gt * scale_x)
        h_gt_disp = int(h_gt * scale_y)
        cv2.rectangle(result_frame, (x_gt_disp, y_gt_disp),
                      (x_gt_disp + w_gt_disp, y_gt_disp + h_gt_disp),
                      (0, 255, 0), 2)
        if tracker_box and tracker_box[2] > 1 and tracker_box[3] > 1: # Рисуем предсказание трекера (красный)
            x_tr, y_tr, w_tr, h_tr = tracker_box
            x_tr_disp = int(x_tr * scale_x)
            y_tr_disp = int(y_tr * scale_y)
            w_tr_disp = int(w_tr * scale_x)
            h_tr_disp = int(h_tr * scale_y)
            cv2.rectangle(result_frame, (x_tr_disp, y_tr_disp),
                          (x_tr_disp + w_tr_disp, y_tr_disp + h_tr_disp),
                          (0, 0, 255), 2)
        y_offset = 30
        cv2.putText(result_frame, f"CamShift", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(result_frame, f"Status: {status}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(result_frame, f"IoU: {iou:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        if cle is not None and cle != float('inf'):
            cv2.putText(result_frame, f"CLE: {cle:.1f} px", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Индикатор качества в углу
        if iou > 0.7:
            color = (0, 255, 0)
        elif iou > 0.3:
            color = (255, 255, 0)
        else:
            color = (0, 0, 255)
        indicator_size = 20
        cv2.rectangle(result_frame,
                      (result_frame.shape[1] - indicator_size - 10, 10),
                      (result_frame.shape[1] - 10, 10 + indicator_size),
                      color, -1)
        return result_frame

    def print_statistics(self):
        if not self.iou_scores:
            print("\nНет данных для статистики!")
            return
        print("\n" + "=" * 50)
        print("СТАТИСТИКА CamShift")
        print("=" * 50)
        valid_indices = [i for i, status in enumerate(self.tracking_status)
                         if status != "LOST"]
        if valid_indices:
            valid_iou = [self.iou_scores[i] for i in valid_indices]
            valid_cle = [self.cle_scores[i] for i in valid_indices]
            print(f"Всего кадров: {len(self.iou_scores)}")
            print(f"Кадров с успешным трекингом: {len(valid_iou)}")
            print(f"Кадров с потерей объекта: {len(self.iou_scores) - len(valid_iou)}")
            print(f"Средний IoU (успешные кадры): {np.mean(valid_iou):.3f}")
            print(f"Средний CLE (успешные кадры): {np.mean(valid_cle):.1f} px")
            successful = sum(1 for iou in valid_iou if iou > 0.5)
            success_rate = (successful / len(valid_iou)) * 100
            print(f"Успешных кадров (IoU > 0.5): {successful}/{len(valid_iou)} ({success_rate:.1f}%)")
        else:
            print("Нет кадров с успешным трекингом!")
        # Сохраняем статистику в JSON
        stats = {
            "total_frames": len(self.iou_scores),
            "frames_with_tracking": len(valid_indices),
            "mean_iou": float(np.mean(valid_iou)) if valid_indices else 0,
            "mean_cle": float(np.mean(valid_cle)) if valid_indices else 0,
            "success_rate": float(success_rate) if valid_indices else 0,
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
    print(f"Запуск тестирования CamShift трекера на видео: {args.video}")
    print("=" * 50)
    tester = CamShiftTester(video_name=args.video)
    tester.test_camshift()
    print("\n" + "=" * 50)
    print("Тестирование завершено!")

if __name__ == "__main__":
    main()