import cv2
import os
import json
import numpy as np
from glob import glob
import argparse

class OpticalFlowTracker:
    def __init__(self, max_points=200, min_points=10):
        self.max_points = max_points
        self.min_points = min_points
        self.feature_params = dict(
            maxCorners=max_points,
            qualityLevel=0.1,
            minDistance=3,
            blockSize=5
        )
        self.lk_params = dict( # Параметры для оптического потока
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        self.prev_frame = None
        self.prev_points = None
        self.bbox = None

    def find_keypoints(self, frame, bbox):
        x, y, w, h = bbox
        roi = frame[y:y + h, x:x + w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(roi_gray, mask=None, maxCorners=50, qualityLevel=0.01, minDistance=5)
        if points is not None:
            points[:, 0, 0] += x
            points[:, 0, 1] += y
            return points
        return None

    def init(self, frame, bbox):
        self.bbox = bbox
        x, y, w, h = bbox
        if (x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]):
            print(f"Ошибка: bbox вне границ кадра!")
            print(f"Кадр: 0-{frame.shape[1]}, 0-{frame.shape[0]}")
            print(f"Bbox: {x}-{x + w}, {y}-{y + h}")
            return False
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            print("Ошибка: ROI пустой!")
            return False
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        debug_roi = roi.copy()
        points = cv2.goodFeaturesToTrack(roi_gray, mask=None, **self.feature_params)
        if points is None:
            print("Не найдено ни одной точки в ROI!")
            cv2.imshow("Debug ROI", roi)
            cv2.waitKey(1000)
            cv2.destroyWindow("Debug ROI")
            return False
        points[:, 0, 0] += x
        points[:, 0, 1] += y
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for point in points:
            px, py = point.ravel()
            cv2.circle(debug_frame, (int(px), int(py)), 3, (0, 0, 255), -1)
        cv2.putText(debug_frame, f"Points: {len(points)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if len(points) < self.min_points:
            print(f"Ошибка: недостаточно точек ({len(points)} < {self.min_points})")
            return False
        self.prev_points = points
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return True

    def update(self, frame):
        if self.prev_points is None or len(self.prev_points) < self.min_points:
            print("Недостаточно точек для обновления!")
            return False, [0, 0, 1, 1]
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Вычисляем оптический поток
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, curr_frame, self.prev_points, None, **self.lk_params
        )
        if curr_points is None:
            print("Ошибка: calcOpticalFlowPyrLK вернул None")
            return False, [0, 0, 1, 1]
        good_prev = self.prev_points[status == 1] # Фильтрация хороших точек
        good_curr = curr_points[status == 1]
        if len(good_curr) < self.min_points:
            print(f"Слишком мало точек после фильтрации ({len(good_curr)} < {self.min_points})")
            return False, [0, 0, 1, 1]
        if len(good_curr) < self.min_points * 2:  # Если точек стало мало
            print(f"Мало точек ({len(good_curr)}), переинициализируем...")
            good_curr_reshaped = good_curr.reshape(-1, 2) # Вычисляем текущий bbox по оставшимся точкам
            x_min = int(np.min(good_curr_reshaped[:, 0]))
            y_min = int(np.min(good_curr_reshaped[:, 1]))
            x_max = int(np.max(good_curr_reshaped[:, 0]))
            y_max = int(np.max(good_curr_reshaped[:, 1]))
            current_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            new_points = self.find_keypoints(frame, current_bbox)
            if new_points is not None and len(new_points) >= self.min_points:
                good_curr = new_points
                print(f"Переинициализировано {len(new_points)} точек")
        good_curr_reshaped = good_curr.reshape(-1, 2)
        if len(good_curr_reshaped) > 5: # Отфильтровываем далекие точки
            median_x = np.median(good_curr_reshaped[:, 0])
            median_y = np.median(good_curr_reshaped[:, 1])
            distances = np.sqrt((good_curr_reshaped[:, 0] - median_x) ** 2 +
                                (good_curr_reshaped[:, 1] - median_y) ** 2)
            median_distance = np.median(distances)
            mask = distances < 2.0 * median_distance
            filtered_points = good_curr_reshaped[mask]
            if len(filtered_points) >= self.min_points:
                good_curr_reshaped = filtered_points
        x_min = int(np.min(good_curr_reshaped[:, 0]))
        y_min = int(np.min(good_curr_reshaped[:, 1]))
        x_max = int(np.max(good_curr_reshaped[:, 0]))
        y_max = int(np.max(good_curr_reshaped[:, 1]))
        if len(good_curr_reshaped) < self.min_points * 1.5:
            print(f"Мало точек ({len(good_curr_reshaped)}), ищем новые...")
            width = x_max - x_min
            height = y_max - y_min
            temp_bbox = [x_min, y_min, width, height]
            new_points = self.find_keypoints(frame, temp_bbox)
            if new_points is not None and len(new_points) > 10:
                new_points_reshaped = new_points.reshape(-1, 2)
                good_curr_reshaped = np.vstack([good_curr_reshaped, new_points_reshaped])
                print(f"Добавлено {len(new_points)} новых точек")
        padding_x = int((x_max - x_min) * 0.1)
        padding_y = int((y_max - y_min) * 0.1)
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(frame.shape[1], x_max + padding_x)
        y_max = min(frame.shape[0], y_max + padding_y)
        width = x_max - x_min
        height = y_max - y_min
        if width < 20 or height < 40:
            print(f"Слишком маленький bbox: {width}x{height}")
            return False, [0, 0, 1, 1]
        self.prev_frame = curr_frame.copy()
        self.prev_points = good_curr.reshape(-1, 1, 2).astype(np.float32)
        self.bbox = [x_min, y_min, width, height]
        return True, self.bbox

class OpticalFlowTester:
    def __init__(self, video_name, frames_dir="my_frames", labels_file="manual_labels.json"):
        self.video_name = video_name
        self.base_dir = os.path.join("experiments", video_name)
        self.frames_dir = os.path.join(self.base_dir, frames_dir)
        self.labels_file = os.path.join(self.base_dir, labels_file)
        self.results_dir = os.path.join("experiments", video_name, "results", "optical_flow")
        if os.path.exists(self.results_dir):
            print(f"Очищаем папку {self.results_dir}...")
            for filename in os.listdir(self.results_dir):
                file_path = os.path.join(self.results_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"Удален: {filename}")
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
        if union <= 0:
            return 0.0
        return intersection / union

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

    def test_optical_flow(self):
        print("=== Тестирование Optical Flow (Lucas-Kanade) ===")
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
        tracker = OpticalFlowTracker(max_points=200, min_points=5)
        if not tracker.init(first_frame, first_box):
            print("Ошибка инициализации трекера!")
            return
        print(f"Трекер инициализирован с {len(tracker.prev_points)} точками")
        print(f"Начальная позиция: x={first_box[0]}, y={first_box[1]}, w={first_box[2]}, h={first_box[3]}")
        print("\nЗапуск трекинга...")
        print("Управление: SPACE - пауза, Q - выход\n")
        all_frames = sorted(glob(os.path.join(self.frames_dir, "*.jpg")))
        self.iou_scores = []
        self.cle_scores = []
        self.frame_numbers = []
        self.tracking_status = []
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
                vis_frame = frame.copy()
                height, width = frame.shape[:2]
                max_display_size = 1200
                if width > max_display_size or height > max_display_size:
                    scale = max_display_size / max(width, height)
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    vis_frame = cv2.resize(vis_frame, (new_w, new_h))
                    scale_x, scale_y = scale, scale
                else:
                    scale_x, scale_y = 1, 1
                x_gt, y_gt, w_gt, h_gt = ground_truth
                x_gt_disp = int(x_gt * scale_x)
                y_gt_disp = int(y_gt * scale_y)
                w_gt_disp = int(w_gt * scale_x)
                h_gt_disp = int(h_gt * scale_y)
                cv2.rectangle(vis_frame, (x_gt_disp, y_gt_disp),
                              (x_gt_disp + w_gt_disp, y_gt_disp + h_gt_disp),
                              (0, 255, 0), 2)
                if success and tracker_box[2] > 1 and tracker_box[3] > 1:
                    x_tr, y_tr, w_tr, h_tr = tracker_box
                    x_tr_disp = int(x_tr * scale_x)
                    y_tr_disp = int(y_tr * scale_y)
                    w_tr_disp = int(w_tr * scale_x)
                    h_tr_disp = int(h_tr * scale_y)
                    cv2.rectangle(vis_frame, (x_tr_disp, y_tr_disp),
                                  (x_tr_disp + w_tr_disp, y_tr_disp + h_tr_disp),
                                  (0, 0, 255), 2)
                if tracker.prev_points is not None and status != "LOST":
                    points_to_show = min(30, len(tracker.prev_points))
                    for point in tracker.prev_points[:points_to_show]:
                        x_pt, y_pt = point[0]
                        x_pt_disp = int(x_pt * scale_x)
                        y_pt_disp = int(y_pt * scale_y)
                        cv2.circle(vis_frame, (x_pt_disp, y_pt_disp),
                                   2, (255, 0, 0), -1)
                y_offset = 30
                cv2.putText(vis_frame, f"IoU: {iou:.3f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
                if status != "LOST":
                    cv2.putText(vis_frame, f"CLE: {cle:.1f} px", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 30
                cv2.putText(vis_frame, f"Status: {status}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
                if tracker.prev_points is not None and status != "LOST":
                    cv2.putText(vis_frame, f"Points: {len(tracker.prev_points)}",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if iou > 0.7:
                    quality_color = (0, 255, 0)  # Зеленый
                elif iou > 0.3:
                    quality_color = (255, 255, 0)  # Желтый
                else:
                    quality_color = (0, 0, 255)  # Красный
                indicator_size = 20 # Индикатор в углу
                cv2.rectangle(vis_frame,
                              (vis_frame.shape[1] - indicator_size - 10, 10),
                              (vis_frame.shape[1] - 10, 10 + indicator_size),
                              quality_color, -1)
                result_path = os.path.join(self.results_dir, f"result_{frame_name}")
                cv2.imwrite(result_path, vis_frame)
                cv2.imshow('Optical Flow Tracking', vis_frame)
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
        print("СТАТИСТИКА Optical Flow")
        print("=" * 50)
        valid_indices = [i for i, status in enumerate(self.tracking_status)
                         if status not in ["LOST", "LOW IoU"]]
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
            successful = sum(1 for iou in valid_iou if iou > 0.5)
            success_rate = (successful / len(valid_iou)) * 100
            print(f"Успешных кадров (IoU > 0.5): {successful}/{len(valid_iou)} ({success_rate:.1f}%)")
            low_error = sum(1 for cle in valid_cle if cle < 20)
            low_error_rate = (low_error / len(valid_cle)) * 100
            print(f"Кадров с низкой ошибкой (CLE < 20px): {low_error}/{len(valid_cle)} ({low_error_rate:.1f}%)")
        else:
            print("Нет кадров с успешным трекингом!")
        total_successful = sum(1 for iou in self.iou_scores if iou > 0.5)
        total_success_rate = (total_successful / len(self.iou_scores)) * 100
        print(f"\nОбщая успешность (все кадры): {total_successful}/{len(self.iou_scores)} ({total_success_rate:.1f}%)")
        # Сохраняем статистику в JSON
        stats = {
            "total_frames": len(self.iou_scores),
            "frames_with_tracking": len(valid_iou) if valid_indices else 0,
            "mean_iou": float(np.mean(valid_iou) if valid_indices else 0),
            "mean_cle": float(np.mean(valid_cle) if valid_indices else 0),
            "median_cle": float(np.median(valid_cle) if valid_indices else 0),
            "min_iou": float(np.min(valid_iou) if valid_indices else 0),
            "max_iou": float(np.max(valid_iou) if valid_indices else 0),
            "success_rate": float(success_rate if valid_indices else 0),
            "low_error_rate": float(low_error_rate if valid_indices else 0),
            "total_success_rate": float(total_success_rate),
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
    parser.add_argument("--video", type=str, required=True, help="Название видео (папка в experiments/)")
    args = parser.parse_args()
    print(f"Запуск тестирования Optical Flow трекера на видео: {args.video}")
    print("=" * 50)
    tester = OpticalFlowTester(video_name=args.video)
    tester.test_optical_flow()
    print("\n" + "=" * 50)
    print("Тестирование завершено!")

if __name__ == "__main__":
    main()