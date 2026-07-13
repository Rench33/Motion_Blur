import cv2
import os
import json
import numpy as np
from glob import glob
from ultralytics import YOLO
import argparse

class YOLOTester:
    def __init__(self, video_name, frames_dir="my_frames", labels_file="manual_labels.json", model_version=None, class_id=0):
        self.class_id = class_id
        self.video_name = video_name
        self.model_version = model_version
        self.base_dir = os.path.join("experiments", video_name)
        self.frames_dir = os.path.join(self.base_dir, frames_dir)
        self.labels_file = os.path.join(self.base_dir, labels_file)
        self.results_dir = os.path.join("experiments", video_name, "results", "yolo")
        os.makedirs(self.results_dir, exist_ok=True) # Создаём основную папку если её нет
        # Очистка папки выбранной модели
        if model_version is not None:
            version_dir = os.path.join(self.results_dir, f"yolo_{model_version}")
            if os.path.exists(version_dir):
                for f in os.listdir(version_dir):
                    if f.endswith('.jpg') or f == 'statistics.json':
                        os.remove(os.path.join(version_dir, f))
            else:
                os.makedirs(version_dir, exist_ok=True)
        self.labels = self.load_labels()
        self.iou_scores = []
        self.cle_scores = []
        self.frame_numbers = []
        self.detection_status = []

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
        distance = np.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)
        return distance

    def find_best_detection(self, detections, ground_truth):
        best_iou = 0
        best_bbox = None
        if detections is None or len(detections) == 0:
            return None, 0
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            if int(cls) != self.class_id: # Проверяем, что это нужный объект
                continue
            det_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            iou = self.calculate_iou(ground_truth, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_bbox = det_bbox
        return best_bbox, best_iou

    def test_yolo(self, model_version='n'):
        print(f"\n=== Тестирование YOLOv8{model_version} ===")
        print("=" * 50)
        model_path = f'yolov8{model_version}.pt'
        print(f"Загрузка модели: {model_path}")
        model = YOLO(model_path)
        print("Модель загружена")
        version_dir = os.path.join(self.results_dir, f"yolo_{model_version}")
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
        all_frames = sorted(glob(os.path.join(self.frames_dir, "*.jpg")))
        print(f"Найдено {len(all_frames)} кадров")
        print(f"Размеченных кадров: {len(self.labels)}")
        self.iou_scores = []
        self.cle_scores = []
        self.frame_numbers = []
        self.detection_status = []
        for i, frame_path in enumerate(all_frames):
            frame_name = os.path.basename(frame_path)
            if frame_name not in self.labels:
                continue
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            ground_truth = self.labels[frame_name]
            results = model(frame, conf=0.25, classes=[self.class_id], verbose=False)
            detections = None
            if results and len(results) > 0:
                detections = results[0].boxes.data.cpu().numpy()
            best_bbox, best_iou = self.find_best_detection(detections, ground_truth)
            if best_iou > 0.3:  # Порог для обнаружения
                status = 'DETECTED'
                iou_score = best_iou
                cle_score = self.calculate_cle(ground_truth, best_bbox)
                if iou_score < 0.5:
                    status = 'LOW IoU'
                elif cle_score > 50:
                    status = 'HIGH ERROR'
            else:
                status = 'NOT DETECTED'
                iou_score = 0.0
                cle_score = float('inf')
                best_bbox = None
            self.iou_scores.append(iou_score)
            self.cle_scores.append(cle_score)
            self.frame_numbers.append(i)
            self.detection_status.append(status)
            if status == 'NOT DETECTED':
                print(f"Кадр {frame_name}: NOT DETECTED")
            else:
                print(f"Кадр {frame_name}: IoU={iou_score:.3f}, CLE={cle_score:.1f}px, Status={status}")
            result_frame = frame.copy()
            height, width = frame.shape[:2]
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
                          (0, 255, 0), 3)
            if best_bbox and status != 'NOT DETECTED':
                x_yl, y_yl, w_yl, h_yl = best_bbox
                if w_yl > 1 and h_yl > 1:
                    x_yl_disp = int(x_yl * scale_x)
                    y_yl_disp = int(y_yl * scale_y)
                    w_yl_disp = int(w_yl * scale_x)
                    h_yl_disp = int(h_yl * scale_y)
                    cv2.rectangle(result_frame, (x_yl_disp, y_yl_disp),
                                  (x_yl_disp + w_yl_disp, y_yl_disp + h_yl_disp),
                                  (255, 0, 0), 2)
            y_offset = 30
            cv2.putText(result_frame, f"YOLOv8{model_version}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            cv2.putText(result_frame, f"Status: {status}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            cv2.putText(result_frame, f"IoU: {iou_score:.3f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            if cle_score != float('inf'):
                cv2.putText(result_frame, f"CLE: {cle_score:.1f} px", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if status == 'DETECTED':
                if iou_score > 0.7:
                    color = (0, 255, 0)  # Зеленый
                elif iou_score > 0.3:
                    color = (255, 255, 0)  # Желтый
                else:
                    color = (0, 0, 255)  # Красный
            elif status == 'LOW IoU':
                color = (255, 165, 0)  # Оранжевый
            else:
                color = (0, 0, 255)  # Красный
            indicator_size = 20
            cv2.rectangle(result_frame,
                          (result_frame.shape[1] - indicator_size - 10, 10),
                          (result_frame.shape[1] - 10, 10 + indicator_size),
                          color, -1)
            result_path = os.path.join(version_dir, f"result_{frame_name}")
            cv2.imwrite(result_path, result_frame)
            cv2.imshow(f'YOLOv8{model_version}', result_frame)
            key = cv2.waitKey(500) & 0xFF  # 500 мс на кадр
            if key == ord('q'):
                print("\nДосрочный выход")
                break
            elif key == ord(' '):
                print("Пауза. Нажмите любую клавишу...")
                cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.print_statistics(model_version)

    def print_statistics(self, model_version):
        if not self.iou_scores:
            print("\nНет данных!")
            return
        print("\n" + "=" * 50)
        print(f"СТАТИСТИКА YOLOv8{model_version}")
        print("=" * 50)
        valid_indices = [i for i, status in enumerate(self.detection_status)
                         if status in ['DETECTED', 'LOW IoU']]
        if valid_indices:
            valid_iou = [self.iou_scores[i] for i in valid_indices]
            valid_cle = [self.cle_scores[i] for i in valid_indices]
            print(f"Всего кадров: {len(self.iou_scores)}")
            print(f"Кадров с обнаружением: {len(valid_iou)}")
            print(f"Кадров без обнаружения: {len(self.iou_scores) - len(valid_iou)}")
            print(f"Средний IoU: {np.mean(valid_iou):.3f}")
            print(f"Средний CLE: {np.mean(valid_cle):.1f} px")
            print(f"Медианный CLE: {np.median(valid_cle):.1f} px")
            print(f"Минимальный IoU: {np.min(valid_iou):.3f}")
            print(f"Максимальный IoU: {np.max(valid_iou):.3f}")
            successful = sum(1 for iou in self.iou_scores if iou > 0.5)
            success_rate = (successful / len(self.iou_scores)) * 100
            print(f"Успешных кадров (IoU > 0.5): {successful}/{len(self.iou_scores)} ({success_rate:.1f}%)")
            low_error = sum(1 for cle in valid_cle if cle < 20)
            low_error_rate = (low_error / len(valid_cle)) * 100 if valid_cle else 0
            print(f"Кадров с CLE < 20px: {low_error}/{len(valid_cle)} ({low_error_rate:.1f}%)")
        else:
            print("Нет обнаружений!")
        # Сохраняем статистику в JSON
        stats = {
            "model_version": model_version,
            "total_frames": len(self.iou_scores),
            "detected_frames": len(valid_iou) if valid_indices else 0,
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
            "detection_status": self.detection_status
        }
        stats_file = os.path.join(self.results_dir, f"yolo_{model_version}", "statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"\nСтатистика сохранена в {stats_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Название видео (папка в experiments/)")
    parser.add_argument("--model", type=str, default=None, help="Версия YOLO (n,s,m,l,x)")
    parser.add_argument("--class", type=int, default=0, dest="class_id",
                        help="Класс объекта (0=человек, 2=автомобиль, 3=мотоцикл)")
    args = parser.parse_args()
    print("Тестирование YOLO на устойчивость к смазыванию")
    print("=" * 50)
    versions = [args.model] if args.model else ['n', 's', 'm', 'l', 'x']
    for version in versions:
        tester = YOLOTester(video_name=args.video, model_version=version, class_id=args.class_id)
        print(f"\n{'=' * 50}")
        print(f"Запуск YOLOv8{version}")
        print(f"{'=' * 50}")
        tester.test_yolo(model_version=version)

if __name__ == "__main__":
    main()