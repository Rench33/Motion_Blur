import cv2
import os
import json
import numpy as np
import argparse
from glob import glob
from ultralytics import YOLO

class HybridYOLOCSRT:
    def __init__(self, video_name, frames_dir="my_frames", labels_file="manual_labels.json", model_version="n", class_id=0):
        self.class_id = class_id
        self.video_name = video_name
        self.model_version = model_version
        self.base_dir = os.path.join("experiments", video_name)
        self.frames_dir = os.path.join(self.base_dir, frames_dir)
        self.labels_file = os.path.join(self.base_dir, labels_file)
        self.results_dir = os.path.join("experiments", video_name, "results", "hybrid_csrt", f"hybrid_csrt_{model_version}")
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
        self.results = []

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

    def find_best_yolo_detection(self, model, frame, ground_truth):
        results = model(frame, conf=0.25, classes=[self.class_id], verbose=False)
        if not results or len(results) == 0 or results[0].boxes is None:
            return None, 0.0
        boxes = results[0].boxes.data.cpu().numpy()
        best_iou = 0.0
        best_bbox = None
        for det in boxes:
            x1, y1, x2, y2, conf, cls = det[:6]
            det_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            iou = self.calculate_iou(ground_truth, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_bbox = det_bbox
        return best_bbox, best_iou

    def draw_result(self, frame, ground_truth, yolo_box, csrt_box, winner_box, winner_name):
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
        if ground_truth is not None: # Ground Truth (зеленый)
            x_gt, y_gt, w_gt, h_gt = ground_truth
            x_gt_disp = int(x_gt * scale_x)
            y_gt_disp = int(y_gt * scale_y)
            w_gt_disp = int(w_gt * scale_x)
            h_gt_disp = int(h_gt * scale_y)
            cv2.rectangle(result_frame, (x_gt_disp, y_gt_disp),
                          (x_gt_disp + w_gt_disp, y_gt_disp + h_gt_disp),
                          (0, 255, 0), 2)
        if yolo_box: # YOLO (синий)
            x_y, y_y, w_y, h_y = yolo_box
            cv2.rectangle(result_frame,
                          (int(x_y * scale_x), int(y_y * scale_y)),
                          (int((x_y + w_y) * scale_x), int((y_y + h_y) * scale_y)),
                          (255, 0, 0), 2)
        if csrt_box and csrt_box[2] > 1 and csrt_box[3] > 1: # CSRT (жёлтый)
            x_c, y_c, w_c, h_c = csrt_box
            cv2.rectangle(result_frame,
                          (int(x_c * scale_x), int(y_c * scale_y)),
                          (int((x_c + w_c) * scale_x), int((y_c + h_c) * scale_y)),
                          (0, 255, 255), 2)
        if winner_box: # Победитель (красный)
            x_w, y_w, w_w, h_w = winner_box
            cv2.rectangle(result_frame,
                          (int(x_w * scale_x), int(y_w * scale_y)),
                          (int((x_w + w_w) * scale_x), int((y_w + h_w) * scale_y)),
                          (0, 0, 255), 3)
        y_offset = 30
        cv2.putText(result_frame, f"HYBRID: YOLOv8{self.model_version} + CSRT", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(result_frame, f"Winner: {winner_name}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return result_frame

    def run(self):
        print(f"=== ГИБРИДНАЯ СИСТЕМА: YOLOv8{self.model_version} + CSRT (без GT) ===")
        print("=" * 60)
        labeled_frames = sorted(self.labels.keys())
        if not labeled_frames:
            print("Нет размеченных кадров!")
            return
        print(f"Найдено {len(labeled_frames)} размеченных кадров")
        model = YOLO(f"yolov8{self.model_version}.pt")
        print("YOLO загружен")
        # Инициализация на первом кадре по GT
        first_frame_path = os.path.join(self.frames_dir, labeled_frames[0])
        first_frame = cv2.imread(first_frame_path)
        first_ground_truth = self.labels[labeled_frames[0]]
        first_box, _ = self.find_best_yolo_detection(model, first_frame, first_ground_truth)
        if first_box is None:
            print("Ошибка: YOLO не нашёл объект на первом кадре!")
            return
        csrt_tracker = cv2.legacy.TrackerCSRT_create()
        csrt_tracker.init(first_frame, tuple(map(int, first_box)))
        csrt_box = first_box
        print(f"Инициализация CSRT по YOLO (max IoU): {first_box}")
        all_frames = sorted(glob(os.path.join(self.frames_dir, "*.jpg")))
        lost_counter = 0
        prev_winner_box = first_box
        for i, frame_path in enumerate(all_frames):
            frame_name = os.path.basename(frame_path)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            ground_truth = self.labels.get(frame_name)
            # 1. YOLO ищет объект, пересекающийся с текущим bbox трекера
            yolo_box = None
            yolo_conf = 0.0
            if csrt_box is not None:
                # Ищем среди всех обнаружений YOLO то, которое максимально пересекается с CSRT
                results = model(frame, conf=0.25, classes=[self.class_id], verbose=False)
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.data.cpu().numpy()
                    best_iou = 0.0
                    best_box = None
                    best_conf = 0.0
                    for det in boxes:
                        x1, y1, x2, y2, conf, cls = det[:6]
                        det_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        iou = self.calculate_iou(csrt_box, det_bbox)
                        if iou > best_iou and iou > 0.3:  # только если пересекается
                            best_iou = iou
                            best_box = det_bbox
                            best_conf = conf
                    yolo_box = best_box
                    yolo_conf = best_conf
            # 2. Обновление CSRT
            csrt_success, csrt_box = csrt_tracker.update(frame)
            if not csrt_success:
                csrt_box = None
            # 3. Логика выбора победителя и переинициализации
            winner_box = None
            winner_name = "NONE"
            if yolo_box is not None and csrt_box is not None:
                winner_box = yolo_box
                winner_name = "YOLO"
                csrt_tracker = cv2.legacy.TrackerCSRT_create()
                csrt_tracker.init(frame, tuple(map(int, yolo_box)))
            elif csrt_box is not None:
                winner_box = csrt_box
                winner_name = "CSRT"
            elif yolo_box is not None:
                winner_box = yolo_box
                winner_name = "YOLO"
                csrt_tracker = cv2.legacy.TrackerCSRT_create()
                csrt_tracker.init(frame, tuple(map(int, yolo_box)))
            else:
                winner_box = prev_winner_box if lost_counter < 5 else None
                winner_name = "PREDICTED" if lost_counter < 5 else "LOST"
            # 4. Метрики
            if ground_truth is not None:
                iou = self.calculate_iou(ground_truth, winner_box) if winner_box else 0
                cle = self.calculate_cle(ground_truth, winner_box) if winner_box else float('inf')
                yolo_iou = self.calculate_iou(ground_truth, yolo_box) if yolo_box else 0
                csrt_iou = self.calculate_iou(ground_truth, csrt_box) if csrt_box else 0
            else:
                iou, cle, yolo_iou, csrt_iou = 0, float('inf'), 0, 0
            # Остановка при потере целевого объекта
            if csrt_iou < 0.1 and csrt_box is not None:
                print(f"\nЦелевой объект потерян на кадре {frame_name}. Остановка.")
                break
            # Сохраняем статистику
            self.results.append({
                "frame": frame_name,
                "frame_num": i,
                "yolo_box": yolo_box,
                "yolo_conf": float(yolo_conf) if yolo_box else 0,
                "yolo_iou": yolo_iou,
                "csrt_box": csrt_box,
                "csrt_iou": csrt_iou,
                "winner": winner_name,
                "winner_box": winner_box,
                "iou": iou,
                "cle": cle
            })
            print(f"Кадр {frame_name}: YOLO conf={yolo_conf:.2f}, "
                  f"Победитель={winner_name}, IoU={iou:.3f}, CLE={cle:.1f}px")
            # Визуализация
            vis_frame = self.draw_result(frame, ground_truth, yolo_box, csrt_box, winner_box, winner_name)
            result_path = os.path.join(self.results_dir, f"result_{frame_name}")
            cv2.imwrite(result_path, vis_frame)
            cv2.imshow(f'Hybrid YOLOv8{self.model_version}+CSRT', vis_frame)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.print_statistics()

    def print_statistics(self):
        if not self.results:
            print("Нет данных!")
            return
        print("\n" + "=" * 60)
        print(f"СТАТИСТИКА ГИБРИДНОЙ СИСТЕМЫ (YOLOv8{self.model_version}+CSRT)")
        print("=" * 60)
        ious = [r["iou"] for r in self.results if r["iou"] > 0]
        cles = [r["cle"] for r in self.results if r["cle"] != float('inf')]
        yolo_wins = sum(1 for r in self.results if r["winner"] == "YOLO")
        csrt_wins = sum(1 for r in self.results if r["winner"] == "CSRT")
        print(f"Всего кадров до потери: {len(self.results)}")
        print(f"Побед YOLO: {yolo_wins}")
        print(f"Побед CSRT: {csrt_wins}")
        print(f"Средний IoU (гибрид): {np.mean(ious) if ious else 0:.3f}")
        print(f"Средний CLE (гибрид): {np.mean(cles):.1f} px")
        successful = sum(1 for r in self.results if r["iou"] > 0.5)
        print(f"Успешных кадров (IoU > 0.5): {successful}/{len(self.results)} "
              f"({successful / len(self.results) * 100:.1f}%)")
        # Сохраняем статистику в JSON
        stats = {
            "model_version": self.model_version,
            "total_frames": len(self.results),
            "yolo_wins": yolo_wins,
            "csrt_wins": csrt_wins,
            "mean_iou": float(np.mean(ious)),
            "mean_cle": float(np.mean(cles)),
            "success_rate": float(successful / len(self.results) * 100),
            "per_frame": self.results
        }
        with open(os.path.join(self.results_dir, "statistics.json"), 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"\nСтатистика сохранена в {self.results_dir}/statistics.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Название видео (папка в experiments/)")
    parser.add_argument("--model", type=str, default=None, help="Версия YOLO (n,s,m,l,x)")
    parser.add_argument("--class", type=int, default=0, dest="class_id",
                        help="Класс объекта (0=человек, 2=автомобиль, 3=мотоцикл)")
    args = parser.parse_args()
    print("Запуск гибридной системы YOLO + CSRT (без привязки к GT)")
    print("=" * 60)
    versions = [args.model] if args.model else ['n', 's', 'm', 'l', 'x']
    for v in versions:
        print(f"\n\n{'=' * 60}")
        print(f"Тестирование модели YOLOv8{v}")
        hybrid = HybridYOLOCSRT(video_name=args.video, model_version=v, class_id=args.class_id)
        hybrid.run()
    print("\nГибридизация завершена!")

if __name__ == "__main__":
    main()