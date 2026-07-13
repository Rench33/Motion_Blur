import os
import sys
import time
import json
import cv2
import argparse
import numpy as np
from glob import glob
from ultralytics import YOLO
from test_csrt import CSRTTester
from test_optical_flow import OpticalFlowTracker
from test_camshift import CamShiftTracker
from hybrid_yolo_csrt import HybridYOLOCSRT
from hybrid_yolo_of import HybridYOLOOF
from hybrid_yolo_camshift import HybridYOLOCamShift

class Benchmark:
    def __init__(self, video_name, frames_dir="my_frames", labels_file="manual_labels.json", class_id=0):
        self.class_id = class_id
        self.video_name = video_name
        self.base_dir = os.path.join("experiments", video_name)
        self.frames_dir = os.path.join(self.base_dir, frames_dir)
        self.labels_file = os.path.join(self.base_dir, labels_file)
        self.labels = self.load_labels()
        self.resolution = None
        self.results = {}

    def load_labels(self):
        with open(self.labels_file, 'r') as f:
            return json.load(f)

    def get_frames(self, max_frames=None):
        all_frames = sorted(glob(os.path.join(self.frames_dir, "*.jpg")))
        frames = []
        for f in all_frames:
            frame_name = os.path.basename(f)
            if frame_name in self.labels:
                frame = cv2.imread(f)
                if self.resolution is None:
                    h, w = frame.shape[:2]
                    self.resolution = {"width": w, "height": h, "megapixels": round(w * h / 1_000_000, 2)}
                frames.append((frame_name, frame))
                if max_frames and len(frames) >= max_frames:
                    break
        return frames

    def compute_stats(self, times):
        if not times:
            return {"mean_ms": 0, "std_ms": 0, "N": 0, "error_ms": 0}
        mean = np.mean(times)
        std = np.std(times, ddof=1) if len(times) > 1 else 0
        N = len(times)
        error = std / np.sqrt(N) if N > 0 else 0
        return {
            "mean_ms": round(mean, 2),
            "std_ms": round(std, 2),
            "N": N,
            "error_ms": round(error, 2)
        }

    def benchmark_csrt(self, frames):
        print("  CSRT...", end=" ", flush=True)
        times = []
        first_frame_name, first_frame = frames[0]
        first_box = self.labels[first_frame_name]
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(first_frame, tuple(first_box))
        for frame_name, frame in frames[1:]:
            start = time.perf_counter()
            success, _ = tracker.update(frame)
            end = time.perf_counter()
            if success:
                times.append((end - start) * 1000)
        stats = self.compute_stats(times)
        print(f"{stats['mean_ms']} ± {stats['error_ms']} мс (N={stats['N']})")
        return stats

    def benchmark_optical_flow(self, frames):
        print("  Optical Flow...", end=" ", flush=True)
        times = []
        first_frame_name, first_frame = frames[0]
        first_box = self.labels[first_frame_name]
        tracker = OpticalFlowTracker(max_points=200, min_points=5)
        tracker.init(first_frame, first_box)
        for frame_name, frame in frames[1:]:
            start = time.perf_counter()
            success, _ = tracker.update(frame)
            end = time.perf_counter()
            if success:
                times.append((end - start) * 1000)
        stats = self.compute_stats(times)
        print(f"{stats['mean_ms']} ± {stats['error_ms']} мс (N={stats['N']})")
        return stats

    def benchmark_camshift(self, frames):
        print("  CamShift...", end=" ", flush=True)
        times = []
        first_frame_name, first_frame = frames[0]
        first_box = self.labels[first_frame_name]
        tracker = CamShiftTracker()
        tracker.init(first_frame, first_box)
        for frame_name, frame in frames[1:]:
            start = time.perf_counter()
            success, _ = tracker.update(frame)
            end = time.perf_counter()
            if success:
                times.append((end - start) * 1000)
        stats = self.compute_stats(times)
        print(f"{stats['mean_ms']} ± {stats['error_ms']} мс (N={stats['N']})")
        return stats

    def benchmark_yolo(self, frames, model_version='n'):
        print(f"  YOLOv8{model_version}...", end=" ", flush=True)
        model = YOLO(f"yolov8{model_version}.pt")
        times = []
        detections_per_frame = []

        for frame_name, frame in frames:
            start = time.perf_counter()
            results = model(frame, conf=0.25, classes=[self.class_id], verbose=False)
            end = time.perf_counter()
            times.append((end - start) * 1000)

            # Подсчет детекций ПОСЛЕ замера времени
            num_dets = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                num_dets = len(results[0].boxes)
            detections_per_frame.append(num_dets)

        stats = self.compute_stats(times)
        avg_dets = np.mean(detections_per_frame) if detections_per_frame else 0

        # Выводим в одну строку (минимальный оверхед)
        print(f"{stats['mean_ms']} ± {stats['error_ms']} мс (N={stats['N']}, dets={avg_dets:.1f})")

        # Сохраняем в результаты для финального отчета
        return {**stats, "avg_detections": round(avg_dets, 1)}

    def benchmark_hybrid_csrt(self, frames):
        print("  Hybrid YOLO+CSRT (n)...", end=" ", flush=True)
        times = []
        first_frame_name, first_frame = frames[0]
        first_box = self.labels[first_frame_name]
        model = YOLO("yolov8n.pt")
        csrt_tracker = cv2.legacy.TrackerCSRT_create()
        csrt_tracker.init(first_frame, tuple(first_box))
        csrt_box = first_box
        for frame_name, frame in frames[1:]:
            start = time.perf_counter()
            yolo_box = None
            results = model(frame, conf=0.25, classes=[self.class_id], verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                best_iou = 0.0
                best_box = None
                for det in boxes:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    det_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    iou = self.calculate_iou(csrt_box, det_bbox)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_box = det_bbox
                yolo_box = best_box
            csrt_success, csrt_box = csrt_tracker.update(frame)
            if not csrt_success:
                csrt_box = None
            if yolo_box is not None and csrt_box is not None:
                csrt_tracker = cv2.legacy.TrackerCSRT_create()
                csrt_tracker.init(frame, tuple(map(int, yolo_box)))
                csrt_box = yolo_box
            end = time.perf_counter()
            times.append((end - start) * 1000)
            ground_truth = self.labels.get(frame_name)
            if ground_truth is not None and csrt_box is not None:
                csrt_iou = self.calculate_iou(ground_truth, csrt_box)
                if csrt_iou < 0.1:
                    break
        stats = self.compute_stats(times)
        print(f"{stats['mean_ms']} ± {stats['error_ms']} мс (N={stats['N']})")
        return stats

    def benchmark_hybrid_of(self, frames):
        print("  Hybrid YOLO+OF (n)...", end=" ", flush=True)
        times = []
        first_frame_name, first_frame = frames[0]
        first_box = self.labels[first_frame_name]
        model = YOLO("yolov8n.pt")
        of_tracker = OpticalFlowTracker(max_points=200, min_points=5)
        of_tracker.init(first_frame, first_box)
        of_box = first_box
        for frame_name, frame in frames[1:]:
            start = time.perf_counter()
            yolo_box = None
            results = model(frame, conf=0.25, classes=[self.class_id], verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                best_iou = 0.0
                best_box = None
                for det in boxes:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    det_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    iou = self.calculate_iou(of_box, det_bbox)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_box = det_bbox
                yolo_box = best_box
            of_success, of_box = of_tracker.update(frame)
            if not of_success:
                of_box = None
            if yolo_box is not None and of_box is not None:
                of_tracker = OpticalFlowTracker(max_points=200, min_points=5)
                of_tracker.init(frame, tuple(map(int, yolo_box)))
                of_box = yolo_box
            end = time.perf_counter()
            times.append((end - start) * 1000)
            ground_truth = self.labels.get(frame_name)
            if ground_truth is not None and of_box is not None:
                of_iou = self.calculate_iou(ground_truth, of_box)
                if of_iou < 0.1:
                    break
        stats = self.compute_stats(times)
        print(f"{stats['mean_ms']} ± {stats['error_ms']} мс (N={stats['N']})")
        return stats

    def benchmark_hybrid_camshift(self, frames):
        print("  Hybrid YOLO+CamShift (n)...", end=" ", flush=True)
        times = []
        first_frame_name, first_frame = frames[0]
        first_box = self.labels[first_frame_name]
        model = YOLO("yolov8n.pt")
        cs_tracker = CamShiftTracker()
        cs_tracker.init(first_frame, first_box)
        cs_box = first_box
        for frame_name, frame in frames[1:]:
            start = time.perf_counter()
            yolo_box = None
            results = model(frame, conf=0.25, classes=[self.class_id], verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                best_iou = 0.0
                best_box = None
                for det in boxes:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    det_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    iou = self.calculate_iou(cs_box, det_bbox)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_box = det_bbox
                yolo_box = best_box
            cs_success, cs_box = cs_tracker.update(frame)
            if not cs_success:
                cs_box = None
            if yolo_box is not None and cs_box is not None:
                cs_tracker = CamShiftTracker()
                cs_tracker.init(frame, tuple(map(int, yolo_box)))
                cs_box = yolo_box
            end = time.perf_counter()
            times.append((end - start) * 1000)
            ground_truth = self.labels.get(frame_name)
            if ground_truth is not None and cs_box is not None:
                cs_iou = self.calculate_iou(ground_truth, cs_box)
                if cs_iou < 0.1:
                    break
        stats = self.compute_stats(times)
        print(f"{stats['mean_ms']} ± {stats['error_ms']} мс (N={stats['N']})")
        return stats

    def calculate_iou(self, box1, box2):
        if box1 is None or box2 is None:
            return 0.0
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0

    def run(self, max_frames=None):
        print(f"\n=== ЗАМЕР ПРОИЗВОДИТЕЛЬНОСТИ: {self.video_name} ===")
        frames = self.get_frames(max_frames)
        if len(frames) < 2:
            print("Недостаточно кадров для замера!")
            return
        print(f"Всего кадров в видео: {len(frames)}")
        print(
            f"Разрешение: {self.resolution['width']}x{self.resolution['height']} ({self.resolution['megapixels']} Мп)")
        print("-" * 60)
        self.results["CSRT"] = self.benchmark_csrt(frames)
        self.results["Optical Flow"] = self.benchmark_optical_flow(frames)
        self.results["CamShift"] = self.benchmark_camshift(frames)
        for v in ['n', 's', 'm', 'l', 'x']:
            self.results[f"YOLOv8{v}"] = self.benchmark_yolo(frames, v)
        self.results["Hybrid YOLO+CSRT (n)"] = self.benchmark_hybrid_csrt(frames)
        self.results["Hybrid YOLO+OF (n)"] = self.benchmark_hybrid_of(frames)
        self.results["Hybrid YOLO+CamShift (n)"] = self.benchmark_hybrid_camshift(frames)
        output_file = os.path.join("experiments", self.video_name, "benchmark_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                "resolution": self.resolution,
                "results": self.results
            }, f, indent=4)
        print("-" * 60)
        print(f"Результаты сохранены в {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Название видео (папка в experiments/)")
    parser.add_argument("--frames", type=int, default=None, help="Количество кадров для замера (по умолчанию все)")
    parser.add_argument("--class", type=int, default=0, dest="class_id",
                        help="Класс объекта (0=человек, 2=автомобиль, 3=мотоцикл)")
    args = parser.parse_args()
    bench = Benchmark(args.video, class_id=args.class_id)
    bench.run(max_frames=args.frames)

if __name__ == "__main__":
    main()