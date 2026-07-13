import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
})

class VideoAnalyzer:
    def __init__(self, video_name, base_dir="experiments"):
        self.video_name = video_name
        self.base_dir = os.path.join(base_dir, video_name, "results")
        self.results = {}
        self.total_frames = 0
        self.common_frames = 0
        self.load_all_results()

    def load_all_results(self):
        trackers = ["csrt", "optical_flow", "camshift"]
        for t in trackers:
            path = os.path.join(self.base_dir, t, "statistics.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.results[t] = json.load(f)
                    if self.total_frames == 0:
                        self.total_frames = self.results[t].get('total_frames', len(self.results[t].get('frame_numbers', [])))
        yolo_versions = ['n', 's', 'm', 'l', 'x']
        for v in yolo_versions:
            path = os.path.join(self.base_dir, "yolo", f"yolo_{v}", "statistics.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.results[f"yolo_{v}"] = json.load(f)
        hybrid_types = ["hybrid_csrt", "hybrid_of"]
        for htype in hybrid_types:
            for v in ['n']:
                key = f"{htype}_{v}"
                path = os.path.join(self.base_dir, htype, f"{htype}_{v}", "statistics.json")
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        self.results[key] = json.load(f)
        n_csrt = len(self.results.get("hybrid_csrt_n", {}).get("per_frame", []))
        n_of = len(self.results.get("hybrid_of_n", {}).get("per_frame", []))
        self.common_frames = min(n_csrt, n_of) if n_csrt and n_of else 0

    def truncate_to_common(self, data, is_hybrid=False):
        if not data or self.common_frames == 0:
            return data
        if is_hybrid:
            if 'per_frame' in data:
                data['per_frame'] = data['per_frame'][:self.common_frames]
                ious = [item['iou'] for item in data['per_frame']]
                cles = [item['cle'] for item in data['per_frame'] if item['cle'] != float('inf')]
                data['mean_iou'] = np.mean(ious) if ious else 0
                data['mean_cle'] = np.mean(cles) if cles else 0
                data['success_rate'] = sum(1 for i in ious if i > 0.5) / len(ious) * 100 if ious else 0
        else:
            if 'iou_scores' in data:
                data['iou_scores'] = data['iou_scores'][:self.common_frames]
                data['cle_scores'] = data['cle_scores'][:self.common_frames]
                data['frame_numbers'] = data['frame_numbers'][:self.common_frames]
                ious = [iou for iou in data['iou_scores'] if iou > 0]
                cles = [cle for cle in data['cle_scores'] if cle != float('inf')]
                data['mean_iou'] = np.mean(ious) if ious else 0
                data['mean_cle'] = np.mean(cles) if cles else 0
                data['success_rate'] = sum(1 for i in data['iou_scores'] if i > 0.5) / len(data['iou_scores']) * 100
        return data

    def plot_trackers_iou(self):
        save_dir = f"plots/{self.video_name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        trackers = {"csrt": "CSRT", "optical_flow": "Optical Flow"}
        for key, name in trackers.items():
            if key in self.results:
                data = self.truncate_to_common(self.results[key].copy())
                if 'iou_scores' in data:
                    plt.plot(data['frame_numbers'], data['iou_scores'],
                             label=f"{name} (ср. IoU: {data.get('mean_iou', 0):.3f})", linewidth=1.5)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Порог успеха (IoU=0.5)')
        plt.xlabel('Номер кадра', fontweight='bold', fontsize=18)
        plt.ylabel('IoU', fontweight='bold', fontsize=18)
        plt.title(f'{self.video_name}: Сравнение трекеров по IoU', fontweight='bold', fontsize=20)
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)

        x_min, x_max = plt.xlim()
        x_ticks = np.arange(0, x_max, 10)
        for x in x_ticks:
            plt.axvline(x=x, color='gray', linestyle=':', linewidth=3.0, alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'trackers_iou.png'), dpi=150)
        plt.close()

    def plot_trackers_cle(self):
        save_dir = f"plots/{self.video_name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        trackers = {"csrt": "CSRT", "optical_flow": "Optical Flow"}
        for key, name in trackers.items():
            if key in self.results:
                data = self.truncate_to_common(self.results[key].copy())
                if 'cle_scores' in data:
                    valid = [(f, c) for f, c in zip(data['frame_numbers'], data['cle_scores']) if c != float('inf')]
                    if valid:
                        frames, cles = zip(*valid)
                        plt.plot(frames, cles, label=f"{name} (ср. CLE: {data.get('mean_cle', 0):.1f}px)", linewidth=1.5)
                        plt.ylim(0, 55)
        plt.axhline(y=20, color='orange', linestyle='--', label='Допустимая ошибка (20px)')
        plt.axhline(y=50, color='r', linestyle='--', label='Критическая ошибка (50px)')
        plt.xlabel('Номер кадра', fontweight='bold', fontsize=18)
        plt.ylabel('CLE (пиксели)', fontweight='bold', fontsize=18)
        plt.title(f'{self.video_name}: Сравнение трекеров по CLE', fontweight='bold')
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'trackers_cle.png'), dpi=150)
        plt.close()

    def plot_yolo_iou(self):
        save_dir = f"plots/{self.video_name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        yolo_versions = ['n', 's', 'm', 'l', 'x']
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for v, color in zip(yolo_versions, colors):
            key = f"yolo_{v}"
            if key in self.results:
                data = self.truncate_to_common(self.results[key].copy())
                if 'iou_scores' in data:
                    plt.plot(data['frame_numbers'], data['iou_scores'],
                             label=f"YOLOv8{v} (ср. IoU: {data.get('mean_iou', 0):.3f})",
                             linewidth=1.5, color=color, alpha=0.7)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Порог успеха (IoU=0.5)')
        plt.xlabel('Номер кадра', fontweight='bold', fontsize=18)
        plt.ylabel('IoU', fontweight='bold', fontsize=18)
        plt.title(f'{self.video_name}: Сравнение моделей YOLO по IoU', fontweight='bold', fontsize=20)
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)

        x_min, x_max = plt.xlim()
        x_ticks = np.arange(0, x_max, 10)
        for x in x_ticks:
            plt.axvline(x=x, color='gray', linestyle=':', linewidth=3.0, alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'yolo_iou.png'), dpi=150)
        plt.close()

    def plot_yolo_cle(self):
        save_dir = f"plots/{self.video_name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        yolo_versions = ['n', 's', 'm', 'l', 'x']
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for v, color in zip(yolo_versions, colors):
            key = f"yolo_{v}"
            if key in self.results:
                data = self.truncate_to_common(self.results[key].copy())
                if 'cle_scores' in data:
                    valid = [(f, c) for f, c in zip(data['frame_numbers'], data['cle_scores']) if c != float('inf')]
                    if valid:
                        frames, cles = zip(*valid)
                        plt.plot(frames, cles, label=f"YOLOv8{v} (ср. CLE: {data.get('mean_cle', 0):.1f}px)",
                                 linewidth=1.5, color=color, alpha=0.7)
                        plt.ylim(0, 25)
        plt.axhline(y=20, color='orange', linestyle='--', label='Допустимая ошибка (20px)')
        plt.axhline(y=50, color='r', linestyle='--', label='Критическая ошибка (50px)')
        plt.xlabel('Номер кадра', fontweight='bold', fontsize=18)
        plt.ylabel('CLE (пиксели)', fontweight='bold', fontsize=18)
        plt.title(f'{self.video_name}: Сравнение моделей YOLO по CLE', fontweight='bold')
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'yolo_cle.png'), dpi=150)
        plt.close()

    def plot_hybrids_iou(self):
        save_dir = f"plots/{self.video_name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        hybrids = {"hybrid_csrt_n": "YOLO+CSRT", "hybrid_of_n": "YOLO+OpticalFlow"}
        for key, name in hybrids.items():
            if key in self.results:
                data = self.truncate_to_common(self.results[key].copy(), is_hybrid=True)
                if 'per_frame' in data:
                    frames = [item['frame_num'] for item in data['per_frame']]
                    ious = [item['iou'] for item in data['per_frame']]
                    mean_iou = np.mean(ious) if ious else 0
                    plt.plot(frames, ious, label=f"{name} (ср. IoU: {mean_iou:.3f})", linewidth=1.5)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Порог успеха (IoU=0.5)')
        plt.xlabel('Номер кадра', fontweight='bold', fontsize=18)
        plt.ylabel('IoU', fontweight='bold', fontsize=18)
        plt.title(f'{self.video_name}: Сравнение гибридных систем по IoU', fontweight='bold', fontsize=20)
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)

        x_min, x_max = plt.xlim()
        x_ticks = np.arange(0, x_max, 10)
        for x in x_ticks:
            plt.axvline(x=x, color='gray', linestyle=':', linewidth=3.0, alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'hybrids_iou.png'), dpi=150)
        plt.close()

    def plot_hybrids_cle(self):
        save_dir = f"plots/{self.video_name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        hybrids = {"hybrid_csrt_n": "YOLO+CSRT", "hybrid_of_n": "YOLO+OpticalFlow"}
        for key, name in hybrids.items():
            if key in self.results:
                data = self.truncate_to_common(self.results[key].copy(), is_hybrid=True)
                if 'per_frame' in data:
                    valid = [(item['frame_num'], item['cle']) for item in data['per_frame'] if item['cle'] != float('inf')]
                    if valid:
                        frames, cles = zip(*valid)
                        mean_cle = np.mean(cles) if cles else 0
                        plt.plot(frames, cles, label=f"{name} (ср. CLE: {mean_cle:.1f}px)", linewidth=1.5)
        plt.axhline(y=20, color='orange', linestyle='--', label='Допустимая ошибка (20px)')
        plt.axhline(y=50, color='r', linestyle='--', label='Критическая ошибка (50px)')
        plt.xlabel('Номер кадра', fontweight='bold', fontsize=18)
        plt.ylabel('CLE (пиксели)', fontweight='bold', fontsize=18)
        plt.title(f'{self.video_name}: Сравнение гибридных систем по CLE', fontweight='bold')
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'hybrids_cle.png'), dpi=150)
        plt.close()

    def print_summary_table(self):
        print("\n" + "=" * 100)
        print(f"СВОДНАЯ ТАБЛИЦА: {self.video_name}")
        print("=" * 100)
        print(f"Общее количество кадров в видео: {self.total_frames}")
        print(f"Общий участок для сравнения (мин. из гибридов): {self.common_frames} кадров")
        print("-" * 100)
        print(f"{'Метод':<25} {'Ср.IoU':<12} {'Ср.CLE':<12} {'Успешность':<12} {'% от всего':<12} {'N кадров':<10}")
        print("-" * 100)
        order = [
            "csrt", "optical_flow", "camshift",
            "yolo_n", "yolo_s", "yolo_m", "yolo_l", "yolo_x",
            "hybrid_csrt_n", "hybrid_of_n"
        ]
        names = {
            "csrt": "CSRT",
            "optical_flow": "Optical Flow",
            "camshift": "CamShift",
            "yolo_n": "YOLOv8n",
            "yolo_s": "YOLOv8s",
            "yolo_m": "YOLOv8m",
            "yolo_l": "YOLOv8l",
            "yolo_x": "YOLOv8x",
            "hybrid_csrt_n": "Гибрид YOLO+CSRT (n)",
            "hybrid_of_n": "Гибрид YOLO+OF (n)",
        }
        for key in order:
            if key in self.results:
                original_data = self.results[key]
                if key.startswith("hybrid"):
                    n_total_real = len(original_data.get('per_frame', []))
                else:
                    n_total_real = len(original_data.get('iou_scores', []))
                percent_of_total = (n_total_real / self.total_frames * 100) if self.total_frames > 0 else 0
                data = original_data.copy()
                if key.startswith("hybrid"):
                    data = self.truncate_to_common(data, is_hybrid=True)
                else:
                    data = self.truncate_to_common(data, is_hybrid=False)
                mean_iou = data.get('mean_iou', 0)
                mean_cle = data.get('mean_cle', 0)
                success = data.get('success_rate', 0)
                print(
                    f"{names[key]:<25} {mean_iou:<12.3f} {mean_cle:<12.1f} {success:<12.2f}% {percent_of_total:<12.1f}% {n_total_real}")

def main():
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        print("Папка experiments не найдена!")
        return
    videos = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]
    if not videos:
        print("В папке experiments нет видеопоследовательностей!")
        return
    print(f"Найдено видео: {', '.join(videos)}")
    for video in videos:
        print(f"\n\nОбработка видео: {video}")
        print("-" * 50)
        analyzer = VideoAnalyzer(video)
        if analyzer.common_frames == 0:
            print(f"Нет данных для общего участка. Пропуск.")
            continue
        analyzer.plot_trackers_iou()
        analyzer.plot_trackers_cle()
        analyzer.plot_yolo_iou()
        analyzer.plot_yolo_cle()
        analyzer.plot_hybrids_iou()
        analyzer.plot_hybrids_cle()
        analyzer.print_summary_table()
    print("\n\nВсе графики сохранены в папке plots/")

if __name__ == "__main__":
    main()