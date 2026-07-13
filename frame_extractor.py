import cv2
import os
import shutil

def extract_frames(video_path, output_dir="extracted_frames", skip_frames=1, start_frame=0, end_frame=None):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка открытия видео!")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    print(f"Оригинальных кадров: {total_frames}")
    print(f"Извлекаем с {start_frame} по {end_frame} (каждый {skip_frames}-й)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # Пропускаем кадры до start_frame
    saved_count = 0
    current_video_frame = start_frame
    while current_video_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (current_video_frame - start_frame) % skip_frames == 0:
            filename = f"frame_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            print(f"Кадр видео {current_video_frame} -> {filename}")
        current_video_frame += 1
        if current_video_frame % 100 == 0:
            print(f"Обработано: {current_video_frame}/{end_frame}")
    cap.release()
    print(f"\n=== Извлечение завершено ===")
    print(f"Извлечено кадров: {saved_count}")
    print(f"Пропущено кадров: {end_frame - start_frame - saved_count}")

if __name__ == "__main__":
    video_path = "match_4k.mov"
    extract_frames("match_4k.mov", output_dir="my_frames", skip_frames=1, start_frame=309, end_frame=355)