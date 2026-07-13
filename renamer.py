import os
from glob import glob
import shutil

def rename_frames(folder_path, output_folder, prefix="frame_", digits=6, start_from=0):
    files = glob(os.path.join(folder_path, "*.jpg"))
    if not files:
        print(f"Файлы .jpg не найдены в {folder_path}")
        return
    os.makedirs(output_folder, exist_ok=True)
    renamed_count = 0
    for new_index, file_path in enumerate(files, start=start_from):
        filename = os.path.basename(file_path)
        new_name = f"{prefix}{new_index:0{digits}d}.jpg"
        new_path = os.path.join(output_folder, new_name)
        if output_folder == folder_path:
            if filename != new_name:
                os.rename(file_path, new_path)
                print(f"{filename} -> {new_name}")
                renamed_count += 1
        else:
            shutil.copy2(file_path, new_path)
            print(f"{filename} -> {new_name} (скопирован в {output_folder})")
            renamed_count += 1
    print(f"\nОбработано {renamed_count} файлов")

def main():
    print("=" * 50)
    print("   ПЕРЕИМЕНОВАТЕЛЬ КАДРОВ")
    print("=" * 50)
    print()
    folder_path = input("Путь к папке с кадрами: ").strip()
    output_folder = input("Куда сохранить (Enter - та же папка): ").strip()
    if not output_folder:
        output_folder = folder_path
    print("\nВыберите формат исходной нумерации:")
    print("1. SportsMOT (6 цифр: 000001.jpg)")
    print("2. OTB100 (4 цифры: 0001.jpg)")
    print("3. Свой вариант")
    choice = input("\nВаш выбор (1/2/3): ").strip()
    if choice == '1':
        digits = 6
        start_from = 1
    elif choice == '2':
        digits = 6
        start_from = 1
    elif choice == '3':
        digits = int(input("Количество цифр в номере (например, 6): ").strip())
        start_from = int(input("Номер первого кадра (0 или 1): ").strip())
    else:
        print("Неверный выбор")
        return
    rename_frames(folder_path, output_folder, prefix="frame_", digits=digits, start_from=start_from)

if __name__ == "__main__":
    main()