import json

def convert_otb_format(input_file, output_file):
    labels = {}
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        line = line.replace(',', ' ')
        parts = line.split()
        if len(parts) >= 4:
            x, y, w, h = map(int, parts[:4])
            frame_name = f"frame_{(idx + 1):06d}.jpg"
            labels[frame_name] = [x, y, w, h]

    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=4)
    print(f"Конвертация OTB завершена. Сохранено {len(labels)} кадров в {output_file}")

def convert_sportsmot_format(input_file, output_file, target_id=0):
    labels = {}
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 6:
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x = int(parts[2])
            y = int(parts[3])
            w = int(parts[4])
            h = int(parts[5])
            if obj_id == target_id:
                frame_name = f"frame_{frame_id:06d}.jpg"
                labels[frame_name] = [x, y, w, h]
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=4)
    print(f"Конвертация SportsMOT завершена. Сохранено {len(labels)} кадров в {output_file}")

def main():
    print("=" * 50)
    print("   КОНВЕРТЕР ФОРМАТОВ РАЗМЕТКИ")
    print("=" * 50)
    print()
    print("Выберите формат исходного файла:")
    print("1. OTB (простой TXT: x y w h на строку)")
    print("2. SportsMOT (CSV: frame,id,x,y,w,h,...)")
    print()
    choice = input("Ваш выбор (1 или 2): ").strip()
    input_file = input("Путь к исходному файлу разметки: ").strip()
    output_file = input("Путь для сохранения JSON (по умолчанию manual_labels.json в папку программы): ").strip()
    if not output_file:
        output_file = "manual_labels.json"
    if choice == '1':
        convert_otb_format(input_file, output_file)
    elif choice == '2':
        target_id = input("ID отслеживаемого объекта (по умолчанию 0): ").strip()
        target_id = int(target_id) if target_id else 0
        convert_sportsmot_format(input_file, output_file, target_id)
    else:
        print("Неверный выбор!")

if __name__ == "__main__":
    main()