import csv
import os

def logging_csv(number, output_dir, landmark_list):
    """Salva dados de treinamento em CSV"""
    if 0 <= number <= 26:
        letra = chr(ord('a') + number)
        csv_path = os.path.join(output_dir, "keypoint.csv")
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        print(f"[Salvando] Letra: {letra} (classe {number})")