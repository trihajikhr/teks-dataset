import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_lines(image_path, output_dir="output_lines"):
    print(f"Membaca gambar: {image_path}")
    
    # Buat folder output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Baca gambar dalam mode grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Tidak dapat membaca gambar {image_path}")
        return

    # 2. Binarization (Ubah ke Hitam & Putih tegas)
    # Kita pakai Inverted Threshold karena kita butuh teks=putih, background=hitam untuk deteksi histogram
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 3. Horizontal Projection Profile
    # Menjumlahkan nilai pixel putih secara horizontal (kiri ke kanan)
    # Area yang banyak putihnya (puncak grafik) berarti ada barisan teks
    hist = np.sum(thresh, axis=1)

    # 4. Cari batas atas dan batas bawah setiap baris teks
    lines = []
    in_line = False
    start_y = 0
    
    # Threshold deteksi: Anggap ada teks jika jumlah pixel putih melebihi batas ini
    # (Nilai ini mungkin perlu disesuaikan tergantung resolusi gambarmu)
    threshold_hist = np.max(hist) * 0.05 

    for y, val in enumerate(hist):
        if not in_line and val > threshold_hist:
            in_line = True
            start_y = y
        elif in_line and val <= threshold_hist:
            in_line = False
            # Simpan baris hanya jika tingginya wajar (misal lebih dari 15 pixel)
            if y - start_y > 15:
                # Tambahkan sedikit margin/padding di atas dan bawah
                margin = 5
                y1 = max(0, start_y - margin)
                y2 = min(img.shape[0], y + margin)
                lines.append((y1, y2))

    # Jika iterasi selesai tapi masih di dalam baris teks
    if in_line and (img.shape[0] - start_y > 15):
        lines.append((max(0, start_y - 5), img.shape[0]))

    # 5. Potong gambar dan simpan
    print(f"🎉 Sukses! Ditemukan {len(lines)} baris teks!")
    
    # Tampilkan gambar asli dengan kotak merah sebagai penanda
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for i, (y1, y2) in enumerate(lines):
        # Gambar kotak merah sebagai penanda
        cv2.rectangle(img_color, (0, y1), (img.shape[1], y2), (0, 0, 255), 2)
        
        # Potong gambar aslinya (bukan yang inverted)
        line_crop = img[y1:y2, 0:img.shape[1]]
        
        # Simpan ke file di folder output
        out_path = os.path.join(output_dir, f"baris_{i+1}.png")
        cv2.imwrite(out_path, line_crop)
        print(f"✅ Menyimpan: {out_path}")

    # 6. Gambar plot hasil Preview
    plt.figure(figsize=(12, 10))
    
    # Tampilkan gambar yang sudah diberi kotak merah
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Hasil Deteksi Baris")
    plt.axis("off")
    
    # Tampilkan grafik Horizontal Projection Profile-nya
    plt.subplot(1, 2, 2)
    plt.plot(hist, range(len(hist)))
    plt.gca().invert_yaxis()
    plt.title("Grafik Kepadatan Teks (Horizontal Profile)")
    
    preview_path = os.path.join(output_dir, "preview_segmentation.png")
    plt.savefig(preview_path)
    print(f"📸 Gambar visualisasi disimpan sebagai: {preview_path}")

if __name__ == "__main__":
    print("=== Eksperimen Fase 1: Line Segmentation ===")
    
    # File sampel (Ubah path ini sesuai gambar yang mau kamu tes)
    sample_img = "../data/sample.jpg"
    out_dir = "../data/images"
    
    if os.path.exists(sample_img):
        segment_lines(sample_img, output_dir=out_dir)
    else:
        print(f"⚠️ Peringatan: Gambar '{sample_img}' tidak ditemukan!")
        print("Silakan masukkan satu gambar catatan (bisa dari internet atau fotomu sendiri)")
        print(f"ke dalam folder 'ai/data/' dan beri nama 'sample.jpg'.")
        print("Lalu jalankan ulang script ini!")
