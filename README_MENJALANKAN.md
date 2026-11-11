# Sistem Prediksi Harga Kendaraan Bekas (Kelompok 9)

## Kebutuhan
- Python dengan Tkinter (bawaan Python untuk Windows). Diuji hanya pada Python 3.12 (numpy 2.5, pandas 3.0, scikit-learn 1.9); versi lain belum diuji.
- Pustaka: `pip install -r requirements.txt`

## Menjalankan aplikasi
1. Letakkan `KELOMPOK_9_PERBAIKAN.py` (boleh diganti nama menjadi `KELOMPOK_9.py`) di satu folder.
2. Buka terminal di folder itu, lalu jalankan: `python KELOMPOK_9_PERBAIKAN.py`
3. Saat pertama dijalankan, `DATASET_KENDARAAN.csv` (dan `.xlsx` jika openpyxl terpasang) dibuat otomatis.
   CSV lama yang tidak valid (harga bekas > harga baru, model tidak sesuai katalog) dibuat ulang otomatis.

## Menjalankan pengujian
`python test_kelompok9.py`  (20 pengujian, tidak membuka jendela GUI)

## Format input
- Harga tawaran dan kilometer: angka bulat, titik sebagai pemisah ribuan (contoh 150.000.000). Koma ditolak.
- Tahun: 4 digit, antara tahun rilis model dan 2026.
- Kilometer maksimum 150.000 (data latih hanya sampai sekitar 121.000 km).

## Catatan penting
Dataset dibuat oleh program dari rumus penyusutan (sintetis), bukan data pasar nyata.
Metrik evaluasi (MAPE, R²) mengukur kemampuan model meniru data tersebut.
