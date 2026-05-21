# Deteksi Gender dan Estimasi Usia Berbasis Citra Wajah Menggunakan CNN

Project final UAS ini menggunakan PyTorch untuk training model CNN dan Flask untuk demo aplikasi dashboard kamera/upload gambar.

## Isi Project

- `kamera gender dan usia.ipynb`: notebook training di Kaggle.
- `app.py`: aplikasi Flask untuk VS Code.
- `requirements.txt`: dependency aplikasi.
- `models/`: folder penyimpanan model yang diupload dari aplikasi.
- `templates/`: halaman dashboard dan halaman hasil prediksi.
- `static/`: CSS, JavaScript, dan aset visual dashboard.
- `static/assets/hologram-face-dashboard.png`: gambar hologram wajah untuk tampilan dashboard.

## Alur Sesuai Panduan UAS

1. Data acquisition menggunakan dataset UTKFace.
2. Data cleaning: validasi nama file, label usia, label gender, gambar corrupt, duplikat, dan missing value.
3. EDA: distribusi gender, histogram usia, boxplot usia, heatmap korelasi, dan contoh gambar.
4. Modeling: SimpleCNN dan ResNet18.
5. Evaluasi: Accuracy, Precision, Recall, F1-score, AUC-ROC, MAE, RMSE, dan confusion matrix.
6. Interpretasi hasil model dan fitur visual yang dipelajari CNN.
7. Final training menggunakan semua data setelah evaluasi selesai.
8. Demo akhir menggunakan dashboard Flask `app.py`.

## Fitur Aplikasi

- Dashboard interaktif untuk deteksi gender dan estimasi usia.
- Upload model `.pth` atau `.pt` dari halaman web, sehingga model tidak otomatis dimuat dari `app.py`.
- Kamera live untuk prediksi realtime setelah model berhasil dimuat.
- Upload gambar untuk prediksi dari file JPG, PNG, atau WEBP.
- Deteksi wajah menggunakan MediaPipe dengan fallback OpenCV Haar Cascade.
- Validasi kualitas wajah seperti ukuran wajah, pencahayaan, dan blur.
- Visual dashboard menggunakan aset hologram wajah, status model, status device, dan panel hasil live.

## Cara Training di Kaggle

1. Upload/import `kamera gender dan usia.ipynb` ke Kaggle.
2. Klik `Add Data`, tambahkan dataset UTKFace.
3. Aktifkan GPU: `Settings -> Accelerator -> GPU`.
4. Jalankan notebook dari atas ke bawah.
5. Download model dari `/kaggle/working/models/final_gender_age_model.pth`.

## Download Model

Model hasil training dapat diunduh melalui Google Drive:

```text
https://drive.google.com/file/d/1pbXLG7TNlAd4IttEh6Ceb1jfvTFZ9G7l/view?usp=drive_link
```

Gunakan file model tersebut pada panel **Upload Model** di dashboard.

## Cara Menjalankan App di VS Code

Jalankan aplikasi:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka browser:

```text
http://127.0.0.1:5000
```

## Alur Menggunakan Dashboard

1. Buka `http://127.0.0.1:5000`.
2. Pada panel **Upload Model**, pilih file model `.pth` atau `.pt` hasil training dari Kaggle atau dari link Google Drive.
3. Klik **Muat Model**.
4. Setelah status berubah menjadi **Model siap**, gunakan salah satu mode prediksi:
   - **Kamera Live** untuk prediksi realtime dari kamera.
   - **Upload Gambar** untuk prediksi dari file gambar.
5. Hasil prediksi menampilkan gender, rentang usia, dan confidence gender.

## Catatan

Evaluasi model tetap berasal dari test set sebelum final training. Final training semua data digunakan untuk membuat model akhir/demo aplikasi.
