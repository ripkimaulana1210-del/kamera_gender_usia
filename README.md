# Deteksi Gender dan Estimasi Usia Berbasis Citra Wajah Menggunakan CNN

Project final UAS ini menggunakan PyTorch untuk training model CNN dan Flask untuk demo aplikasi kamera/upload gambar.

## Isi Project

- `final_kaggle_gender_age_pytorch_cnn.ipynb`: notebook training di Kaggle.
- `app.py`: aplikasi Flask untuk VS Code.
- `requirements.txt`: dependency aplikasi.
- `models/`: folder model final `final_gender_age_model.pth`.

## Alur Sesuai Panduan UAS

1. Data acquisition menggunakan dataset UTKFace.
2. Data cleaning: validasi nama file, label usia, label gender, gambar corrupt, duplikat, dan missing value.
3. EDA: distribusi gender, histogram usia, boxplot usia, heatmap korelasi, dan contoh gambar.
4. Modeling: SimpleCNN dan ResNet18.
5. Evaluasi: Accuracy, Precision, Recall, F1-score, AUC-ROC, MAE, RMSE, dan confusion matrix.
6. Interpretasi hasil model dan fitur visual yang dipelajari CNN.
7. Final training menggunakan semua data setelah evaluasi selesai.
8. Demo akhir menggunakan Flask `app.py`.

## Cara Training di Kaggle

1. Upload/import `final_kaggle_gender_age_pytorch_cnn.ipynb` ke Kaggle.
2. Klik `Add Data`, tambahkan dataset UTKFace.
3. Aktifkan GPU: `Settings -> Accelerator -> GPU`.
4. Jalankan notebook dari atas ke bawah.
5. Download model dari `/kaggle/working/models/final_gender_age_model.pth`.

## Cara Menjalankan App di VS Code

Masukkan model hasil Kaggle ke:

```text
models/final_gender_age_model.pth
```

Lalu jalankan:

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

## Catatan

Evaluasi model tetap berasal dari test set sebelum final training. Final training semua data digunakan untuk membuat model akhir/demo aplikasi.
