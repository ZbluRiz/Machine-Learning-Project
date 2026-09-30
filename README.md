# 🚀 Machine Learning & Data Science Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-15B064?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

Koleksi proyek Machine Learning komprehensif yang mencakup permasalahan **Supervised Learning** (Regresi Linear, Tree-based Models, Ensemble Methods, dan Gradient Boosting) serta **Unsupervised Learning** (Clustering & Segmentasi Pasar). Setiap proyek dirancang dengan alur kerja Data Science standar industri: mulai dari *Exploratory Data Analysis* (EDA), pra-pemrosesan data, validasi silang (*cross-validation*), *hyperparameter tuning*, hingga evaluasi model dan *inference pipeline*.

---

## 📌 Daftar Proyek

| No | Proyek | Domain | Algoritma / Metode Utama | Metrik Utama | Status |
|:---:|---|---|---|---|:---:|
| 1 | [🌾 Crop Yield Prediction](#1-crop-yield-prediction-using-ensemble-learning) | AgTech / Pertanian | Decision Tree, Random Forest, Gradient Boosting, GridSearchCV | **$R^2 = 0.9860$** | Selesai |
| 2 | [🪙 Bitcoin Trading Volume](#2-bitcoin-trading-volume-prediction) | FinTech / Kripto | Linear Regression, Random Forest, XGBoost, RandomizedSearchCV | **$R^2 = 0.6829$** | Selesai |
| 3 | [🎓 Student Performance](#3-student-academic-performance-prediction) | EdTech / Pendidikan | Multiple Linear Regression (OLS) | **$R^2 = 0.9890$** | Selesai |
| 4 | [👥 Customer Segmentation](#4-customer-segmentation-analysis) | Retail / Pemasaran | K-Means Clustering, Silhouette & Davies-Bouldin Analysis | **Silhouette = 0.543** | Selesai |

---

## 🔬 Detail Proyek & Hasil Eksperimen

### 1. Crop Yield Prediction using Ensemble Learning
📁 **File:** [`Crop_Yield_Prediction_Ensemble.ipynb`](./Crop_Yield_Prediction_Ensemble.ipynb)  
📊 **Dataset:** `yield_df.csv` (28.242 baris data multinasional hasil panen tanaman)

* **Tujuan:** Mengestimasi produktivitas panen tanaman (`hg/ha_yield`) berdasarkan negara (*Area*), komoditas (*Item*), tahun, curah hujan rata-rata, volume pestisida, dan temperatur rata-rata.
* **Alur Eksperimen:**
  * **EDA Menyeluruh:** Analisis distribusi (*Histogram & KDE*), deteksi pencilan (*Boxplot & Violin Plot*), tren temporal tahunan, matriks korelasi pairplot, dan analisis agregat antar wilayah/komoditas.
  * **Feature Engineering & Preprocessing:** One-Hot Encoding fitur kategorikal + filtrasi outlier target berbasis Z-Score ($|Z| < 3$).
  * **Model Tuning:** Optimasi hyperparameter menggunakan `GridSearchCV` 3-Fold.
* **Hasil Evaluasi:**

| Model | MAE | RMSE | $R^2$ Score |
|---|:---:|:---:|:---:|
| **Random Forest (Tuned)** 🏆 | **3.053,90** | **8.524,30** | **0.9860** |
| Decision Tree (Tuned) | 3.340,78 | 11.223,33 | 0.9758 |
| Gradient Boosting (Tuned) | 18.063,29 | 28.827,29 | 0.8402 |

* **Insight Utama:** Random Forest menghasilkan akurasi tertinggi berkat kemampuannya memodelkan interaksi non-linear yang kompleks antara jenis komoditas dan kondisi iklim mikro. Analisis *Feature Importance* menunjukkan jenis komoditas (*Item*) dan negara (*Area*) merupakan prediktor paling signifikan terhadap hasil panen.

---

### 2. Bitcoin Trading Volume Prediction
📁 **File:** [`Bitcoin_Volume_Prediction_XGBoost.ipynb`](./Bitcoin_Volume_Prediction_XGBoost.ipynb)  
📊 **Dataset:** `btc.csv` (3.171 baris data historis transaksi Bitcoin harian)

* **Tujuan:** Memprediksi volume perdagangan harian (*Trading Volume*) berdasarkan struktur harga pasar (*Open, High, Low, Close, Adjusted Close*).
* **Alur Eksperimen:**
  * Standarisasi multivariat menggunakan `StandardScaler` untuk fitur input dan target volume.
  * Benchmarking baseline model: Ordinary Least Squares vs. Bagging (Random Forest) vs. Boosting (XGBoost).
  * Hyperparameter tuning berbasis `RandomizedSearchCV` (20 iterasi, 3-fold cross validation).
  * Pembuatan fungsi inferensi data baru terintegrasi dengan pemulihan skala (*inverse scaling*).
* **Hasil Evaluasi:**

| Model | MAE | RMSE | $R^2$ Score |
|---|:---:|:---:|:---:|
| Linear Regression | 0.4532 | 0.5894 | 0.5960 |
| Baseline Random Forest | 0.3208 | 0.5295 | 0.6739 |
| Baseline XGBoost | 0.3246 | 0.5655 | 0.6280 |
| **Tuned XGBoost** 🏆 | **0.3270** | **0.5221** | **0.6829** |

* **Best Parameters (XGBoost):** `n_estimators=500`, `max_depth=10`, `learning_rate=0.1`, `gamma=0.3`, `reg_alpha=1`, `reg_lambda=10`, `colsample_bytree=0.6`.

---

### 3. Student Academic Performance Prediction
📁 **File:** [`Student_Performance_Linear_Regression.ipynb`](./Student_Performance_Linear_Regression.ipynb)  
📊 **Dataset:** `Student_Performance.csv` (10.000 baris data performa siswa)

* **Tujuan:** Memodelkan faktor-faktor yang mempengaruhi capaian indeks prestasi belajar siswa (*Performance Index*).
* **Fitur:** Jam belajar (*Hours Studied*), nilai sebelumnya (*Previous Scores*), partisipasi ekstrakurikuler, durasi tidur, dan latihan soal.
* **Hasil Evaluasi:**
  * **$R^2$ Score:** `0.9890` (Model menjelaskan 98,9% variansi nilai siswa)
  * **Mean Absolute Error (MAE):** `1.6111`
  * **Root Mean Squared Error (RMSE):** `2.0206`
  * **Mean Absolute Percentage Error (MAPE):** `3.50%`
* **Insight:** Hasil evaluasi mengonfirmasi korelasi linear yang sangat kuat antara jam belajar terfokus dan riwayat nilai sebelumnya terhadap indeks performa akhir.

---

### 4. Customer Segmentation Analysis
📁 **File:** [`Customer_Segmentation_KMeans.ipynb`](./Customer_Segmentation_KMeans.ipynb)  
📊 **Dataset:** `segmentation.csv` (2.000 profil konsumen)

* **Tujuan:** Segmentasi demografis dan ekonomi pelanggan untuk strategi pemasaran berbasis kluster.
* **Fitur Utama:** Jenis kelamin, status perkawinan, usia, pendapatan tahunan (*Income*), dan jenis pekerjaan.
* **Metode & Metrik:**
  * Algoritma K-Means Clustering dengan konfigurasi awal $K=4$.
  * Evaluasi kualitas partisi kluster:
    * **Silhouette Score:** `0.5430` (Kluster terpisah secara wajar dan kohesif)
    * **Davies-Bouldin Index:** `0.5392` (Rasio dispersi intra-kluster rendah)
  * Analisis eksplorasi $K=2$ hingga $K=10$ untuk menentukan jumlah kluster optimal.

---

## 🛠️ Tech Stack & Library

* **Programming Language:** Python 3.10+
* **Machine Learning & Modeling:** `scikit-learn`, `xgboost`
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Environment:** Jupyter Notebook / Google Colab / VS Code

---

## 💻 Cara Menjalankan Proyek Secara Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/USERNAME/project-machine-learning.git
cd project-machine-learning
```

### 2. Buat & Aktifkan Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Jupyter Notebook
```bash
jupyter notebook
# atau
jupyter lab
```

---

## 📂 Struktur Repositori

```text
├── Bitcoin_Volume_Prediction_XGBoost.ipynb        # Proyek Prediksi Volume BTC (XGBoost)
├── Crop_Yield_Prediction_Ensemble.ipynb           # Proyek Prediksi Hasil Panen (Ensemble)
├── Customer_Segmentation_KMeans.ipynb             # Proyek Segmentasi Pelanggan (K-Means)
├── Student_Performance_Linear_Regression.ipynb    # Proyek Analisis Nilai Siswa (OLS)
├── Dataset/                                       # Direktori Dataset
│   ├── Student_Performance.csv                    # Dataset Nilai Siswa
│   └── btc.csv                                    # Dataset Historis Bitcoin
├── requirements.txt                               # Daftar dependensi Python
├── .gitignore                                     # Filter file yang diabaikan Git
└── README.md                                      # Dokumentasi utama proyek
```

---

## 👤 Profil Penulis

* **GitHub:** [@your-github-username](https://github.com/)
* **LinkedIn:** [Profil LinkedIn](https://www.linkedin.com/)
* **Email:** email.anda@example.com

⭐ *Jika repositori ini bermanfaat, jangan ragu untuk memberikan bintang (Star)!*
