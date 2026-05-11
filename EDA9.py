# ============================================================
# EDA FLIGHTS DATASET
# Fokus: Time Series, Trend, Seasonality, Lineplot, Boxplot, Heatmap
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Load dataset flights dari seaborn.
# Dataset ini berisi jumlah penumpang pesawat berdasarkan tahun dan bulan.
#
# Kolom:
# year       = tahun
# month      = bulan
# passengers = jumlah penumpang
df = sns.load_dataset('flights')

print("Dataset:")
print(df)

print("\nInfo dataset:")
print(df.info())

print("\nStatistik deskriptif:")
print(df.describe())

print("\nShape dataset:")
print(df.shape)

# Insight awal:
# Dataset flights memiliki 144 baris dan 3 kolom.
# Data terdiri dari year, month, dan passengers.
# Dataset ini cocok untuk latihan EDA time series sederhana.


# ============================================================
# 2. CEK MISSING VALUE
# ============================================================

print("\nMissing value per kolom:")
print(df.isnull().sum())

missing_percentage = df.isnull().sum() / len(df) * 100

print("\nPersentase missing value:")
print(missing_percentage)

# Insight:
# Tidak ada missing value pada dataset.
# Artinya, data lengkap dan tidak perlu dilakukan missing value handling.


# ============================================================
# 3. CEK DATA DUPLIKAT
# ============================================================

print("\nJumlah data duplikat:")
print(df.duplicated().sum())

# Insight:
# Tidak ada data duplikat.
# Setiap kombinasi year dan month hanya muncul satu kali.


# ============================================================
# 4. CEK JUMLAH DATA PER TAHUN
# ============================================================

# value_counts().sort_index() digunakan untuk mengecek apakah
# setiap tahun memiliki jumlah data yang sama.
print("\nJumlah data per tahun:")
print(df['year'].value_counts().sort_index())

plt.figure(figsize=(10, 4))
sns.countplot(data=df, x='year')
plt.title("Jumlah Data per Tahun")
plt.xlabel("Year")
plt.ylabel("Jumlah Baris")
plt.xticks(rotation=45)
plt.show()

# Insight:
# Setiap tahun memiliki 12 data.
# Artinya, data bulanan lengkap dari Januari sampai Desember untuk setiap tahun.
#
# Countplot ini bukan untuk melihat jumlah penumpang,
# tetapi untuk mengecek kelengkapan jumlah bulan per tahun.


# ============================================================
# 5. HISTOGRAM PASSENGERS
# ============================================================

# Histogram digunakan untuk melihat distribusi jumlah penumpang.
# KDE membantu melihat bentuk distribusi secara halus.

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x='passengers', kde=True)
plt.title("Distribusi Jumlah Penumpang")
plt.xlabel("Passengers")
plt.ylabel("Frequency")
plt.show()

# Insight:
# Distribusi passengers cenderung right-skewed ringan.
# Jumlah penumpang paling sering berada pada rentang rendah-menengah,
# kira-kira sekitar 100 sampai 300.
#
# Nilai passengers yang tinggi muncul lebih sedikit,
# terutama pada tahun-tahun akhir ketika jumlah penumpang makin besar.


# ============================================================
# 6. TOTAL PASSENGERS PER TAHUN
# ============================================================

# Groupby year digunakan untuk menghitung total penumpang per tahun.
penumpang_tahun = df.groupby('year')['passengers'].sum().reset_index()

print("\nTotal passengers per tahun:")
print(penumpang_tahun)

plt.figure(figsize=(10, 5))
sns.lineplot(data=penumpang_tahun, x='year', y='passengers', marker='o')
plt.title("Total Passengers per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Total Passengers")
plt.show()

# Insight:
# Total passengers meningkat secara jelas dari tahun ke tahun.
# Dari 1950 sampai 1960 terlihat trend naik yang cukup kuat.
#
# Ini menunjukkan adanya pertumbuhan jumlah penumpang pesawat
# dalam periode tersebut.


# ============================================================
# 7. RATA-RATA PASSENGERS PER TAHUN
# ============================================================

# Selain total, rata-rata per tahun juga bisa digunakan
# untuk melihat apakah jumlah penumpang bulanan meningkat.
mean_passengers = df.groupby('year')['passengers'].mean().reset_index()

print("\nRata-rata passengers per tahun:")
print(mean_passengers)

plt.figure(figsize=(10, 5))
sns.lineplot(data=mean_passengers, x='year', y='passengers', marker='o')
plt.title("Rata-rata Passengers per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Average Passengers")
plt.show()

# Insight:
# Rata-rata passengers per tahun juga meningkat.
# Ini memperkuat insight bahwa pertumbuhan bukan hanya karena satu bulan tertentu,
# tetapi secara umum jumlah penumpang bulanan makin tinggi dari tahun ke tahun.


# ============================================================
# 8. TOTAL PASSENGERS PER BULAN
# ============================================================

# Groupby month digunakan untuk melihat bulan mana yang paling ramai
# secara total dari seluruh tahun.
penumpang_bulan = df.groupby('month')['passengers'].sum().reset_index()

print("\nTotal passengers per bulan:")
print(penumpang_bulan)

plt.figure(figsize=(10, 5))
sns.lineplot(data=penumpang_bulan, x='month', y='passengers', marker='o')
plt.title("Total Passengers per Bulan")
plt.xlabel("Bulan")
plt.ylabel("Total Passengers")
plt.xticks(rotation=45)
plt.show()

# Insight:
# Jumlah penumpang tertinggi terjadi pada bulan Juli dan Agustus.
# Setelah Agustus, jumlah penumpang turun cukup jelas hingga November.
#
# Ini menunjukkan adanya pola musiman:
# pertengahan tahun cenderung menjadi periode paling ramai.


# ============================================================
# 9. PASSENGERS PER BULAN BERDASARKAN TAHUN
# ============================================================

# Lineplot dengan hue='year' digunakan untuk melihat pola bulanan
# pada setiap tahun.
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='month', y='passengers', marker='o', hue='year')
plt.title("Passengers per Bulan Berdasarkan Tahun")
plt.xlabel("Bulan")
plt.ylabel("Passengers")
plt.xticks(rotation=45)
plt.show()

# Insight:
# Bentuk garis antar tahun terlihat mirip.
# Hampir setiap tahun mengalami kenaikan pada pertengahan tahun,
# terutama sekitar Juli dan Agustus.
#
# Perbedaannya, tahun-tahun yang lebih baru memiliki jumlah passengers
# yang lebih tinggi dibanding tahun-tahun awal.
#
# Jadi ada dua pola penting:
# 1. Trend naik dari tahun ke tahun.
# 2. Seasonality, yaitu kenaikan berulang pada bulan tertentu.


# ============================================================
# 10. BOXPLOT PASSENGERS BERDASARKAN BULAN
# ============================================================

# Boxplot digunakan untuk membandingkan distribusi passengers
# pada setiap bulan.
#
# Cara baca:
# - Garis tengah box = median
# - Box = 50% data tengah
# - Whisker = rentang data normal
# - Titik di luar whisker = kandidat outlier

plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='month', y='passengers')
plt.title("Distribusi Passengers Berdasarkan Bulan")
plt.xlabel("Bulan")
plt.ylabel("Passengers")
plt.xticks(rotation=45)
plt.show()

# Insight:
# Juli dan Agustus memiliki median passengers paling tinggi.
# Artinya, pada kebanyakan tahun, dua bulan ini memang cenderung lebih ramai.
#
# Box pada beberapa bulan juga cukup tinggi,
# menunjukkan adanya variasi jumlah penumpang antar tahun.
#
# Bulan awal tahun seperti Januari dan Februari cenderung memiliki
# median passengers lebih rendah dibanding pertengahan tahun.


# ============================================================
# 11. PIVOT TABLE YEAR X MONTH
# ============================================================

# Pivot table mengubah data dari format panjang menjadi format matrix.
#
# Baris  = month
# Kolom  = year
# Nilai  = passengers
#
# Format ini sangat cocok untuk membuat heatmap.
pivot_flight = df.pivot(index='month', columns='year', values='passengers')

print("\nPivot table flights:")
print(pivot_flight)


# ============================================================
# 12. HEATMAP PASSENGERS PER BULAN DAN TAHUN
# ============================================================

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_flight, annot=True, fmt='d')
plt.title("Heatmap Passengers per Bulan dan Tahun")
plt.xlabel("Year")
plt.ylabel("Bulan")
plt.show()

# Insight:
# Heatmap memperlihatkan dua pola sekaligus:
#
# 1. Trend naik:
#    Warna semakin kuat dari tahun awal ke tahun akhir.
#    Ini berarti jumlah penumpang meningkat dari waktu ke waktu.
#
# 2. Seasonality:
#    Juli dan Agustus cenderung memiliki nilai tinggi di banyak tahun.
#    Ini menunjukkan bahwa dua bulan tersebut adalah periode ramai secara konsisten.
#
# Heatmap sangat bagus untuk membaca pola time series bulanan
# karena kombinasi tahun dan bulan terlihat dalam satu visual.


# ============================================================
# 13. MEMBUAT KOLOM DATE
# ============================================================

# Membuat kolom date dari gabungan year dan month.
# Ini membuat data lebih terasa seperti time series asli.
df['date'] = pd.to_datetime(
    df['year'].astype(str) + '-' + df['month'].astype(str) + '-01'
)

print("\nContoh kolom date:")
print(df[['year', 'month', 'date', 'passengers']].head())

print("\nData dengan kolom date:")
print(df[['year', 'month', 'date', 'passengers']])


# ============================================================
# 14. LINEPLOT TIME SERIES
# ============================================================

# Lineplot ini menampilkan perubahan passengers dari waktu ke waktu
# secara berurutan berdasarkan date.
plt.figure(figsize=(12, 5))
sns.lineplot(data=df, x='date', y='passengers', marker='o')
plt.title("Passengers Time Series")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.show()

# Insight:
# Lineplot time series menunjukkan trend naik jangka panjang.
# Selain itu, terlihat pola naik-turun yang berulang setiap tahun.
#
# Kenaikan musiman paling terlihat sekitar Juli dan Agustus.
# Ini menunjukkan adanya seasonality yang konsisten.


# ============================================================
# 15. FINAL INSIGHT
# ============================================================

print("""
FINAL INSIGHT FLIGHTS EDA:

1. Dataset flights memiliki 144 baris dan 3 kolom:
   year, month, dan passengers.

2. Dataset tidak memiliki missing value dan tidak memiliki data duplikat.

3. Setiap tahun memiliki 12 data, sehingga data bulanan lengkap
   dari Januari sampai Desember.

4. Distribusi passengers cenderung right-skewed ringan,
   karena jumlah penumpang tinggi lebih banyak muncul pada tahun-tahun akhir.

5. Total passengers meningkat dari tahun ke tahun.
   Ini menunjukkan adanya upward trend.

6. Rata-rata passengers per tahun juga meningkat,
   sehingga pertumbuhan terlihat konsisten secara bulanan.

7. Juli dan Agustus adalah bulan dengan jumlah passengers tertinggi.
   Ini menunjukkan adanya pola musiman atau seasonality.

8. Boxplot berdasarkan bulan menunjukkan bahwa median passengers
   pada Juli dan Agustus lebih tinggi dibanding bulan lain.

9. Heatmap memperjelas dua pola utama:
   trend naik dari tahun ke tahun dan seasonality pada bulan tertentu.

10. Lineplot time series menunjukkan pola naik jangka panjang
    dan fluktuasi berulang setiap tahun.

11. Dataset flights sangat cocok untuk belajar:
    - time series EDA
    - lineplot
    - groupby berdasarkan waktu
    - pivot table
    - heatmap
    - trend dan seasonality analysis
""")
