# Penanganan dua kolom: sebelum vs sesudah

Di-generate oleh `make before-after` dari PDF asli.

## KB_01: tanpa text layer, dua kolom (kiri 1-6, kanan 7-8)

**1. `page.get_text()` mentah** (0 karakter):

```text
(kosong: halaman hanya berisi gambar)
```

**2. OCR dibaca per baris visual, kiri ke kanan** (10 baris pertama). Section 1 dan 7 tercampur dalam satu baris:

```text
Penanganan Tiket Layanan
1. Tujuan 7. Penutupan
Panduan ini digunakan oleh petugas layanan untuk Tiket dapat ditutup setelah penyelesaian
menerima, mengelompokkan, menangani, dan dikonfirmasi pelanggan atau setelah dua hari kerja
menutup tiket pelanggan secara konsisten. Setiap tanpa tanggapan sejak notifikasi selesai, kecuali
tiket wajib memiliki nomor referensi dan waktu tiket P1 yang memerlukan konfirmasi eksplisit.
pencatatan.
8. Catatan Khusus
2. Pencatatan Tiket
Apabila ada indikasi kebocoran data, jangan
```

**3. Sesudah urutan kolom dan potong per section:**

```text
doc_title: Penanganan Tiket Layanan
1. Tujuan: Panduan ini digunakan oleh petugas layanan untuk menerima, mengelompok...
2. Pencatatan Tiket: Catat identitas pelanggan, kanal laporan, ringkasan masalah, waktu kej...
3. Tingkat Prioritas: P1 Kritis: layanan utama tidak dapat digunakan oleh banyak pelanggan. ...
4. Target Respons Awal: P1: 15 menit: P2: 1 jam: P3: 4 jam kerja: P4: 1 hari kerja. Respons aw...
5. Penugasan: Petugas layanan melakukan triase dan menetapkan penanggung jawab. Tim ...
6. Status Tiket: Status yang digunakan adalah Baru, Diproses, Menunggu Pelanggan, Menun...
7. Penutupan: Tiket dapat ditutup setelah penyelesaian dikonfirmasi pelanggan atau s...
8. Catatan Khusus: Apabila ada indikasi kebocoran data, jangan mencantumkan informasi rah...
```

## KB_03: ada text layer, dua kolom (kiri 1-6, kanan 7-8)

**1. `page.get_text()` mentah** (6 baris pertama). Nomor halaman `1` menempel tepat setelah judul:

```text
Panduan Operasional Layanan
1
1. Jam Layanan
Layanan reguler beroperasi Senin sampai Jumat
pukul 09.00-17.00 WIB, tidak termasuk hari libur
nasional. Gangguan kritis diteruskan melalui jalur
```

**2. Sesudah:**

```text
doc_title: Panduan Operasional Layanan
1. Jam Layanan: Layanan reguler beroperasi Senin sampai Jumat pukul 09.00-17.00 WIB, t...
2. Serah Terima Shift: Petugas mencatat tiket aktif, pekerjaan tertunda, hambatan, PIC, dan t...
3. Pemeriksaan Harian: Pada awal hari kerja, petugas memeriksa antrean tiket baru, tiket mend...
4. Pembaruan Berkala: Tiket P1 diperbarui kepada pelanggan setidaknya setiap 30 menit selama...
5. Kanal Komunikasi: Informasi resmi diberikan melalui kanal layanan yang tercatat. Permint...
6. Pengelolaan Dokumen: Simpan bukti dan catatan kerja di lokasi yang disetujui. Berkas pelang...
7. Pelaporan Mingguan: Koordinator menyusun ringkasan jumlah tiket, kategori masalah, ketepat...
8. Situasi Tidak Biasa: Jika sistem tiket tidak tersedia, catat laporan dalam formulir sementa...
```
