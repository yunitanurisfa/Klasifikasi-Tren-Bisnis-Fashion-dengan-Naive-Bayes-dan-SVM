# Klasifikasi-Tren-Bisnis-Fashion-dengan-Naive-Bayes-dan-SVM
KLASIFIKASI TREN BISNIS BERDASARKAN SENTIMEN PENGGUNA TWITTER DENGAN MACHINE LEARNING

Saya melakukan penelitian klasifikasi tren bisnis berdasarkan pengguna twitter dengan metode machie learning untuk tugas akhir kuliah. Dalam penelitian tersebut penelitian dilakukan dengan menggunakan hasil crawling data twitter dilakukan pada tweet yang mengandung kata ‘fashion’, ‘kuliner’, ‘jasa’, ‘properti’, dan ‘kerajinan tangan’. Data tweet tersebut diambil pada rrentang waktu tanggal 1 Januari 2022 hingga 31 Desember 2022. Data yang diambil yaitu data username, waktu, dan juga tweet. Dari data tweet yang diperoleh tersebut kemudian diberi label sesuai dengan kategori pencarian jenis bisnis popular pada twitter (‘fashion’, ‘kuliner’, ‘jasa’, ‘properti’, dan ‘kerajinan tangan’) dan dilengkapi dengan label positif dan negatif (‘fashion positif’, ‘fashion negatif’, ‘kuliner positif’, ‘kuliner negatif’, ‘jasa posif’, ‘jasa negatif’, ‘ property positif’, ‘properti negatif’, ‘kuliner positif’, ‘kuliner negatif’) yang menyatakan komentar berisi pernyataan positif atau negatif. Pada penelitian ini metode support vector machine dan naïve bayes akan dibandingkan untuk melakukan analisis sentiment tren bisinis berdasarkan twitt pengguna twitter. Data yang digunakan yaitu data twitt dari tanggal 1 Januari 2022 hingga 31 Desember 2022. Sebelum dilakukan pemodelan SVM dan Naïve bayes data dilakukan preprocessing dengan beberapa tahap yaitu filtering, case folding dan data cleaning, normalization, tokenization, stopword removal dan stemming. Pada penelotian ini stemming dilakukan menggunakan library Sastrawi dan Nondeterministic Context Stemmer. Perbandingan akurasi antara metode SVM dan Naïve bayes dilakukan dengan menggunakan metode confusion matrix. Hasil pengujian yang didapatkan menghasilkan bahwa akurasi SVM lebih tinggi jika dibandingkan dengan akurasi naïve bayes, yaitu SVM mendapatkan akurasi sebesar 88% pada proses stemming menggunakan sastrawi dan 87% pada proses stemming menggunakan Nondeterministic Context Stemmer sedangkan Naïve bayes sebesar 75% pada proses stemming menggunakan Sastrawi dan Nondeterministic Context Stemmer.
