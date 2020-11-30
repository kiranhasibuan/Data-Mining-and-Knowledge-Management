# Data-Mining-and-Knowledge-Management
Penyakit jantung menggambarkan berbagai kondisi yang memengaruhi jantung Anda. Penyakit jantung yang termasuk diantaranya yaitu penyakit pembuluh darah, seperti penyakit arteri koroner, masalah irama jantung (aritmia), dan cacat jantung sejak lahir (cacat jantung bawaan), dan lainnya.
Istilah "penyakit jantung" sering digunakan secara bergantian dengan istilah "penyakit kardiovaskular". Penyakit kardiovaskular umumnya mengacu pada kondisi yang melibatkan penyempitan atau penyumbatan pembuluh darah yang dapat menyebabkan serangan jantung, nyeri dada (angina), atau stroke. Kondisi jantung lainnya, seperti yang memengaruhi otot, katup, atau ritme jantung Anda, juga dianggap sebagai bentuk penyakit jantung.
Penyakit jantung merupakan salah satu penyebab morbiditas dan mortalitas terbesar di antara penduduk dunia. Prediksi penyakit kardiovaskular dianggap sebagai salah satu subjek terpenting dalam bagian analisis data klinis. Jumlah data di industri perawatan kesehatan sangat besar. Penambangan data mengubah kumpulan besar data perawatan kesehatan mentah menjadi informasi yang dapat membantu membuat keputusan dan prediksi yang tepat.
Menurut sebuah artikel berita, penyakit jantung terbukti menjadi penyebab utama kematian bagi wanita dan pria. Artikel tersebut menyatakan sebagai berikut:
•	Sekitar 610.000 orang meninggal karena penyakit jantung di Amerika Serikat setiap tahun – itu berarti 1 dari setiap 4 kematian.
•	Penyakit jantung adalah penyebab utama kematian bagi pria dan wanita. Lebih dari separuh kematian akibat penyakit jantung pada tahun 2009 terjadi pada pria.
•	Penyakit Jantung Koroner (PJK) adalah jenis penyakit jantung yang paling umum, menewaskan lebih dari 370.000 orang setiap tahun.
•	Setiap tahun sekitar 735.000 orang Amerika mengalami serangan jantung. Dari jumlah tersebut, 525.000 adalah serangan jantung pertama dan 210.000 terjadi pada orang yang pernah mengalami serangan jantung.
Hal ini membuat penyakit jantung menjadi perhatian utama yang harus ditangani. Tetapi sulit untuk mengidentifikasi penyakit jantung karena beberapa faktor risiko yang berkontribusi seperti diabetes, tekanan darah tinggi, kolesterol tinggi, denyut nadi yang tidak normal, dan banyak faktor lainnya. Karena kendala tersebut, para ilmuwan telah beralih ke pendekatan modern seperti Data Mining dan Machine Learning untuk memprediksi penyakit.
Machine Learning (ML) terbukti efektif dalam membantu pengambilan keputusan dan prediksi dari sejumlah besar data yang dihasilkan oleh industri perawatan kesehatan.
Pada artikel ini, saya akan menerapkan pendekatan Machine Learning (dan akhirnya membandingkannya) untuk mengklasifikasikan apakah seseorang menderita penyakit jantung atau tidak, menggunakan salah satu kumpulan data - Heart_Disease_Classification dataset dari https://www.kaggle.com/datasets.
Dataset
Kumpulan data ini berasal dari tahun 1988 dan terdiri dari empat database: Cleveland, Hongaria, Swiss, dan Long Beach V. Ini berisi 76 atribut, termasuk atribut yang diprediksi, tetapi semua eksperimen yang diterbitkan mengacu pada penggunaan subset dari 14 di antaranya. Bidang "target" mengacu pada keberadaan penyakit jantung pada pasien. Ini bilangan bulat bernilai 0 = tidak ada penyakit dan 1 = penyakit.
Dataset terdiri dari 1025 data individu. Ada 14 atribut dalam dataset, yang dijelaskan di bawah ini:
	Age (age)
menampilkan usia individu.
	Sex (sex)
menampilkan jenis kelamin individu menggunakan format berikut: 1 = Pria; 0 = Wanita
	Chest-pain type (cp)
menampilkan jenis nyeri dada yang dialami oleh individu dengan menggunakan format berikut: 0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptotic
	Resting Blood Pressure (trestbps)	
menampilkan nilai tekanan darah istirahat individu dalam mmHg(unit)
	Serum Cholestrol (chol)
menampilkan kolesterol serum dalam mg / dl (unit)
	Fasting Blood Sugar (fbs)	
membandingkan nilai gula darah puasa seseorang dengan 120mg / dl.
Jika gula darah puasa> 120mg / dl maka: 1 (true)
lainnya: 0 (false)
	Resting ECG (restecg)
menampilkan hasil elektrokardiografi istirahat 
0 = normal
1 = memiliki ST-T wave abnormality
2 = left ventricular hyperthrophy
	Max heart rate achieved (thalach)	
menampilkan detak jantung maksimal yang dicapai oleh individu.
	Exercise induced angina (exang)
latihan angina yang diinduksi: 1=yes; 0=no
	ST depression induced by exercise relative to rest (oldpeak)	
menampilkan nilai yang merupakan integer atau float.
	Peak exercise ST segment (slope)	
0 = upsloping; 1 = flat; 2 = downsloping
	Number of major vessel (0-4) colored by flurosopy (ca)
menampilkan nilai sebagai integer atau float
	Thal (thal)
menampilkan thalassemia
	Diagnosis of heart disease (target)
Menampilkan apakah individu tersebut menderita penyakit jantung atau tidak:
1 = absence; 0 = present
Parameter Penyakit Jantung
	Age : Usia adalah faktor risiko terpenting dalam mengembangkan penyakit kardiovaskular atau jantung, dengan risiko sekitar tiga kali lipat pada setiap dekade kehidupan. Garis lemak koroner dapat mulai terbentuk pada masa remaja. Diperkirakan 82 persen orang yang meninggal karena penyakit jantung koroner berusia 65 tahun ke atas. Secara bersamaan, risiko stroke berlipat ganda setiap dekade setelah usia 55 tahun.
	Sex : Pria berisiko lebih besar terkena penyakit jantung dibandingkan wanita pra-menopause. Setelah melewati masa menopause, terdapat pendapat bahwa risiko wanita sama dengan pria meskipun data terbaru dari WHO dan PBB membantahnya. Jika seorang wanita menderita diabetes, dia lebih mungkin mengembangkan penyakit jantung daripada pria dengan diabetes.
	Angina(Chest Pain) : Angina adalah nyeri dada atau ketidaknyamanan yang disebabkan ketika otot jantung Anda tidak mendapatkan cukup darah yang kaya oksigen. Mungkin terasa seperti ada tekanan atau tekanan di dada Anda. Ketidaknyamanan juga bisa terjadi di bahu, lengan, leher, rahang, atau punggung. Nyeri angina bahkan mungkin terasa seperti gangguan pencernaan.
	 Resting Blood Pressure : Seiring waktu, tekanan darah tinggi dapat merusak arteri yang memberi makan jantung Anda. Tekanan darah tinggi yang terjadi dengan kondisi lain, seperti obesitas, kolesterol tinggi atau diabetes, semakin meningkatkan risiko Anda.
	Serum Cholesterol : Kadar kolesterol LDL (low-density lipoprotein / LDL) yang tinggi (kolesterol "jahat") kemungkinan besar akan mempersempit arteri. Tingkat trigliserida yang tinggi, sejenis lemak darah yang terkait dengan diet Anda, juga meningkatkan risiko serangan jantung. Namun, kolesterol high-density lipoprotein (HDL) tingkat tinggi (kolesterol "baik") menurunkan risiko serangan jantung.
	Fasting Blood Sugar : Tidak cukup memproduksi hormon yang disekresikan oleh pankreas (insulin) atau tidak merespons insulin dengan benar menyebabkan kadar gula darah tubuh Anda meningkat, meningkatkan risiko serangan jantung.
	Resting ECG : Untuk orang dengan risiko rendah penyakit kardiovaskular, USPSTF menyimpulkan dengan kepastian sedang bahwa potensi bahaya skrining dengan istirahat atau olahraga EKG sama atau melebihi potensi manfaat. Untuk orang dengan risiko menengah hingga tinggi, bukti saat ini tidak cukup untuk menilai keseimbangan manfaat dan bahaya skrining.
	Max heart rate achieved : Peningkatan risiko kardiovaskular, terkait dengan akselerasi detak jantung, sebanding dengan peningkatan risiko yang diamati pada tekanan darah tinggi. Telah terbukti bahwa peningkatan denyut jantung sebesar 10 denyut per menit dikaitkan dengan peningkatan risiko kematian jantung setidaknya 20%, dan peningkatan risiko ini serupa dengan yang diamati dengan peningkatan darah sistolik. tekanan sebesar 10 mm Hg.
	Exercise induced angina : Rasa sakit atau ketidaknyamanan yang terkait dengan angina biasanya terasa kencang, mencengkeram, atau tertekan, dan dapat bervariasi dari ringan hingga berat. Angina biasanya dirasakan di tengah dada tetapi bisa menyebar ke salah satu atau kedua bahu, punggung, leher, rahang, atau lengan. Bahkan bisa dirasakan di tangan Anda. o Jenis Angina a. Angina Stabil / Angina Pectoris b. Angina tidak stabil c. Varian (Prinzmetal) Angina d. Angina mikrovaskuler.
	Peak exercise ST segment : Tes stres EKG treadmill dianggap tidak normal jika ada penurunan segmen ST horizontal atau miring ke bawah ≥ 1 mm pada 60–80 ms setelah titik J. EKG latihan dengan depresi segmen ST yang miring ke atas biasanya dilaporkan sebagai tes 'samar-samar'. Secara umum, terjadinya depresi segmen ST horizontal atau miring ke bawah pada beban kerja yang lebih rendah (dihitung dalam MET) atau detak jantung menunjukkan prognosis yang lebih buruk dan kemungkinan penyakit multi-pembuluh yang lebih tinggi. Durasi depresi segmen ST juga penting, karena pemulihan yang lama setelah stres puncak konsisten dengan uji stres EKG treadmill positif. Temuan lain yang sangat mengindikasikan CAD yang signifikan adalah terjadinya elevasi segmen ST> 1 mm (sering menunjukkan iskemia transmural); pasien ini sering dirujuk segera untuk angiografi koroner.
