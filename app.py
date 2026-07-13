import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from sqlalchemy import text
from tensorflow.keras.models import load_model as keras_load_model
import pickle
import pandas as pd
import os
import io
from datetime import datetime
from zoneinfo import ZoneInfo

# Membatasi TensorFlow agar tidak rakus RAM/CPU di server Streamlit
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ========== JURUS SUNTIK PAKSA (MONKEY PATCHING) ==========
# 1. Simpan fungsi bawaan asli Keras ke dalam memori
original_dense_init = tf.keras.layers.Dense.__init__

# 2. Buat fungsi pencegat (interceptor)
def patched_dense_init(self, *args, **kwargs):
    # Buang paksa penyakitnya sebelum Keras sempat membacanya
    kwargs.pop('quantization_config', None)
    # Lanjutkan proses normal menggunakan fungsi asli
    original_dense_init(self, *args, **kwargs)

# 3. Ganti otak Keras dengan fungsi pencegat kita
tf.keras.layers.Dense.__init__ = patched_dense_init
# ==========================================================


# ==========================================
# 1. SETTING HALAMAN & THEME DASHBOARD
# ==========================================
st.set_page_config(
    page_title="Sistem Deteksi Gizi Makanan - Dashboard Skripsi",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: bold; color: #6C63FF; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #455A64; text-align: center; margin-bottom: 25px; }
    .metric-box { background-color: #E8F5E9; padding: 15px; border-radius: 8px; border-left: 5px solid #2E7D32; margin-bottom: 10px; text-align: center; }
    .step-box { background-color: #ffffff; padding: 15px; border-radius: 8px; border-left: 4px solid #FFA000; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .scan-btn { display: flex; justify-content: center; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INISIALISASI DATABASE & LOADER MODEL AI
# ==========================================
# Inisialisasi Koneksi ke TiDB Cloud
conn = st.connection('mysql', type='sql')

# Buat tabel otomatis jika belum ada di database
with conn.session as s:
    s.execute(text("""
        CREATE TABLE IF NOT EXISTS riwayat_mbg (
            id INT AUTO_INCREMENT PRIMARY KEY,
            waktu VARCHAR(255),
            target_grup VARCHAR(100),
            item_terdeteksi TEXT,
            total_kalori FLOAT,
            ketercapaian FLOAT
        )
    """))
    s.commit()

@st.cache_data
def load_nutrition_data():
    if os.path.exists('nutrition.csv'):
        return pd.read_csv('nutrition.csv')
    return pd.DataFrame(columns=['name', 'calories', 'proteins', 'carbohydrate', 'fat'])

def load_history_data():
    try:
        df = conn.query("SELECT waktu as Waktu, target_grup as `Target Grup`, item_terdeteksi as `Item Terdeteksi`, total_kalori as `Total Kalori`, ketercapaian as `Ketercapaian (%)` FROM riwayat_mbg ORDER BY id DESC", ttl=0)
        return df
    except Exception:
        return pd.DataFrame(columns=["Waktu", "Target Grup", "Item Terdeteksi", "Total Kalori", "Ketercapaian (%)"])

@st.cache_resource
def load_model():
    return tf.saved_model.load('model_ssd_skripsi_tfod/saved_model')

@st.cache_resource
def load_mlp_model():
    mlp = keras_load_model('model_mlp_skripsi_terbaru.keras', compile=False)
    
    with open('scaler_area.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return mlp, scaler

# Muat data CSV ke memori
nutrition_df = load_nutrition_data()


# ==========================================
# 3. KAMUS KELAS, STATE, & FUNGSI LOGIN
# ==========================================
CATEGORY_INDEX = {
    1: 'anggur', 2: 'apel', 3: 'ayam_goreng', 4: 'buah_naga', 5: 'buah_susu',
    6: 'daging_sapi', 7: 'edamame', 8: 'ikan', 9: 'jeruk', 10: 'kelengkeng',
    11: 'kentang', 12: 'lauk_pendamping', 13: 'mangga', 14: 'melon', 15: 'mie_goreng',
    16: 'nasi', 17: 'pisang', 18: 'rambutan', 19: 'roti', 20: 'salak',
    21: 'sayur', 22: 'selada', 23: 'semangka', 24: 'stroberi', 25: 'susu',
    26: 'tahu', 27: 'telur_ayam', 28: 'tempe', 29: 'ubi', 30: 'udang'
}

if 'history' not in st.session_state:
    st.session_state.history = []

if 'menu' not in st.session_state:
    st.session_state.menu = "🏠 Halaman Utama"

if 'akg_df' not in st.session_state:
    st.session_state.akg_df = pd.DataFrame([
        {"Grup": "SD (7-12 Tahun)", "kalori": 1900, "protein": 50, "karbohidrat": 275, "lemak": 60},
        {"Grup": "SMP (13-15 Tahun)", "kalori": 2200, "protein": 65, "karbohidrat": 320, "lemak": 75},
        {"Grup": "SMA/K (16-18 Tahun)", "kalori": 2400, "protein": 70, "karbohidrat": 350, "lemak": 85}
    ])

def check_password():
    """Fungsi otentikasi admin"""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if st.session_state["password_correct"]:
        return True
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 Login Admin (Kelola Data)")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if password == "admin_gizi_2026": 
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.sidebar.error("Password salah")
    return False

# ==========================================
# 4. SIDEBAR NAVIGASI & KONTROL
# ==========================================
if os.path.exists("hero_image.png"):
    st.sidebar.image("hero_image.png", width='stretch')

st.sidebar.title("📌 Menu Navigasi")
selected_menu = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Halaman Utama", 
    "🔍 Deteksi & Analisis Gizi", 
    "🕰️ Riwayat Deteksi",
    "📊 Arsitektur Sistem", 
    "🧠 Info Model & Cara Kerja AI",
    "🔧 Manajemen Data & Gizi"
], index=["🏠 Halaman Utama", "🔍 Deteksi & Analisis Gizi", "🕰️ Riwayat Deteksi", "📊 Arsitektur Sistem", "🧠 Info Model & Cara Kerja AI", "🔧 Manajemen Data & Gizi"].index(st.session_state.menu))

if selected_menu != st.session_state.menu:
    st.session_state.menu = selected_menu
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pengaturan AI")
threshold = st.sidebar.slider("Batas Keyakinan (Threshold)", 0.1, 0.9, 0.40, 0.05)
iou_threshold = st.sidebar.slider("Toleransi Tumpukan (IoU NMS)", 0.1, 1.0, 0.40, 0.05)

# ==========================================
# 5. ROUTING HALAMAN
# ==========================================

# HALAMAN 1: HALAMAN UTAMA
if st.session_state.menu == "🏠 Halaman Utama":
    st.markdown("<div class='main-title'>Sistem Deteksi Objek Makanan dan Estimasi Nilai Gizi</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Menggunakan Algoritma SSD dan MLP pada Program Makan Bergizi Gratis</div>", unsafe_allow_html=True)
    
    # Membagi layar menjadi 3 kolom (kiri 1 part, tengah 1.5 part, kanan 1 part)
    # Ini akan membuat gambar di tengah menjadi lebih kecil dan proporsional
    col_img1, col_img2, col_img3 = st.columns([1, 1.5, 1])
    with col_img2:
        if os.path.exists("hero_image.png"):
            st.image("hero_image.png", width='stretch')
    
    st.markdown("""
    📝 Tentang Aplikasi
    Sistem cerdas ini dirancang secara khusus untuk mendukung program **Makan Bergizi Gratis**. Dengan menggunakan arsitektur *end-to-end Machine Learning*, aplikasi ini mampu mendeteksi letak dan jenis lauk pada nampan secara *real-time* menggunakan **SSD (Single Shot Multibox Detector)**, kemudian mengestimasi nilai kalorinya melalui komputasi **MLP (Multi-Layer Perceptron)**.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Mulai Scan Makanan Sekarang", use_container_width=True):
            st.session_state.menu = "🔍 Deteksi & Analisis Gizi"
            st.rerun()
            
# HALAMAN 2: DETEKSI & ANALISIS GIZI (HYBRID)
elif st.session_state.menu == "🔍 Deteksi & Analisis Gizi":
    
    # Muat Model SSD
    try:
        detect_fn = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.sidebar.error(f"⚠️ Model SSD gagal dimuat: {e}")

    # Muat Model MLP
    try:
        mlp_model, area_scaler = load_mlp_model()
        mlp_loaded = True
    except Exception as e:
        mlp_loaded = False
        st.sidebar.error(f"⚠️ Model MLP gagal dimuat: {e}")

    
    st.markdown("<h2 style='text-align: center; color: #6C63FF;'>Analisis & Estimasi Gizi Cerdas (Hybrid AI)</h2>", unsafe_allow_html=True)
    
    target_grup = st.selectbox("🎯 Pilih Target Penerima MBG (Standar Kemenkes):", st.session_state.akg_df['Grup'].tolist())
    
    tab1, tab2 = st.tabs(["📁 Unggah Foto", "📸 Gunakan Kamera Real-Time"])
    
    image_source = None
    
    with tab1:
        uploaded_file = st.file_uploader("Unggah foto menu makanan Anda:", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])
        if uploaded_file is not None:
            image_source = uploaded_file
            
    with tab2:
        st.info("💡 Pastikan memberikan izin akses kamera pada browser Anda.")
        camera_file = st.camera_input("Ambil foto langsung dari nampan makanan:")
        if camera_file is not None:
            image_source = camera_file
    
    if image_source is not None and model_loaded:
        image = Image.open(image_source).convert('RGB')
        image_np = np.array(image)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption='📸 Input', width='stretch')
            
        with col2:
            with st.spinner("Model SSD sedang melokalisasi objek..."):
                input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
                detections = detect_fn(input_tensor)
                
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                
                draw = ImageDraw.Draw(image)
                im_width, im_height = image.size
                
                # List untuk menyimpan data per-kotak bounding box
                detected_instances = []
                
                boxes = detections['detection_boxes']
                scores = detections['detection_scores']
                classes = detections['detection_classes']
                
                selected_indices = tf.image.non_max_suppression(
                    boxes, 
                    scores, 
                    max_output_size=20, 
                    iou_threshold=iou_threshold, 
                    score_threshold=threshold
                )
                
                for i in selected_indices.numpy():
                    class_id = classes[i]
                    score = scores[i]
                    class_name = CATEGORY_INDEX.get(class_id, f'Unknown_{class_id}')
                        
                    ymin, xmin, ymax, xmax = boxes[i]
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                    
                    # Ekstraksi Fitur Geometris (Luas Area dalam Piksel Persegi)
                    pixel_area = (xmax - xmin) * im_width * (ymax - ymin) * im_height
                    
                    # Simpan instance
                    detected_instances.append({
                        'name': class_name,
                        'area': pixel_area,
                        'score': score
                    })
                    
                    draw.rectangle([(left, top), (right, bottom)], outline="#E65100", width=5)
                    draw.text((left + 5, top + 5), f"{class_name.replace('_', ' ').title()} {int(score*100)}%", fill="#E65100")
            
            st.image(image, caption='🎯 Hasil Lokalisasi', width=400)
            
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button("💾 Unduh Gambar Hasil Deteksi", byte_im, "hasil_deteksi.jpg", "image/jpeg", use_container_width=True)
            
        st.markdown("---")
        
        if detected_instances:
            st.markdown("### 📝 Rincian Gizi Dinamis (Dihitung Otomatis oleh AI MLP)")
            
            total_kalori = 0.0
            total_protein = 0.0
            total_karbo = 0.0
            total_lemak = 0.0
            
            unique_item_names = set()
            
            for inst in detected_instances:
                nama_bersih = str(inst['name']).replace('_', ' ').title()
                unique_item_names.add(nama_bersih)
                
                # Cari data dasar di CSV berdasarkan nama
                row_gizi = nutrition_df[nutrition_df['name'] == inst['name']]
                
                if not row_gizi.empty:
                    row_gizi = row_gizi.iloc[0]
                    csv_id = int(row_gizi['id'])
                    kalori_dasar = float(row_gizi['calories'])
                    
                    # ========================================================
                    # PROSES PREDIKSI MLP REGRESSION (INTI SKRIPSI)
                    # ========================================================
                    if mlp_loaded:
                        try:
                            # 1. Normalisasi Luas Area
                            area_scaled = area_scaler.transform([[inst['area']]])
                            
                            # 2. One-Hot Encoding untuk ID Kelas (Matriks 1x30)
                            class_onehot = np.zeros((1, 30))
                            class_onehot[0, csv_id - 1] = 1
                            
                            # 3. Lempar ke Model MLP
                            prediksi = mlp_model.predict([area_scaled, class_onehot], verbose=0)[0][0]
                            estimasi_kalori = max(0.0, float(prediksi))
                            
                        except Exception as e:
                            st.warning(f"MLP Error pada {nama_bersih}: {e}. Menggunakan nilai statis.")
                            estimasi_kalori = kalori_dasar
                    else:
                        estimasi_kalori = kalori_dasar
                        
                    # Kalkulasi makronutrien secara proporsional mengikuti kalori MLP
                    rasio = estimasi_kalori / kalori_dasar if kalori_dasar > 0 else 1.0
                    estimasi_protein = float(row_gizi['proteins']) * rasio
                    estimasi_karbo = float(row_gizi['carbohydrate']) * rasio
                    estimasi_lemak = float(row_gizi['fat']) * rasio
                    
                    # Tambahkan ke total keseluruhan
                    total_kalori += estimasi_kalori
                    total_protein += estimasi_protein
                    total_karbo += estimasi_karbo
                    total_lemak += estimasi_lemak
                    
                    # Tampilkan rincian per-objek
                    st.info(f"🍽️ **{nama_bersih}** (Akurasi SSD: *{int(inst['score']*100)}%* | Luas Piksel: *{int(inst['area']):,} px²*)\n\n"
                            f"⚡ Kalori AI: **{estimasi_kalori:.1f} kkal** | 🥩 Protein: **{estimasi_protein:.1f} g** | 🍞 Karbohidrat: **{estimasi_karbo:.1f} g** | 🥑 Lemak: **{estimasi_lemak:.1f} g**")
                else:
                    st.warning(f"⚠️ Data '{nama_bersih}' belum terdaftar di database Admin.")
            
            st.markdown("### 📊 Akumulasi Total Gizi (Hybrid Output)")
            
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                st.markdown(f"<div class='metric-box'>❤️ <b>Total Kalori</b><br><span style='font-size:22px;'>{total_kalori:.1f} kkal</span></div>", unsafe_allow_html=True)
            with meta_col2:
                st.markdown(f"<div class='metric-box'>🥩 <b>Total Protein</b><br><span style='font-size:22px;'>{total_protein:.1f} g</span></div>", unsafe_allow_html=True)
            with meta_col3:
                st.markdown(f"<div class='metric-box'>🍞 <b>Total Karbohidrat</b><br><span style='font-size:22px;'>{total_karbo:.1f} g</span></div>", unsafe_allow_html=True)
            with meta_col4:
                st.markdown(f"<div class='metric-box'>🥑 <b>Total Lemak</b><br><span style='font-size:22px;'>{total_lemak:.1f} g</span></div>", unsafe_allow_html=True)
            
            # -------------------------------------------------------------
            # EVALUASI KEMENKES & SARAN TINDAK LANJUT
            # -------------------------------------------------------------
            st.markdown(f"#### ⚖️ Evaluasi Pemenuhan Gizi 1x Makan MBG untuk {target_grup}")
            st.info("Asumsi: 1 porsi MBG ditargetkan memenuhi ±33% (sepertiga) dari Angka Kecukupan Gizi (AKG) harian.")
            
            target_row = st.session_state.akg_df[st.session_state.akg_df['Grup'] == target_grup].iloc[0]
            target_1x_makan = target_row['kalori'] / 3
            persentase_kalori = (total_kalori / target_1x_makan) * 100
            
            col_eval1, col_eval2 = st.columns(2)
            with col_eval1:
                st.metric(label="Ketercapaian Kalori", value=f"{persentase_kalori:.1f}%", delta=f"{total_kalori - target_1x_makan:.1f} kkal dari target {target_1x_makan:.0f} kkal")
            
            with col_eval2:
                if persentase_kalori < 80:
                    st.warning("⚠️ Perhatian: Porsi kalori kurang dari target ideal.")
                    st.markdown("**💡 Saran Tindakan:** Tambahkan porsi karbohidrat atau protein padat.")
                elif persentase_kalori > 120:
                    st.warning("⚠️ Perhatian: Porsi kalori melebihi target ideal.")
                    st.markdown("**💡 Saran Tindakan:** Porsi terlalu besar. Kurangi takaran nasi atau lauk.")
                else:
                    st.success("✅ Hebat! Porsi gizi ideal dan sesuai standar Kemenkes.")
                    st.markdown("**💡 Saran Tindakan:** Komposisi gizi sudah seimbang. Pertahankan takaran ini.")
            if st.button("💾 Simpan Hasil ke Riwayat Permanen"):
                try:
                    # Menarik waktu lokal (WIB)
                    zona_wib = ZoneInfo("Asia/Jakarta")
                    waktu_sekarang = datetime.now(zona_wib)
                    waktu_simpan = waktu_sekarang.strftime("%Y-%m-%d %H:%M:%S")

                    with conn.session as s:
                        s.execute(text(
                            "INSERT INTO riwayat_mbg (waktu, target_grup, item_terdeteksi, total_kalori, ketercapaian) VALUES (:w, :tg, :it, :tk, :kc)"
                        ), {
                            "w": waktu_simpan,  # <-- Menggunakan variabel waktu_simpan yang baru
                            "tg": target_grup,
                            "it": ", ".join(list(unique_item_names)),
                            "tk": round(total_kalori, 2),
                            "kc": round(persentase_kalori, 2)
                        })
                        s.commit()
                    st.success("✅ Data berhasil disimpan secara permanen ke Database Cloud TiDB! Silakan cek menu Riwayat Deteksi.")
                except Exception as e:
                    st.error(f"Gagal menyimpan ke database: {e}")
                    
        else:
            st.info("Tidak ada makanan yang terdeteksi dengan jelas. Coba atur ulang sudut kamera atau ubah nilai slider 'Batas Keyakinan'.")

# HALAMAN 3: RIWAYAT DETEKSI
elif st.session_state.menu == "🕰️ Riwayat Deteksi":
    st.markdown("<h2 style='text-align: center; color: #6C63FF;'>Riwayat Analisis Gizi</h2>", unsafe_allow_html=True)
    
    df_history = load_history_data()
    
    if not df_history.empty:
        # Fitur Filter dan Sorting
        with st.expander("🛠️ Panel Filter & Urutkan Data", expanded=True):
            col_filt1, col_filt2 = st.columns(2)
            with col_filt1:
                # Filter berdasarkan Grup
                grup_unik = df_history['Target Grup'].unique().tolist()
                pilih_grup = st.multiselect("Filter Target Grup:", options=grup_unik, default=grup_unik)
            
            with col_filt2:
                # Sortir berdasarkan metrik
                pilih_sort = st.selectbox("Urutkan Berdasarkan:", [
                    "Waktu (Terbaru ke Terlama)", 
                    "Waktu (Terlama ke Terbaru)", 
                    "Kalori (Tertinggi ke Terendah)", 
                    "Kalori (Terendah ke Tertinggi)"
                ])
        
        # Terapkan Filter
        df_filtered = df_history[df_history['Target Grup'].isin(pilih_grup)].copy()
        
        # Terapkan Sort
        if pilih_sort == "Waktu (Terbaru ke Terlama)":
            pass 
        elif pilih_sort == "Waktu (Terlama ke Terbaru)":
            df_filtered = df_filtered.iloc[::-1] 
        elif pilih_sort == "Kalori (Tertinggi ke Terendah)":
            df_filtered = df_filtered.sort_values(by="Total Kalori", ascending=False)
        elif pilih_sort == "Kalori (Terendah ke Tertinggi)":
            df_filtered = df_filtered.sort_values(by="Total Kalori", ascending=True)

        # Tampilkan tabel
        st.dataframe(df_filtered, width='stretch', hide_index=True)
        
        # Tombol Unduh & Hapus
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            csv_history = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Data Tampil (CSV)", csv_history, "riwayat_mbg_terfilter.csv", "text/csv", use_container_width=True)
        with col_act2:
            if check_password(): 
                if st.button("🗑️ Kosongkan Seluruh Riwayat Permanen", use_container_width=True):
                    try:
                        with conn.session as s:
                            s.execute(text("TRUNCATE TABLE riwayat_mbg"))
                            s.commit()
                        st.success("Riwayat di Database Cloud berhasil dikosongkan. Silakan muat ulang halaman.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal mengosongkan database: {e}")
            else:
                st.info("🔒 Login Admin diperlukan untuk menghapus riwayat dari Cloud.")
    else:
        st.info("Belum ada data riwayat yang tersimpan di Database Cloud TiDB. Lakukan analisis dan simpan hasilnya terlebih dahulu.")


# HALAMAN 4: ARSITEKTUR SISTEM
elif st.session_state.menu == "📊 Arsitektur Sistem":
    st.markdown("<h2 style='text-align: center; color: #6C63FF;'>Arsitektur & Pemodelan Sistem</h2>", unsafe_allow_html=True)
    
    st.markdown("### 🛠️ 1. Arsitektur Jaringan Saraf Tiruan")
    if os.path.exists("arsitektur_lengkap.png"):
        st.image("arsitektur_lengkap.png", width='stretch')
        
    st.markdown("---")
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("### 🧠 2. Skema Jaringan MLP")
        if os.path.exists("arsitektur_mlp.png"):
            st.image("arsitektur_mlp.png", width='stretch')
    with col_arch2:
        st.markdown("### 🎯 3. Lapisan Topologi SSD")
        if os.path.exists("arsitektur_ssd.png"):
            st.image("arsitektur_ssd.png", width='stretch')


# HALAMAN 5: INFO MODEL & CARA KERJA AI
elif st.session_state.menu == "🧠 Info Model & Cara Kerja AI":
    st.markdown("<h2 style='text-align: center; color: #6C63FF;'>Model AI: SSD MobileNet V2</h2>", unsafe_allow_html=True)
    st.write("Sistem ini dibangun menggunakan algoritma **Single Shot Multibox Detector (SSD)** yang dipadukan dengan arsitektur ekstraktor fitur **MobileNet V2** melalui TensorFlow Object Detection API.")
    
    st.markdown("### 💡 Mengapa Menggunakan Model Ini?")
    st.info("Kombinasi **SSD** dan **MobileNet V2** sangat ideal untuk aplikasi pendeteksi objek secara *real-time*. MobileNet V2 dirancang sangat ringan sehingga proses komputasi menjadi cepat dan tidak membebani memori, sementara algoritma SSD mampu menemukan letak makanan sekaligus menebak jenis makanannya hanya dalam satu kali proses tebakan (*Single Shot*).")

    st.markdown("### ⚙️ Bagaimana Cara Kerja AI Mendeteksi Makanan?")
    st.markdown("<div class='step-box'><b>1. Tahap Input Gambar (Masukan)</b><br>Gambar nampan makanan yang Anda ambil melalui kamera atau unggah akan diubah menjadi deretan angka (Tensor matriks piksel) agar bisa dipahami oleh komputer.</div>", unsafe_allow_html=True)
    st.markdown("<div class='step-box'><b>2. Tahap Ekstraksi Fitur (Oleh MobileNet V2)</b><br>Jaringan MobileNet V2 bertugas sebagai 'mata' AI. Ia memindai gambar tersebut dan mencari ciri khas (fitur) dari makanan, seperti warna, tekstur, dan bentuk ujung pinggiran piring/makanan.</div>", unsafe_allow_html=True)
    st.markdown("<div class='step-box'><b>3. Tahap Lokalisasi & Klasifikasi (Oleh SSD)</b><br>Setelah fitur ditemukan, algoritma SSD menyebar ribuan kotak acuan (*anchor boxes*) di atas gambar untuk memposisikan letak objek dan mengklasifikasikan nama makanannya.</div>", unsafe_allow_html=True)
    st.markdown("<div class='step-box'><b>4. Tahap Integrasi Data Gizi (Modul Kalkulasi MLP)</b><br>Nama lauk dari hasil deteksi AI dikirim ke database CSV. Sistem mencocokkan namanya, menarik nilai gizi yang sesuai, menghitung akumulasinya, lalu menampilkannya di layar.</div>", unsafe_allow_html=True)
    
    st.markdown("### 🧹 Apa itu NMS (Non-Maximum Suppression)?")
    st.info("Sistem ini dilengkapi algoritma NMS untuk mengatasi masalah *multiple bounding boxes* (kotak yang bertumpuk pada satu makanan). NMS akan menyeleksi kotak berdasarkan skor tertinggi dan menghapus kotak lain yang bertumpang tindih melebihi batas IoU (*Intersection over Union*).")


# HALAMAN 6: MANAJEMEN DATA & GIZI (DENGAN AKSES LOGIN)
elif st.session_state.menu == "🔧 Manajemen Data & Gizi":
    st.markdown("<h2 style='text-align: center; color: #6C63FF;'>Panel Kelola Database Gizi</h2>", unsafe_allow_html=True)
    
    st.markdown("### 📝 Daftar Basis Data Gizi Makanan")
    
    is_admin = check_password()
    
    if is_admin:
        st.success("✅ Login Berhasil. Anda memiliki akses untuk mengedit data.")
        st.write("Perubahan pada tabel di bawah ini akan langsung tersimpan ke dalam file `nutrition.csv`.")
        
        edited_df = st.data_editor(nutrition_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("💾 Simpan Perubahan Gizi ke CSV"):
            try:
                edited_df.to_csv('nutrition.csv', index=False)
                st.success("✅ Database gizi berhasil diperbarui!")
                st.cache_data.clear() 
            except Exception as e:
                st.error(f"Gagal menyimpan data gizi: {e}")
                
        st.markdown("---")
        st.markdown("### 📊 Tabel Standar Angka Kecukupan Gizi (AKG) Kemenkes")
        st.write("Data di bawah merujuk pada Permenkes No. 28 Tahun 2019. Anda dapat mengedit atau menambahkan rentang usia target penerima MBG lainnya.")
        
        st.session_state.akg_df = st.data_editor(st.session_state.akg_df, num_rows="dynamic", use_container_width=True)
        
    else:
        st.info("👁️ Mode Lihat (Read-Only). Silakan login di panel sebelah kiri untuk menambah, mengedit, atau menghapus data.")
        st.dataframe(nutrition_df, width='stretch')
        
        st.markdown("---")
        st.markdown("### 📊 Tabel Standar Angka Kecukupan Gizi (AKG) Kemenkes")
        st.write("Data di bawah merujuk pada Permenkes No. 28 Tahun 2019.")
        st.dataframe(st.session_state.akg_df, width='stretch')
