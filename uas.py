import re
import os
import math
import streamlit as st
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


KAMUS_POSITIF = [
    # Rasa
    'enak', 'lezat', 'mantap', 'nikmat', 'gurih', 'sedap', 'maknyus', 'juara',
    'empuk', 'renyah', 'crispy', 'fresh', 'segar', 'harum', 'wangi',
    # Kualitas
    'bagus', 'berkualitas', 'premium', 'istimewa', 'spesial', 'hebat',
    'top', 'unggul', 'recommended', 'rekomendasi', 'favorit',
    # Kepuasan
    'puas', 'senang', 'suka', 'cinta', 'favorit', 'recommend', 'worth',
    'memuaskan', 'menyenangkan', 'nyaman', 'oke', 'ok', 'mantul',
    # Porsi & Harga
    'murah', 'terjangkau', 'ekonomis', 'banyak', 'besar', 'jumbo', 'melimpah',
    'cukup', 'pas', 'sesuai', 'sepadan', 'worthit',
    # Pelayanan
    'ramah', 'sopan', 'cepat', 'sigap', 'tanggap', 'responsif', 'profesional',
    # Tempat
    'bersih', 'rapi', 'nyaman', 'cozy', 'asri', 'strategis', 'luas',
    # Lainnya
    'konsisten', 'stabil', 'andal', 'terpercaya', 'cocok', 'sesuai', 'khas',
    'autentik', 'original', 'nagih', 'ketagihan', 'bikin', 'nambah'
]

KAMUS_NEGATIF = [
    # Rasa
    'tidak enak', 'gak enak', 'hambar', 'tawar', 'pahit', 'asam', 'basi',
    'amis', 'anyir', 'tengik', 'keras', 'alot', 'kering', 'gosong', 'mentah',
    # Waktu
    'lama', 'lambat', 'telat', 'terlambat', 'menunggu', 'antri', 'nunggu',
    # Harga
    'mahal', 'kemahalan', 'overprice', 'pricey', 'tidak worth', 'gak worth',
    # Kualitas
    'buruk', 'jelek', 'mengecewakan', 'zonk', 'gagal', 'tidak bagus',
    # Emosi
    'kecewa', 'kesal', 'jengkel', 'tidak puas', 'gak puas', 'kapok',
    # Tempat
    'kotor', 'jorok', 'kumuh', 'sempit', 'sesak', 'pengap', 'panas', 'gerah',
    'berantakan', 'acak', 'bau',
    # Porsi
    'sedikit', 'kurang', 'minim', 'kecil', 'mini', 'pelit',
    # Pelayanan
    'tidak ramah', 'gak ramah', 'cuek', 'jutek', 'kasar', 'tidak sopan',
    'tidak profesional', 'tidak sigap', 'lemot',
    # Lainnya
    'tidak recommend', 'gak recommend', 'tidak konsisten', 'menurun',
    'tidak sesuai', 'mengecewakan', 'overrated'
]

KAMUS_SARAN = [
    # Usulan
    'sebaiknya', 'harusnya', 'seharusnya', 'baiknya', 'lebih baik',
    'saran', 'usul', 'masukan', 'kritik', 'suggest', 'suggestion',
    # Permintaan
    'tolong', 'mohon', 'minta', 'harap', 'dimohon', 'please',
    # Kebutuhan
    'perlu', 'butuh', 'harus', 'wajib', 'penting', 'dibutuhkan',
    # Perbaikan
    'ditambah', 'ditingkatkan', 'diperbaiki', 'dibenahi', 'dikembangkan',
    'diperluas', 'ditambahi', 'diupgrade', 'diperhatikan', 'dijaga',
    'dimaintain', 'maintenance', 'renovasi',
    # Lainnya
    'kalau', 'andai', 'seandainya', 'coba', 'cobalah', 'semoga',
    'harapan', 'ekspektasi', 'improvement', 'enhance',
    # Hasil stemming dari kata saran (untuk mencocokkan dengan token yang sudah di-stem)
    'baik', 'harus', 'saran', 'usul', 'kritik', 'tambah', 'tingkat',
    'perbaik', 'benah', 'kembang', 'luas', 'upgrade', 'perhatikan',
    'jaga', 'harap'
]

# ========================================
# INISIALISASI SASTRAWI
# ========================================

@st.cache_resource
def init_sastrawi():
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    
    factory_stopword = StopWordRemoverFactory()
    stopword_remover = factory_stopword.create_stop_word_remover()
    
    return stemmer, stopword_remover

stemmer, stopword_remover = init_sastrawi()

# ========================================
# PREPROCESSING
# ========================================

def case_folding(text):
    return text.lower()

def tokenizing(text):
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

def stopword_removal(tokens):
    text = ' '.join(tokens)
    cleaned = stopword_remover.remove(text)
    return cleaned.split()

def stemming(tokens):
    text = ' '.join(tokens)
    stemmed = stemmer.stem(text)
    return stemmed.split()

def preprocessing(text):
    text = case_folding(text)
    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return tokens

# ========================================
# PERHITUNGAN SKOR
# ========================================

def hitung_tf(tokens, kamus):
    count = sum(1 for token in tokens if token in kamus)
    if len(tokens) == 0:
        return 0.0
    return count / len(tokens)

def hitung_similarity_score(tokens):
    tf_positif = hitung_tf(tokens, KAMUS_POSITIF)
    tf_negatif = hitung_tf(tokens, KAMUS_NEGATIF)
    tf_saran = hitung_tf(tokens, KAMUS_SARAN)
    
    total_tf = tf_positif + tf_negatif + tf_saran
    
    if total_tf == 0:
        return {
            'positif': 0.0,
            'negatif': 0.0,
            'saran': 0.0
        }
    
    return {
        'positif': tf_positif / total_tf,
        'negatif': tf_negatif / total_tf,
        'saran': tf_saran / total_tf
    }

def tentukan_kategori_dominan(scores):
    if all(score == 0 for score in scores.values()):
        return 'TIDAK TERKLASIFIKASI'
    
    max_score = max(scores.values())
    for kategori, score in scores.items():
        if score == max_score:
            return kategori.upper()
    
    return 'TIDAK TERKLASIFIKASI'

# ========================================
# BACA DATA
# ========================================

@st.cache_data
def baca_komentar_dari_folder(folder_path='Data/Raw'):
    komentar_data = []
    
    if not os.path.exists(folder_path):
        return []
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not files:
        return []
    
    files.sort()
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                komentar = f.read().strip()
                
            if komentar:
                komentar_data.append({
                    'nama_file': filename,
                    'komentar': komentar
                })
        except Exception as e:
            st.warning(f"Gagal membaca file '{filename}': {e}")
    
    return komentar_data

# ========================================
# PROSES DATA
# ========================================

@st.cache_data
def proses_semua_komentar(komentar_data):
    hasil = []
    
    for idx, item in enumerate(komentar_data, 1):
        komentar = item['komentar']
        nama_file = item['nama_file']
        
        tokens = preprocessing(komentar)
        scores = hitung_similarity_score(tokens)
        kategori_dominan = tentukan_kategori_dominan(scores)
        
        hasil.append({
            'id': idx,
            'nama_file': nama_file,
            'komentar_asli': komentar,
            'tokens': tokens,
            'scores': scores,
            'kategori_dominan': kategori_dominan
        })
    
    return hasil

# ========================================
# SEARCH ENGINE
# ========================================

def hitung_relevance_score(tokens_query, tokens_doc, scores_doc):
    # Jaccard Similarity
    set_query = set(tokens_query)
    set_doc = set(tokens_doc)
    
    if len(set_query.union(set_doc)) == 0:
        jaccard = 0.0
    else:
        jaccard = len(set_query.intersection(set_doc)) / len(set_query.union(set_doc))
    
    # Sentiment alignment
    query_scores = hitung_similarity_score(tokens_query)
    
    dot_product = (query_scores['positif'] * scores_doc['positif'] +
                   query_scores['negatif'] * scores_doc['negatif'] +
                   query_scores['saran'] * scores_doc['saran'])
    
    mag_query = math.sqrt(query_scores['positif']**2 + 
                          query_scores['negatif']**2 + 
                          query_scores['saran']**2)
    mag_doc = math.sqrt(scores_doc['positif']**2 + 
                        scores_doc['negatif']**2 + 
                        scores_doc['saran']**2)
    
    if mag_query * mag_doc == 0:
        sentiment_sim = 0.0
    else:
        sentiment_sim = dot_product / (mag_query * mag_doc)
    
    relevance = 0.7 * jaccard + 0.3 * sentiment_sim
    return relevance

def search_engine(hasil_proses, query):
    tokens_query = preprocessing(query)
    
    if not tokens_query:
        return []
    
    hasil_search = []
    for item in hasil_proses:
        relevance = hitung_relevance_score(tokens_query, item['tokens'], item['scores'])
        
        if relevance > 0:
            hasil_search.append({
                **item,
                'relevance_score': relevance
            })
    
    hasil_search.sort(key=lambda x: x['relevance_score'], reverse=True)
    return hasil_search

# ========================================
# STREAMLIT APP
# ========================================

def main():
    st.set_page_config(
        page_title="Search Engine Sentimen Komentar",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Search Engine - Analisis Sentimen Komentar")
    st.subheader("Rumah Makan Bebek Pak Slamet")
    st.markdown("---")
    
    # Baca data
    folder_path = 'Data/Raw'
    komentar_data = baca_komentar_dari_folder(folder_path)
    
    if not komentar_data:
        st.error(f"❌ Tidak ada data di folder '{folder_path}'")
        st.info("📌 Pastikan struktur folder: Data/Raw/ dan letakkan file-file .txt di folder tersebut.")
        return
    
    # Proses data
    hasil_proses = proses_semua_komentar(komentar_data)
    
    # Sidebar - Ringkasan Data
    with st.sidebar:
        st.header("📊 Ringkasan Data")
        
        counter = Counter([item['kategori_dominan'] for item in hasil_proses])
        total = len(hasil_proses)
        
        st.metric("Total Komentar", total)
        
        st.markdown("### Distribusi Kategori")
        for kategori in ['POSITIF', 'NEGATIF', 'SARAN', 'TIDAK TERKLASIFIKASI']:
            count = counter.get(kategori, 0)
            persen = count/total*100 if total > 0 else 0
            
            # Warna badge
            if kategori == 'POSITIF':
                color = "🟢"
            elif kategori == 'NEGATIF':
                color = "🔴"
            elif kategori == 'SARAN':
                color = "🟡"
            else:
                color = "⚪"
            
            st.write(f"{color} **{kategori}**: {count} ({persen:.1f}%)")
    
    # Main Content - Search
    st.header("🔎 Pencarian Komentar")
    
    query = st.text_input(
        "Masukkan kata kunci pencarian:",
        placeholder="Contoh: enak, mahal, lambat, dll."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Cari", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Mencari komentar..."):
            hasil_search = search_engine(hasil_proses, query)
        
        st.markdown("---")
        
        if not hasil_search:
            st.warning(f"❌ Tidak ditemukan komentar yang relevan dengan '{query}'")
        else:
            st.success(f"✓ Ditemukan **{len(hasil_search)}** komentar relevan")
            
            # Batasi tampilan maksimal 10 hasil
            max_display = min(10, len(hasil_search))
            
            for idx, item in enumerate(hasil_search[:max_display], 1):
                with st.container():
                    # Header hasil
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"### [{idx}] {item['nama_file']}")
                    with col_b:
                        st.metric("Relevance", f"{item['relevance_score']:.3f}")
                    
                    # Kategori badge
                    kategori = item['kategori_dominan']
                    if kategori == 'POSITIF':
                        badge_color = "green"
                    elif kategori == 'NEGATIF':
                        badge_color = "red"
                    elif kategori == 'SARAN':
                        badge_color = "orange"
                    else:
                        badge_color = "gray"
                    
                    st.markdown(f"**Kategori:** :{badge_color}[{kategori}]")
                    
                    # Komentar
                    st.markdown(f"**Komentar:** {item['komentar_asli']}")
                    
                    # Skor detail
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positif", f"{item['scores']['positif']:.3f}")
                    with col2:
                        st.metric("Negatif", f"{item['scores']['negatif']:.3f}")
                    with col3:
                        st.metric("Saran", f"{item['scores']['saran']:.3f}")
                    
                    st.markdown("---")
            
            if len(hasil_search) > max_display:
                st.info(f"... dan {len(hasil_search) - max_display} hasil lainnya")

if __name__ == "__main__":
    main()