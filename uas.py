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
    'bagus', 'baik', 'berkualitas', 'premium', 'istimewa', 'spesial', 'hebat',
    'top', 'terbaik', 'unggul', 'recommended', 'rekomendasi', 'favorit',
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
    'harapan', 'ekspektasi', 'improvement', 'enhance'
]

# ========================================
# INISIALISASI SASTRAWI
# ========================================

print("\n⚙️  Menginisialisasi Sastrawi...")
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()
print("✓ Sastrawi siap\n")

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

def baca_komentar_dari_folder(folder_path='Data/Raw'):
    komentar_data = []
    
    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder '{folder_path}' tidak ditemukan!")
        return []
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not files:
        print(f"❌ ERROR: Tidak ada file .txt di folder '{folder_path}'")
        return []
    
    files.sort()
    print(f"📂 Ditemukan {len(files)} file komentar\n")
    
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
            print(f"⚠️  Warning: Gagal membaca file '{filename}': {e}")
    
    return komentar_data

# ========================================
# PROSES DATA
# ========================================

def proses_semua_komentar(komentar_data):
    hasil = []
    
    print("⚙️  Memproses komentar...", end='', flush=True)
    
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
    
    print(" ✓\n")
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
# TAMPILAN OUTPUT
# ========================================

def tampilkan_header():
    print("\n" + "=" * 70)
    print("SEARCH ENGINE - ANALISIS SENTIMEN KOMENTAR")
    print("Rumah Makan Bebek Pak Slamet")
    print("=" * 70 + "\n")

def tampilkan_ringkasan(hasil_proses):
    counter = Counter([item['kategori_dominan'] for item in hasil_proses])
    total = len(hasil_proses)
    
    print("=" * 70)
    print("RINGKASAN DATA")
    print("=" * 70)
    print(f"Total Komentar: {total}")
    print(f"\nDistribusi Kategori Dominan:")
    for kategori in ['POSITIF', 'NEGATIF', 'SARAN', 'TIDAK TERKLASIFIKASI']:
        count = counter.get(kategori, 0)
        persen = count/total*100 if total > 0 else 0
        print(f"  {kategori:20s}: {count:3d} ({persen:5.1f}%)")
    print("=" * 70 + "\n")

def tampilkan_hasil_search(hasil_search, query):
    print("\n" + "=" * 70)
    print(f"HASIL PENCARIAN: '{query}'")
    print("=" * 70)
    
    if not hasil_search:
        print(f"\n❌ Tidak ditemukan komentar yang relevan dengan '{query}'\n")
        return
    
    print(f"\n✓ Ditemukan {len(hasil_search)} komentar relevan:\n")
    
    # Batasi tampilan maksimal 10 hasil
    max_display = min(10, len(hasil_search))
    
    for idx, item in enumerate(hasil_search[:max_display], 1):
        print(f"[{idx}] Relevance: {item['relevance_score']:.3f} | {item['nama_file']}")
        print(f"    Komentar: {item['komentar_asli']}")
        print(f"    Kategori: {item['kategori_dominan']}")
        print(f"    Skor: P={item['scores']['positif']:.3f}, "
              f"N={item['scores']['negatif']:.3f}, "
              f"S={item['scores']['saran']:.3f}")
        print()
    
    if len(hasil_search) > max_display:
        print(f"... dan {len(hasil_search) - max_display} hasil lainnya")
    
    print("=" * 70)

# ========================================
# MENU INTERAKTIF
# ========================================

def tampilkan_menu():
    print("\n" + "=" * 70)
    print("🔍 MENU SEARCH ENGINE")
    print("=" * 70)
    print("1. Cari komentar")
    print("2. Lihat ringkasan data")
    print("3. Keluar")
    print("=" * 70)

def menu_interaktif(hasil_proses):
    """
    Menu interaktif untuk search engine di terminal
    """
    while True:
        tampilkan_menu()
        pilihan = input("\nPilih menu (1-3): ").strip()
        
        if pilihan == '1':
            # Mode pencarian
            print("\n" + "-" * 70)
            print("MODE PENCARIAN")
            print("Ketik 'back' untuk kembali ke menu")
            print("-" * 70)
            
            while True:
                query = input("\n🔎 Masukkan kata pencarian: ").strip()
                
                if query.lower() == 'back':
                    break
                
                if not query:
                    print("⚠️  Query tidak boleh kosong!")
                    continue
                
                hasil_search = search_engine(hasil_proses, query)
                tampilkan_hasil_search(hasil_search, query)
                
                print("\nIngin mencari lagi? (ketik kata kunci atau 'back' untuk menu)")
        
        elif pilihan == '2':
            # Tampilkan ringkasan
            tampilkan_ringkasan(hasil_proses)
            input("\nTekan Enter untuk kembali ke menu...")
        
        elif pilihan == '3':
            # Keluar
            print("\n" + "=" * 70)
            print("Terima kasih telah menggunakan Search Engine!")
            print("=" * 70 + "\n")
            break
        
        else:
            print("\n❌ Pilihan tidak valid! Silakan pilih 1-3")

# ========================================
# PROGRAM UTAMA
# ========================================

def main():
    tampilkan_header()
    
    # Baca data
    folder_path = 'Data/Raw'
    print(f"📂 Membaca data dari '{folder_path}'...")
    komentar_data = baca_komentar_dari_folder(folder_path)
    
    if not komentar_data:
        print("\n⚠️  Program dihentikan karena tidak ada data.")
        print("\n📌 Pastikan struktur folder: Data/Raw/")
        print("   Letakkan file-file .txt di folder tersebut.")
        return
    
    print(f"✓ Berhasil membaca {len(komentar_data)} komentar\n")
    
    # Proses data
    hasil_proses = proses_semua_komentar(komentar_data)
    
    # Tampilkan ringkasan awal
    tampilkan_ringkasan(hasil_proses)
    
    # Jalankan menu interaktif
    menu_interaktif(hasil_proses)

# ========================================
# JALANKAN PROGRAM
# ========================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program dihentikan oleh user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")