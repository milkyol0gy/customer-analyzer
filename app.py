import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import unicodedata
from datetime import datetime
from sqlalchemy import create_engine, text
import warnings
from werkzeug.security import generate_password_hash, check_password_hash

warnings.filterwarnings("ignore")

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

try:
    from prefixspan import PrefixSpan
    PREFIXSPAN_AVAILABLE = True
except ImportError:
    PREFIXSPAN_AVAILABLE = False


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Analisis Transaksi Shopee",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] { min-width: 240px; max-width: 270px; }
.block-container { padding-top: 1.4rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTS
# =============================================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Kolom kunci untuk mendeteksi duplikat saat menyimpan ke database
DEDUP_KEY_COLS = ["No. Pesanan", "Nama Produk"]

# Kolom yang dibaca sebagai string agar tidak di-parse otomatis oleh pandas
NUMERIC_COLS = [
    "Jumlah", "Total Harga Produk", "Total Diskon",
    "Diskon Dari Penjual", "Diskon Dari Shopee", "Voucher Ditanggung Penjual",
    "Voucher Ditanggung Shopee", "Total Pembayaran", "Harga Awal", "Harga Setelah Diskon",
]

DATETIME_COLS = [
    "Waktu Pesanan Dibuat", "Waktu Pesanan Selesai",
    "Pesanan Harus Dikirimkan Sebelum (Menghindari keterlambatan)",
    "Waktu Pengiriman Diatur", "Waktu Pembayaran Dilakukan",
]

# Kolom yang tidak dibutuhkan dalam analisis dan akan dihapus saat preprocessing
COLUMNS_TO_DROP = [
    "Alasan Pembatalan", "Status Pembatalan/ Pengembalian", "No. Resi",
    "Antar ke counter/ pick-up",
    "Pesanan Harus Dikirimkan Sebelum (Menghindari keterlambatan)",
    "Waktu Pengiriman Diatur", "SKU Induk", "Nomor Referensi SKU",
    "Berat Produk", "Total Berat", "Catatan dari Pembeli", "Catatan",
    "Nama Penerima", "No. Telepon", "Alamat Pengiriman", "Cashback Koin",
    "Paket Diskon (Diskon dari Shopee)", "Paket Diskon (Diskon dari Penjual)",
    "Paket Diskon", "Metode Pembayaran", "Returned quantity",
    "Diskon Kartu Kredit", "Ongkos Kirim Dibayar oleh Pembeli",
    "Estimasi Potongan Biaya Pengiriman", "Ongkos Kirim Pengembalian Barang",
    "Perkiraan Ongkos Kirim", "Product_Age_Months", "Jam_Mentah",
    "Waktu Pembayaran Dilakukan", "Potongan Koin Shopee", "Nama Variasi",
]

# Pemetaan nama produk yang tidak konsisten ke nama standar
PRODUCT_NAME_MAPPING = {
    "[PROMO] Quick Fresh Winter Melon Tea Lemon 160gr":
        "Quick Fresh Winter Melon Tea / Teh Buah Kundur Rasa Lemon 160GR",
    "[PROMO] Quick Fresh Winter Melon Tea Original 160gr":
        "Quick Fresh Winter Melon Tea / Teh Buah Kundur Original 160GR",
    "Finega Apple Vinegar / Cuka Apel Alami 250ml (Botol Plastik)":
        "Finega Apple Vinegar 250ml / Cuka Apel Alami",
    "NIMS Crispy Choco Cup 60GR / Cokelat Cereal": "NIMS Crispy Choco Cup 60GR",
    "NIMS Crispy Choco Tub 250GR / Cokelat Cereal": "NIMS Crispy Choco Tub 250GR",
    "Quick Fresh Black Sesame Bar 113GR / Snack Biji Wijen Hitam":
        "Quick Fresh Black Sesame Bar 113GR / Permen Enting-Enting",
    "Quick Fresh Black Sesame Bar 113GR / Snack Enting-Enting":
        "Quick Fresh Black Sesame Bar 113GR / Permen Enting-Enting",
    "Quick Fresh Honey Bottle Diamond 875gr (Botol)":
        "Madu Murni Asli Quick Fresh Honey 875 gram (Botol)",
    "Quick Fresh Peanut Bar 140GR / Snack  Enting Kacang":
        "Quick Fresh Peanut Bar 140GR / Permen Enting-Enting",
    "Quick Fresh Peanut Bar 140GR / Snack Enting-Enting":
        "Quick Fresh Peanut Bar 140GR / Permen Enting-Enting",
    "Quick Fresh Sesame Bar 126GR / Snack Biji Wijen":
        "Quick Fresh Sesame Bar 126GR / Permen Enting-Enting",
    "Quick Fresh Sesame Bar 126GR / Snack Enting-Enting":
        "Quick Fresh Sesame Bar 126GR / Permen Enting-Enting",
    "[Paket Bundling] Quick Fresh Bar / Permen Enting-Enting":
        "Paket Bundling Quick Fresh Bar / Permen Enting-Enting",
}

# Singkatan nama brand untuk mempersingkat nama produk
BRAND_REPLACEMENTS = {
    "Quick Fresh Honey": "QF Honey",
    "Quick Fresh": "QF",
    "NIMS": "NM",
    "Hundred Seeds": "HS",
    "Mae fu": "MF",
    "Shake Club House": "SH",
    "[FREE SHAKE CLUB HOUSE": "[FREE SH",
    "Finega": "FN",
    "Quickfresh": "QF",
}

# Threshold parameter default untuk setiap level analisis
DEFAULT_MAR_PARAMS = {
    "L1 – Jenis Produk": {"support": 0.0007, "confidence": 0.02, "lift": 1.0},
    "L2 – Tipe Produk":  {"support": 0.0006, "confidence": 0.03, "lift": 1.0},
    "L3 – Nama Produk":  {"support": 0.0004, "confidence": 0.09, "lift": 1.0},
}

DEFAULT_MDAR_SUPPORT    = 0.02
DEFAULT_MDAR_CONFIDENCE = 0.3
DEFAULT_MDAR_LIFT       = 1.2
DEFAULT_SPM_SUPPORT     = 0.017
BUNDLING_DISCOUNT_RATE  = 0.05

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei",  6: "Jun",
    7: "Jul", 8: "Agt", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des",
}

# Prefix yang digunakan untuk memberi label item per dimensi pada MDAR
MDAR_DIMENSION_PREFIXES = {
    "Nama Produk":      "PROD=",
    "Waktu (Jam)":      "TIME=",
    "Hari":             "DAY=",
    "Bulan":            "MONTH=",
    "Diskon":           "DISC=",
    "Voucher":          "VOUCHER=",
    "Kota/Kabupaten":   "CITY=",
    "Provinsi":         "PROV=",
}

ALL_MDAR_DIMENSIONS = {
    "Nama Produk":    ("item_product",  "Nama Produk"),
    "Waktu (Jam)":    ("item_time",     "Jam"),
    "Hari":           ("item_day",      "Hari"),
    "Bulan":          ("item_month",    "Bulan"),
    "Diskon":         ("item_discount", "_computed_"),
    "Voucher":        ("item_voucher",  "_computed_"),
    "Kota/Kabupaten": ("item_city",     "Kota/Kabupaten"),
    "Provinsi":       ("item_province", "Provinsi"),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_product_name(name):
    # Normalisasi nama produk: hapus spasi berlebih, konversi unicode, lowercase.
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"\s+", " ", name)
    return name.lower()


def parse_indonesian_number(series: pd.Series) -> pd.Series:
    # Konversi kolom numerik dari format Indonesia (titik sebagai pemisah ribuan,
    # koma sebagai desimal) ke integer Python.
    # Contoh: '1.500,50' → 1500, 'Rp 25.000' → 25000
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).round().astype(int)

    cleaned = series.astype(str).str.strip().str.replace(r"[Rp\s]", "", regex=True)

    def convert_single_value(value):
        if value in ("", "nan", "None", "-"):
            return 0
        # Jika ada koma, asumsi format: 1.500,50 → ubah ke 1500.50
        if "," in value:
            value = value.replace(".", "").replace(",", ".")
        else:
            # Jika format 1.500 (hanya titik sebagai ribuan tanpa desimal)
            if re.fullmatch(r"\d{1,3}(\.\d{3})+", value):
                value = value.replace(".", "")
        try:
            return int(round(float(value)))
        except (ValueError, TypeError):
            return 0

    return cleaned.apply(convert_single_value)


def categorize_hour_to_session(hour):
    # Konversi jam (0-23) ke sesi waktu: Morning, Afternoon, Evening, Night.
    if 5 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 16:
        return "Afternoon"
    elif 17 <= hour <= 20:
        return "Evening"
    else:
        return "Night"


def combine_product_and_variation(row):
    # Gabungkan nama produk dengan kata pertama variasi jika ada.
    product_name = str(row["Nama Produk"]).strip()
    variation    = row.get("Nama Variasi", None)
    if pd.isna(variation) or str(variation).lower() == "nan":
        return product_name
    return f"{product_name} {str(variation).split()[0]}"


def add_product_maturity_label(df, product_col, date_col, threshold_months=3):
    # Tambahkan kolom 'Product Maturity' berdasarkan rentang kemunculan produk.
    # Produk dengan usia < threshold_months dikategorikan sebagai 'New Product'.
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    product_lifecycle = (
        df.groupby(product_col)[date_col]
        .agg(first_seen="min", last_seen="max")
        .reset_index()
    )
    product_lifecycle["age_months"] = (
        (product_lifecycle["last_seen"] - product_lifecycle["first_seen"])
        / pd.Timedelta(days=30)
    )
    product_lifecycle["Product Maturity"] = product_lifecycle["age_months"].apply(
        lambda age: "New Product" if age < threshold_months else "Established Product"
    )

    return df.merge(
        product_lifecycle[[product_col, "age_months", "Product Maturity"]],
        on=product_col, how="left"
    )


def frozenset_to_readable_string(value):
    # Konversi frozenset hasil association rules ke string yang bisa dibaca.
    if isinstance(value, frozenset):
        return ", ".join(sorted(value))
    return str(value)


def filter_dataframe_by_period(df, date_col, year_range, month_range):
    # Filter dataframe berdasarkan rentang tahun dan bulan yang dipilih.
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    mask = (
        (df[date_col].dt.year  >= year_range[0])  &
        (df[date_col].dt.year  <= year_range[1])  &
        (df[date_col].dt.month >= month_range[0]) &
        (df[date_col].dt.month <= month_range[1])
    )
    return df[mask]


def render_period_filter_widgets(df, date_col="Waktu Pesanan Dibuat", key_prefix=""):
    # Tampilkan selectbox tahun dan bulan awal/akhir, kembalikan tuple range.
    if date_col not in df.columns:
        return (2023, 2025), (1, 12)

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    min_year = int(parsed_dates.dt.year.min()) if not parsed_dates.isna().all() else 2023
    max_year = int(parsed_dates.dt.year.max()) if not parsed_dates.isna().all() else 2025

    col1, col2, col3, col4 = st.columns(4)
    year_start  = col1.selectbox("Tahun Awal",  range(min_year, max_year + 1),
                                  index=0, key=f"{key_prefix}_year_start")
    year_end    = col2.selectbox("Tahun Akhir", range(min_year, max_year + 1),
                                  index=max_year - min_year, key=f"{key_prefix}_year_end")
    month_start = col3.selectbox("Bulan Awal",  range(1, 13), index=0,
                                  key=f"{key_prefix}_month_start",
                                  format_func=lambda x: f"{x}–{MONTH_NAMES[x]}")
    month_end   = col4.selectbox("Bulan Akhir", range(1, 13), index=11,
                                  key=f"{key_prefix}_month_end",
                                  format_func=lambda x: f"{x}–{MONTH_NAMES[x]}")
    return (year_start, year_end), (month_start, month_end)


def show_metric_explanation():
    # Tampilkan penjelasan support, confidence, dan lift dalam expander.
    with st.expander("Penjelasan Support, Confidence, dan Lift", expanded=False):
        st.markdown("""
**Support** = seberapa sering kombinasi produk muncul di seluruh transaksi.
Nilai 0.05 berarti kombinasi tersebut muncul di 5% dari semua transaksi.

**Confidence** = jika pelanggan membeli A, seberapa besar kemungkinan juga membeli B.
Nilai 0.5 berarti kemungkinan 50%.

**Lift** = kekuatan asosiasi. Lift > 1 berarti hubungan nyata (bukan kebetulan).
Nilai minimum default adalah 1.
        """)


def estimate_bundling_price(antecedent_str, consequent_str, df, discount=BUNDLING_DISCOUNT_RATE):
    
    # Estimasi harga bundling dari dua itemset dengan mengurangi diskon.
    # Mengambil harga terakhir (terbaru) dari setiap produk di dataset.
    if "Nama Produk" not in df.columns or "Harga Awal" not in df.columns:
        return "N/A"

    all_product_names = [p.strip() for p in (antecedent_str + ", " + consequent_str).split(",")]
    df_copy = df.copy()
    df_copy["Waktu Pesanan Dibuat"] = pd.to_datetime(df_copy["Waktu Pesanan Dibuat"], errors="coerce")

    total_price = 0
    found_count = 0
    for product_name in all_product_names:
        name_mask = df_copy["Nama Produk"].str.lower().str.contains(
            re.escape(product_name.lower()), na=False
        )
        product_rows = df_copy[name_mask & (df_copy["Harga Awal"] > 0)]
        if product_rows.empty:
            continue
        latest_price = product_rows.sort_values("Waktu Pesanan Dibuat", ascending=False)["Harga Awal"].iloc[0]
        total_price += latest_price
        found_count += 1

    if found_count == 0 or total_price == 0:
        return "N/A"

    bundling_price = total_price * (1 - discount)
    return f"Rp {bundling_price:,.0f}".replace(",", ".")


def categorize_promotion_window(median_days):
    # Kategorikan median jarak waktu antar pembelian (dalam hari)
    # ke dalam jendela promosi yang direkomendasikan.
    if median_days is None or pd.isna(median_days):
        return "Tidak Diketahui"
    if median_days <= 7:
        return "Sangat Cepat (≤7 Hari)"
    elif median_days <= 30:
        return "Cepat (8–30 Hari)"
    elif median_days <= 60:
        return "Sedang (31–60 Hari)"
    elif median_days <= 90:
        return "Lambat (61–90 Hari)"
    else:
        return "Sangat Lambat (>90 Hari)"


def parse_threshold_input(raw_input, default_value):
    # Parse input threshold dari text_input.
    # Kembalikan (value, is_using_default).
    # Kembalikan (None, False) jika input tidak valid.
    stripped = raw_input.strip() if raw_input else ""
    if not stripped:
        return default_value, True
    try:
        return float(stripped), False
    except ValueError:
        return None, False


def run_association_rules_for_level(df, item_col, min_support, min_confidence, min_lift=1.0):
    # Jalankan FP-Growth + association rules untuk satu level hierarki produk.
    # Mengembalikan (rules_df, error_message, covered_items_list).
    if item_col not in df.columns:
        return pd.DataFrame(), f"Kolom '{item_col}' tidak ditemukan.", []

    # Bangun transaksi: setiap pesanan → list item
    transactions = df.groupby("No. Pesanan")[item_col].apply(list).tolist()

    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    encoded_df    = pd.DataFrame(encoded_array, columns=encoder.columns_)

    frequent_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame(), "Tidak ada frequent itemset. Coba turunkan Min Support.", []

    covered_items = sorted({item for itemset in frequent_itemsets["itemsets"] for item in itemset})

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] > min_lift].sort_values("lift", ascending=False).reset_index(drop=True)

    for col in ["antecedents", "consequents"]:
        rules[col] = rules[col].apply(frozenset_to_readable_string)

    return rules, None, covered_items


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def create_db_engine():
    # Buat SQLAlchemy engine untuk koneksi ke PostgreSQL.
    connection_string = st.secrets["DATABASE_URL"]
    return create_engine(connection_string)


def save_dataframe_to_db(df, engine, table_name):
    # Simpan dataframe ke tabel PostgreSQL dengan deduplication otomatis.
    # Hanya baris yang belum ada di DB (berdasarkan No. Pesanan + Nama Produk) yang diinsert.
    # Mengembalikan dict ringkasan hasil operasi.

    result_info = {"duplicates_skipped": 0, "rows_inserted": 0, "error": None}

    try:
        df_to_insert = df.drop(columns=["id", "created_at"], errors="ignore").copy()

        # Cek apakah tabel sudah ada di database
        with engine.connect() as conn:
            table_exists = engine.dialect.has_table(conn, table_name)

        if table_exists:
            # Ambil hanya kolom kunci dari tabel yang ada untuk dibandingkan
            key_col_query = ", ".join(f'"{col}"' for col in DEDUP_KEY_COLS)
            existing_keys_df = pd.read_sql(
                f'SELECT {key_col_query} FROM "{table_name}"', engine
            )

            if len(existing_keys_df) > 0:
                # Buat set pasangan (No. Pesanan, Nama Produk) yang sudah ada di DB
                existing_key_set = set(
                    zip(*(existing_keys_df[col].astype(str) for col in DEDUP_KEY_COLS
                          if col in existing_keys_df.columns))
                )

                available_key_cols = [col for col in DEDUP_KEY_COLS if col in df_to_insert.columns]
                if available_key_cols:
                    # Filter: pertahankan hanya baris yang belum ada di DB
                    is_new_row = ~df_to_insert.apply(
                        lambda row: tuple(str(row[col]) for col in available_key_cols) in existing_key_set,
                        axis=1
                    )
                    result_info["duplicates_skipped"] = int((~is_new_row).sum())
                    df_to_insert = df_to_insert[is_new_row].copy()

        if len(df_to_insert) == 0:
            result_info["error"] = "no_new_data"
            return result_info

        # Tambahkan timestamp upload dan simpan ke DB
        df_to_insert["created_at"] = datetime.now()
        df_to_insert.to_sql(table_name, engine, if_exists="append", index=False)
        result_info["rows_inserted"] = len(df_to_insert)

        # Pastikan kolom id (primary key) ada di tabel
        with engine.begin() as conn:
            conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY'))

    except Exception as error:
        result_info["error"] = str(error)

    return result_info


def load_dataframe_from_db(engine, table_name="fact_sales"):
    # Load seluruh data dari tabel PostgreSQL. Kembalikan (df, error_message).
    try:
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
        return df, None
    except Exception as error:
        return None, str(error)

def authenticate_user(engine, username, password):

    query = text("""
        SELECT username, password
        FROM users
        WHERE username = :username
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(
            query,
            {"username": username}
        ).fetchone()

    if result is None:
        return False

    return check_password_hash(
        result.password,
        password
    )

# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

def run_preprocessing_pipeline(raw_df, engine=None):
    # Pipeline preprocessing lengkap untuk data transaksi Shopee.
    # Tahapan: filter status → konversi tipe data → standardisasi nama produk
    #          → join kategori produk → feature engineering → hapus kolom tidak perlu.
    df = raw_df.copy()

    # 1. Hapus transaksi yang dibatalkan atau belum selesai
    df = df[~df["Status Pesanan"].isin(["Batal", "Pesanan Diterima"])]

    # 2. Konversi kolom tanggal ke tipe datetime
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 3. Konversi kolom numerik dari format Indonesia ke integer
    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        df[col] = parse_indonesian_number(df[col])

    # 4. Standardisasi nama produk
    if "Nama Produk" in df.columns:
        df["Nama Produk"] = df["Nama Produk"].str.strip()
        # Ganti nama produk yang tidak konsisten ke nama standar
        df["Nama Produk"] = df["Nama Produk"].replace(
            {k.strip(): v for k, v in PRODUCT_NAME_MAPPING.items()}
        )
        # Gabungkan dengan variasi jika ada
        if "Nama Variasi" in df.columns:
            df["Nama Produk"] = df.apply(combine_product_and_variation, axis=1)
        # Singkat nama brand
        for brand_old, brand_new in sorted(BRAND_REPLACEMENTS.items(), key=lambda x: -len(x[0])):
            df["Nama Produk"] = df["Nama Produk"].str.replace(brand_old, brand_new, regex=False)

    # 5. Join dengan tabel master produk untuk mendapatkan Jenis dan Tipe Produk
    if engine is not None and "Nama Produk" in df.columns:
        try:
            master_product_df = pd.read_sql(
                "SELECT nama_produk, jenis_produk, tipe_produk FROM dim_master_produk", engine
            )
            # Normalisasi nama produk untuk pencocokan yang lebih toleran
            df["_name_normalized"]               = df["Nama Produk"].apply(normalize_product_name)
            master_product_df["_name_normalized"] = master_product_df["nama_produk"].apply(normalize_product_name)

            df = df.merge(
                master_product_df[["_name_normalized", "jenis_produk", "tipe_produk"]],
                on="_name_normalized", how="left"
            )
            df.drop(columns=["_name_normalized"], inplace=True)
            df.rename(columns={"jenis_produk": "Jenis Produk", "tipe_produk": "Tipe Produk"}, inplace=True)
        except Exception:
            df["Jenis Produk"] = "Belum Dikategorikan"
            df["Tipe Produk"]  = ""
    else:
        df["Jenis Produk"] = "Belum Dikategorikan"
        df["Tipe Produk"]  = ""

    df["Jenis Produk"] = df["Jenis Produk"].fillna("Belum Dikategorikan")
    df["Tipe Produk"]  = df["Tipe Produk"].fillna("")

    # 6. Tandai apakah produk termasuk bundling atau satuan
    if "Nama Produk" in df.columns:
        df["Jenis"] = np.where(
            df["Nama Produk"].str.contains("free|bundling", case=False, na=False),
            "Bundling", "Single"
        )

    # 7. Ekstrak fitur waktu dari kolom tanggal pesanan
    if "Waktu Pesanan Dibuat" in df.columns:
        df["Bulan"]   = df["Waktu Pesanan Dibuat"].dt.month
        df["Tahun"]   = df["Waktu Pesanan Dibuat"].dt.year
        df["Tanggal"] = pd.to_datetime(df["Waktu Pesanan Dibuat"].dt.date, errors="coerce")
        df["Hari"]    = df["Waktu Pesanan Dibuat"].dt.day_name()
        df["Jam"]     = df["Waktu Pesanan Dibuat"].dt.hour.apply(categorize_hour_to_session)

    # 8. Hitung kematangan produk berdasarkan lama produk muncul di data
    if "Nama Produk" in df.columns and "Waktu Pesanan Dibuat" in df.columns:
        df = add_product_maturity_label(df, "Nama Produk", "Waktu Pesanan Dibuat", threshold_months=3)

    # 9. Hapus kolom yang tidak diperlukan
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], errors="ignore").copy()

    return df


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

session_defaults = {
    "authenticated":           False,
    "df_raw":                  None,
    "df_clean":                None,
    "db_engine":               None,
    "db_connected":            False,
    "mar_results":             {},      # hasil Multilevel AR per level
    "mar_covered_items":       {},      # produk tercakup per level
    "mdar_rules_all":          None,    # semua rules MDAR
    "mdar_rules_with_product": None,    # rules MDAR yang melibatkan produk
    "mdar_covered_products":   [],
    "spm_result":              None,
}
for key, default_val in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# Auto-connect ke database saat pertama kali aplikasi dijalankan
if not st.session_state.db_connected:
    try:
        engine = create_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.session_state.db_engine    = engine
        st.session_state.db_connected = True
    except Exception:
        pass


# =============================================================================
# HALAMAN LOGIN
# =============================================================================

if not st.session_state.authenticated:
    st.markdown("""
    <div style='max-width:400px;margin:80px auto;padding:2rem;
                border:1px solid #e0e0e0;border-radius:12px;
                box-shadow:0 4px 16px rgba(0,0,0,0.08)'>
        <h2 style='text-align:center;margin-bottom:1.5rem'>🛒 Shopee Analyzer</h2>
    </div>
    """, unsafe_allow_html=True)
    st.write(type(st.session_state.db_engine))
    st.write(st.session_state.db_engine)

    _, col_center, _ = st.columns([1, 1.5, 1])
    with col_center:
        st.markdown("### Sign In")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Masuk", type="primary", use_container_width=True):
            if authenticate_user(st.session_state.db_engine, username,password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Username atau password salah.")
    st.stop()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🛒 Shopee Analyzer")
    st.caption("Login sebagai **admin**")
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.markdown("---")
    selected_menu = st.radio("Pilih Menu", options=[
        "Upload & Preprocessing",
        "Multilevel Association Rules",
        "Multidimensional Association Rules",
        "Sequential Pattern Mining",
    ])

    st.markdown("---")
    st.markdown("**Konfigurasi Database**")

    if st.session_state.db_connected:
        st.success("● DB Terhubung")
    else:
        st.warning("● DB Tidak Terhubung")

    st.markdown("---")
    if st.session_state.df_clean is not None:
        st.info(f"Data tersedia\n{len(st.session_state.df_clean):,} baris")
    else:
        st.caption("Belum ada data")


# =============================================================================
# HALAMAN 1 — UPLOAD & PREPROCESSING
# =============================================================================

if selected_menu == "Upload & Preprocessing":
    st.title("Upload & Preprocessing Data")

    tab_upload, tab_load_db = st.tabs(["Upload File", "Load dari Database"])

    # ── Tab 1: Upload File ───────────────────────────────────────────────────
    with tab_upload:
        st.subheader("1. Upload File Transaksi")
        st.write("Upload file CSV atau XLSX hasil ekspor transaksi Shopee.")

        uploaded_file = st.file_uploader("Pilih file CSV atau XLSX", type=["csv", "xlsx"])

        if uploaded_file:
            with st.spinner("Membaca file..."):
                try:
                    filename  = uploaded_file.name
                    force_str = {col: str for col in NUMERIC_COLS}

                    if filename.endswith(".csv"):
                        # Coba separator titik koma dulu (format ekspor Shopee Indonesia)
                        try:
                            raw_df = pd.read_csv(uploaded_file, sep=";", low_memory=False, dtype=force_str)
                            if raw_df.shape[1] <= 1:
                                uploaded_file.seek(0)
                                raw_df = pd.read_csv(uploaded_file, sep=",", low_memory=False, dtype=force_str)
                        except Exception:
                            uploaded_file.seek(0)
                            raw_df = pd.read_csv(uploaded_file, low_memory=False, dtype=force_str)
                    else:
                        raw_df = pd.read_excel(uploaded_file, dtype=force_str)

                    st.session_state.df_raw = raw_df
                    st.success(f"'{filename}' berhasil dimuat — {len(raw_df):,} baris, {len(raw_df.columns)} kolom")
                except Exception as error:
                    st.error(f"Gagal membaca file: {error}")

        if st.session_state.df_raw is not None:
            raw_df = st.session_state.df_raw
            col1, col2 = st.columns(2)
            col1.metric("Total Baris",  f"{len(raw_df):,}")
            col2.metric("Total Kolom",  len(raw_df.columns))

            st.divider()
            st.subheader("2. Preprocessing")
            if st.button("Jalankan Preprocessing", type="primary"):
                with st.spinner("Memproses data..."):
                    clean_df = run_preprocessing_pipeline(
                        st.session_state.df_raw,
                        engine=st.session_state.db_engine,
                    )
                    st.session_state.df_clean = clean_df
                st.success("Preprocessing selesai!")

            if st.session_state.df_clean is not None:
                clean_df = st.session_state.df_clean
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Baris Bersih",  f"{len(clean_df):,}")
                col2.metric("Baris Dihapus", f"{len(raw_df) - len(clean_df):,}")
                col3.metric("Kolom Tersisa", len(clean_df.columns))
                col4.metric(
                    "Jenis Produk Terisi",
                    f"{(clean_df.get('Jenis Produk', '') != 'Belum Dikategorikan').sum():,}"
                )

                with st.expander("Preview Data Bersih (50 baris pertama)"):
                    st.dataframe(clean_df.head(50), use_container_width=True)

                # Download hasil preprocessing sebagai XLSX
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    clean_df.to_excel(writer, index=False, sheet_name="Data Bersih")
                st.download_button(
                    "⬇ Download Hasil Preprocessing (XLSX)",
                    excel_buffer.getvalue(),
                    "data_bersih.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

                st.divider()
                st.subheader("3. Simpan ke Database")

                if not st.session_state.db_connected:
                    st.warning("Database belum terhubung. Konfigurasikan koneksi di sidebar.")
                else:
                    target_table = st.text_input(
                        "Nama Tabel",
                        value="fact_sales",
                        help="Nama tabel tujuan di PostgreSQL. Duplikat akan otomatis dilewati."
                    )

                    if st.button("Simpan ke Database", type="primary"):
                        with st.spinner("Menyimpan ke database..."):
                            save_result = save_dataframe_to_db(
                                clean_df,
                                st.session_state.db_engine,
                                target_table
                            )

                        if save_result["error"] == "no_new_data":
                            st.info(
                                f"Semua data sudah ada di database — "
                                f"{save_result['duplicates_skipped']:,} baris duplikat dilewati."
                            )
                        elif save_result["error"]:
                            st.error(f"Gagal menyimpan: {save_result['error']}")
                        else:
                            skipped = save_result["duplicates_skipped"]
                            inserted = save_result["rows_inserted"]
                            if skipped > 0:
                                st.info(f"{skipped:,} baris duplikat dilewati.")
                            st.success(f"{inserted:,} baris baru berhasil disimpan ke tabel '{target_table}'.")

    # ── Tab 2: Load dari Database ────────────────────────────────────────────
    with tab_load_db:
        st.subheader("Load Data dari Database")
        st.write("Muat data yang sudah tersimpan di PostgreSQL untuk digunakan dalam analisis.")

        if not st.session_state.db_connected:
            st.warning("Database belum terhubung. Konfigurasikan koneksi di sidebar.")
        else:
            load_table_name = st.text_input("Nama Tabel", value="fact_sales", key="load_table_name")

            if st.button("Load Data dari Database", type="primary"):
                with st.spinner("Memuat data dari database..."):
                    loaded_df, load_error = load_dataframe_from_db(
                        st.session_state.db_engine, load_table_name
                    )

                if load_error:
                    st.error(f"Gagal memuat data: {load_error}")
                else:
                    st.session_state.df_clean = loaded_df
                    st.success(f"{len(loaded_df):,} baris berhasil dimuat dari tabel '{load_table_name}'.")
                    col1, col2 = st.columns(2)
                    col1.metric("Total Baris",  f"{len(loaded_df):,}")
                    col2.metric("Total Kolom",  len(loaded_df.columns))

                    with st.expander("Preview Data (50 baris pertama)"):
                        st.dataframe(loaded_df.head(50), use_container_width=True)


# =============================================================================
# HALAMAN 2 — MULTILEVEL ASSOCIATION RULES
# =============================================================================

elif selected_menu == "Multilevel Association Rules":
    st.title("Multilevel Association Rules")
    st.write("Analisis asosiasi pada tiga level: **L1 – Jenis Produk**, **L2 – Tipe Produk**, **L3 – Nama Produk**.")

    if not MLXTEND_AVAILABLE:
        st.error("Library `mlxtend` belum terinstall. Jalankan: `pip install mlxtend`")
        st.stop()
    if st.session_state.df_clean is None:
        st.warning("Data belum tersedia. Upload file atau load dari database terlebih dahulu.")
        st.stop()

    show_metric_explanation()
    full_df = st.session_state.df_clean.copy()

    st.subheader("Rentang Waktu Data")
    year_range, month_range = render_period_filter_widgets(full_df, key_prefix="mar")
    st.divider()

    # Konfigurasi tiga level analisis
    level_configs = {
        "L1 – Jenis Produk": {"col": "Jenis Produk", "key": "l1", "state_key": "mar_l1"},
        "L2 – Tipe Produk":  {"col": "Tipe Produk",  "key": "l2", "state_key": "mar_l2"},
        "L3 – Nama Produk":  {"col": "Nama Produk",  "key": "l3", "state_key": "mar_l3"},
    }
    for config in level_configs.values():
        if config["state_key"] not in st.session_state:
            st.session_state[config["state_key"]] = None
        covered_key = f"mar_covered_{config['key']}"
        if covered_key not in st.session_state:
            st.session_state[covered_key] = []

    RESULT_COLS = ["antecedents", "consequents", "support", "confidence", "lift", "Rekomendasi Harga Bundling"]

    tab_l1, tab_l2, tab_l3, tab_rules_load_db = st.tabs(["L1 – Jenis Produk","L2 – Tipe Produk","L3 – Nama Produk", "Load Data dari DB"])

    for tab_widget, (level_label, config) in zip([tab_l1, tab_l2, tab_l3], level_configs.items()):
        with tab_widget:
            default_params = DEFAULT_MAR_PARAMS[level_label]
            covered_key    = f"mar_covered_{config['key']}"

            st.markdown(f"### {level_label}")
            st.caption(
                f"Default — support: **{default_params['support']}**, "
                f"confidence: **{default_params['confidence']}**, "
                f"lift: **{default_params['lift']}**"
            )

            col_sup, col_conf, col_lift = st.columns(3)
            raw_support    = col_sup.text_input("Min Support",    placeholder=f"default: {default_params['support']}",    key=f"mar_sup_{config['key']}")
            raw_confidence = col_conf.text_input("Min Confidence", placeholder=f"default: {default_params['confidence']}", key=f"mar_conf_{config['key']}")
            raw_lift       = col_lift.text_input("Min Lift",       placeholder=f"default: {default_params['lift']}",       key=f"mar_lift_{config['key']}")

            min_support,    using_default_sup  = parse_threshold_input(raw_support,    default_params["support"])
            min_confidence, using_default_conf = parse_threshold_input(raw_confidence, default_params["confidence"])
            min_lift,       using_default_lift = parse_threshold_input(raw_lift,       default_params["lift"])

            if None in (min_support, min_confidence, min_lift):
                st.error("Masukkan angka desimal yang valid untuk semua threshold.")
            else:
                # Tampilkan info jika menggunakan nilai default
                default_labels = []
                if using_default_sup:  default_labels.append(f"support={default_params['support']} (default)")
                if using_default_conf: default_labels.append(f"confidence={default_params['confidence']} (default)")
                if using_default_lift: default_labels.append(f"lift={default_params['lift']} (default)")
                if default_labels:
                    st.info("Menggunakan: " + ", ".join(default_labels))

                if st.button(f"Jalankan {level_label}", type="primary", key=f"run_mar_{config['key']}"):
                    filtered_df = filter_dataframe_by_period(
                        full_df, "Waktu Pesanan Dibuat", year_range, month_range
                    )
                    if len(filtered_df) == 0:
                        st.error("Tidak ada data pada rentang waktu yang dipilih.")
                    else:
                        with st.spinner("Menjalankan FP-Growth..."):
                            rules_df, error_msg, covered_items = run_association_rules_for_level(
                                filtered_df, config["col"], min_support, min_confidence, min_lift
                            )

                        if error_msg:
                            st.warning(error_msg)
                            st.session_state[config["state_key"]] = pd.DataFrame()
                            st.session_state[covered_key] = []
                        else:
                            # Hitung rekomendasi harga bundling untuk level nama produk
                            if config["col"] == "Nama Produk":
                                with st.spinner("Menghitung estimasi harga bundling..."):
                                    rules_df["Rekomendasi Harga Bundling"] = rules_df.apply(
                                        lambda row: estimate_bundling_price(
                                            row["antecedents"], row["consequents"], filtered_df
                                        ), axis=1
                                    )
                            else:
                                rules_df["Rekomendasi Harga Bundling"] = "N/A"

                            st.session_state[config["state_key"]] = rules_df
                            st.session_state[covered_key]         = covered_items
                            st.success(f"Ditemukan {len(rules_df)} rules dari {len(covered_items)} item.")

            # Tampilkan hasil jika sudah ada
            saved_rules    = st.session_state.get(config["state_key"])
            covered_items  = st.session_state.get(covered_key, [])

            if saved_rules is not None:
                st.markdown("---")
                if saved_rules.empty:
                    st.info("Tidak ada rules yang memenuhi threshold.")
                else:
                    # Filter produk yang ditampilkan
                    if covered_items:
                        selected_items = st.multiselect(
                            f"Filter {config['col']} ({len(covered_items)} item):",
                            options=covered_items,
                            default=covered_items,
                            key=f"filter_mar_{config['key']}"
                        )
                        if selected_items:
                            show_mask = (
                                saved_rules["antecedents"].apply(lambda x: any(p in x for p in selected_items)) |
                                saved_rules["consequents"].apply(lambda x: any(p in x for p in selected_items))
                            )
                            display_rules = saved_rules[show_mask]
                        else:
                            display_rules = saved_rules
                    else:
                        display_rules = saved_rules

                    st.write(f"**{len(display_rules)} rules** (lift ≥ {min_lift})")
                    visible_cols = [c for c in RESULT_COLS if c in display_rules.columns]
                    st.dataframe(display_rules[visible_cols], use_container_width=True, hide_index=True)

                    # Download
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        display_rules[visible_cols].to_excel(writer, index=False)
                    st.download_button(
                        f"⬇ Download {level_label}",
                        excel_buf.getvalue(),
                        f"mar_rules_{config['key']}.xlsx",
                        key=f"download_mar_{config['key']}"
                    )

    with tab_rules_load_db:
        st.subheader("Load Data dari Database")
        st.write("Muat data hasil analisis yang tersimpan di DB.")
        level_configs = {
            "L1 – Jenis Produk": {"db_name": "mar_rules_lvl1", "support": 0.0007, "confidence": 0.02, "lift": 1.0},
            "L2 – Tipe Produk":  {"db_name": "mar_rules_lvl2",  "support": 0.0006, "confidence": 0.03, "lift": 1.0},
            "L3 – Nama Produk":  {"db_name": "mar_rules_lvl3",  "support": 0.0004, "confidence": 0.09, "lift": 1.0},
        }

        if not st.session_state.db_connected:
            st.warning("Database belum terhubung. Konfigurasikan koneksi di sidebar.")
        else:

            if st.button("Load Data dari Database", type="primary"):
                for level_name, config in level_configs.items():
                    st.markdown(f"### {level_name}")
                    st.caption(
                        f"Support: **{config['support']}**, "
                        f"Confidence: **{config['confidence']}**, "
                        f"Lift: **{config['lift']}**"
                    )
                    with st.spinner("Memuat data dari database..."):
                        loaded_df, load_error = load_dataframe_from_db(
                            st.session_state.db_engine, config['db_name']
                        )

                    if load_error:
                        st.error(f"Gagal memuat data: {load_error}")
                    else:
                        st.session_state.rules_mar_db = loaded_df
                        st.success(f"{len(loaded_df):,} baris berhasil dimuat dari tabel {config['db_name']}.")
                        not_show_cols = ['id','created_at']
                        st.dataframe(loaded_df.drop(columns=not_show_cols,errors='ignore'), use_container_width=True)

# =============================================================================
# HALAMAN 3 — MULTIDIMENSIONAL ASSOCIATION RULES
# =============================================================================

elif selected_menu == "Multidimensional Association Rules":
    st.title("Multidimensional Association Rules (MDAR)")
    st.write("Gabungkan dimensi produk, waktu, diskon, dan dimensi lainnya dalam satu analisis asosiasi.")

    if not MLXTEND_AVAILABLE:
        st.error("Library `mlxtend` belum terinstall.")
        st.stop()
    if st.session_state.df_clean is None:
        st.warning("Data belum tersedia. Upload file atau load dari database terlebih dahulu.")
        st.stop()

    show_metric_explanation()
    tab_process_rules, tab_load_rules = st.tabs(["Membentuk Rules Baru", "Load Rules dari DB"])


    with tab_process_rules:
        full_df = st.session_state.df_clean.copy()

        with st.expander("Atur Dimensi, Parameter & Rentang Waktu", expanded=True):
            st.markdown("**Rentang Waktu**")
            year_range, month_range = render_period_filter_widgets(full_df, key_prefix="mdar")

            # Filter berdasarkan jenis produk sebelum analisis
            st.markdown("**Filter Jenis Produk**")
            if "Jenis Produk" in full_df.columns:
                all_product_types = sorted(full_df["Jenis Produk"].dropna().unique().tolist())
                use_all_types = st.checkbox("Gunakan semua jenis produk", value=True, key="mdar_use_all_types")
                if not use_all_types:
                    selected_product_types = st.multiselect(
                        "Pilih jenis produk yang akan dianalisis:",
                        options=all_product_types,
                        default=all_product_types[:1] if all_product_types else [],
                        key="mdar_product_type_filter"
                    )
                    if not selected_product_types:
                        st.warning("Pilih minimal satu jenis produk.")
                else:
                    selected_product_types = all_product_types
            else:
                st.info("Kolom 'Jenis Produk' tidak ditemukan — semua produk digunakan.")
                selected_product_types = None

            st.markdown("**Pilih Dimensi**")
            use_all_dimensions = st.checkbox("Gunakan semua dimensi", value=True, key="mdar_use_all_dims")
            if not use_all_dimensions:
                selected_dimension_labels = st.multiselect(
                    "Dimensi:",
                    options=["Nama Produk","Bulan","Waktu (Jam)","Hari","Diskon","Voucher","Kota/Kabupaten","Provinsi"],
                    default=["Nama Produk","Diskon", "Bulan"]
                )
            else:
                selected_dimension_labels = list(ALL_MDAR_DIMENSIONS.keys())

            st.markdown("**Threshold**")
            col_sup, col_conf, col_lift = st.columns(3)
            raw_support    = col_sup.text_input("Min Support",    placeholder=f"disarankan: {DEFAULT_MDAR_SUPPORT}",    key="mdar_sup")
            raw_confidence = col_conf.text_input("Min Confidence", placeholder=f"disarankan: {DEFAULT_MDAR_CONFIDENCE}", key="mdar_conf")
            raw_lift       = col_lift.text_input("Min Lift",       placeholder=f"default: {DEFAULT_MDAR_LIFT}",         key="mdar_lift")

        min_support,    _ = parse_threshold_input(raw_support,    DEFAULT_MDAR_SUPPORT)
        min_confidence, _ = parse_threshold_input(raw_confidence, DEFAULT_MDAR_CONFIDENCE)
        min_lift,       _ = parse_threshold_input(raw_lift,       DEFAULT_MDAR_LIFT)

        if None in (min_support, min_confidence, min_lift):
            st.error("Masukkan angka desimal yang valid.")
            st.stop()
        if len(selected_dimension_labels) < 2:
            st.warning("Pilih minimal 2 dimensi.")
            st.stop()

        if st.button("Jalankan MDAR", type="primary"):
            filtered_df = filter_dataframe_by_period(full_df, "Waktu Pesanan Dibuat", year_range, month_range)
            if len(filtered_df) == 0:
                st.error("Tidak ada data pada rentang waktu yang dipilih.")
                st.stop()

            with st.spinner("Membangun dimensi item..."):
                working_df = filtered_df.copy()

                # Hitung kategori diskon — quantile dihitung dari seluruh data periode (sebelum filter jenis)
                # agar threshold konsisten dan tidak berubah tergantung filter produk
                if "Diskon" in selected_dimension_labels:
                    if "Harga Awal" in working_df.columns and "Harga Setelah Diskon" in working_df.columns:
                        working_df["_discount_pct"] = working_df.apply(
                            lambda row: (row["Harga Awal"] - row["Harga Setelah Diskon"]) / row["Harga Awal"] * 100
                            if row["Harga Awal"] > 0 else 0, axis=1
                        )
                        # Ambil distribusi diskon dari baris yang memang mendapat diskon
                        nonzero_discounts = working_df[working_df["_discount_pct"] > 0]["_discount_pct"]
                        quantile_values   = nonzero_discounts.quantile([0.33, 0.66]).values if len(nonzero_discounts) > 0 else [10, 20]
                        q33 = round(quantile_values[0], 2)
                        q66 = round(quantile_values[1], 2)
                        st.caption(f"ℹ Threshold diskon: Low ≤ {q33}%, Medium {q33}–{q66}%, High > {q66}%")

                        def categorize_discount(pct):
                            if pct == 0:        return "No_Discount"
                            elif pct <= q33:    return f"Low_Discount (0-{q33}%)"
                            elif pct <= q66:    return f"Medium_Discount ({q33}-{q66}%)"
                            else:               return f"High_Discount (>{q66}%)"

                        working_df["item_discount"] = "DISC=" + working_df["_discount_pct"].apply(categorize_discount)
                    else:
                        st.warning("Kolom harga tidak ditemukan — dimensi Diskon dilewati.")
                        selected_dimension_labels = [d for d in selected_dimension_labels if d != "Diskon"]

                # Buat label voucher: kombinasi, penjual, shopee, atau tidak ada voucher
                if "Voucher" in selected_dimension_labels:
                    if "Voucher Ditanggung Penjual" in working_df.columns:
                        def categorize_voucher(row):
                            seller_voucher = row.get("Voucher Ditanggung Penjual", 0)
                            shopee_voucher = row.get("Voucher Ditanggung Shopee",  0)
                            if seller_voucher > 0 and shopee_voucher > 0: return "voucher_kombinasi"
                            elif seller_voucher > 0: return "voucher_penjual"
                            elif shopee_voucher > 0: return "voucher_shopee"
                            else:                    return "no_voucher"

                        working_df["item_voucher"] = "VOUCHER=" + working_df.apply(categorize_voucher, axis=1)
                    else:
                        st.warning("Kolom voucher tidak ditemukan — dimensi Voucher dilewati.")
                        selected_dimension_labels = [d for d in selected_dimension_labels if d != "Voucher"]

            # Terapkan filter jenis produk SETELAH encoding dimensi
            # agar quantile diskon tidak berubah saat filter berbeda
            if selected_product_types is not None and "Jenis Produk" in working_df.columns:
                before_filter_count = len(working_df)
                working_df = working_df[working_df["Jenis Produk"].isin(selected_product_types)]
                st.info(
                    f"Filter jenis produk ({', '.join(selected_product_types)}): "
                    f"{before_filter_count:,} → {len(working_df):,} baris."
                )

            if len(working_df) == 0:
                st.error("Tidak ada data setelah filter jenis produk.")
                st.stop()

            with st.spinner("Membangun item per dimensi..."):
                # Peta dimensi ke kolom item dan prefix label
                simple_dimension_map = {
                    "Nama Produk":      ("item_product",  "Nama Produk",      "PROD="),
                    "Waktu (Jam)":      ("item_time",     "Jam",              "TIME="),
                    "Hari":             ("item_day",      "Hari",             "DAY="),
                    "Bulan":            ("item_month",    "Bulan",            "MONTH="),
                    "Kota/Kabupaten":   ("item_city",     "Kota/Kabupaten",   "CITY="),
                    "Provinsi":         ("item_province", "Provinsi",         "PROV="),
                }
                skipped_dimensions = []
                for dim_label, (item_col, source_col, prefix) in simple_dimension_map.items():
                    if dim_label in selected_dimension_labels:
                        if source_col in working_df.columns:
                            working_df[item_col] = prefix + working_df[source_col].astype(str)
                        else:
                            st.warning(f"Kolom '{source_col}' tidak ada — dimensi {dim_label} dilewati.")
                            skipped_dimensions.append(dim_label)

                selected_dimension_labels = [d for d in selected_dimension_labels if d not in skipped_dimensions]

                # Kumpulkan kolom item yang valid
                valid_item_cols = [
                    ALL_MDAR_DIMENSIONS[label][0]
                    for label in selected_dimension_labels
                    if ALL_MDAR_DIMENSIONS[label][0] in working_df.columns
                ]
                product_item_cols = [c for c in valid_item_cols if c == "item_product"]
                context_item_cols = [c for c in valid_item_cols if c != "item_product"]

                # Agregasi per pesanan: item_product menjadi list, dimensi konteks menjadi nilai tunggal
                agg_rules = {}
                if product_item_cols:
                    agg_rules["item_product"] = list
                for col in context_item_cols:
                    agg_rules[col] = "first"

                grouped_by_order = working_df.groupby("No. Pesanan").agg(agg_rules).reset_index()

                def build_transaction_items(row):
                    """Gabungkan semua item produk dan item konteks menjadi satu list transaksi."""
                    items = list(row["item_product"]) if "item_product" in grouped_by_order.columns else []
                    for col in context_item_cols:
                        items.append(str(row[col]))
                    return items

                grouped_by_order["transaction_items"] = grouped_by_order.apply(build_transaction_items, axis=1)

            with st.spinner("Menjalankan FP-Growth..."):
                encoder        = TransactionEncoder()
                encoded_array  = encoder.fit(grouped_by_order["transaction_items"]).transform(grouped_by_order["transaction_items"])
                encoded_df     = pd.DataFrame(encoded_array, columns=encoder.columns_)
                frequent_items = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)

            if frequent_items.empty:
                st.warning("Tidak ada frequent itemset. Coba turunkan min_support.")
                st.stop()

            with st.spinner("Membuat association rules..."):
                rules_df = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)
                rules_df = rules_df[rules_df["lift"] >= min_lift].sort_values("lift", ascending=False).reset_index(drop=True)

                def get_context_dimensions(row):
                    """Dimensi non-produk yang terlibat dalam rule ini."""
                    all_items = list(row["antecedents"]) + list(row["consequents"])
                    context_dims = set()
                    for item in all_items:
                        for dim_name, prefix in MDAR_DIMENSION_PREFIXES.items():
                            if dim_name != "Nama Produk" and str(item).startswith(prefix):
                                context_dims.add(dim_name)
                                break
                    return ", ".join(sorted(context_dims)) if context_dims else "—"

                rules_df["Dimensi Konteks"] = rules_df.apply(get_context_dimensions, axis=1)

                # Ambil hanya rules yang minimal salah satu sisinya melibatkan produk
                rules_with_product = rules_df[
                    rules_df["antecedents"].apply(lambda x: any("PROD=" in item for item in x)) |
                    rules_df["consequents"].apply(lambda x: any("PROD=" in item for item in x))
                ].copy()

                covered_products = sorted({
                    item.replace("PROD=", "")
                    for _, row in rules_with_product.iterrows()
                    for item in list(row["antecedents"]) + list(row["consequents"])
                    if "PROD=" in item
                })

                # Konversi frozenset ke string untuk ditampilkan di tabel
                for col in ["antecedents", "consequents"]:
                    rules_df[col]            = rules_df[col].apply(frozenset_to_readable_string)
                    rules_with_product[col]  = rules_with_product[col].apply(frozenset_to_readable_string)

            st.session_state.mdar_rules_all          = rules_df
            st.session_state.mdar_rules_with_product = rules_with_product
            st.session_state.mdar_covered_products   = covered_products

            st.write(rules_df.head())

            st.success(
                f"Selesai! {len(rules_with_product)} rules melibatkan produk dari {len(covered_products)} produk "
            )

        # Tampilkan hasil MDAR jika sudah ada
        if st.session_state.mdar_rules_with_product is not None:
            rules_with_product = st.session_state.mdar_rules_with_product
            covered_products   = st.session_state.mdar_covered_products
            tab_rules, tab_context = st.tabs(["Rules", "Informasi Dimensi Kontekstual"])
            context_configs = {
                "Chi-Square Test": {"db_name": "chi2_testing_mdar", "caption": """
                    Tabel ini menampilkan hasil uji Chi-Square untuk mengukur signifikansi hubungan antara dimensi kontekstual (waktu, voucher, diskon, dan atribut transaksi lainnya) dengan dimensi produk. Nilai p-value digunakan untuk menentukan apakah hubungan yang ditemukan terjadi secara kebetulan atau memiliki keterkaitan yang signifikan secara statistik. Dimensi dengan p-value < 0,05 dianggap memiliki hubungan yang signifikan terhadap perilaku pembelian konsumen.
                """},
                "Lift Comparison":  {"db_name": "lift_comparison_testing_mdar",  "caption": """
                    Tabel ini menampilkan perbandingan nilai lift antara aturan asosiasi tanpa konteks dan aturan asosiasi dengan penambahan dimensi kontekstual (Multidimensional Association Rule). Pengujian ini digunakan untuk mengukur apakah penambahan dimensi seperti waktu, diskon, atau voucher mampu memperkuat hubungan antar item. Semakin besar peningkatan lift yang diperoleh, semakin besar kontribusi konteks dalam menjelaskan pola pembelian konsumen.
                """},
            }

            MDAR_DISPLAY_COLS = ["antecedents", "consequents", "support", "confidence", "lift"]
            with tab_rules:
                st.subheader("Rules MDAR")

                if rules_with_product.empty:
                    st.info("Tidak ada rules yang melibatkan produk. Pastikan dimensi 'Nama Produk' dipilih.")
                else:
                    if covered_products:
                        selected_products = st.multiselect(
                            f"Filter produk ({len(covered_products)} produk tercakup):",
                            options=covered_products,
                            default=covered_products,
                            key="mdar_product_filter"
                        )
                        if selected_products:
                            product_filter_mask = (
                                rules_with_product["antecedents"].apply(lambda x: any(p in x for p in selected_products)) |
                                rules_with_product["consequents"].apply(lambda x: any(p in x for p in selected_products))
                            )
                            display_rules = rules_with_product[product_filter_mask]
                        else:
                            display_rules = rules_with_product
                    else:
                        display_rules = rules_with_product

                    st.write(f"**{len(display_rules)} rules**")
                    visible_cols = [c for c in MDAR_DISPLAY_COLS if c in display_rules.columns]
                    st.dataframe(display_rules[visible_cols], use_container_width=True, hide_index=True)

                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        display_rules[visible_cols].to_excel(writer, index=False)
                    st.download_button("⬇ Download Rules (XLSX)", excel_buf.getvalue(), "mdar_rules.xlsx")
            
            with tab_context:
                for context_name, config in context_configs.items():
                    st.markdown(f"### {context_name}")
                    st.caption(
                        f"Keterangan: {config['caption']}"
                    )
                    with st.spinner("Memuat data dari database..."):
                        loaded_df, load_error = load_dataframe_from_db(
                            st.session_state.db_engine, config['db_name']
                        )

                    if load_error:
                        st.error(f"Gagal memuat data: {load_error}")
                    else:
                        st.session_state.context_mdar_db = loaded_df
                        not_show_cols = ['id','created_at']
                        st.dataframe(loaded_df.drop(columns=not_show_cols,errors='ignore'), use_container_width=True)

    with tab_load_rules:
        st.subheader("Load Data dari Database")
        st.write("Muat data hasil analisis yang tersimpan di DB.")
        jenis_configs = {
            "Madu": {"db_name": "mdar_rules_madu", "support": 0.02, "confidence": 0.3, "lift": 1.2},
            "Minuman":  {"db_name": "mdar_rules_minuman",  "support": 0.02, "confidence": 0.3, "lift": 1.2},
            "Healthy Snack":  {"db_name": "mdar_rules_snack",  "support": 0.02, "confidence": 0.3, "lift": 1.2},
            "Bumbu Dapur":  {"db_name": "mdar_rules_bumbu",  "support": 0.02, "confidence": 0.5, "lift": 1.2},
        }

        context_configs = {
            "Chi-Square Test": {"db_name": "chi2_testing_mdar", "caption": """
                Tabel ini menampilkan hasil uji Chi-Square untuk mengukur signifikansi hubungan antara dimensi kontekstual (waktu, voucher, diskon, dan atribut transaksi lainnya) dengan dimensi produk. Nilai p-value digunakan untuk menentukan apakah hubungan yang ditemukan terjadi secara kebetulan atau memiliki keterkaitan yang signifikan secara statistik. Dimensi dengan p-value < 0,05 dianggap memiliki hubungan yang signifikan terhadap perilaku pembelian konsumen.
            """},
            "Lift Comparison":  {"db_name": "lift_comparison_testing_mdar",  "caption": """
            Tabel ini menampilkan perbandingan nilai lift antara aturan asosiasi tanpa konteks dan aturan asosiasi dengan penambahan dimensi kontekstual (Multidimensional Association Rule). Pengujian ini digunakan untuk mengukur apakah penambahan dimensi seperti waktu, diskon, atau voucher mampu memperkuat hubungan antar item. Semakin besar peningkatan lift yang diperoleh, semakin besar kontribusi konteks dalam menjelaskan pola pembelian konsumen.
            """},
        }
        if not st.session_state.db_connected:
            st.warning("Database belum terhubung. Konfigurasikan koneksi di sidebar.")
        else:

            if st.button("Load Data dari Database", type="primary"):
                tab_rules, tab_context = st.tabs(["Rules MDAR","Informasi Dimensi Kontekstual"])

                with tab_rules:
                    for jenis_name, config in jenis_configs.items():
                        st.markdown(f"### {jenis_name}")
                        st.caption(
                            f"Support: **{config['support']}**, "
                            f"Confidence: **{config['confidence']}**, "
                            f"Lift: **{config['lift']}**"
                        )
                        with st.spinner("Memuat data dari database..."):
                            loaded_df, load_error = load_dataframe_from_db(
                                st.session_state.db_engine, config['db_name']
                            )

                        if load_error:
                            st.error(f"Gagal memuat data: {load_error}")
                        else:
                            st.session_state.rules_mdar_db = loaded_df
                            st.success(f"{len(loaded_df):,} baris berhasil dimuat dari tabel {config['db_name']}.")
                            not_show_cols = ['id','created_at']
                            st.dataframe(loaded_df.drop(columns=not_show_cols,errors='ignore'), use_container_width=True)
                
                with tab_context:
                    for context_name, config in context_configs.items():
                        st.markdown(f"### {context_name}")
                        st.caption(
                            f"Keterangan: {config['caption']}"
                        )
                        with st.spinner("Memuat data dari database..."):
                            loaded_df, load_error = load_dataframe_from_db(
                                st.session_state.db_engine, config['db_name']
                            )

                        if load_error:
                            st.error(f"Gagal memuat data: {load_error}")
                        else:
                            st.session_state.context_mdar_db = loaded_df
                            not_show_cols = ['id','created_at']
                            st.dataframe(loaded_df.drop(columns=not_show_cols,errors='ignore'), use_container_width=True)


# =============================================================================
# HALAMAN 4 — SEQUENTIAL PATTERN MINING
# =============================================================================

elif selected_menu == "Sequential Pattern Mining":
    st.title("Sequential Pattern Mining")
    st.write("Temukan urutan pembelian produk lintas transaksi menggunakan algoritma PrefixSpan.")

    if not PREFIXSPAN_AVAILABLE:
        st.error("Library `prefixspan` belum terinstall.")
        st.stop()
    if st.session_state.df_clean is None:
        st.warning("Data belum tersedia. Upload file atau load dari database terlebih dahulu.")
        st.stop()

    tab_rules, tab_rules_spm_load_db = st.tabs(["Membentuk Rules Baru", "Load Rules dari DB"])

    with tab_rules:
        full_df = st.session_state.df_clean.copy()

        with st.expander("Atur Parameter & Rentang Waktu", expanded=True):
            st.markdown("**Rentang Waktu**")
            year_range, month_range = render_period_filter_widgets(full_df, key_prefix="spm")
            st.markdown("**Threshold**")
            col_sup, col_len = st.columns(2)
            raw_support     = col_sup.text_input("Min Support (0–1)", placeholder=f"default: {DEFAULT_SPM_SUPPORT}", key="spm_sup")
            min_pattern_len = col_len.number_input("Panjang pola minimum", min_value=2, max_value=10, value=2)

        min_support_frac, is_default = parse_threshold_input(raw_support, DEFAULT_SPM_SUPPORT)
        if min_support_frac is None or not (0 < min_support_frac <= 1):
            st.error("Masukkan angka antara 0 dan 1.")
            st.stop()
        if is_default:
            st.info(f"Support default: {DEFAULT_SPM_SUPPORT} ({DEFAULT_SPM_SUPPORT * 100:.0f}% pembeli unik)")

        if st.button("Jalankan Sequential Pattern Mining", type="primary"):
            filtered_df = filter_dataframe_by_period(full_df, "Waktu Pesanan Dibuat", year_range, month_range)
            if len(filtered_df) == 0:
                st.error("Tidak ada data pada rentang waktu yang dipilih.")
                st.stop()

            required_cols = {"Username (Pembeli)", "Nama Produk", "Waktu Pesanan Dibuat", "No. Pesanan"}
            missing_cols  = required_cols - set(filtered_df.columns)
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
                st.stop()

            with st.spinner("Menyiapkan data sekuensial..."):
                spm_df = filtered_df[["Username (Pembeli)", "Nama Produk", "Waktu Pesanan Dibuat", "No. Pesanan"]].copy()
                spm_df.columns = ["user_id", "item", "timestamp", "order_id"]
                spm_df["timestamp"] = pd.to_datetime(spm_df["timestamp"])
                spm_df = spm_df.sort_values(["user_id", "timestamp"])

                # Bangun sekuens per pengguna: setiap order jadi satu event berisi list produk
                orders_grouped = spm_df.groupby(["user_id", "order_id", "timestamp"])["item"].apply(list).reset_index()
                user_sequences = orders_grouped.groupby("user_id")["item"].apply(list).tolist()

                # Encode nama produk ke integer (PrefixSpan butuh integer)
                item_to_id = {}
                id_to_item = {}
                current_id = 0
                for sequence in user_sequences:
                    for event in sequence:
                        for item in event:
                            if item not in item_to_id:
                                item_to_id[item]       = current_id
                                id_to_item[current_id] = item
                                current_id += 1

                # Encode sekuens: setiap event di-flatten menjadi list integer
                encoded_sequences = []
                for sequence in user_sequences:
                    encoded_sequence = []
                    for event in sequence:
                        for item in event:
                            encoded_sequence.append(item_to_id[item])
                    encoded_sequences.append(encoded_sequence)

                encoded_sequences = [seq for seq in encoded_sequences if len(seq) >= 2]

                total_unique_users = len(encoded_sequences)
                min_support_count  = max(1, int(min_support_frac * total_unique_users))

                items_per_order_df = (
                    spm_df.groupby("order_id")["item"]
                    .count().reset_index()
                    .rename(columns={"item": "Jumlah Produk"})
                )

            with st.spinner(f"Menjalankan PrefixSpan (min_sup={min_support_count} dari {total_unique_users} pembeli)..."):
                prefixspan = PrefixSpan(encoded_sequences)
                found_patterns = prefixspan.frequent(min_support_count)

            with st.spinner("Menghitung timing antar pembelian & kategori Promotion Window..."):
                # Buat mapping user → list (item, timestamp) untuk hitung jarak waktu
                user_timeline_map = {}
                for user_id, user_group in spm_df.groupby("user_id"):
                    sorted_group = user_group.sort_values("timestamp")
                    user_timeline_map[user_id] = [
                        (row["item"], row["timestamp"]) for _, row in sorted_group.iterrows()
                    ]

                def compute_inter_purchase_gaps(pattern_items):
                    """
                    Hitung jarak hari antar pembelian berurutan yang cocok dengan pola.
                    Menelusuri tiap pengguna dan mencari kemunculan pola secara berurutan.
                    """
                    gap_days_list = []
                    for user_timeline in user_timeline_map.values():
                        i = 0
                        while i < len(user_timeline):
                            if user_timeline[i][0] == pattern_items[0]:
                                current_idx = i
                                current_ts  = user_timeline[i][1]
                                pattern_matched = True
                                for next_item in pattern_items[1:]:
                                    found_next = False
                                    for j in range(current_idx + 1, len(user_timeline)):
                                        if user_timeline[j][0] == next_item:
                                            gap_days = (user_timeline[j][1] - current_ts).days
                                            if gap_days == 0:
                                                gap_days = (user_timeline[j][1] - current_ts).total_seconds / (24*3600)
                                            gap_days_list.append(gap_days)
                                            current_idx = j
                                            current_ts  = user_timeline[j][1]
                                            found_next  = True
                                            break
                                    if not found_next:
                                        pattern_matched = False
                                        break
                            i += 1

                    if not gap_days_list:
                        return None

                    gap_array = np.array(gap_days_list)
                    return {
                        "mean":   round(float(np.mean(gap_array)),   1),
                        "median": round(float(np.median(gap_array)), 1),
                        "min":    int(np.min(gap_array)),
                        "max":    int(np.max(gap_array)),
                    }

                pattern_rows = []
                for support_count, pattern_ids in found_patterns:
                    if len(pattern_ids) < min_pattern_len:
                        continue
                    decoded_pattern = [id_to_item[pid] for pid in pattern_ids]
                    timing_stats    = compute_inter_purchase_gaps(decoded_pattern)

                    # Gunakan median (bukan mean) untuk kategori Promotion Window
                    # karena median lebih tahan terhadap outlier jarak waktu ekstrem
                    median_days    = timing_stats["median"] if timing_stats else None
                    promo_category = categorize_promotion_window(median_days)

                    pattern_rows.append({
                        "Sequence Rule":                decoded_pattern[0] + " → " + " → ".join(decoded_pattern[1:]) if len(decoded_pattern) > 1 else decoded_pattern[0],
                        "Panjang Pola":                 len(decoded_pattern),
                        "Support (count)":              support_count,
                        "Support (%)":                  round(support_count / total_unique_users * 100, 2),
                        "Rata-rata Jarak Waktu (hari)": timing_stats["mean"]   if timing_stats else None,
                        "Median Jarak Waktu (hari)":    timing_stats["median"] if timing_stats else None,
                        "Min Jarak Waktu (hari)":       timing_stats["min"]    if timing_stats else None,
                        "Max Jarak Waktu (hari)":       timing_stats["max"]    if timing_stats else None,
                        "Promotion Window":             promo_category,
                    })

                patterns_df = pd.DataFrame(pattern_rows).sort_values("Support (count)", ascending=False)

            st.session_state.spm_result = {
                "patterns_df":       patterns_df,
                "items_per_order":   items_per_order_df,
                "total_users":       total_unique_users,
                "min_support_count": min_support_count,
                "min_pattern_len":   min_pattern_len,
            }
            st.success("Selesai!")

        # Tampilkan hasil SPM
        spm_data = st.session_state.spm_result
        if spm_data is not None:
            patterns_df     = spm_data["patterns_df"]
            items_per_order = spm_data["items_per_order"]
            total_users     = spm_data["total_users"]

            st.subheader("Statistik per Transaksi")
            qty = items_per_order["Jumlah Produk"]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Transaksi",    f"{len(items_per_order):,}")
            col2.metric("Rata-rata item/trx", f"{qty.mean():.2f}")
            col3.metric("Median",             f"{int(qty.median())}")
            col4.metric("Maks item/trx",      f"{int(qty.max())}")
            col5.metric("Pembeli Unik",        f"{total_users:,}")

            with st.expander("Distribusi jumlah item per transaksi"):
                dist_df = (
                    items_per_order["Jumlah Produk"].value_counts()
                    .sort_index().reset_index()
                )
                dist_df.columns = ["Jumlah Produk", "Jumlah Transaksi"]
                st.dataframe(dist_df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Frequent Sequential Patterns")
            st.write(f"Ditemukan **{len(patterns_df)} pola** (panjang ≥ {spm_data['min_pattern_len']}, support ≥ {spm_data['min_support_count']})")

            with st.expander("Keterangan Kategori Promotion Window", expanded=False):
                st.markdown("""
    | Kategori | Rentang Waktu |
    |---|---|
    | Sangat Cepat | ≤ 7 Hari |
    | Cepat | 8 – 30 Hari |
    | Sedang | 31 – 60 Hari |
    | Lambat | 61 – 90 Hari |
    | Sangat Lambat | > 90 Hari |

    *Kategori didasarkan pada **Median Jarak Waktu (hari)** antar pembelian dalam satu pola urutan.*
                """)

            if patterns_df.empty:
                st.info("Tidak ada pola yang ditemukan. Coba turunkan min_support.")
            else:
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                max_length    = int(patterns_df["Panjang Pola"].max())
                length_range  = col_f1.slider("Panjang pola:", 2, max(2, max_length), (2, max(2, max_length)))
                top_n         = col_f2.number_input("Top-N:", min_value=5, max_value=1000, value=min(100, len(patterns_df)))
                sort_column   = col_f3.selectbox("Urutkan:", [
                    "Support (count)", "Support (%)", "Rata-rata Jarak Waktu (hari)",
                    "Median Jarak Waktu (hari)", "Min Jarak Waktu (hari)", "Max Jarak Waktu (hari)",
                ])

                # Filter berdasarkan kategori Promotion Window
                if "Promotion Window" in patterns_df.columns:
                    available_windows = sorted(patterns_df["Promotion Window"].dropna().unique().tolist())
                    selected_windows  = col_f4.multiselect(
                        "Filter Promotion Window:",
                        options=available_windows,
                        default=available_windows,
                        key="spm_window_filter"
                    )
                else:
                    selected_windows = None

                display_patterns = patterns_df[
                    (patterns_df["Panjang Pola"] >= length_range[0]) &
                    (patterns_df["Panjang Pola"] <= length_range[1])
                ]
                if selected_windows is not None and "Promotion Window" in display_patterns.columns:
                    display_patterns = display_patterns[display_patterns["Promotion Window"].isin(selected_windows)]

                sort_ascending   = "Jarak" in sort_column
                display_patterns = display_patterns.sort_values(sort_column, ascending=sort_ascending).head(top_n)

                output_cols = [
                    "Sequence Rule", "Panjang Pola", "Support (count)", "Support (%)",
                    "Rata-rata Jarak Waktu (hari)", "Median Jarak Waktu (hari)",
                    "Min Jarak Waktu (hari)", "Max Jarak Waktu (hari)", "Promotion Window"
                ]
                output_cols = [c for c in output_cols if c in display_patterns.columns]
                st.dataframe(display_patterns[output_cols].reset_index(drop=True), use_container_width=True)

                # Ringkasan distribusi Promotion Window
                if "Promotion Window" in patterns_df.columns:
                    st.markdown("**Distribusi Promotion Window**")
                    window_dist_df = (
                        patterns_df["Promotion Window"].value_counts()
                        .reset_index()
                        .rename(columns={"index": "Promotion Window", "Promotion Window": "Jumlah Pola"})
                    )
                    if window_dist_df.columns.tolist() == ["Promotion Window", "count"]:
                        window_dist_df.columns = ["Promotion Window", "Jumlah Pola"]
                    st.dataframe(window_dist_df, use_container_width=True, hide_index=True)

                # Download hasil
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    patterns_df[output_cols].to_excel(writer, index=False, sheet_name="Patterns")
                    dist_df.to_excel(writer, index=False, sheet_name="Item per Transaksi")
                st.download_button("⬇ Download XLSX", excel_buf.getvalue(), "sequential_patterns.xlsx")
    
    with tab_rules_spm_load_db:
        st.subheader("Load Data dari Database")
        st.write("Muat data hasil analisis yang tersimpan di DB.")
        spm_configs = {
            "Hasil Rules": {"db_name": "spm_rules", "support": 0.017},
        }

        if not st.session_state.db_connected:
            st.warning("Database belum terhubung. Konfigurasikan koneksi di sidebar.")
        else:

            if st.button("Load Data dari Database", type="primary"):
                for spm_name, config in spm_configs.items():
                    st.markdown(f"### {spm_name}")
                    st.caption(
                        f"Support: **{config['support']}** "
                    )
                    with st.spinner("Memuat data dari database..."):
                        loaded_df, load_error = load_dataframe_from_db(
                            st.session_state.db_engine, config['db_name']
                        )

                    if load_error:
                        st.error(f"Gagal memuat data: {load_error}")
                    else:
                        st.session_state.spm_rules_db = loaded_df
                        spm_db_df = loaded_df
                        st.success(f"{len(spm_db_df):,} baris berhasil dimuat dari tabel {config['db_name']}.")
                        not_show_cols = ['id','created_at']
                        spm_db_df = spm_db_df.drop(columns=not_show_cols,errors='ignore')
                        
                        # FILTER
                        col1, col2, col3, col4 = st.columns(4)
                        len_range = col1.slider("Filter Panjang Pola", spm_db_df['Panjang Pola'].min(), spm_db_df['Panjang Pola'].max(), (spm_db_df['Panjang Pola'].min(), spm_db_df['Panjang Pola'].max()))
                        top_n = col2.number_input("Top-N:", 5, 10000, min(100, len(spm_db_df)))
                        sort_column = col3.selectbox("Ururtkan Berdasarkan:",[
                            "Support (count)", "Support (%)", "Rata-rata Jarak Waktu (hari)",
                            "Median Jarak Waktu (hari)", "Min Jarak Waktu (hari)", "Max Jarak Waktu (hari)",
                        ])
                        active_windows = spm_db_df['Promotion Window'].dropna().unique().tolist()
                        selected_windows  = col4.multiselect(
                            "Filter Promotion Window:",
                            options=active_windows,
                            default=active_windows,
                            key="spm_window_filter"
                        )
                        display_patterns = spm_db_df[
                            (spm_db_df["Panjang Pola"] >= len_range[0]) &
                            (spm_db_df["Panjang Pola"] <= len_range[1])
                        ]
                        if selected_windows is not None and "Promotion Window" in display_patterns.columns:
                            display_patterns = display_patterns[display_patterns["Promotion Window"].isin(selected_windows)]

                        sort_ascending   = "Jarak" in sort_column
                        display_patterns = display_patterns.sort_values(sort_column, ascending=sort_ascending).head(top_n)

                        st.dataframe(display_patterns, use_container_width=True)