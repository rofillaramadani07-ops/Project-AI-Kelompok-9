# ==========================================
# SISTEM PREDIKSI HARGA KENDARAAN BEKAS
# Praktikum AI 2026
# Anggota Kelompok 9: Surya, Rofilla, Rafli
# ==========================================
# Modul praktikum yang diimplementasikan :
#   MODUL III -> Fuzzy Logic (Sugeno)
#                menilai kewajaran harga tawaran
#                (fuzzifikasi -> inferensi Rule Base Sugeno -> defuzzifikasi Weighted Average)
#   MODUL IV  -> Teknik Preprocessing Data
#                mean removal + scaling (StandardScaler), penskalaan kilometer
#   MODUL V   -> Teknik Labelling Data (HANYA KONSEP)
#                kategori Motor/Mobil diubah menjadi angka 0/1 (fitur Is_Mobil),
#                dilakukan manual, tidak memakai LabelEncoder
#   MODUL VI  -> Supervised Learning (bagian Linear Regression)
#                dikembangkan menjadi Polynomial Ridge Regression
# ==========================================

# ------------------------------------------------------------------
# BAGIAN A - IMPORT LIBRARY
# ------------------------------------------------------------------
import os
import re
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge                              # MODUL VI: Supervised Learning
from sklearn.metrics import mean_absolute_error, r2_score           # MODUL VI: evaluasi model
from sklearn.model_selection import train_test_split                # MODUL VI: pembagian data latih/uji
from sklearn.pipeline import make_pipeline                          # MODUL IV + VI: rangkaian preprocessing -> model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # MODUL VI: Polynomial / MODUL IV: StandardScaler

# ------------------------------------------------------------------
# BAGIAN A - KONSTANTA
# ------------------------------------------------------------------
TAHUN_SEKARANG = 2026
CSV_FILE = "DATASET_KENDARAAN.csv"
XLSX_FILE = "DATASET_KENDARAAN.xlsx"
KM_MAKS = 150000   # batas atas input; data latih maksimum sekitar 121.000 km
KM_MAKS_TXT = f"{KM_MAKS:,}".replace(",", ".")
BATAS_WAJAR = 0.05  # satu sumber kebenaran: tawaran dalam +/-5% estimasi dianggap "Wajar"

# ------------------------------------------------------------------
# BAGIAN A - KATALOG KENDARAAN
# Katalog kendaraan
# (kategori, merek, model, tahun rilis, cc, transmisi, harga baru dalam JUTA Rp)
# Harga baru = perkiraan harga on-the-road saat unit baru dirilis.
# ------------------------------------------------------------------
CATALOG = [
    ("Motor", "Honda", "BeAT", 2008, 110, "Automatic", 14.0),
    ("Motor", "Honda", "BeAT Sporty CBS", 2020, 110, "Automatic", 17.5),
    ("Motor", "Honda", "Scoopy Stylish", 2021, 110, "Automatic", 20.5),
    ("Motor", "Honda", "Vario 125", 2018, 125, "Automatic", 21.0),
    ("Motor", "Yamaha", "NMAX 155 Connected", 2020, 155, "Automatic", 33.0),
    ("Motor", "Yamaha", "Aerox 155", 2021, 155, "Automatic", 28.5),
    ("Motor", "Kawasaki", "Ninja 250 ABS", 2019, 250, "Manual", 66.0),
    ("Motor", "Suzuki", "Satria F150", 2016, 150, "Manual", 23.5),
    ("Mobil", "Toyota", "Avanza 1.3 G M/T", 2019, 1300, "Manual", 215.0),
    ("Mobil", "Toyota", "Innova Reborn 2.4 G", 2018, 2400, "Automatic", 400.0),
    ("Mobil", "Honda", "Brio Satya E M/T", 2021, 1200, "Manual", 165.0),
    ("Mobil", "Honda", "HR-V 1.5 E CVT", 2021, 1500, "Automatic", 365.0),
    ("Mobil", "Daihatsu", "Xenia 1.3 X M/T", 2020, 1300, "Manual", 190.0),
    ("Mobil", "Daihatsu", "Terios R A/T", 2021, 1500, "Automatic", 270.0),
    ("Mobil", "Mitsubishi", "Xpander Ultimate A/T", 2020, 1500, "Automatic", 285.0),
    ("Mobil", "Mitsubishi", "Pajero Sport Dakar", 2019, 2400, "Automatic", 560.0),
]

REQUIRED_COLS = ["Kategori", "Merek", "Model", "Tahun", "Kilometer",
                 "Kapasitas_Mesin", "Transmisi", "Harga_Baru", "Harga"]


# ------------------------------------------------------------------
# BAGIAN A - FUNGSI BANTU (format Rupiah, pembaca angka, kilometer wajar)
# ------------------------------------------------------------------
def rupiah(juta):
    """Ubah nilai juta rupiah menjadi teks 'Rp 12.500.000'."""
    return "Rp " + f"{int(round(juta * 1_000_000)):,}".replace(",", ".")


def parse_angka(teks):
    """Ubah teks '150.000.000' / 'Rp 150000000' menjadi int.
    Titik dianggap pemisah ribuan. Selain digit 0-9 -> None (ditolak)."""
    bersih = teks.strip().lower().replace("rp", "").replace(".", "").replace(" ", "")
    return int(bersih) if re.fullmatch(r"[0-9]+", bersih) else None


def km_wajar(kat, umur):
    """Perkiraan kilometer normal berdasarkan umur kendaraan."""
    per_tahun = 4000 if kat == "Motor" else 10000
    return int(round(max(umur, 0) * per_tahun / 500.0) * 500)


# ==================================================================
# ### MODUL III - FUZZY LOGIC (SUGENO)
# ==================================================================
# ------------------------------------------------------------------
# FUZZY LOGIC (Sugeno) - PENILAIAN KEWAJARAN HARGA TAWARAN
# Alur: fuzzifikasi -> inferensi aturan IF-THEN (AND = perkalian)
#       -> defuzzifikasi rata-rata berbobot (weighted average)
# ------------------------------------------------------------------

# --- MODUL III (a): FUNGSI KEANGGOTAAN ---
def mf_segitiga(x, a, b, c):
    """Keanggotaan segitiga: 0 di a, puncak 1 di b, kembali 0 di c."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def mf_trapesium(x, a, b, c, d):
    """Keanggotaan trapesium: naik a->b, bernilai 1 di b..c, turun c->d."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    return (x - a) / (b - a) if x < b else (d - x) / (d - c)


def mf_bahu_kiri(x, a, b):
    """Bahu kiri: bernilai 1 sampai a, turun linear menjadi 0 di b."""
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def mf_bahu_kanan(x, a, b):
    """Bahu kanan: bernilai 0 sampai a, naik linear menjadi 1 di b."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


class FuzzyPenilaianHarga:
    """Sistem fuzzy Sugeno dua input untuk menilai harga tawaran.

    Input 1  rasio harga  = harga tawaran / harga prediksi model ML
             himpunan: Murah, Wajar, Mahal. "Wajar" = tawaran dalam +/-5%
             dari estimasi (BATAS_WAJAR; sama dengan rentang wajar yang ditampilkan).
    Input 2  rasio pakai  = kilometer / kilometer wajar menurut umur
             himpunan: Rendah, Normal, Tinggi
    Output   skor kelayakan beli 0-100 (konstanta Sugeno, ditetapkan
             berdasarkan pertimbangan kelompok, bukan hasil pelatihan).
    """

    # --- MODUL III (b): BASIS ATURAN IF-THEN (9 aturan) ---
    # (harga, pemakaian) -> skor kelayakan beli
    ATURAN = {
        ("Murah", "Rendah"): 95, ("Murah", "Normal"): 88, ("Murah", "Tinggi"): 70,
        ("Wajar", "Rendah"): 78, ("Wajar", "Normal"): 65, ("Wajar", "Tinggi"): 45,
        ("Mahal", "Rendah"): 40, ("Mahal", "Normal"): 25, ("Mahal", "Tinggi"): 10,
    }

    # --- MODUL III (c): FUZZIFIKASI (mengubah rasio menjadi derajat keanggotaan) ---
    @staticmethod
    def fuzzifikasi_harga(rasio):
        b = BATAS_WAJAR
        return {"Murah": mf_bahu_kiri(rasio, 1 - b - 0.05, 1 - b),
                "Wajar": mf_trapesium(rasio, 1 - b - 0.05, 1 - b, 1 + b, 1 + b + 0.05),
                "Mahal": mf_bahu_kanan(rasio, 1 + b, 1 + b + 0.05)}

    @staticmethod
    def fuzzifikasi_pakai(rasio):
        return {"Rendah": mf_bahu_kiri(rasio, 0.60, 1.00),
                "Normal": mf_segitiga(rasio, 0.60, 1.00, 1.60),
                "Tinggi": mf_bahu_kanan(rasio, 1.00, 1.60)}

    # --- MODUL III (d): LABEL & STATUS (mengubah skor/rasio menjadi teks) ---
    @staticmethod
    def label(skor):
        """Label kelayakan beli berdasarkan skor 0-100."""
        if skor >= 75:
            return "Sangat Layak"
        if skor >= 55:
            return "Layak"
        if skor >= 35:
            return "Kurang Layak"
        return "Tidak Layak"

    @staticmethod
    def status_harga(rasio):
        """Status harga tawaran terhadap estimasi AI (sudut pandang pembeli)."""
        b = BATAS_WAJAR
        if rasio < 0.50:
            return "Terlalu Murah (Waspada)"
        if rasio < 1 - b - 0.05:
            return "Murah"
        if rasio < 1 - b:
            return "Agak Murah"
        if rasio <= 1 + b:
            return "Wajar"
        if rasio < 1 + b + 0.05:
            return "Agak Mahal"
        return "Terlalu Mahal"

    # --- MODUL III (e): INFERENSI & DEFUZZIFIKASI ---
    def nilai(self, harga_tawaran, harga_prediksi, km, km_normal):
        """harga_* dalam juta rupiah; km_normal = kilometer wajar menurut umur."""
        rasio_harga = harga_tawaran / harga_prediksi
        rasio_pakai = km / max(km_normal, 2000)          # hindari bagi nol (unit baru)

        mu_h = self.fuzzifikasi_harga(rasio_harga)       # 1. fuzzifikasi
        mu_p = self.fuzzifikasi_pakai(rasio_pakai)

        pembilang = penyebut = 0.0                        # 2. inferensi (AND = perkalian)
        for (h, p), z in self.ATURAN.items():
            alfa = mu_h[h] * mu_p[p]
            pembilang += alfa * z
            penyebut += alfa
        skor = pembilang / penyebut                       # 3. defuzzifikasi (rata-rata berbobot)

        return {"rasio_harga": rasio_harga, "rasio_pakai": rasio_pakai,
                "mu_harga": mu_h, "mu_pakai": mu_p,
                "skor": skor, "label": self.label(skor),
                "status_harga": self.status_harga(rasio_harga),
                "kategori_harga": max(mu_h, key=mu_h.get)}


# ------------------------------------------------------------------
# MODEL MACHINE LEARNING
# Class ini memuat BAGIAN B (dataset), MODUL IV (preprocessing)
# dan MODUL VI (supervised learning)
# ------------------------------------------------------------------
class VehiclePriceModel:
    def __init__(self):
        self.df = self.load_dataset()
        self.train_model()

    # ==============================================================
    # BAGIAN B - DATASET
    # ==============================================================
    # ---------- dataset ----------
    @staticmethod
    def generate_dataset():
        """Membuat dataset pasaran harga bekas (estimasi) dengan pola
        penyusutan yang realistis: turun cepat di tahun-tahun awal lalu
        melandai, dan makin murah jika kilometer lebih tinggi dari wajar."""
        rng = np.random.default_rng(2026)
        rows = []
        for kat, merek, model, rilis, cc, trans, baru in CATALOG:
            k = 0.12 if kat == "Motor" else 0.08          # laju penyusutan
            floor = 0.28 if kat == "Motor" else 0.35      # nilai sisa minimum
            c_km = 0.030 if kat == "Motor" else 0.015     # pengaruh kilometer
            for tahun in range(rilis, TAHUN_SEKARANG + 1):
                umur = TAHUN_SEKARANG - tahun
                for _ in range(6):
                    wajar = km_wajar(kat, umur)
                    km = max(500, int(wajar * rng.uniform(0.5, 1.6)) + int(rng.integers(300, 3000)))
                    faktor_umur = floor + (1 - floor) * np.exp(-k * umur)
                    faktor_km = np.exp(-c_km * (km - wajar) / 10000.0)
                    noise = rng.normal(1.0, 0.03)
                    harga = baru * faktor_umur * faktor_km * noise
                    harga = min(harga, baru * 0.98)   # harga bekas tidak boleh >= harga baru
                    rows.append([kat, merek, model, tahun, km, cc, trans,
                                 baru, round(float(harga), 2)])
        return pd.DataFrame(rows, columns=REQUIRED_COLS)

    def load_dataset(self):
        df = None
        if os.path.exists(CSV_FILE):
            try:
                tmp = pd.read_csv(CSV_FILE)
                model_katalog = {c[2] for c in CATALOG}
                if (all(c in tmp.columns for c in REQUIRED_COLS) and len(tmp) >= 50
                        and set(tmp["Model"]) == model_katalog          # CSV sesuai katalog
                        and (tmp["Harga"] <= tmp["Harga_Baru"]).all()   # tidak ada harga bekas > harga baru
                        and tmp[REQUIRED_COLS].notna().all().all()):
                    df = tmp
            except Exception:
                df = None
        if df is None:
            df = self.generate_dataset()
            df.to_csv(CSV_FILE, index=False)
            try:
                df.to_excel(XLSX_FILE, index=False)
            except Exception:
                pass  # openpyxl belum terpasang -> lewati berkas Excel
        return df

    # ==============================================================
    # ### MODUL IV - TEKNIK PREPROCESSING DATA
    #     (rekayasa fitur di sini + StandardScaler di pipeline Modul 2)
    # ==============================================================
    # ---------- training ----------
    @staticmethod
    def make_features(kat, harga_baru, tahun, km):
        umur = np.maximum(0, TAHUN_SEKARANG - np.asarray(tahun, dtype=float))   # fitur Umur
        is_mobil = (np.asarray(kat) == "Mobil").astype(float)                   # MODUL V (konsep Labelling Data): kategori -> angka, 1 = Mobil, 0 = Motor
        return np.column_stack([
            np.log(np.asarray(harga_baru, dtype=float)),       # transformasi logaritma harga baru
            umur,
            np.asarray(km, dtype=float) / 10000.0,             # penskalaan kilometer (per 10.000 km)
            is_mobil,
        ])

    # ==============================================================
    # ### MODUL VI - SUPERVISED LEARNING
    #     (Linear Regression -> Polynomial Ridge Regression)
    # ==============================================================
    def train_model(self):
        d = self.df
        X = self.make_features(d["Kategori"].values, d["Harga_Baru"].values,
                               d["Tahun"].values, d["Kilometer"].values)
        y = np.log(d["Harga"].values.astype(float))             # target dilatih dalam bentuk log(harga)

        # MODUL VI: pembagian data 80% latih : 20% uji
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # MODUL IV (StandardScaler = mean removal + scaling) -> MODUL VI (PolynomialFeatures + Ridge)
        self.model = make_pipeline(StandardScaler(),
                                   PolynomialFeatures(degree=2, include_bias=False),
                                   Ridge(alpha=0.01))
        self.model.fit(X_tr, y_tr)

        # MODUL VI: evaluasi model pada data uji (MAE, R2, MAPE)
        pred = np.exp(self.model.predict(X_te))
        actual = np.exp(y_te)
        self.mae = mean_absolute_error(actual, pred)
        self.r2 = r2_score(actual, pred)
        self.mape = float(np.mean(np.abs(pred - actual) / actual))

        # latih ulang memakai seluruh data agar prediksi final maksimal
        self.model.fit(X, y)

    # ---------- prediksi (MODUL VI: dipakai oleh GUI) ----------
    def predict(self, kat, harga_baru, tahun, km):
        X = self.make_features([kat], [harga_baru], [tahun], [km])
        hasil = float(np.exp(self.model.predict(X))[0])
        return max(hasil, harga_baru * 0.10)


# ==================================================================
# BAGIAN C - APLIKASI GUI (TKINTER)
# Menggabungkan: MODUL IV + VI (prediksi harga) dan MODUL III (penilaian tawaran)
# ==================================================================
class VehicleApp:
    BG = "#0b1220"
    CARD = "#162033"
    CARD2 = "#1d2a42"
    BORDER = "#2b3a57"
    TEXT = "#f1f5f9"
    MUTED = "#94a3b8"
    ACCENT = "#22d3ee"
    ACCENT2 = "#6366f1"
    GOOD = "#4ade80"
    WARN = "#fbbf24"
    BAD = "#f87171"
    FONT = "Segoe UI"

    def __init__(self, root):
        self.root = root
        self.root.title("Prediksi Harga Kendaraan Bekas - KELOMPOK 9")
        self.root.geometry("1100x720")
        self.root.minsize(900, 640)
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)          # bisa diperbesar / maximize

        self.ml = VehiclePriceModel()            # MODUL IV + VI: model prediksi harga
        self.fuzzy = FuzzyPenilaianHarga()       # MODUL III: sistem fuzzy penilai tawaran
        self.build_catalog()
        self.setup_style()
        self.create_widgets()
        self.update_merek_options()
        self.root.bind("<Return>", lambda e: self.predict_price())

    # ---------- data ----------
    def build_catalog(self):
        self.vehicle_data = {}
        for kat, merek, model, rilis, cc, trans, baru in CATALOG:
            self.vehicle_data.setdefault(kat, {}).setdefault(merek, {})[model] = {
                "rilis": rilis, "cc": cc, "transmisi": trans, "baru": baru}

    # ---------- style ----------
    def setup_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        F = self.FONT

        s.configure("TFrame", background=self.BG)
        s.configure("Card.TFrame", background=self.CARD)
        s.configure("Header.TFrame", background=self.ACCENT2)

        s.configure("Title.TLabel", background=self.ACCENT2, foreground="#ffffff", font=(F, 20, "bold"))
        s.configure("Sub.TLabel", background=self.ACCENT2, foreground="#e0e7ff", font=(F, 10))
        s.configure("CardTitle.TLabel", background=self.CARD, foreground=self.ACCENT, font=(F, 13, "bold"))
        s.configure("Form.TLabel", background=self.CARD, foreground=self.TEXT, font=(F, 10, "bold"))
        s.configure("Hint.TLabel", background=self.CARD, foreground=self.MUTED, font=(F, 9))
        s.configure("ChipKey.TLabel", background=self.CARD2, foreground=self.MUTED, font=(F, 9))
        s.configure("ChipVal.TLabel", background=self.CARD2, foreground=self.TEXT, font=(F, 12, "bold"))

        s.configure("TCombobox", fieldbackground="#25344f", background="#33466a",
                    foreground=self.TEXT, arrowcolor=self.ACCENT, bordercolor=self.BORDER,
                    padding=6)
        s.map("TCombobox", fieldbackground=[("readonly", "#25344f")],
              foreground=[("readonly", self.TEXT)],
              selectbackground=[("readonly", "#25344f")],
              selectforeground=[("readonly", self.TEXT)])
        s.configure("TEntry", fieldbackground="#25344f", foreground=self.TEXT,
                    insertcolor=self.TEXT, bordercolor=self.BORDER, padding=6)
        s.configure("TSpinbox", fieldbackground="#25344f", foreground=self.TEXT,
                    background="#33466a", arrowcolor=self.ACCENT, bordercolor=self.BORDER,
                    insertcolor=self.TEXT, padding=6)
        self.root.option_add("*TCombobox*Listbox.background", "#25344f")
        self.root.option_add("*TCombobox*Listbox.foreground", self.TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.ACCENT2)
        self.root.option_add("*TCombobox*Listbox.font", (F, 10))

        s.configure("Accent.TButton", font=(F, 12, "bold"), background=self.ACCENT2,
                    foreground="#ffffff", borderwidth=0, padding=12)
        s.map("Accent.TButton", background=[("active", "#4f46e5"), ("pressed", "#4338ca")])
        s.configure("Ghost.TButton", font=(F, 10, "bold"), background="#33466a",
                    foreground=self.TEXT, borderwidth=0, padding=10)
        s.map("Ghost.TButton", background=[("active", "#415a85")])

        s.configure("Treeview", background=self.CARD2, fieldbackground=self.CARD2,
                    foreground=self.TEXT, rowheight=28, borderwidth=0, font=(F, 10))
        s.configure("Treeview.Heading", background="#33466a", foreground=self.TEXT,
                    font=(F, 10, "bold"), borderwidth=0, padding=6)
        s.map("Treeview", background=[("selected", self.ACCENT2)])

    # ---------- widget ----------
    def card(self, parent, title):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=20)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        return frame

    def chip(self, parent, key, col):
        box = tk.Frame(parent, bg=self.CARD2, padx=12, pady=8)
        box.grid(row=0, column=col, sticky="nsew", padx=4)
        parent.columnconfigure(col, weight=1)
        ttk.Label(box, text=key, style="ChipKey.TLabel").pack(anchor="w")
        var = tk.StringVar(value="-")
        ttk.Label(box, textvariable=var, style="ChipVal.TLabel").pack(anchor="w")
        return var

    def create_widgets(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ----- header -----
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 16))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Sistem Prediksi Harga Kendaraan Bekas", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Project AI 2026  •  KELOMPOK 9: Surya · Rofilla · Rafli",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        # ----- body -----
        body = ttk.Frame(self.root, padding=16)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=5, uniform="col")
        body.columnconfigure(1, weight=6, uniform="col")
        body.rowconfigure(0, weight=1)

        # ===== kolom kiri : form =====
        form = self.card(body, "Parameter Kendaraan")
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        form.columnconfigure(1, weight=1)

        def label(text, r):
            ttk.Label(form, text=text, style="Form.TLabel").grid(
                row=r, column=0, sticky="w", pady=8, padx=(0, 14))

        label("Kategori", 1)
        self.combo_kat = ttk.Combobox(form, values=list(self.vehicle_data.keys()),
                                      state="readonly", font=(self.FONT, 11))
        self.combo_kat.grid(row=1, column=1, sticky="ew", pady=8)
        self.combo_kat.current(0)
        self.combo_kat.bind("<<ComboboxSelected>>", lambda e: self.update_merek_options())

        label("Merek", 2)
        self.combo_merek = ttk.Combobox(form, state="readonly", font=(self.FONT, 11))
        self.combo_merek.grid(row=2, column=1, sticky="ew", pady=8)
        self.combo_merek.bind("<<ComboboxSelected>>", lambda e: self.update_model_options())

        label("Tipe / Model", 3)
        self.combo_model = ttk.Combobox(form, state="readonly", font=(self.FONT, 11))
        self.combo_model.grid(row=3, column=1, sticky="ew", pady=8)
        self.combo_model.bind("<<ComboboxSelected>>", lambda e: self.update_spec())

        label("Tahun Pembuatan", 4)
        self.var_tahun = tk.StringVar()
        self.spin_tahun = ttk.Spinbox(form, from_=1990, to=TAHUN_SEKARANG,
                                      textvariable=self.var_tahun, font=(self.FONT, 11))
        self.spin_tahun.grid(row=4, column=1, sticky="ew", pady=8)

        label("Jarak Tempuh (km)", 5)
        self.var_km = tk.StringVar()
        self.entry_km = ttk.Entry(form, textvariable=self.var_km, font=(self.FONT, 11))
        self.entry_km.grid(row=5, column=1, sticky="ew", pady=8)

        label("Harga Tawaran (Rp)", 6)
        self.var_tawaran = tk.StringVar()
        self.entry_tawaran = ttk.Entry(form, textvariable=self.var_tawaran, font=(self.FONT, 11))
        self.entry_tawaran.grid(row=6, column=1, sticky="ew", pady=8)

        self.hint = ttk.Label(form, text="", style="Hint.TLabel", wraplength=340, justify="left")
        self.hint.grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # chip spesifikasi otomatis
        chips = tk.Frame(form, bg=self.CARD)
        chips.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        self.var_rilis = self.chip(chips, "Rilis", 0)
        self.var_cc = self.chip(chips, "Mesin", 1)
        self.var_trans = self.chip(chips, "Transmisi", 2)

        btns = ttk.Frame(form, style="Card.TFrame")
        btns.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        btns.columnconfigure(0, weight=3)
        btns.columnconfigure(1, weight=1)
        ttk.Button(btns, text="🚀  HITUNG ESTIMASI HARGA", style="Accent.TButton",
                   command=self.predict_price).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(btns, text="↺  Reset", style="Ghost.TButton",
                   command=self.reset_form).grid(row=0, column=1, sticky="ew")

        # ===== kolom kanan : hasil + riwayat =====
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        result = self.card(right, "Hasil Prediksi")
        result.grid(row=0, column=0, sticky="ew")
        result.columnconfigure(0, weight=1)

        self.lbl_model = tk.Label(result, text="Pilih kendaraan lalu tekan tombol hitung",
                                  font=(self.FONT, 11), bg=self.CARD, fg=self.MUTED)
        self.lbl_model.grid(row=1, column=0, columnspan=2, sticky="w")
        self.lbl_price = tk.Label(result, text="Rp 0", font=(self.FONT, 32, "bold"),
                                  bg=self.CARD, fg=self.GOOD)
        self.lbl_price.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.lbl_range = tk.Label(result, text="Rentang wajar: -", font=(self.FONT, 11),
                                  bg=self.CARD, fg=self.TEXT)
        self.lbl_range.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 8))
        # label hasil penilaian MODUL III (Fuzzy Logic)
        self.lbl_fuzzy = tk.Label(result, text="Penilaian tawaran: isi kolom Harga Tawaran (opsional)",
                                  font=(self.FONT, 12, "bold"), bg=self.CARD, fg=self.MUTED)
        self.lbl_fuzzy.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 2))
        self.lbl_fuzzy_detail = tk.Label(result, text="", font=(self.FONT, 9), bg=self.CARD,
                                         fg=self.MUTED, justify="left", wraplength=470)
        self.lbl_fuzzy_detail.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))
        # label evaluasi MODUL VI (MAE, MAPE, R2)
        self.lbl_metric = tk.Label(
            result,
            text=(f"Evaluasi AI (data uji, dataset sintetis)  •  MAE = Rp {self.ml.mae:.2f} Juta  •  "
                  f"MAPE = {self.ml.mape*100:.1f}%  •  R² = {self.ml.r2*100:.1f}%"),
            font=(self.FONT, 9, "italic"), bg=self.CARD, fg=self.MUTED)
        self.lbl_metric.grid(row=6, column=0, columnspan=2, sticky="w")

        hist = self.card(right, "Riwayat Prediksi")
        hist.grid(row=1, column=0, sticky="nsew", pady=(16, 0))
        hist.columnconfigure(0, weight=1)
        hist.rowconfigure(1, weight=1)

        cols = ("model", "tahun", "km", "harga", "nilai")
        self.tree = ttk.Treeview(hist, columns=cols, show="headings", height=6)
        for c, t, w in (("model", "Model", 160), ("tahun", "Tahun", 55),
                        ("km", "Kilometer", 80), ("harga", "Estimasi Harga", 130),
                        ("nilai", "Penilaian Tawaran", 170)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w" if c == "model" else "center")
        self.tree.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(hist, orient="vertical", command=self.tree.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)

    # ---------- event ----------
    def update_merek_options(self):
        kat = self.combo_kat.get()
        merek_list = list(self.vehicle_data[kat].keys())
        self.combo_merek["values"] = merek_list
        self.combo_merek.current(0)
        self.update_model_options()

    def update_model_options(self):
        kat, merek = self.combo_kat.get(), self.combo_merek.get()
        model_list = list(self.vehicle_data[kat][merek].keys())
        self.combo_model["values"] = model_list
        self.combo_model.current(0)
        self.update_spec()

    def current_info(self):
        return self.vehicle_data[self.combo_kat.get()][self.combo_merek.get()][self.combo_model.get()]

    def update_spec(self):
        kat = self.combo_kat.get()
        info = self.current_info()
        rilis = info["rilis"]

        self.var_rilis.set(str(rilis))
        self.var_cc.set(f"{info['cc']} cc")
        self.var_trans.set(info["transmisi"])

        self.spin_tahun.config(from_=rilis, to=TAHUN_SEKARANG)
        default_tahun = min(TAHUN_SEKARANG, rilis + 2) if kat == "Mobil" else rilis
        self.var_tahun.set(str(default_tahun))
        self.var_km.set(str(km_wajar(kat, TAHUN_SEKARANG - default_tahun)))
        self.hint.config(text=f"Tahun dapat diisi {rilis} – {TAHUN_SEKARANG}. "
                              f"Kilometer terisi otomatis sebagai perkiraan wajar (maks. {KM_MAKS_TXT} km). "
                              f"Harga tawaran opsional (rupiah penuh), contoh 150.000.000.")

    def reset_form(self):
        self.combo_kat.current(0)
        self.update_merek_options()
        self.var_tawaran.set("")
        self.lbl_model.config(text="Pilih kendaraan lalu tekan tombol hitung")
        self.lbl_price.config(text="Rp 0")
        self.lbl_range.config(text="Rentang wajar: -")
        self.lbl_fuzzy.config(text="Penilaian tawaran: isi kolom Harga Tawaran (opsional)", fg=self.MUTED)
        self.lbl_fuzzy_detail.config(text="")

    def predict_price(self):
        # ----- BAGIAN C (1): VALIDASI INPUT -----
        try:
            kat = self.combo_kat.get()
            model_tipe = self.combo_model.get()
            info = self.current_info()

            tahun_txt = self.var_tahun.get().strip()
            if not re.fullmatch(r"[0-9]{4}", tahun_txt):
                raise ValueError("tahun")
            tahun = int(tahun_txt)
            km_val = parse_angka(self.var_km.get()) if self.var_km.get().strip() else 0
            if km_val is None:
                raise ValueError("km")
            km = float(km_val)
        except ValueError:
            messagebox.showerror("Error", "Tahun dan Jarak Tempuh harus berupa angka yang valid.")
            return

        if tahun < info["rilis"] or tahun > TAHUN_SEKARANG:
            messagebox.showwarning("Peringatan",
                                   f"Tahun {model_tipe} harus antara {info['rilis']} dan {TAHUN_SEKARANG}.")
            return
        if km < 0:
            messagebox.showwarning("Peringatan", "Jarak tempuh (kilometer) tidak boleh negatif.")
            return
        if km > KM_MAKS:
            messagebox.showwarning(
                "Peringatan",
                f"Jarak tempuh melebihi {KM_MAKS_TXT} km, di luar rentang data latih sehingga "
                f"prediksi tidak andal. Masukkan nilai yang lebih kecil.")
            return

        # harga tawaran bersifat opsional (rupiah penuh)
        tawaran = None
        if self.var_tawaran.get().strip():
            tawaran = parse_angka(self.var_tawaran.get())
            if tawaran is None:
                messagebox.showerror("Error", "Harga tawaran harus berupa angka, contoh 150.000.000.")
                return
            if tawaran <= 0:
                messagebox.showwarning("Peringatan",
                                       "Harga tawaran harus lebih besar dari 0 (rupiah penuh, contoh 150.000.000).")
                return

        # ----- BAGIAN C (2): PREDIKSI HARGA (MODUL VI: Supervised Learning) -----
        harga = self.ml.predict(kat, info["baru"], tahun, km)
        margin = BATAS_WAJAR
        low, high = harga * (1 - margin), harga * (1 + margin)

        self.lbl_model.config(text=f"{self.combo_merek.get()} {model_tipe}  •  {tahun}  •  {int(km):,} km".replace(",", "."))
        self.lbl_price.config(text=rupiah(harga))
        self.lbl_range.config(text=f"Rentang wajar: {rupiah(low)}  –  {rupiah(high)}   ({harga:.2f} Juta)")

        # ----- BAGIAN C (3): penilaian harga tawaran dengan Fuzzy Logic (MODUL III) -----
        penilaian = "-"
        if tawaran is None:
            self.lbl_fuzzy.config(text="Penilaian tawaran: isi kolom Harga Tawaran (opsional)", fg=self.MUTED)
            self.lbl_fuzzy_detail.config(text="")
        else:
            h = self.fuzzy.nilai(tawaran / 1_000_000, harga, km,
                                 km_wajar(kat, TAHUN_SEKARANG - tahun))
            status = h["status_harga"]
            warna = {"Sangat Layak": self.GOOD, "Layak": self.GOOD,
                     "Kurang Layak": self.WARN, "Tidak Layak": self.BAD}[h["label"]]
            if status.startswith("Terlalu Murah"):
                warna = self.WARN          # tawaran janggal murah -> beri tanda waspada
            self.lbl_fuzzy.config(
                text=f"Penilaian tawaran: {status}  •  Skor {h['skor']:.0f}/100", fg=warna)

            mh, mp = h["mu_harga"], h["mu_pakai"]
            catatan = ""
            if status.startswith("Terlalu Murah"):
                catatan = "\nTawaran jauh di bawah estimasi, periksa kondisi unit dan kelengkapan dokumen."
            self.lbl_fuzzy_detail.config(text=(
                f"Kelayakan beli: {h['label']}  •  Tawaran = {h['rasio_harga']*100:.0f}% dari estimasi\n"
                f"Murah {mh['Murah']:.2f} · Wajar {mh['Wajar']:.2f} · Mahal {mh['Mahal']:.2f}  |  "
                f"Pemakaian = {h['rasio_pakai']*100:.0f}% dari km wajar: "
                f"Rendah {mp['Rendah']:.2f} · Normal {mp['Normal']:.2f} · Tinggi {mp['Tinggi']:.2f}"
                f"{catatan}"))
            penilaian = status

        # ----- BAGIAN C (4): simpan ke tabel riwayat -----
        self.tree.insert("", 0, values=(f"{self.combo_merek.get()} {model_tipe}", tahun,
                                        f"{int(km):,}".replace(",", "."), rupiah(harga), penilaian))


if __name__ == "__main__":
    root = tk.Tk()
    VehicleApp(root)
    root.mainloop()