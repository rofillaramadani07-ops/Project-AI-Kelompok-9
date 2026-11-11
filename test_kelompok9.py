"""Pengujian otomatis KELOMPOK 9 (unittest, tanpa jendela GUI).
Jalankan:  python test_kelompok9.py
Berkas program harus berada di folder yang sama (KELOMPOK_9.py atau KELOMPOK_9_PERBAIKAN.py)."""
import importlib.util, os, sys, tempfile, time, types, unittest
from unittest import mock

try:
    import tkinter  # noqa
except ImportError:                                   # lingkungan tanpa Tkinter -> stub
    tk = types.ModuleType("tkinter"); tk.ttk = types.ModuleType("tkinter.ttk")
    tk.messagebox = types.ModuleType("tkinter.messagebox")
    sys.modules.update({"tkinter": tk, "tkinter.ttk": tk.ttk, "tkinter.messagebox": tk.messagebox})

HERE = os.path.dirname(os.path.abspath(__file__))
for nama in ("KELOMPOK_9.py", "KELOMPOK_9_PERBAIKAN.py"):
    path = os.path.join(HERE, nama)
    if os.path.exists(path):
        break
spec = importlib.util.spec_from_file_location("app", path)
m = importlib.util.module_from_spec(spec)
os.chdir(tempfile.mkdtemp())                          # CSV uji dibuat di folder sementara
spec.loader.exec_module(m)
ML = m.VehiclePriceModel()


class Var:
    def __init__(self, v=""): self.v = v
    def get(self): return self.v
    def set(self, v): self.v = v

class Widget:
    def __init__(self, v=""): self.v = v; self.cfg = {}
    def get(self): return self.v
    def config(self, **kw): self.cfg.update(kw)

class Tree:
    def __init__(self): self.rows = []
    def insert(self, parent, idx, values): self.rows.insert(0, values)

def make_app(kat="Motor", merek="Honda", model="Scoopy Stylish", tahun="2022", km="2000", tawaran=""):
    a = m.VehicleApp.__new__(m.VehicleApp)
    a.ml, a.fuzzy = ML, m.FuzzyPenilaianHarga()
    a.build_catalog()
    a.combo_kat, a.combo_merek, a.combo_model = Widget(kat), Widget(merek), Widget(model)
    a.var_tahun, a.var_km, a.var_tawaran = Var(tahun), Var(km), Var(tawaran)
    for n in ("lbl_model", "lbl_price", "lbl_range", "lbl_fuzzy", "lbl_fuzzy_detail"):
        setattr(a, n, Widget())
    a.tree = Tree()
    return a

def jalankan(app):
    with mock.patch.object(m.messagebox, "showerror", create=True) as e, \
         mock.patch.object(m.messagebox, "showwarning", create=True) as w:
        app.predict_price()
    return e.called, w.called


class TestParsing(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(m.parse_angka("150.000.000"), 150000000)
        self.assertEqual(m.parse_angka("Rp 5"), 5)
        self.assertEqual(m.parse_angka("0"), 0)
    def test_invalid(self):
        for t in ("²", "1e6", "12,5", "abc", "  ", "-5", "1.5e3"):
            self.assertIsNone(m.parse_angka(t), t)


class TestDataset(unittest.TestCase):
    def test_dataset_valid(self):
        d = ML.df
        self.assertEqual(len(d), 780)
        self.assertEqual(int((d.Harga > d.Harga_Baru).sum()), 0)
        self.assertEqual(int(d.isna().sum().sum()), 0)
        self.assertEqual(set(d.Model), {c[2] for c in m.CATALOG})
    def test_csv_lama_tidak_valid_dibuat_ulang(self):
        import pandas as pd
        bad = ML.df.copy(); bad.loc[0, "Harga"] = bad.loc[0, "Harga_Baru"] + 1
        bad.to_csv(m.CSV_FILE, index=False)
        ulang = m.VehiclePriceModel()
        self.assertEqual(int((ulang.df.Harga > ulang.df.Harga_Baru).sum()), 0)


class TestModel(unittest.TestCase):
    def test_metrik_wajar(self):
        self.assertLess(ML.mape, 0.05); self.assertGreater(ML.r2, 0.95)
    def test_prediksi_masuk_akal(self):
        for kat, baru, th, km in [("Motor", 20.5, 2022, 2000), ("Mobil", 400.0, 2018, 100000),
                                  ("Motor", 20.5, 2026, 500), ("Motor", 14.0, 2008, 150000)]:
            p = ML.predict(kat, baru, th, km)
            self.assertGreater(p, 0); self.assertLessEqual(p, baru)
    def test_km_naik_harga_turun(self):
        v = [ML.predict("Motor", 20.5, 2022, k) for k in range(0, 150001, 10000)]
        self.assertTrue(all(a >= b for a, b in zip(v, v[1:])))
    def test_waktu_pelatihan(self):
        t = time.time(); m.VehiclePriceModel(); self.assertLess(time.time() - t, 10)


class TestFuzzy(unittest.TestCase):
    fz = m.FuzzyPenilaianHarga()
    def test_keanggotaan_berjumlah_satu(self):
        for i in range(0, 301):
            x = i / 100
            self.assertAlmostEqual(sum(self.fz.fuzzifikasi_harga(x).values()), 1.0)
            self.assertAlmostEqual(sum(self.fz.fuzzifikasi_pakai(x).values()), 1.0)
    def test_status_dan_batas_wajar(self):
        b = m.BATAS_WAJAR
        self.assertEqual(self.fz.status_harga(1.0), "Wajar")
        self.assertEqual(self.fz.status_harga(1 + b), "Wajar")
        self.assertEqual(self.fz.status_harga(4.52), "Terlalu Mahal")
        self.assertEqual(self.fz.status_harga(0.45), "Terlalu Murah (Waspada)")
        self.assertEqual(self.fz.status_harga(0.77), "Murah")
    def test_ekstrem_tidak_error(self):
        for tw in (1e-9, 1e-6, 1e6, 1e12):
            r = self.fz.nilai(tw, 15.5, 0, 0)
            self.assertTrue(0 <= r["skor"] <= 100)
    def test_skor_dalam_rentang(self):
        for tw in range(0, 200, 5):
            for km in (0, 2000, 20000, 150000):
                s = self.fz.nilai(tw / 10, 15.5, km, 4000)["skor"]
                self.assertTrue(0 <= s <= 100)


class TestAlurPrediksi(unittest.TestCase):
    def test_normal(self):
        a = make_app(tawaran="12000000"); e, w = jalankan(a)
        self.assertFalse(e or w); self.assertEqual(len(a.tree.rows), 1)
        self.assertIn("Murah", a.tree.rows[0][-1])
    def test_tanpa_tawaran(self):
        a = make_app(); e, w = jalankan(a)
        self.assertFalse(e or w); self.assertEqual(a.tree.rows[0][-1], "-")
    def test_tawaran_satuan_ratusan_ribuan(self):
        for t in ("5", "500", "5000"):
            a = make_app(tawaran=t); e, w = jalankan(a)
            self.assertFalse(e or w, t); self.assertIn("Terlalu Murah", a.tree.rows[0][-1])
    def test_tawaran_sangat_besar(self):
        a = make_app(tawaran="9" * 30); e, w = jalankan(a)
        self.assertFalse(e or w); self.assertEqual(a.tree.rows[0][-1], "Terlalu Mahal")
    def test_input_tidak_valid_ditolak_tanpa_crash(self):
        kasus = [dict(tawaran="²"), dict(tawaran="abc"), dict(tawaran="0"),
                 dict(km="abc"), dict(km="-5"), dict(km="2000.5,3"), dict(km="150001"),
                 dict(tahun=""), dict(tahun="abc"), dict(tahun="2022.9"),
                 dict(tahun="2020"), dict(tahun="2027")]          # Scoopy rilis 2021
        for k in kasus:
            a = make_app(**k); e, w = jalankan(a)
            self.assertTrue(e or w, k); self.assertEqual(len(a.tree.rows), 0, k)
    def test_km_kosong_dianggap_nol(self):
        a = make_app(km=""); e, w = jalankan(a)
        self.assertFalse(e or w); self.assertEqual(len(a.tree.rows), 1)
    def test_batas_km(self):
        a = make_app(km="150.000"); e, w = jalankan(a); self.assertFalse(e or w)
    def test_semua_kendaraan_katalog(self):
        for kat, merek, model, rilis, *_ in m.CATALOG:
            for th in (rilis, m.TAHUN_SEKARANG):
                a = make_app(kat, merek, model, str(th), "1000", "1000000"); e, w = jalankan(a)
                self.assertFalse(e or w, (model, th)); self.assertEqual(len(a.tree.rows), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
