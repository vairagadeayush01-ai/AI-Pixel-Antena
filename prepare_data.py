"""
Prepare deployment data: filter CSVs to keep only the 'New in-band' model.
Run once before deploying.
"""
import pandas as pd, pathlib, json

SRC = pathlib.Path(__file__).resolve().parent.parent  # inband_finetune_results
DST = pathlib.Path(__file__).resolve().parent / "data"

# --- inverse_design_em_targets.csv  (keep only model=="new") ---
df = pd.read_csv(SRC / "inverse_design_em_targets.csv")
df = df[df["model"] == "new"].drop(columns=["model"])
df.to_csv(DST / "inverse_design_em_targets.csv", index=False)
print(f"em_targets: {len(df)} rows")

# --- inverse_design_em_spectra.csv (keep only new_* columns + Frequency_GHz) ---
sp = pd.read_csv(SRC / "inverse_design_em_spectra.csv")
keep = ["Frequency_GHz"] + [c for c in sp.columns if c.startswith("new_")]
sp = sp[keep]
# rename new_13.5GHz -> 13.5GHz
sp.columns = [c.replace("new_", "") for c in sp.columns]
sp.to_csv(DST / "inverse_design_em_spectra.csv", index=False)
print(f"em_spectra: {sp.shape}")

# --- full_test_metrics.csv (keep only New in-band) ---
ft = pd.read_csv(SRC / "full_test_metrics.csv")
ft = ft[ft.iloc[:, 0] == "New in-band"]
ft.to_csv(DST / "full_test_metrics.csv", index=False)
print(f"full_test: {len(ft)} rows")

# --- failure_types (keep only New in-band) ---
for fname in ["failure_types_in_band.csv", "failure_types_outside_band.csv"]:
    df = pd.read_csv(SRC / fname)
    df = df[df.iloc[:, 0] == "New in-band"]
    df.to_csv(DST / fname, index=False)
    print(f"{fname}: {len(df)} rows")

# --- per_bin_comparison.csv (keep only new-model columns) ---
pb = pd.read_csv(SRC / "per_bin_comparison.csv")
keep_cols = ["bin_start_GHz", "n_test", "New in-band success_%", "New in-band median_res_err_GHz"]
pb = pb[keep_cols]
pb.columns = ["bin_start_GHz", "n_test", "success_%", "median_res_err_GHz"]
pb.to_csv(DST / "per_bin_performance.csv", index=False)
print(f"per_bin: {len(pb)} rows")

# --- EM validation results (keep only model=="new") ---
em_path = SRC / "v4_validation" / "EM_validation_results.csv"
if em_path.exists():
    em = pd.read_csv(em_path)
    em = em[em["Model"] == "new"].drop(columns=["Model"])
    em.to_csv(DST / "em_validation_results.csv", index=False)
    print(f"em_validation: {len(em)} rows")

# --- model_comparison.csv (keep only New in-band) ---
mc = pd.read_csv(SRC / "model_comparison.csv")
mc = mc[mc["model"] == "New in-band"].drop(columns=["model"])
mc.to_csv(DST / "model_metrics.csv", index=False)
print(f"model_metrics: {len(mc)} rows")

print("\n✅ All deployment data prepared!")
