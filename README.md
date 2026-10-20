# AI Pixel Antenna — Inverse Design Dashboard

ML-driven inverse design of 12×12 pixel antennas using an in-band fine-tuned model (13–17 GHz).

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure

```
deploy/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # Theme configuration
├── data/                   # All CSV/JSON data files
│   ├── summary.json
│   ├── training_history.csv
│   ├── bin_stats.csv
│   ├── hist_resonance.csv
│   ├── subset_bin_counts.csv
│   ├── inverse_design_em_targets.csv
│   ├── inverse_design_em_spectra.csv
│   ├── per_bin_performance.csv
│   ├── full_test_metrics.csv
│   ├── model_metrics.csv
│   ├── failure_types_in_band.csv
│   ├── failure_types_outside_band.csv
│   └── em_validation_results.csv
├── models/                 # Model checkpoints (Git LFS)
│   └── inverse_inband_finetuned.pth (21 MB)
├── masks/                  # 12×12 antenna mask CSVs
│   └── antenna_mask_*.csv
└── images/                 # Visualization PNGs
    ├── *.png
    └── em_validation/
        └── *.png
```

## 🌐 Deploy to Streamlit Cloud

1. Clone or connect repository `https://github.com/vairagadeayush01-ai/AI-Pixel-Antena`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo and select `app.py`
