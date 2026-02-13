# Gait and Balance Streamlined Analysis

A clinical balance and gait analysis research repository evaluating the feasibility and efficacy of the **GnB (Gait & Balance) smartphone-based balance assessment app** in a clinical rehabilitation setting.

## 📊 Study Overview

- **Duration:** 21 November 2024 – 4 April 2025 (134 days)
- **Participants:** 23 older adults in a clinical balance program
- **Data Collection:** iPhone/Android accelerometer and gyroscope sensors
- **Psychological Assessment:** DASS-21 (Depression, Anxiety, Stress Scale)

## 🎯 Key Findings

### Feasibility Success (All 5 Thresholds Met ✓)
- **Recruitment:** 23 enrolled (100% within 30 days)
- **Retention:** 61% retained ≥60 days (target: ≥50%)
- **Adherence:** 57% completed ≥6 sessions (target: ≥50%)
- **Sensor Data Completeness:** 100%
- **Questionnaire Completeness:** 100%

### Primary Clinical Outcomes
- **Walking Speed:** Significant improvement (d=0.59, p=0.030; 78% of participants improved)
- **Step Time:** Trending improvement (d=-0.48, p=0.084; 67% improved)
- **Static Stability:** Unchanged (negligible effects)
- **Psychological Correlations:**
  - Higher stress/anxiety → shorter steps, slower walking speed
  - Higher depression → stiffer postural control

## 📁 Repository Structure

```
├── Pipeline_Streamlined/       # Main MATLAB sensor processing pipeline
│   ├── run_gb_pipeline.m       # Core data processing script
│   └── README.md               # Pipeline documentation
│
├── Feasibility_Analysis/       # Feasibility study analysis (Python)
│   ├── run_feasibility_analysis.py
│   └── feasibility_methods_results.md
│
├── Primary_Analysis/           # Clinical outcomes analysis (Python)
│   ├── run_primary_analysis.py
│   └── primary_methods_results.md
│
├── Pipeline/                   # Legacy MATLAB outcome functions
│   ├── AGTRCT_StaticBalanceOutcomes.m
│   ├── AGTRCT_ComfortableGaitOutcomes.m
│   └── AGTRCT_appDataTable.m
│
├── Raw_Sensor_Data/           # Raw CSV sensor files (gitignored)
├── Outcome_Data/              # Processed outcome summaries (gitignored)
└── Outcome_Data_Replicated/   # Validation outputs (gitignored)
```

## 🔬 Technologies

| Technology | Usage |
|-----------|-------|
| **MATLAB** | Sensor signal processing (SincMotion library), balance/gait algorithms |
| **Python 3** | Statistical analysis, visualization, report generation |
| **C/C++** | SincMotion wavelib dependency (MEX compilation) |
| **CSV** | Data interchange format |

## 🚀 Getting Started

### Prerequisites

**MATLAB Requirements:**
- MATLAB R2018a or later
- Signal Processing Toolbox
- SincMotion library (included in project dependencies)
- wavelib MEX functions compiled

**Python Requirements:**
- Python 3.8+
- pandas, scipy, matplotlib, seaborn
- Install via: `pip install pandas scipy matplotlib seaborn`

### Running the Pipeline

#### 1. Process Raw Sensor Data → Outcomes
```matlab
% In MATLAB, navigate to Pipeline_Streamlined/
cd Pipeline_Streamlined
run_gb_pipeline()

% Or with custom options:
run_gb_pipeline('path/to/Raw_Sensor_Data', 'path/to/Outcome_Data', struct(...
    'fs', 100, ...                    % Sampling frequency (Hz)
    'isAndroid', 0, ...               % 0=iOS, 1=Android
    'heightsFile', '../Participant HeightWeight.xlsx', ...
    'fillMissingHeightWithMean', true))
```

**Output:** Per-participant CSV files with balance and gait metrics

#### 2. Run Feasibility Analysis
```python
# In Python, navigate to Feasibility_Analysis/
cd Feasibility_Analysis
python run_feasibility_analysis.py
```

**Output:** 
- 10 publication-quality figures (recruitment, retention, adherence, completeness)
- 15+ CSV summary tables

#### 3. Run Primary Clinical Analysis
```python
# In Python, navigate to Primary_Analysis/
cd Primary_Analysis
python run_primary_analysis.py
```

**Output:**
- 10 figures (longitudinal trajectories, effect sizes, DASS correlations)
- 6 CSV statistical summaries

## 📈 Metrics Computed

### Balance Metrics (Static Tasks)
- **Postural Stability:** Overall, Medial-Lateral (ML), Anterior-Posterior (AP)
- **Tasks:** Firm Eyes Open/Closed, Compliant Surface Eyes Open/Closed

### Gait Metrics (Dynamic Tasks)
- **Walking Balance** (percentage)
- **Step Length** (cm)
- **Step Time** (seconds)
- **Walking Speed** (cm/s)
- **Tasks:** Walk Heel-to-Toe, Walk Head Turns

## 📄 Data Format

### Input: Raw Sensor Data
```
Raw_Sensor_Data/
  ├── John Smith Test set 1 on 21-01-2025 Firm EO.csv
  ├── John Smith Test set 1 on 21-01-2025 Firm EC.csv
  └── ... (6 tasks per session)
```

### Output: Processed Outcomes
```
Outcome_Data/
  └── Outcomes for John Smith.csv
```

**Columns:** Session, TestSet, Date, FirmEO_Stability, FirmEC_Stability, WalkHF_Speed, ...

## 🔐 Data Privacy

**Participant data is excluded from version control:**
- ✅ All `Outcome_Data/` files ignored
- ✅ All `Raw_Sensor_Data/` files ignored
- ✅ All `.xlsx` participant data files ignored

Only analysis scripts and documentation are version controlled.

## 📚 Citation

If you use this code or methodology, please cite:

```bibtex
@article{gnb-feasibility-2025,
  title={Feasibility of Smartphone-Based Balance Assessment in Clinical Rehabilitation},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  note={Manuscript in preparation}
}
```

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact the research team.

## 📜 License

[Specify license - e.g., MIT, GPL-3.0, or proprietary]

---

**Status:** Active research project | Last updated: February 2026
