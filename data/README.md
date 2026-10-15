# Data Directory

## Dataset: UCI Heart Disease (Cleveland Subset)

### Source
- **Name:** Heart Disease  
- **UCI Dataset ID:** 45  
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease  
- **Donated:** 30 June 1988  
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
- **Introductory Paper:** Detrano R. et al., *International application of a new probability algorithm
  for the diagnosis of coronary artery disease*, American Journal of Cardiology, 1989.

### Download

The dataset is fetched automatically at runtime via the `ucimlrepo` Python package:

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=45)
```

No manual download is required. The raw data is **not committed** to this repository.

### Dataset Details

| Property          | Value                             |
|-------------------|-----------------------------------|
| Subset used       | Cleveland only                    |
| Instances         | 303                               |
| Input features    | 13                                |
| Target variable   | `num` (angiographic disease status) |
| Missing values    | Yes — `ca` and `thal` columns     |

### Feature Schema

| Feature    | Type        | Description                                              | Units   |
|------------|-------------|----------------------------------------------------------|---------|
| `age`      | Integer     | Age in years                                             | years   |
| `sex`      | Categorical | 1 = male, 0 = female                                     | —       |
| `cp`       | Categorical | Chest pain type (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic) | — |
| `trestbps` | Integer     | Resting blood pressure on hospital admission             | mm Hg   |
| `chol`     | Integer     | Serum cholesterol                                        | mg/dl   |
| `fbs`      | Categorical | Fasting blood sugar > 120 mg/dl (1=true, 0=false)        | —       |
| `restecg`  | Categorical | Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy) | — |
| `thalach`  | Integer     | Maximum heart rate achieved                              | bpm     |
| `exang`    | Categorical | Exercise-induced angina (1=yes, 0=no)                    | —       |
| `oldpeak`  | Float       | ST depression induced by exercise relative to rest       | —       |
| `slope`    | Categorical | Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping) | — |
| `ca`       | Integer     | Number of major vessels coloured by fluoroscopy (0–3)    | —       |
| `thal`     | Categorical | Thalassemia (3=normal, 6=fixed defect, 7=reversable defect) | —   |

### Binary Target Mapping

The raw target `num` takes values 0–4 (number of vessels with >50% narrowing).
It is mapped to a binary classification target as follows:

| Raw `num` | Binary label | Meaning                                  |
|-----------|-------------|------------------------------------------|
| 0         | **0**       | No disease — < 50% diameter narrowing   |
| 1, 2, 3, 4 | **1**     | Disease present — ≥ 50% narrowing in ≥ 1 vessel |

**Rationale:** This mapping is the standard protocol for all published Cleveland-subset experiments,
following Detrano et al. (1989). Values 1–4 are not ordinally meaningful for a binary classifier.
The mapping is explicit and documented — not a silent decision.

### Privacy

Patient names and social security numbers were removed from the original database and replaced
with dummy values before public release.
