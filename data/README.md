# Data

## 1. Data Source

This project analyzes historical luxury wristwatch auction sales collected through the ALT/FNData API.

The dataset includes completed auction sales for three luxury watch brands—Rolex, Cartier, and Omega—at Sotheby's and Bonhams.

The data was collected using capped, sale-date-ordered API requests. Consequently, the sample is not a random representation of the entire luxury watch market.

**Source:** ALT/FNData API

**Redistribution status:** Public redistribution rights have not yet been confirmed. Therefore, the dataset is not committed to GitHub or distributed through a public GitHub release.

## 2. Data Access and Distribution

The dataset is available to authorized project team members through the team's shared OneDrive folder.

To obtain the dataset:

1. Clone the project repository: `https://github.com/lflanigan-sudo/aii600-final-project`
2. Access the team's shared OneDrive folder using an authorized account.
3. Navigate to the `processed` folder.
4. Download `expanded_pull_eligible.csv`.
5. Place the file in the local project's `data/` directory.

The expected local file path is:

`data/expanded_pull_eligible.csv`

Access to the shared OneDrive folder requires team authorization. No API key is required to read the downloaded CSV locally. The team must confirm that its data-sharing arrangement complies with the provider's licensing terms.

The dataset is not published through a GitHub release.

## 3. Dataset Description

| Property | Description |
|---|---|
| Filename | `expanded_pull_eligible.csv` |
| Local file path | `data/expanded_pull_eligible.csv` |
| Source | ALT/FNData API |
| Auction houses | Sotheby's and Bonhams |
| Brands | Rolex, Cartier, Omega |
| Unit of observation | One eligible individual wristwatch auction sale |
| Number of observations | 4,664 |
| Number of columns | 212 |
| Observation period | February 5, 2021 – July 28, 2026 |
| Target variable | `usd_price_decimal_num` |
| Target units | USD |

### Observations by Brand

| Brand | Number of Sales |
|---|---:|
| Rolex | 1,786 |
| Cartier | 1,594 |
| Omega | 1,284 |
| **Total** | **4,664** |

## 4. SHA-256 Checksum

To ensure reproducibility, the dataset is identified by its SHA-256 checksum.

**File:** `expanded_pull_eligible.csv`

**SHA-256:**

`cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a`

The notebook verifies this checksum before fitting prediction models.

The verification code in `analysis.ipynb` is:

```python
import hashlib
from pathlib import Path

data_path = Path("data/expanded_pull_eligible.csv")

expected_sha256 = "cf3b41d890af1feba9348fa9a73d954fcab193ed890a39044bd0355198fc5a8a"

actual_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()

assert actual_sha256 == expected_sha256, "Dataset checksum mismatch!"

print("Dataset SHA-256 verified successfully.")
```

If the checksum does not match, the team will investigate the difference before modeling.

If the dataset is intentionally updated, the team will recalculate the checksum, document the changes, and update this README in the same Git commit.

## 5. Loading the Dataset

The notebook reads the dataset from the local `data/` folder using Python and pandas.

```python
import pandas as pd
from pathlib import Path

data_path = Path("data/expanded_pull_eligible.csv")

df = pd.read_csv(data_path, low_memory=False)

print("Rows:", len(df))
print("Columns:", len(df.columns))
```

For the documented dataset version, the expected dimensions are 4,664 rows and 212 columns.

## 6. Data Quality and Limitations

The dataset includes normalized watch specifications and data-quality indicators. Some candidate predictors contain missing values.

| Variable | Missing Values |
|---|---:|
| Case material | 39.11% |
| Movement | 28.24% |
| Model | 12.03% |
| Production year | 7.42% |
| Brand | 0.00% |
| Sale date | 0.00% |
| Auction sale price | 0.00% |

The following limitations are relevant to the analysis:

- **Sampling:** Capped, sale-date-ordered API retrieval means the observations are not a random sample of all luxury watch sales.
- **Coverage:** The dataset covers only three brands and two auction houses.
- **Missing information:** Some watch characteristics are incomplete, potentially limiting prediction accuracy.
- **Price interpretation:** Recorded auction prices may differ from total acquisition or resale costs, which can include buyer's premiums, taxes, shipping, servicing, and resale fees. The precise price definition requires confirmation.
- **Time period:** The dataset covers historical auction sales from February 2021 through July 2026. Auction prices may change over time due to market conditions, collector demand, and economic factors. Therefore, models trained on historical data may not predict future auction prices with the same accuracy.

## 7. Description-Based Data Cleaning and Quality Audit

### 7.1 Objective

To improve data completeness for luxury-watch auction price prediction, we developed a Python cleaning process that uses information from auction catalogue descriptions to recover missing watch specifications.

The process preserves the original dataset and creates a separate cleaned CSV. Existing normalized values are retained, and missing values are populated only when recognizable supporting information is available in `norm_item_description`.

### 7.2 Cleaning Methodology

The original dataset contains 4,664 auction records and 212 columns, including 66 normalized columns identified by the `norm_` prefix.

The cleaning process:

1. Verifies the original CSV against its approved SHA-256 checksum.
2. Preserves all 212 original columns.
3. Creates 66 additional columns using the naming convention `norm_<field>_cleaned`.
4. Copies existing normalized values into the corresponding cleaned columns.
5. Attempts to recover missing specifications from `norm_item_description` using rule-based text extraction.
6. Does not overwrite existing populated normalized values.
7. Saves the results to a new CSV without modifying the original dataset.

The extraction targets explicitly identifiable specifications, including watch model, reference number, movement, production year, dial information, case descriptions, and bracelet or strap information.

Ambiguous or unsupported information is not intentionally imputed.

### 7.3 Cleaned Dataset

| Property | Original Dataset | Cleaned Dataset |
|---|---|---|
| Filename | `expanded_pull_eligible.csv` | `expanded_pull_eligible_cleaned.csv` |
| Local directory | `data/` | `data/` |
| Records | 4,664 | 4,664 |
| Original columns | 212 | 212 |
| Additional cleaned columns | 0 | 66 |
| Total columns | 212 | 278 |
| Original normalized values overwritten | — | 0 |

The cleaned dataset retains the original information and adds separate cleaned fields for analysis. The original file remains the approved source dataset.


### 7.4 Missing-Data Improvements

The revised conservative cleaning procedure recovered 1,734 missing values across six normalized fields.

| Normalized Field | Missing Before | Missing After | Values Recovered |
|---|---:|---:|---:|
| `norm_dial` | 3,123 | 2,473 | 650 |
| `norm_watch_case` | 3,123 | 2,555 | 568 |
| `norm_reference` | 1,192 | 947 | 245 |
| `norm_strap_bracelet` | 3,146 | 2,930 | 216 |
| `norm_dial_color` | 1,766 | 1,759 | 7 |
| `norm_model` | 561 | 513 | 48 |
| **Total** | **12,911** | **11,177** | **1,734** |

The six improved fields experienced a combined 13.43% relative reduction in missing values.

Of 4,664 auction records, 664 (14.24%) received at least one newly populated field.

Across all 66 normalized fields, missingness decreased by 1.35%.

These results describe improved data completeness, not demonstrated improvements in predictive accuracy. The new enrichment fields are not used in the current fitted regression.



### 7.5 Data Integrity and Quality Review

The revised cleaning process was evaluated using the quality-audit section in `analysis.ipynb`.

The audit confirmed:

- All 4,664 original records were preserved.
- All 212 original columns were preserved without modification.
- All 667 literal `None` values in `item_provenance` were preserved.
- No existing normalized values were overwritten.
- The candidate contains 66 additional cleaned columns, for 278 columns total.
- The saved candidate file was checked against its generated SHA-256 checksum.

The extraction rules were made more conservative to reduce the risk of confusing dial colors with hands or markers and primary case materials with bezel or case-back materials.

The seven newly recovered dial colors and the recovered case-material values still require manual semantic validation. Ambiguous extractions should remain missing.

The cleaned dataset remains a candidate and is not used in the current fitted regression.


### 7.6 Modeling Implications

The additional cleaned columns provide candidate predictors for the watch-price prediction models. They are not automatically included in model training.

Predictor selection will consider information available before the auction purchase decision, missingness, extraction reliability, and validation performance.

The realized auction sale price remains the prediction target and is not used as an input predictor.

The team will evaluate whether the additional cleaned specifications improve predictive performance relative to a simpler baseline. Any statistical imputation or preprocessing parameters learned from data will be fitted on the training partition only.

### 7.7 Data Confidentiality and Reproducibility

The original and cleaned datasets are proprietary project data and are not distributed through the public GitHub repository.

Both files are maintained in the authorized project storage environment.

The notebook displays aggregate cleaning statistics and charts rather than individual auction records or catalogue descriptions.

The original dataset is verified using its approved SHA-256 fingerprint. The cleaned CSV receives its own SHA-256 fingerprint after generation.

The separate candidate file is
`expanded_pull_eligible_cleaned.csv`.

**Candidate SHA-256:**

`74bf8efd5279afc4b59013618063c78292c166c491b922bdaf3db6e347ac9824`

This fingerprint identifies the revised candidate generated and audited in the notebook. It does not indicate that the enrichment has been approved for modeling.
