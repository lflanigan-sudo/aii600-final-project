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

`008790f43b08461b27b01cd496d5d2b5c7e8c5214bb609bf5a53579aec21109c`

The notebook verifies this checksum before fitting prediction models.

The verification code in `analysis.ipynb` is:

```python
import hashlib
from pathlib import Path

data_path = Path("data/expanded_pull_eligible.csv")

expected_sha256 = "008790f43b08461b27b01cd496d5d2b5c7e8c5214bb609bf5a53579aec21109c"

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
