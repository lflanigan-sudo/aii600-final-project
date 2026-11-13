"""One small implicit quantile network for the frozen watch-price benchmark.

The linear q33 model is refit only as a check. This script does not write
model_q33, validation_predictions, or the reserved test partition.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from scripts.predictors import INPUT_SHA256, construct_record

SEED = 20261024
TAU = 1 / 3
FROZEN_LINEAR_LOSS = 3975.7439
N_COSINE = 16
MAX_EPOCHS = 300
PATIENCE = 40
TAUS_PER_ROW = 8
FEATURE_COLS = ["brand", "norm_model", "case_body", "movement", "age_at_sale"]
CATEGORICAL = ["brand", "norm_model", "case_body", "movement"]


def pinball_loss(actual, predicted, tau):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    return float(np.mean(np.maximum(tau * error, (tau - 1) * error)))


def load_partitions(data_root):
    eligible_path = data_root / "expanded_pull_eligible.csv"
    assignment_path = data_root / "chronological_split_assignments.csv"
    digest = hashlib.sha256(eligible_path.read_bytes()).hexdigest()
    if digest != INPUT_SHA256:
        raise RuntimeError("Eligible file checksum does not match the approved file.")
    with eligible_path.open(newline="", encoding="utf-8-sig") as handle:
        eligible_rows = list(csv.DictReader(handle))
    with assignment_path.open(newline="", encoding="utf-8-sig") as handle:
        assignments = list(csv.DictReader(handle))
    eligible_by_id = {row["record_id"]: row for row in eligible_rows}
    rows = []
    predictors = {}
    for assignment in assignments:
        if assignment["partition"] == "test":
            continue
        record_id = assignment["record_id"]
        source = eligible_by_id[record_id]
        if source["designer"] != assignment["brand"]:
            raise RuntimeError(f"Brand mismatch for record {record_id}")
        predictor = construct_record(source)
        predictors[record_id] = predictor
        rows.append(
            {
                "record_id": record_id,
                "brand": assignment["brand"],
                "partition": assignment["partition"],
                "usd_price": float(source["usd_price_decimal_num"]),
            }
        )
    frame = pd.DataFrame(rows)
    train = frame.loc[frame["partition"].eq("train")].reset_index(drop=True)
    validation = frame.loc[frame["partition"].eq("validation")].reset_index(drop=True)
    if len(train) != 2490 or len(validation) != 802:
        raise RuntimeError("Expected 2,490 training rows and 802 validation rows.")
    if set(train["record_id"]) & set(validation["record_id"]):
        raise RuntimeError("Training and validation overlap.")
    return train, validation, predictors


def assemble_features(partition, predictors):
    rows = []
    for record_id, brand in zip(partition["record_id"], partition["brand"]):
        predictor = predictors[record_id]
        row = {col: predictor.get(col) for col in FEATURE_COLS if col != "brand"}
        row["brand"] = brand
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in CATEGORICAL:
        out[col] = out[col].map(
            lambda value: "Missing"
            if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""
            else str(value).strip()
        )
    out["age_at_sale"] = pd.to_numeric(out["age_at_sale"], errors="coerce")
    out.loc[out["age_at_sale"] < 0, "age_at_sale"] = np.nan
    return out


def collapse_rare_models(x_train, x_val):
    counts = x_train["norm_model"].value_counts()
    frequent = set(counts[counts >= 15].index)
    for frame in (x_train, x_val):
        frame["norm_model"] = frame["norm_model"].where(frame["norm_model"].isin(frequent), "Other")
    return frequent


def fit_frozen_linear(x_train, x_val, y_train, y_val):
    """Refit the approved linear q33 specification as a check, not a replacement."""
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("age", SimpleImputer(strategy="median", add_indicator=True), ["age_at_sale"]),
        ]
    )
    model = Pipeline(
        [
            ("features", preprocessor),
            ("quantile", QuantileRegressor(quantile=TAU, alpha=0.01, solver="highs")),
        ]
    )
    model.fit(x_train, np.log(y_train))
    prediction = np.exp(model.predict(x_val))
    loss = pinball_loss(y_val, prediction, TAU)
    if abs(loss - FROZEN_LINEAR_LOSS) > 0.02:
        raise RuntimeError(
            f"Recomputed linear q33 loss {loss:.4f} does not match the frozen score {FROZEN_LINEAR_LOSS:.4f}."
        )
    return loss, prediction


class FeatureCoder:
    def fit(self, frame):
        self.levels = {col: sorted(frame[col].unique()) for col in CATEGORICAL}
        observed = frame["age_at_sale"].dropna()
        self.age_median = float(observed.median())
        imputed = frame["age_at_sale"].fillna(self.age_median).to_numpy(dtype=float)
        self.age_mean = float(imputed.mean())
        self.age_scale = float(imputed.std(ddof=0))
        if self.age_scale <= 0:
            raise RuntimeError("Training age has no variation after imputation.")
        self.columns = []
        for col in CATEGORICAL:
            self.columns.extend(f"{col}={level}" for level in self.levels[col])
        self.columns.extend(["age_at_sale_z", "age_missing"])
        return self

    def transform(self, frame):
        pieces = []
        for col in CATEGORICAL:
            values = frame[col].to_numpy()
            for level in self.levels[col]:
                pieces.append((values == level).astype(float))
        age = frame["age_at_sale"].to_numpy(dtype=float)
        missing = np.isnan(age).astype(float)
        filled = np.where(np.isnan(age), self.age_median, age)
        pieces.append((filled - self.age_mean) / self.age_scale)
        pieces.append(missing)
        return np.column_stack(pieces)


def initial_params(rng, n_inputs, width, log_price_mean):
    def he(fan_in, fan_out):
        return rng.normal(0, np.sqrt(2 / fan_in), size=(fan_in, fan_out))

    return {
        "W_x": he(n_inputs, width),
        "b_x": np.zeros(width),
        "W_tau": he(N_COSINE, width),
        "b_tau": np.zeros(width),
        "W1": he(width, width),
        "b1": np.zeros(width),
        "W2": rng.normal(0, 0.01, size=width),
        "b2": np.array(log_price_mean),
    }


def cosine_basis(tau):
    frequencies = np.arange(1, N_COSINE + 1)
    return np.cos(np.pi * tau[:, None] * frequencies[None, :])


def forward(x, tau, params):
    cosine = cosine_basis(tau)
    z_tau = cosine @ params["W_tau"] + params["b_tau"]
    psi = np.maximum(z_tau, 0)
    z_x = x @ params["W_x"] + params["b_x"]
    hx = np.maximum(z_x, 0)
    hidden = hx * psi
    z2 = hidden @ params["W1"] + params["b1"]
    h2 = np.maximum(z2, 0)
    log_q = h2 @ params["W2"] + params["b2"]
    cache = (x, cosine, z_tau, psi, z_x, hx, hidden, z2, h2)
    return log_q, cache


def backward(cache, params, dlog_q):
    x, cosine, z_tau, psi, z_x, hx, hidden, z2, h2 = cache
    grads = {
        "W2": h2.T @ dlog_q,
        "b2": np.array(dlog_q.sum()),
    }
    dh2 = dlog_q[:, None] * params["W2"][None, :]
    dz2 = dh2 * (z2 > 0)
    grads["W1"] = hidden.T @ dz2
    grads["b1"] = dz2.sum(axis=0)
    dhidden = dz2 @ params["W1"].T
    dhx = dhidden * psi
    dpsi = dhidden * hx
    dz_x = dhx * (z_x > 0)
    grads["W_x"] = x.T @ dz_x
    grads["b_x"] = dz_x.sum(axis=0)
    dz_tau = dpsi * (z_tau > 0)
    grads["W_tau"] = cosine.T @ dz_tau
    grads["b_tau"] = dz_tau.sum(axis=0)
    return grads


def dollar_grad(y, q, tau):
    # Mean dollar pinball, through the log-price output.
    d_q = (y < q).astype(float) - tau
    return d_q * q / len(y)


def predict_usd(x, tau, params):
    tau_values = np.full(len(x), tau, dtype=float)
    log_q, _ = forward(x, tau_values, params)
    return np.exp(log_q)


def check_gradient(rng):
    x = rng.normal(size=(4, 6))
    tau = np.full(4, TAU)
    y = np.exp(rng.normal(size=4))
    params = initial_params(rng, 6, 3, 0.0)
    log_q, cache = forward(x, tau, params)
    q = np.exp(log_q)
    base = pinball_loss(y, q, TAU)
    analytic = backward(cache, params, dollar_grad(y, q, tau))
    step = 1e-5
    numerical = np.zeros_like(params["W2"])
    for index in range(len(params["W2"])):
        params["W2"][index] += step
        nudged, _ = forward(x, tau, params)
        numerical[index] = (pinball_loss(y, np.exp(nudged), TAU) - base) / step
        params["W2"][index] -= step
    scale = max(np.max(np.abs(analytic["W2"])), 1e-8)
    if np.max(np.abs(analytic["W2"] - numerical)) / scale > 1e-4:
        raise RuntimeError("IQN gradient check failed.")


def train_one(x_train, y_train, x_val, y_val, width, learning_rate, weight_decay, rng):
    params = initial_params(rng, x_train.shape[1], width, float(np.mean(np.log(y_train))))
    moments = {name: np.zeros_like(value) for name, value in params.items()}
    seconds = {name: np.zeros_like(value) for name, value in params.items()}
    best_params = {name: value.copy() for name, value in params.items()}
    best_loss = np.inf
    best_epoch = 0
    wait = 0
    step = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        tau = rng.uniform(size=len(x_train) * TAUS_PER_ROW)
        x_rep = np.repeat(x_train, TAUS_PER_ROW, axis=0)
        y_rep = np.repeat(y_train, TAUS_PER_ROW)
        log_q, cache = forward(x_rep, tau, params)
        q = np.exp(log_q)
        if not np.isfinite(q).all():
            return None
        grads = backward(cache, params, dollar_grad(y_rep, q, tau))
        step += 1
        for name, value in params.items():
            grad = grads[name]
            if name.startswith("W"):
                grad = grad + weight_decay * value
            moments[name] = 0.9 * moments[name] + 0.1 * grad
            seconds[name] = 0.999 * seconds[name] + 0.001 * grad * grad
            corrected_m = moments[name] / (1 - 0.9**step)
            corrected_v = seconds[name] / (1 - 0.999**step)
            value -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        prediction = predict_usd(x_val, TAU, params)
        if not np.isfinite(prediction).all():
            return None
        loss = pinball_loss(y_val, prediction, TAU)
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_params = {name: value.copy() for name, value in params.items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return {
        "params": best_params,
        "loss": best_loss,
        "epoch": best_epoch,
        "width": width,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }


def fit_iqn(x_train, y_train, x_val, y_val):
    rng = np.random.default_rng(SEED)
    check_gradient(rng)
    runs = []
    for width in (16, 32):
        for learning_rate in (1e-3, 3e-3):
            for weight_decay in (0.0, 1e-4):
                result = train_one(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    width,
                    learning_rate,
                    weight_decay,
                    np.random.default_rng(SEED + width + int(learning_rate * 1e6) + int(weight_decay * 1e8)),
                )
                if result is None:
                    print(
                        f"width {width}, lr {learning_rate}, decay {weight_decay}: diverged"
                    )
                    continue
                runs.append(result)
                print(
                    f"width {width:2d}  lr {learning_rate:.4f}  decay {weight_decay:.1e}"
                    f"  epoch {result['epoch']:3d}  q33 pinball {result['loss']:,.4f}"
                )
    if not runs:
        raise RuntimeError("Every IQN setting diverged.")
    runs.sort(key=lambda run: (run["loss"], run["width"], -run["weight_decay"], run["learning_rate"]))
    return runs[0], runs


def representative_contrasts(coder, params, x_train):
    observed = x_train.dropna(subset=["age_at_sale"]).copy()
    rows = []
    for brand, brand_rows in observed.groupby("brand"):
        named = brand_rows.loc[brand_rows["norm_model"].ne("Other")]
        counts = (
            named.groupby(["norm_model", "case_body", "movement"], observed=True)
            .size()
            .sort_values(ascending=False)
        )
        model_name, case_body, movement = counts.index[0]
        group = brand_rows.loc[
            brand_rows["norm_model"].eq(model_name)
            & brand_rows["case_body"].eq(case_body)
            & brand_rows["movement"].eq(movement)
        ]
        age = float(group["age_at_sale"].median())
        base = {"brand": brand, "norm_model": model_name, "case_body": case_body, "movement": movement}
        base_q = prototype_q33(coder, params, base, age)
        older = prototype_q33(coder, params, base, age + 1)
        younger = prototype_q33(coder, params, base, age - 1)

        def training_count(values):
            mask = pd.Series(True, index=observed.index)
            for key, value in values.items():
                mask &= observed[key].eq(value)
            return int(mask.sum())

        rows.append(
            {
                "brand": brand,
                "prototype": f"{model_name}, {case_body}, {movement}",
                "support": int(counts.iloc[0]),
                "age": age,
                "contrast": "age, per year",
                "changed_support": int(counts.iloc[0]),
                "q33_usd": base_q,
                "change_usd": (older - younger) / 2,
            }
        )
        movement_counts = brand_rows["movement"].value_counts()
        for alternative in movement_counts.index:
            if alternative == movement:
                continue
            swapped = dict(base, movement=alternative)
            rows.append(
                {
                    "brand": brand,
                    "prototype": f"{model_name}, {case_body}, {movement}",
                    "support": int(counts.iloc[0]),
                    "age": age,
                    "contrast": f"movement -> {alternative}",
                    "changed_support": training_count(swapped),
                    "q33_usd": base_q,
                    "change_usd": prototype_q33(coder, params, swapped, age) - base_q,
                }
            )
            if len([row for row in rows if row["brand"] == brand and row["contrast"].startswith("movement")]) >= 2:
                break
        case_counts = brand_rows["case_body"].value_counts()
        for alternative in case_counts.index:
            if alternative == case_body:
                continue
            swapped = dict(base, case_body=alternative)
            rows.append(
                {
                    "brand": brand,
                    "prototype": f"{model_name}, {case_body}, {movement}",
                    "support": int(counts.iloc[0]),
                    "age": age,
                    "contrast": f"case body -> {alternative}",
                    "changed_support": training_count(swapped),
                    "q33_usd": base_q,
                    "change_usd": prototype_q33(coder, params, swapped, age) - base_q,
                }
            )
            break
    return pd.DataFrame(rows)


def prototype_q33(coder, params, values, age):
    frame = pd.DataFrame([{**values, "age_at_sale": age}])
    return float(predict_usd(coder.transform(frame), TAU, params)[0])


def main():
    data_root = Path(__file__).resolve().parents[1] / "data"
    train, validation, predictors = load_partitions(data_root)
    x_train = assemble_features(train, predictors)
    x_val = assemble_features(validation, predictors)
    collapse_rare_models(x_train, x_val)
    y_train = train["usd_price"].to_numpy(dtype=float)
    y_val = validation["usd_price"].to_numpy(dtype=float)
    linear_loss, linear_prediction = fit_frozen_linear(x_train, x_val, y_train, y_val)
    brand_q33 = train.groupby("brand")["usd_price"].quantile(TAU)
    baseline_prediction = validation["brand"].map(brand_q33).to_numpy(dtype=float)
    baseline_loss = pinball_loss(y_val, baseline_prediction, TAU)

    coder = FeatureCoder().fit(x_train)
    chosen, runs = fit_iqn(coder.transform(x_train), y_train, coder.transform(x_val), y_val)
    iqn_prediction = predict_usd(coder.transform(x_val), TAU, chosen["params"])
    iqn_loss = pinball_loss(y_val, iqn_prediction, TAU)
    train_prediction = predict_usd(coder.transform(x_train), TAU, chosen["params"])
    train_loss = pinball_loss(y_train, train_prediction, TAU)
    calibration = float(np.mean(y_val <= iqn_prediction))
    by_brand = []
    for brand in ["Cartier", "Omega", "Rolex"]:
        mask = validation["brand"].eq(brand).to_numpy()
        by_brand.append(
            {
                "brand": brand,
                "n": int(mask.sum()),
                "brand_q33": pinball_loss(y_val[mask], baseline_prediction[mask], TAU),
                "linear_q33": pinball_loss(y_val[mask], linear_prediction[mask], TAU),
                "iqn_q33": pinball_loss(y_val[mask], iqn_prediction[mask], TAU),
                "iqn_fraction_below": float(np.mean(y_val[mask] <= iqn_prediction[mask])),
            }
        )

    print("\nQ33 DOLLAR PINBALL ON THE SAME 802 VALIDATION SALES")
    comparison = pd.DataFrame(
        {
            "Method": ["Brand q33", "Linear q33", "IQN q33"],
            "Validation pinball loss (USD)": [baseline_loss, linear_loss, iqn_loss],
        }
    )
    print(comparison.to_string(index=False, formatters={"Validation pinball loss (USD)": lambda value: f"{value:,.4f}"}))
    print(
        f"Selected IQN: width {chosen['width']}, learning rate {chosen['learning_rate']}, "
        f"weight decay {chosen['weight_decay']}, stopped at epoch {chosen['epoch']}."
    )
    print(f"IQN fraction of validation prices at or below q33: {calibration:.4f}")
    print(f"IQN training q33 pinball: {train_loss:,.4f}")
    print("The linear loss is the frozen benchmark recomputed on these rows.")
    print("The IQN loss is the best loss in the 2024 search, not an untouched check.")
    print("Test partition was not used.")
    print("\nPINBALL BY BRAND ON THE SAME VALIDATION SALES")
    print(pd.DataFrame(by_brand).to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    print("\nLOCAL AGE SLOPE AND CATEGORY SWAPS")
    contrasts = representative_contrasts(coder, chosen["params"], x_train)
    print(
        contrasts.to_string(
            index=False,
            formatters={"age": "{:.1f}".format, "q33_usd": "{:,.2f}".format, "change_usd": "{:,.2f}".format},
        )
    )
    return {
        "linear_loss": linear_loss,
        "linear_prediction": linear_prediction,
        "baseline_loss": baseline_loss,
        "iqn_loss": iqn_loss,
        "iqn_prediction": iqn_prediction,
        "train_loss": train_loss,
        "calibration": calibration,
        "by_brand": by_brand,
        "chosen": {key: value for key, value in chosen.items() if key != "params"},
        "contrasts": contrasts,
        "validation_ids": validation["record_id"].tolist(),
    }


if __name__ == "__main__":
    main()
