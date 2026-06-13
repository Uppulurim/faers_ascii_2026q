import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma

BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/ASCII"

# =========================================================
# EBGM: Empirical Bayes Geometric Mean
# Based on DuMouchel (1999) two-component gamma-Poisson model
# =========================================================

def compute_expected(n_ij, n_i, n_j, N):
    """Expected count under independence assumption."""
    return (n_i * n_j) / N

def neg_log_likelihood(params, observed, expected):
    """Two-component gamma mixture negative log-likelihood."""
    alpha1, beta1, alpha2, beta2, p = params
    if any(x <= 0 for x in [alpha1, beta1, alpha2, beta2]):
        return np.inf
    if not (0 < p < 1):
        return np.inf
    try:
        mu = observed / expected
        mu = np.clip(mu, 1e-10, None)
        # Two-component mixture
        comp1 = p * gamma.pdf(mu, a=alpha1, scale=1/beta1)
        comp2 = (1-p) * gamma.pdf(mu, a=alpha2, scale=1/beta2)
        mixture = comp1 + comp2
        mixture = np.clip(mixture, 1e-300, None)
        return -np.sum(np.log(mixture))
    except Exception:
        return np.inf

def fit_prior(observed, expected):
    """Fit the two-component gamma prior using MLE."""
    print("[EBGM] Fitting prior distribution...")
    # Multiple starting points to avoid local minima
    best_result = None
    best_val = np.inf
    starts = [
        [0.2, 0.1, 2.0, 4.0, 0.5],
        [0.5, 0.5, 3.0, 3.0, 0.3],
        [0.1, 0.2, 1.5, 2.0, 0.7],
    ]
    for start in starts:
        try:
            result = minimize(
                neg_log_likelihood, start,
                args=(observed, expected),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_result = result
        except Exception:
            continue
    if best_result is None:
        raise ValueError("Prior fitting failed")
    alpha1, beta1, alpha2, beta2, p = best_result.x
    print(f"[EBGM] Prior fitted: alpha1={alpha1:.3f}, beta1={beta1:.3f}, "
          f"alpha2={alpha2:.3f}, beta2={beta2:.3f}, p={p:.3f}")
    return alpha1, beta1, alpha2, beta2, p

def compute_ebgm(n, e, alpha1, beta1, alpha2, beta2, p):
    """
    Compute EBGM (posterior geometric mean of lambda=n/e).
    Uses the posterior mixture weights Q_n.
    """
    ebgm_vals = []
    eb05_vals = []  # 5th percentile (lower credible bound)

    for ni, ei in zip(n, e):
        if ei <= 0:
            ebgm_vals.append(np.nan)
            eb05_vals.append(np.nan)
            continue

        # Posterior parameters (conjugate update for Poisson-Gamma)
        a1_post = alpha1 + ni
        b1_post = beta1 + ei
        a2_post = alpha2 + ni
        b2_post = beta2 + ei

        # Mixture weight update
        w1 = p * gamma.pdf(ni/ei, a=alpha1, scale=1/beta1) + 1e-300
        w2 = (1-p) * gamma.pdf(ni/ei, a=alpha2, scale=1/beta2) + 1e-300
        Q = w1 / (w1 + w2)

        # EBGM = exp(E[log(lambda)|n])
        # For Gamma(a,b): E[log(X)] = digamma(a) - log(b)
        log_mean1 = float(np.log(a1_post) - np.log(b1_post))
        log_mean2 = float(np.log(a2_post) - np.log(b2_post))
        ebgm = np.exp(Q * log_mean1 + (1-Q) * log_mean2)
        ebgm_vals.append(round(float(ebgm), 3))

        # EB05: 5th percentile of posterior (conservative lower bound)
        # Approximate as weighted 5th percentile
        q05_1 = gamma.ppf(0.05, a=a1_post, scale=1/b1_post)
        q05_2 = gamma.ppf(0.05, a=a2_post, scale=1/b2_post)
        eb05  = Q * q05_1 + (1-Q) * q05_2
        eb05_vals.append(round(float(eb05), 3))

    return ebgm_vals, eb05_vals

def run_ebgm(target_drug, min_count=3, top_n=2000):
    """Run EBGM for a target drug across all events."""
    print(f"\n[EBGM] Loading FAERS data for {target_drug}...")
    drug = pd.read_csv(f"{BASE}/DRUG26Q1.txt", sep="$", low_memory=False)
    reac = pd.read_csv(f"{BASE}/REAC26Q1.txt", sep="$", low_memory=False)

    drug["drugname_clean"] = drug["drugname"].str.upper().str.strip()
    merged = drug[["primaryid","drugname_clean"]].merge(
        reac[["primaryid","pt"]], on="primaryid"
    ).drop_duplicates(subset=["primaryid","drugname_clean","pt"])

    N = len(merged)
    total_drug  = len(merged[merged["drugname_clean"] == target_drug])
    event_counts = merged["pt"].value_counts()
    drug_events  = merged[merged["drugname_clean"] == target_drug]["pt"].value_counts()

    print(f"[EBGM] Total rows: {N}, Drug rows: {total_drug}, Events to score: {len(drug_events)}")

    # Build observed/expected arrays
    rows = []
    for event, n_ij in drug_events.items():
        if n_ij < min_count:
            continue
        n_j = event_counts.get(event, 0)
        e_ij = compute_expected(n_ij, total_drug, n_j, N)
        if e_ij > 0:
            rows.append({"event": event, "n": n_ij, "e": round(e_ij, 4), "ror": round(n_ij/e_ij, 3)})

    df = pd.DataFrame(rows)
    print(f"[EBGM] {len(df)} drug-event pairs to score")

    # Fit prior on all pairs (use top_n most common for speed)
    fit_df = df.nlargest(top_n, "n")
    alpha1, beta1, alpha2, beta2, p = fit_prior(fit_df["n"].values, fit_df["e"].values)

    # Score all pairs
    ebgm_vals, eb05_vals = compute_ebgm(
        df["n"].values, df["e"].values,
        alpha1, beta1, alpha2, beta2, p
    )
    df["ebgm"] = ebgm_vals
    df["eb05"] = eb05_vals

    # Signal threshold: EB05 >= 2 (DuMouchel standard)
    df["ebgm_signal"] = df["eb05"] >= 2.0
    df["drug"] = target_drug
    df = df.sort_values("ebgm", ascending=False)

    print(f"\n[EBGM] Results for {target_drug}:")
    print(f"  Total scored: {len(df)}")
    print(f"  EBGM signals (EB05>=2): {df['ebgm_signal'].sum()}")
    print(f"\nTop 10 by EBGM:")
    print(df[["event","n","e","ebgm","eb05","ebgm_signal"]].head(10).to_string(index=False))
    return df

if __name__ == "__main__":
    results = []
    for drug in ["METHOTREXATE", "ACTEMRA", "DUPIXENT", "TYMLOS"]:
        df = run_ebgm(drug)
        results.append(df)

    combined = pd.concat(results, ignore_index=True)
    out_path = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/ebgm_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n[EBGM] All results saved to ebgm_results.csv")
    print(f"Total EBGM signals across all drugs: {combined['ebgm_signal'].sum()}")