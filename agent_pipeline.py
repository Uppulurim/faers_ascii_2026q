import pandas as pd
import numpy as np
import requests
import time
import os

BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/ASCII"
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

# =========================================================
# AGENT 1: INGESTION AGENT
# =========================================================
def ingestion_agent():
    print("[Ingestion Agent] Loading FAERS files...")
    drug = pd.read_csv(f"{BASE}/DRUG26Q1.txt", sep="$", low_memory=False)
    reac = pd.read_csv(f"{BASE}/REAC26Q1.txt", sep="$", low_memory=False)

    drug["drugname_clean"] = drug["drugname"].str.upper().str.strip()
    merged = drug[["primaryid", "drugname_clean"]].merge(
    reac[["primaryid", "pt"]], on="primaryid"
).drop_duplicates(subset=["primaryid", "drugname_clean", "pt"])
    print(f"[Ingestion Agent] Loaded {len(merged)} drug-event rows.")
    return merged


# =========================================================
# AGENT 2: MAPPING AGENT (RxNorm normalization)
# =========================================================
RXNORM_CACHE_FILE = "rxnorm_cache.csv"

def get_rxcui(drug_name):
    try:
        r = requests.get(f"{RXNORM_BASE}/rxcui.json", params={"name": drug_name}, timeout=5)
        r.raise_for_status()
        ids = r.json().get("idGroup", {}).get("rxnormId")
        return ids[0] if ids else None
    except Exception:
        return None

def get_ingredient_name(rxcui):
    try:
        r = requests.get(f"{RXNORM_BASE}/rxcui/{rxcui}/property.json",
                         params={"propName": "RxNorm Name"}, timeout=5)
        r.raise_for_status()
        return r.json().get("propConceptGroup", {}).get("propConcept", [{}])[0].get("propValue")
    except Exception:
        return None

def mapping_agent(drug_names, min_score=70):
    print(f"[Mapping Agent] Mapping {len(drug_names)} unique drug names via RxNorm...")

    if os.path.exists(RXNORM_CACHE_FILE):
        cache = pd.read_csv(RXNORM_CACHE_FILE)
    else:
        cache = pd.DataFrame(columns=["original_name", "rxcui", "normalized_name", "status"])

    cached_names = set(cache["original_name"])
    new_names = [n for n in drug_names if n not in cached_names]
    print(f"[Mapping Agent] {len(new_names)} new names to query, {len(cached_names)} cached.")

    results = []
    for i, name in enumerate(new_names):
        rxcui = get_rxcui(name)
        if rxcui:
            norm_name = get_ingredient_name(rxcui)
            status = "MAPPED" if norm_name else "NEEDS_REVIEW"
        else:
            norm_name = None
            status = "NEEDS_REVIEW"
        results.append({"original_name": name, "rxcui": rxcui,
                         "normalized_name": norm_name, "status": status})
        if (i+1) % 25 == 0:
            print(f"[Mapping Agent]   ...{i+1}/{len(new_names)} done")
        time.sleep(0.05)

    if results:
        new_df = pd.DataFrame(results)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache.to_csv(RXNORM_CACHE_FILE, index=False)

    n_review = (cache["status"] == "NEEDS_REVIEW").sum()
    print(f"[Mapping Agent] Done. {n_review} names flagged NEEDS_REVIEW (no exact RxNorm match).")
    return cache


# =========================================================
# AGENT 3: SIGNAL DETECTION AGENT
# =========================================================
def calc_ror(a, b, c, d):
    if b == 0 or c == 0 or a == 0:
        return None, None, None
    ror = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    return ror, np.exp(np.log(ror) - 1.96*se), np.exp(np.log(ror) + 1.96*se)

def calc_prr(a, b, c, d):
    if (a+b) == 0 or (c+d) == 0 or c == 0:
        return None, None
    a, b, c, d = float(a), float(b), float(c), float(d)
    prr = (a/(a+b)) / (c/(c+d))
    chi2 = ((a*d - b*c)**2 * (a+b+c+d)) / ((a+b)*(c+d)*(a+c)*(b+d))
    return prr, chi2

def signal_detection_agent(merged, target_drug, min_count=3):
    print(f"[Signal Detection Agent] Scanning all events for {target_drug}...")
    drug_rows = merged[merged["drugname_clean"] == target_drug]

    total = len(merged)
    drug_total = len(drug_rows)
    event_counts_all = merged["pt"].value_counts()
    event_counts_drug = drug_rows["pt"].value_counts()

    results = []
    for event, a in event_counts_drug.items():
        if a < min_count:
            continue
        b = drug_total - a
        event_total = event_counts_all.get(event, 0)
        c = event_total - a
        d = total - a - b - c
        ror, ci_l, ci_u = calc_ror(a, b, c, d)
        prr, chi2 = calc_prr(a, b, c, d)
        if ror is None:
            continue
        flagged = (ci_l > 1) and (prr is not None and prr >= 2) and (chi2 is not None and chi2 >= 4)
        results.append({
            "drug": target_drug, "event": event, "count": a,
            "ror": round(ror, 2), "ror_ci_lower": round(ci_l, 2), "ror_ci_upper": round(ci_u, 2),
            "prr": round(prr, 2), "chi2": round(chi2, 2), "signal_flagged": flagged
        })

    df = pd.DataFrame(results).sort_values("ror_ci_lower", ascending=False)
    print(f"[Signal Detection Agent] {df['signal_flagged'].sum()} signals flagged out of {len(df)} events.")
    return df


# =========================================================
# AGENT 4: EXPLAINABILITY AGENT
# =========================================================
def explainability_agent(signals_df):
    print("[Explainability Agent] Generating rationales for flagged signals...")
    flagged = signals_df[signals_df["signal_flagged"]].copy()
    rationales = []
    for _, row in flagged.iterrows():
        rationale = (
            f"{row['drug']} was reported with '{row['event']}' in {row['count']} cases. "
            f"This combination occurs {row['ror']}x more often than expected "
            f"(95% CI: {row['ror_ci_lower']}-{row['ror_ci_upper']}, entirely above 1, "
            f"meaning the result is statistically significant). "
            f"PRR={row['prr']} (≥2 threshold met), chi2={row['chi2']} (≥4 threshold met)."
        )
        rationales.append(rationale)
    flagged["rationale"] = rationales
    return flagged


# =========================================================
# AGENT 5: REVIEWER AGENT (Human-in-the-loop)
# =========================================================
REVIEW_LOG_FILE = "review_log.csv"

def reviewer_agent(flagged_df, auto_mode=True):
    print(f"[Reviewer Agent] {len(flagged_df)} signals require human review.")
    flagged_df = flagged_df.copy()

    if auto_mode:
        flagged_df["review_status"] = "PENDING_REVIEW"
        flagged_df["reviewer_notes"] = ""
    else:
        statuses, notes = [], []
        for _, row in flagged_df.iterrows():
            print("\n" + "="*60)
            print(row["rationale"])
            decision = input("Accept (a) / Reject (r) / Defer (d)? ").strip().lower()
            status = {"a": "ACCEPTED", "r": "REJECTED", "d": "DEFERRED"}.get(decision, "DEFERRED")
            note = input("Optional note: ").strip()
            statuses.append(status)
            notes.append(note)
        flagged_df["review_status"] = statuses
        flagged_df["reviewer_notes"] = notes

    return flagged_df


# =========================================================
# MAIN PIPELINE
# =========================================================
if __name__ == "__main__":
    TOP_N_DRUGS = 10
    TOP_N_REVIEW = 25

    merged = ingestion_agent()

    top_drugs = merged["drugname_clean"].value_counts().head(TOP_N_DRUGS).index.tolist()
    print(f"\n[Main] Running signal detection for top {TOP_N_DRUGS} drugs: {top_drugs}\n")

    rxnorm_map = mapping_agent(top_drugs)

    all_signals = []
    for drug in top_drugs:
        signals_df = signal_detection_agent(merged, drug, min_count=5)
        flagged = explainability_agent(signals_df)
        flagged = flagged[(flagged["prr"] >= 3) & (flagged["chi2"] >= 10)]
        all_signals.append(flagged)
        print(f"  -> {drug}: {len(flagged)} signals after stricter filter\n")

    combined = pd.concat(all_signals, ignore_index=True)
    combined = combined.sort_values("chi2", ascending=False).reset_index(drop=True)
    combined.to_csv("signals_all_drugs.csv", index=False)
    print(f"\n[Main] Total signals across {TOP_N_DRUGS} drugs: {len(combined)}")
    print(f"[Main] Saved to signals_all_drugs.csv")

    top_signals = combined.head(TOP_N_REVIEW).copy()
    rest_signals = combined.iloc[TOP_N_REVIEW:].copy()

    print(f"\n[Main] Interactively reviewing top {len(top_signals)} signals by chi2...")
    reviewed_top = reviewer_agent(top_signals, auto_mode=False)

    rest_signals["review_status"] = "PENDING_REVIEW"
    rest_signals["reviewer_notes"] = ""

    final = pd.concat([reviewed_top, rest_signals], ignore_index=True)
    final.to_csv("review_log.csv", index=False)

    print(f"\nDone. {len(reviewed_top)} signals reviewed interactively, "
          f"{len(rest_signals)} logged as PENDING_REVIEW.")
    print("Files written: signals_all_drugs.csv, review_log.csv")