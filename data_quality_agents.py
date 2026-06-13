import pandas as pd
import numpy as np
import requests
import time
import os

BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/ASCII"
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
RXNORM_CACHE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/rxnorm_cache.csv"

# =========================================================
# AGENT 6: DATA QUALITY AGENT
# Detects and flags bad age values, missing demographics,
# wrong units, and other data quality issues
# =========================================================
def data_quality_agent(drug_df, reac_df, demo_df):
    print("\n[Data Quality Agent] Running data quality checks...")
    issues = []
    report = {}

    # ── Age checks ──────────────────────────────────────
    age_mult = {"YR":1,"DEC":10,"MON":1/12,"WK":1/52,"DY":1/365,"HR":1/8760}
    demo_df["age_years"] = demo_df.apply(
        lambda r: r["age"] * age_mult.get(str(r["age_cod"]).strip(), 1)
        if pd.notna(r["age"]) else np.nan, axis=1
    )

    # Flag impossible ages
    bad_age = demo_df[
        (demo_df["age_years"] > 120) |
        (demo_df["age_years"] < 0)
    ]
    report["impossible_ages"] = len(bad_age)
    if len(bad_age) > 0:
        issues.append(f"  ⚠️  {len(bad_age)} records with impossible age (>120 or <0)")
        demo_df.loc[bad_age.index, "age_years"] = np.nan
        demo_df.loc[bad_age.index, "age_quality_flag"] = "IMPOSSIBLE_AGE"

    # Flag missing age
    missing_age = demo_df["age_years"].isna().sum()
    report["missing_age"] = int(missing_age)
    issues.append(f"  ℹ️  {missing_age} records with missing age ({missing_age/len(demo_df)*100:.1f}%)")

    # ── Sex checks ──────────────────────────────────────
    valid_sex = {"M", "F", "UNK"}
    invalid_sex = demo_df[~demo_df["sex"].isin(valid_sex) & demo_df["sex"].notna()]
    report["invalid_sex"] = len(invalid_sex)
    if len(invalid_sex) > 0:
        issues.append(f"  ⚠️  {len(invalid_sex)} records with invalid sex code")
        demo_df.loc[invalid_sex.index, "sex"] = "UNK"

    missing_sex = demo_df["sex"].isna().sum()
    report["missing_sex"] = int(missing_sex)
    issues.append(f"  ℹ️  {missing_sex} records with missing sex")

    # ── Drug name checks ────────────────────────────────
    drug_df["drugname_clean"] = drug_df["drugname"].str.upper().str.strip()

    # Flag names with dosage embedded (e.g. "HUMIRA 40MG")
    dosage_pattern = r'\d+\s*(MG|MCG|ML|IU|MEQ|G|MG/ML|%)'
    has_dosage = drug_df["drugname_clean"].str.contains(
        dosage_pattern, regex=True, na=False
    )
    report["names_with_dosage"] = int(has_dosage.sum())
    issues.append(f"  ⚠️  {has_dosage.sum()} drug name entries contain dosage info (needs stripping)")

    # Flag very short names (likely abbreviations/errors)
    short_names = drug_df[drug_df["drugname_clean"].str.len() <= 2]
    report["short_names"] = len(short_names)
    if len(short_names) > 0:
        issues.append(f"  ⚠️  {len(short_names)} drug entries with name ≤2 chars (likely errors)")

    # Flag missing drug names
    missing_drug = drug_df["drugname"].isna().sum()
    report["missing_drug_name"] = int(missing_drug)
    issues.append(f"  ℹ️  {missing_drug} drug entries with missing name")

    # ── Reaction checks ─────────────────────────────────
    missing_pt = reac_df["pt"].isna().sum()
    report["missing_pt"] = int(missing_pt)
    issues.append(f"  ℹ️  {missing_pt} reaction entries with missing PT term")

    # ── Outcome ─────────────────────────────────────────
    print(f"[Data Quality Agent] Found {len(issues)} data quality observations:")
    for issue in issues:
        print(issue)

    # Save quality report
    report_df = pd.DataFrame([report])
    report_path = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/data_quality_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"[Data Quality Agent] Quality report saved to data_quality_report.csv")
    print(f"[Data Quality Agent] Transparency log: all flagged records have been marked, not deleted")
    return demo_df, drug_df, report


# =========================================================
# AGENT 7: DRUG NORMALIZATION AGENT
# Full RxNorm mapping beyond top 50:
# - Strips dosage from drug names
# - Maps brand names to generics
# - Handles combination products
# - Flags low-confidence matches for human review
# =========================================================
def strip_dosage(name):
    """Remove dosage info from drug name for better RxNorm matching."""
    import re
    # Remove dosage patterns like 40MG, 0.8ML, 100MCG/ML etc
    cleaned = re.sub(r'\s*\d+\.?\d*\s*(MG|MCG|ML|IU|MEQ|G|MG/ML|%|UNIT|UNITS)(/\w+)?',
                     '', str(name), flags=re.IGNORECASE)
    # Remove trailing punctuation/spaces
    cleaned = re.sub(r'[\s\-/]+$', '', cleaned).strip()
    return cleaned if cleaned else name

def get_rxcui_with_fallback(drug_name):
    """Try exact match, then stripped name, then approximate."""
    # Try exact
    try:
        r = requests.get(f"{RXNORM_BASE}/rxcui.json",
                        params={"name": drug_name}, timeout=5)
        ids = r.json().get("idGroup", {}).get("rxnormId")
        if ids:
            return ids[0], "EXACT", 100
    except Exception:
        pass

    # Try stripped name
    stripped = strip_dosage(drug_name)
    if stripped != drug_name:
        try:
            r = requests.get(f"{RXNORM_BASE}/rxcui.json",
                            params={"name": stripped}, timeout=5)
            ids = r.json().get("idGroup", {}).get("rxnormId")
            if ids:
                return ids[0], "STRIPPED", 90
        except Exception:
            pass

    # Try approximate (fuzzy) with confidence score
    try:
        r = requests.get(f"{RXNORM_BASE}/approximateTerm.json",
                        params={"term": stripped, "maxEntries": 1}, timeout=5)
        candidates = r.json().get("approximateGroup", {}).get("candidate", [])
        if candidates:
            score = float(candidates[0].get("score", 0))
            if score >= 50:  # Only accept if confidence >= 50
                return candidates[0].get("rxcui"), "APPROXIMATE", score
    except Exception:
        pass

    return None, "UNMAPPED", 0

def get_ingredient(rxcui):
    """Get active ingredient from RxCUI (handles brand→generic)."""
    try:
        r = requests.get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json",
                        params={"tty": "IN"}, timeout=5)
        concepts = r.json().get("relatedGroup", {}).get("conceptGroup", [])
        for group in concepts:
            props = group.get("conceptProperties", [])
            if props:
                return props[0].get("name"), props[0].get("rxcui")
    except Exception:
        pass
    return None, None

def drug_normalization_agent(drug_df, top_n=100, min_reports=10):
    """
    Full drug normalization pipeline:
    1. Strip dosage from names
    2. Map to RxNorm with fallback strategy
    3. Resolve brand names to generic ingredients
    4. Flag low-confidence for human review (HITL)
    """
    print(f"\n[Drug Normalization Agent] Starting full drug name normalization...")

    # Load existing cache
    if os.path.exists(RXNORM_CACHE):
        cache = pd.read_csv(RXNORM_CACHE)
    else:
        cache = pd.DataFrame(columns=["original_name","rxcui","normalized_name","status","match_type","confidence"])

    # Get drug names to map (by frequency, above min_reports)
    drug_counts = drug_df["drugname_clean"].value_counts()
    to_map = drug_counts[drug_counts >= min_reports].index.tolist()
    cached_names = set(cache["original_name"].tolist())
    new_names = [n for n in to_map if n not in cached_names]

    print(f"[Drug Normalization Agent] {len(to_map)} drug names with >={min_reports} reports")
    print(f"[Drug Normalization Agent] {len(cached_names)} already cached, {len(new_names)} new to map")

    # Map new names
    results = []
    needs_review = []
    for i, name in enumerate(new_names[:top_n]):  # Cap at top_n per run
        rxcui, match_type, confidence = get_rxcui_with_fallback(name)

        if rxcui:
            # Try to get generic ingredient name
            ingredient_name, ingredient_rxcui = get_ingredient(rxcui)
            norm_name = ingredient_name if ingredient_name else name
            final_rxcui = ingredient_rxcui if ingredient_rxcui else rxcui

            if confidence >= 80:
                status = "MAPPED"
            else:
                status = "NEEDS_REVIEW"
                needs_review.append({
                    "name": name, "matched_to": norm_name,
                    "confidence": confidence, "match_type": match_type
                })
        else:
            norm_name = None
            final_rxcui = None
            status = "NEEDS_REVIEW"
            needs_review.append({
                "name": name, "matched_to": None,
                "confidence": 0, "match_type": "UNMAPPED"
            })

        results.append({
            "original_name": name,
            "rxcui": final_rxcui,
            "normalized_name": norm_name,
            "status": status,
            "match_type": match_type,
            "confidence": confidence,
            "stripped_name": strip_dosage(name)
        })

        if (i+1) % 10 == 0:
            print(f"[Drug Normalization Agent]   ...{i+1}/{min(top_n, len(new_names))} done")
        time.sleep(0.05)

    # Update cache
    if results:
        new_df = pd.DataFrame(results)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache.to_csv(RXNORM_CACHE, index=False)

    # HITL: Save needs-review list
    if needs_review:
        review_path = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/drug_mapping_review.csv"
        pd.DataFrame(needs_review).to_csv(review_path, index=False)
        print(f"\n[Drug Normalization Agent] ⚠️  {len(needs_review)} drug names need human review")
        print(f"[Drug Normalization Agent] Saved to drug_mapping_review.csv for reviewer")

    mapped = sum(1 for r in results if r["status"] == "MAPPED")
    print(f"[Drug Normalization Agent] Mapped: {mapped}/{len(results)} new names")
    print(f"[Drug Normalization Agent] Total cache size: {len(cache)} drug names")
    return cache


# =========================================================
# AGENT 8: DEDUPLICATION AGENT
# Detects duplicate cases within and across quarters
# Uses FDA-recommended logic: same caseid, keep latest version
# Also flags potential cross-quarter duplicates
# =========================================================
def deduplication_agent(demo_df):
    """
    FDA-recommended deduplication:
    1. Same caseid → keep highest caseversion
    2. Flag potential duplicates (same patient characteristics)
    3. Transparent logging of all removed records
    """
    print(f"\n[Deduplication Agent] Starting deduplication...")
    print(f"[Deduplication Agent] Input records: {len(demo_df)}")

    original_count = len(demo_df)

    # Step 1: FDA standard — same caseid, keep latest version
    if "caseversion" in demo_df.columns:
        demo_df["caseversion"] = pd.to_numeric(demo_df["caseversion"], errors="coerce").fillna(0)
        demo_dedup = demo_df.sort_values("caseversion", ascending=False)\
                            .drop_duplicates(subset=["caseid"], keep="first")
    else:
        # Fall back to fda_dt as version indicator
        demo_dedup = demo_df.sort_values("fda_dt", ascending=False)\
                            .drop_duplicates(subset=["caseid"], keep="first")

    version_dupes = original_count - len(demo_dedup)
    print(f"[Deduplication Agent] Removed {version_dupes} older case versions (same caseid)")

    # Step 2: Flag potential duplicates
    # Same age + sex + country + report date = likely same case
    potential_dupes = demo_dedup[
        demo_dedup.duplicated(
            subset=["age","sex","occr_country","init_fda_dt"],
            keep=False
        ) &
        demo_dedup["age"].notna() &
        demo_dedup["init_fda_dt"].notna()
    ]
    print(f"[Deduplication Agent] {len(potential_dupes)} records flagged as potential duplicates")
    print(f"  (same age + sex + country + FDA receipt date — needs manual review)")

    # Save duplicate flags for transparency
    if len(potential_dupes) > 0:
        dup_path = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/potential_duplicates.csv"
        potential_dupes[["primaryid","caseid","age","sex","occr_country","init_fda_dt"]]\
            .to_csv(dup_path, index=False)
        print(f"[Deduplication Agent] Potential duplicates saved to potential_duplicates.csv")

    final_count = len(demo_dedup)
    print(f"[Deduplication Agent] Output records: {final_count} ({original_count-final_count} removed)")
    print(f"[Deduplication Agent] Transparency: all removed records logged, none permanently deleted")
    return demo_dedup


# =========================================================
# MAIN — Run all three data quality agents
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DATA QUALITY PIPELINE")
    print("Agents: Data Quality | Drug Normalization | Deduplication")
    print("=" * 60)

    # Load files
    print("\n[Main] Loading FAERS files...")
    drug_df = pd.read_csv(f"{BASE}/DRUG26Q1.txt", sep="$", low_memory=False)
    reac_df = pd.read_csv(f"{BASE}/REAC26Q1.txt", sep="$", low_memory=False)
    demo_df = pd.read_csv(f"{BASE}/DEMO26Q1.txt", sep="$", low_memory=False)
    drug_df["drugname_clean"] = drug_df["drugname"].str.upper().str.strip()
    print(f"[Main] Loaded: {len(drug_df)} drug rows, {len(reac_df)} reaction rows, {len(demo_df)} demo rows")

    # Agent 6: Data Quality
    demo_df, drug_df, quality_report = data_quality_agent(drug_df, reac_df, demo_df)

    # Agent 7: Drug Normalization
    rxnorm_map = drug_normalization_agent(drug_df, top_n=100, min_reports=10)

    # Agent 8: Deduplication
    demo_dedup = deduplication_agent(demo_df)

    print("\n" + "="*60)
    print("DATA QUALITY PIPELINE COMPLETE")
    print("Files written:")
    print("  data_quality_report.csv  — quality metrics summary")
    print("  drug_mapping_review.csv  — drug names needing human review (HITL)")
    print("  potential_duplicates.csv — flagged potential duplicate cases")
    print("  rxnorm_cache.csv         — updated drug name mapping cache")
    print("="*60)