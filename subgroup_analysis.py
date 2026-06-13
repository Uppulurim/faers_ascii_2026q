import pandas as pd
import numpy as np

BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/ASCII"
SIGNALS = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/review_log.csv"

def load_data():
    print("[Subgroup Agent] Loading FAERS files...")
    drug = pd.read_csv(f"{BASE}/DRUG26Q1.txt", sep="$", low_memory=False)
    reac = pd.read_csv(f"{BASE}/REAC26Q1.txt", sep="$", low_memory=False)
    demo = pd.read_csv(f"{BASE}/DEMO26Q1.txt", sep="$", low_memory=False)

    drug["drugname_clean"] = drug["drugname"].str.upper().str.strip()

    merged = drug[["primaryid","drugname_clean"]].merge(
        reac[["primaryid","pt"]], on="primaryid"
    ).drop_duplicates(subset=["primaryid","drugname_clean","pt"])

    merged = merged.merge(
        demo[["primaryid","age","age_cod","sex","occr_country"]], on="primaryid", how="left"
    )

    # Normalize age to years
    age_mult = {"YR":1,"DEC":10,"MON":1/12,"WK":1/52,"DY":1/365,"HR":1/8760}
    merged["age_years"] = merged.apply(
        lambda r: r["age"] * age_mult.get(str(r["age_cod"]).strip(), 1)
        if pd.notna(r["age"]) else np.nan, axis=1
    )

    # Age groups
    merged["age_group"] = pd.cut(
        merged["age_years"],
        bins=[0,17,44,64,74,150],
        labels=["Pediatric (0-17)","Adult (18-44)","Middle-aged (45-64)","Older adult (65-74)","Elderly (75+)"]
    )

    print(f"[Subgroup Agent] Loaded {len(merged)} rows with demographics.")
    return merged

def subgroup_analysis(merged, drug, event, min_count=3):
    """Analyse a single drug-event pair across demographic subgroups."""
    print(f"\n[Subgroup Agent] Analysing {drug} + {event}")
    subset = merged[(merged["drugname_clean"] == drug) & (merged["pt"] == event)]
    total  = merged[merged["drugname_clean"] == drug]

    if len(subset) < min_count:
        print(f"  Too few cases ({len(subset)}) — skipping.")
        return None

    results = {"drug": drug, "event": event, "total_cases": len(subset)}

    # Sex distribution
    sex_map = {"M":"Male","F":"Female","UNK":"Unknown"}
    sex_counts = subset["sex"].map(sex_map).fillna("Unknown").value_counts()
    results["sex_distribution"] = sex_counts.to_dict()
    total_sex  = total["sex"].map(sex_map).fillna("Unknown").value_counts()
    results["sex_vs_all_drug_reports"] = total_sex.to_dict()

    # Age group distribution
    age_counts = subset["age_group"].value_counts().dropna()
    results["age_group_distribution"] = age_counts.to_dict()
    results["median_age"] = round(subset["age_years"].median(), 1) if subset["age_years"].notna().any() else "N/A"
    results["age_range"]  = f"{subset['age_years'].min():.0f}-{subset['age_years'].max():.0f}" if subset["age_years"].notna().any() else "N/A"

    # Top countries
    country_counts = subset["occr_country"].value_counts().head(5)
    results["top_countries"] = country_counts.to_dict()

    return results

def print_subgroup_report(result):
    if not result:
        return
    print(f"\n{'='*60}")
    print(f"DRUG: {result['drug']}  |  EVENT: {result['event']}")
    print(f"Total cases: {result['total_cases']}")
    print(f"Median age: {result['median_age']} years  |  Range: {result['age_range']}")
    print(f"\nSex distribution:")
    for k, v in result["sex_distribution"].items():
        print(f"  {k}: {v}")
    print(f"\nAge group distribution:")
    for k, v in result["age_group_distribution"].items():
        print(f"  {k}: {v}")
    print(f"\nTop reporting countries:")
    for k, v in result["top_countries"].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    merged = load_data()
    review = pd.read_csv(SIGNALS)
    accepted = review[review["review_status"] == "ACCEPTED"].head(10)

    all_results = []
    for _, row in accepted.iterrows():
        result = subgroup_analysis(merged, row["drug"], row["event"])
        if result:
            print_subgroup_report(result)
            all_results.append(result)

    # Save to CSV
    out = []
    for r in all_results:
        out.append({
            "drug": r["drug"],
            "event": r["event"],
            "total_cases": r["total_cases"],
            "median_age": r["median_age"],
            "age_range": r["age_range"],
            "sex_distribution": str(r["sex_distribution"]),
            "age_group_distribution": str(r["age_group_distribution"]),
            "top_countries": str(r["top_countries"])
        })
    pd.DataFrame(out).to_csv(
        "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/subgroup_results.csv",
        index=False
    )
    print("\n[Subgroup Agent] Results saved to subgroup_results.csv")