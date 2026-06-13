import os
import pandas as pd
import anthropic

BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/ASCII"
REVIEW_LOG = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/review_log.csv"

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def load_cases(drug, event, max_cases=20):
    """Load actual case details for a drug-event pair from FAERS."""
    print(f"  Loading cases for {drug} + {event}...")
    drug_df = pd.read_csv(f"{BASE}/DRUG26Q1.txt", sep="$", low_memory=False)
    reac_df = pd.read_csv(f"{BASE}/REAC26Q1.txt", sep="$", low_memory=False)
    demo_df = pd.read_csv(f"{BASE}/DEMO26Q1.txt", sep="$", low_memory=False)
    outc_df = pd.read_csv(f"{BASE}/OUTC26Q1.txt", sep="$", low_memory=False)

    drug_df["drugname_clean"] = drug_df["drugname"].str.upper().str.strip()

    # Get primaryids for this drug-event pair
    drug_pids = set(drug_df[drug_df["drugname_clean"] == drug]["primaryid"])
    reac_pids = set(reac_df[reac_df["pt"] == event]["primaryid"])
    target_pids = list(drug_pids & reac_pids)[:max_cases]

    if not target_pids:
        return None

    # Pull demographics and outcomes
    demo = demo_df[demo_df["primaryid"].isin(target_pids)][
        ["primaryid", "age", "age_cod", "sex", "occr_country", "rept_dt", "init_fda_dt"]
    ].drop_duplicates("primaryid")

    outc = outc_df[outc_df["primaryid"].isin(target_pids)][
        ["primaryid", "outc_cod"]
    ]

    # Outcome code mapping
    outc_map = {"DE":"Death","LT":"Life-threatening","HO":"Hospitalization",
                "DS":"Disability","CA":"Congenital anomaly","RI":"Required intervention","OT":"Other"}
    outc["outcome"] = outc["outc_cod"].map(outc_map).fillna("Unknown")
    outc_summary = outc.groupby("primaryid")["outcome"].apply(lambda x: ", ".join(x)).reset_index()

    # All drugs per case
    all_drugs = drug_df[drug_df["primaryid"].isin(target_pids)][
        ["primaryid", "drugname_clean", "role_cod"]
    ]
    drug_summary = all_drugs.groupby("primaryid")["drugname_clean"].apply(
        lambda x: ", ".join(x.unique()[:5])
    ).reset_index()
    drug_summary.columns = ["primaryid", "all_drugs"]

    # All reactions per case
    all_reac = reac_df[reac_df["primaryid"].isin(target_pids)][["primaryid", "pt"]]
    reac_summary = all_reac.groupby("primaryid")["pt"].apply(
        lambda x: ", ".join(x.unique()[:8])
    ).reset_index()
    reac_summary.columns = ["primaryid", "all_reactions"]

    merged = demo.merge(outc_summary, on="primaryid", how="left")
    merged = merged.merge(drug_summary, on="primaryid", how="left")
    merged = merged.merge(reac_summary, on="primaryid", how="left")
    return merged


def summarize_signal(drug, event, stats, cases_df):
    """Use Claude to generate a medical reviewer summary."""
    cases_text = cases_df.to_string(index=False) if cases_df is not None else "No case details available."

    prompt = f"""You are assisting a pharmacovigilance medical reviewer at a pharmaceutical company.

A safety signal has been detected in FAERS Q1 2026 data:
- Drug: {drug}
- Adverse Event: {event}
- Reporting Odds Ratio (ROR): {stats.get('ror', 'N/A')} (95% CI: {stats.get('ror_ci_lower','N/A')}-{stats.get('ror_ci_upper','N/A')})
- Proportional Reporting Ratio (PRR): {stats.get('prr', 'N/A')}
- Chi²: {stats.get('chi2', 'N/A')}
- Number of cases: {stats.get('count', 'N/A')}

Case-level data (up to 20 cases):
{cases_text}

Please provide a structured medical review summary covering:
1. SIGNAL SUMMARY: Brief description of the statistical finding
2. PATIENT DEMOGRAPHICS: Age range, sex distribution, countries
3. OUTCOME SEVERITY: Types of outcomes reported (death, hospitalization, etc.)
4. CONCOMITANT MEDICATIONS: Notable co-medications that might confound
5. CLINICAL ASSESSMENT: Is this signal biologically plausible? Known mechanism?
6. RECOMMENDED ACTION: Further investigation needed? Label review? Signal close?

Important: Describe patterns only. Do not draw causal conclusions.
Keep the summary concise and suitable for a medical reviewer."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def run_llm_summarizer():
    """Run LLM summarization on all ACCEPTED signals from review log."""
    review = pd.read_csv(REVIEW_LOG)
    accepted = review[review["review_status"] == "ACCEPTED"].head(5)  # top 5 to start

    print(f"[LLM Summarizer] Generating summaries for {len(accepted)} accepted signals...\n")

    summaries = []
    for _, row in accepted.iterrows():
        drug  = row["drug"]
        event = row["event"]
        stats = {"ror": row.get("ror"), "ror_ci_lower": row.get("ror_ci_lower"),
                 "ror_ci_upper": row.get("ror_ci_upper"), "prr": row.get("prr"),
                 "chi2": row.get("chi2"), "count": row.get("count")}

        print(f"[LLM Summarizer] Summarizing: {drug} + {event}")
        cases_df = load_cases(drug, event)
        summary  = summarize_signal(drug, event, stats, cases_df)

        summaries.append({
            "drug": drug, "event": event,
            "ror": stats["ror"], "count": stats["count"],
            "llm_summary": summary
        })
        print(f"  Done.\n")

    # Save summaries
    out_path = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/llm_summaries.txt"
    with open(out_path, "w") as f:
        for s in summaries:
            f.write("="*70 + "\n")
            f.write(f"DRUG: {s['drug']}  |  EVENT: {s['event']}\n")
            f.write(f"ROR: {s['ror']}  |  Cases: {s['count']}\n")
            f.write("-"*70 + "\n")
            f.write(s["llm_summary"])
            f.write("\n\n")

    print(f"[LLM Summarizer] Summaries saved to llm_summaries.txt")
    return summaries


if __name__ == "__main__":
    # First install anthropic if needed
    summaries = run_llm_summarizer()
    for s in summaries:
        print("\n" + "="*70)
        print(f"DRUG: {s['drug']}  |  EVENT: {s['event']}")
        print("="*70)
        print(s["llm_summary"])