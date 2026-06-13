import pandas as pd

BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/ASCII"

drug = pd.read_csv(f"{BASE}/DRUG26Q1.txt", sep="$", low_memory=False)
reac = pd.read_csv(f"{BASE}/REAC26Q1.txt", sep="$", low_memory=False)
demo = pd.read_csv(f"{BASE}/DEMO26Q1.txt", sep="$", low_memory=False)

drug["drugname_clean"] = drug["drugname"].str.upper().str.strip()
merged = drug[["primaryid", "caseid", "drugname_clean"]].merge(
    reac[["primaryid", "pt"]], on="primaryid"
)

def get_case_ids(drugname, event):
    rows = merged[(merged["drugname_clean"] == drugname) & (merged["pt"] == event)]
    return set(rows["primaryid"]), set(rows["caseid"])

# 201-case batch
events_201 = ["Urinary bladder herniation", "Hepatitis B virus test positive",
              "Vascular access site pain", "Bladder neoplasm"]

print("=== 201-case batch ===")
sets_201 = {}
for e in events_201:
    pids, cids = get_case_ids("ACTEMRA", e)
    sets_201[e] = pids
    print(f"{e}: {len(pids)} primaryids")

# Check overlap
base_set = sets_201[events_201[0]]
for e in events_201[1:]:
    overlap = base_set & sets_201[e]
    print(f"Overlap between '{events_201[0]}' and '{e}': {len(overlap)} / {len(base_set)}")

# 122-case batch
events_122 = ["Fallopian tube disorder", "Adnexal torsion", "Infertility", "Joint ankylosis"]

print("\n=== 122-case batch ===")
sets_122 = {}
for e in events_122:
    pids, cids = get_case_ids("ACTEMRA", e)
    sets_122[e] = pids
    print(f"{e}: {len(pids)} primaryids")

base_set2 = sets_122[events_122[0]]
for e in events_122[1:]:
    overlap = base_set2 & sets_122[e]
    print(f"Overlap between '{events_122[0]}' and '{e}': {len(overlap)} / {len(base_set2)}")

# Sample some case details from demo for the 201-batch
print("\n=== Sample demographic info for 201-case batch ===")
sample_pids = list(sets_201[events_201[0]])[:5]
sample_demo = demo[demo["primaryid"].isin(sample_pids)]
print(sample_demo[["primaryid", "caseid", "age", "sex", "occr_country", "occp_cod", "rept_dt"]].to_string(index=False))

# Check reporter/country pattern across full 201 set
print("\n=== Country/reporter distribution for full 201-case set ===")
full_demo = demo[demo["primaryid"].isin(sets_201[events_201[0]])]
print("Countries:", full_demo["occr_country"].value_counts().to_dict())
print("Reporter type (occp_cod):", full_demo["occp_cod"].value_counts().to_dict())
print("Report dates (top 5):", full_demo["rept_dt"].value_counts().head(5).to_dict())