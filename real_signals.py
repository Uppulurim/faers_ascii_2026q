import pandas as pd
import numpy as np

BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1/ASCII"

def calc_ror(a, b, c, d):
    if b == 0 or c == 0 or a == 0:
        return None, None, None
    ror = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = np.exp(np.log(ror) - 1.96 * se)
    ci_upper = np.exp(np.log(ror) + 1.96 * se)
    return ror, ci_lower, ci_upper

def calc_prr(a, b, c, d):
    if (a + b) == 0 or (c + d) == 0 or c == 0:
        return None, None
    prr = (a / (a + b)) / (c / (c + d))
    chi2 = ((a*d - b*c)**2 * (a+b+c+d)) / ((a+b)*(c+d)*(a+c)*(b+d))
    return prr, chi2

# Load drug and reaction files
drug = pd.read_csv(f"{BASE}/DRUG26Q1.txt", sep="$", low_memory=False)
reac = pd.read_csv(f"{BASE}/REAC26Q1.txt", sep="$", low_memory=False)

print(f"Drug rows: {len(drug)}, Reaction rows: {len(reac)}")

# Clean drug names
drug["drugname_clean"] = drug["drugname"].str.upper().str.strip()

# Join on primaryid
merged = drug[["primaryid", "drugname_clean"]].merge(
    reac[["primaryid", "pt"]], on="primaryid"
)

print(f"Merged rows: {len(merged)}")
print(merged.head())

# ---- Choose a drug and event to test ----
TARGET_DRUG = "METHOTREXATE"
TARGET_EVENT = "Nausea"

a = len(merged[(merged["drugname_clean"] == TARGET_DRUG) & (merged["pt"] == TARGET_EVENT)])
b = len(merged[(merged["drugname_clean"] == TARGET_DRUG) & (merged["pt"] != TARGET_EVENT)])
c = len(merged[(merged["drugname_clean"] != TARGET_DRUG) & (merged["pt"] == TARGET_EVENT)])
d = len(merged[(merged["drugname_clean"] != TARGET_DRUG) & (merged["pt"] != TARGET_EVENT)])

print(f"\na={a}, b={b}, c={c}, d={d}")

ror, ci_l, ci_u = calc_ror(a, b, c, d)
prr, chi2 = calc_prr(a, b, c, d)

if ror:
    print(f"\n{TARGET_DRUG} + {TARGET_EVENT}")
    print(f"ROR={ror:.2f} (95% CI {ci_l:.2f}-{ci_u:.2f})")
    print(f"PRR={prr:.2f}, chi2={chi2:.2f}")
else:
    print(f"\nNo data found for {TARGET_DRUG} + {TARGET_EVENT} — try different names.")
    print("Top 10 drugs by report count:")
    print(merged["drugname_clean"].value_counts().head(10))
    print("\nTop 10 events by report count:")
    print(merged["pt"].value_counts().head(10))