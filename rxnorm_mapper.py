import requests
import time
import pandas as pd

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

def get_rxcui(drug_name):
    try:
        r = requests.get(f"{RXNORM_BASE}/rxcui.json", params={"name": drug_name}, timeout=5)
        r.raise_for_status()
        data = r.json()
        ids = data.get("idGroup", {}).get("rxnormId")
        return ids[0] if ids else None
    except Exception:
        return None

def get_approximate_match(drug_name):
    try:
        r = requests.get(f"{RXNORM_BASE}/approximateTerm.json",
                         params={"term": drug_name, "maxEntries": 1}, timeout=5)
        r.raise_for_status()
        candidates = r.json().get("approximateGroup", {}).get("candidate", [])
        if candidates:
            return candidates[0].get("rxcui"), candidates[0].get("score")
        return None, None
    except Exception:
        return None, None

def get_ingredient_name(rxcui):
    try:
        r = requests.get(f"{RXNORM_BASE}/rxcui/{rxcui}/property.json",
                         params={"propName": "RxNorm Name"}, timeout=5)
        r.raise_for_status()
        return r.json().get("propConceptGroup", {}).get("propConcept", [{}])[0].get("propValue")
    except Exception:
        return None

def map_drug_names(unique_drug_names, delay=0.1):
    results = []
    for name in unique_drug_names:
        rxcui = get_rxcui(name)
        score = 100
        if not rxcui:
            rxcui, score = get_approximate_match(name)
        norm_name = get_ingredient_name(rxcui) if rxcui else None
        results.append({
            "original_name": name,
            "rxcui": rxcui,
            "normalized_name": norm_name,
            "match_score": score
        })
        time.sleep(delay)
    return pd.DataFrame(results)

if __name__ == "__main__":
    test_drugs = ["TYLENOL", "advil", "Lipitor", "metfromin", "ASPIRIN 81MG"]
    df = map_drug_names(test_drugs)
    print(df)