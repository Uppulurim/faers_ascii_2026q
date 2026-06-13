 # AI-Driven Pharmacovigilance Signal Detection System (FAERS Analytics Pipeline)

## Overview

This project is an end-to-end pharmacovigilance analytics pipeline built on FDA FAERS (FDA Adverse Event Reporting System) data. It processes raw adverse event reports, normalizes drug names using RxNorm mapping, and generates safety signals to identify potential drug–adverse event associations.

The system demonstrates how large-scale healthcare data can be transformed into actionable insights using Python-based data engineering and signal detection techniques.

---

## Key Features

* **FAERS Data Processing Pipeline**

  * Parses and structures raw FAERS quarterly datasets
  * Cleans and standardizes multi-table adverse event data

* **Drug Normalization (RxNorm Mapping)**

  * Maps raw drug names to standardized RxNorm identifiers
  * Improves consistency across reports and datasets

* **Signal Detection Engine**

  * Aggregates drug–event co-occurrence patterns
  * Generates signal strength metrics for pharmacovigilance review

* **Batch Investigation Framework**

  * Enables scalable processing of multiple FAERS batches
  * Supports iterative analysis across reporting periods

* **Signal Output Generation**

  * Produces structured outputs for downstream analysis and reporting

---

## Project Structure

```
faers_ascii_2026q1/
│
├── agent_pipeline.py        # Main pipeline orchestration
├── investigate_batches.py   # Batch-level processing and analysis
├── pipleline.py             # Core ETL workflow
├── real_signals.py          # Signal detection logic
├── rxnorm_mapper.py         # Drug normalization using RxNorm
│
├── signals_all.csv         # Aggregated signal outputs
├── signals_all_drugs.csv   # Drug-level signal dataset
├── review_log.csv          # Processing logs
├── rxnorm_cache.csv        # Cached RxNorm mappings
│
├── ASCII/                  # Raw FAERS datasets (excluded from GitHub)
├── Deleted/                # Legacy or removed files
├── FAQs.pdf
└── Readme.pdf
```

---

## Technologies Used

* Python 3
* Pandas
* Data Engineering Pipelines
* RxNorm Drug Ontology
* FAERS Public Dataset (FDA)
* CSV-based signal analytics

---

## Key Concepts

* Pharmacovigilance
* Adverse Event Reporting (FAERS)
* Drug Safety Signal Detection
* Entity Normalization (RxNorm)
* Healthcare Data Analytics
* Batch Data Processing Pipelines

---

## Important Note

Large FAERS raw datasets are excluded from this repository due to GitHub file size limitations. Only processed code, pipelines, and derived outputs are included.

---

## Future Improvements

* Integrate machine learning models for predictive signal detection
* Add statistical disproportionality analysis (PRR, ROR, Bayesian methods)
* Build interactive dashboard for signal exploration
* Deploy pipeline using cloud infrastructure (AWS/GCP)

---

## Author

Developed as part of an AI-driven pharmacovigilance analytics project focusing on real-world drug safety signal detection.
# FAERS Pharmacovigilance Platform
