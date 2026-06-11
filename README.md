# 🏥 Healthcare AI Risk Assessment — Patient Prioritization Bias

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AIF360](https://img.shields.io/badge/IBM%20AIF360-052FAD?style=for-the-badge&logo=ibm&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![Risk](https://img.shields.io/badge/Risk%20Level-HIGH-red?style=for-the-badge)
![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-High%20Risk%20AI-orange?style=for-the-badge)

> **Detecting racial bias in patient prioritization AI — based on the real Optum Algorithm Bias case (Science, 2019)**

---

## 📌 Real World Reference

In **2019**, researchers published in **Science journal** that a widely-used healthcare AI algorithm developed by **Optum** was racially biased.

The algorithm used **annual healthcare cost** as a proxy for medical need. But Black patients had historically **lower costs** — not because they were healthier, but because of **unequal access to care**.

**Result:** Black patients were assigned lower risk scores → received less preventive care → worse health outcomes.

This project simulates and audits such bias using IBM AIF360.

---

## 🎯 Problem Statement

| Issue | Detail |
|-------|--------|
| **Bias Type** | Racial Bias (Black patients under-prioritized) |
| **Root Cause** | Annual cost used as biased proxy for medical need |
| **Impact** | Patients miss critical preventive care |
| **Legal Risk** | Civil Rights Act Section 1557 violation |
| **EU AI Act** | High-risk AI — requires fundamental rights assessment |

---

## 📊 Fairness Metrics Used

| Metric | Result | Threshold |
|--------|--------|-----------|
| **Statistical Parity Difference (SPD)** | -0.18 → 0.01 | ≈ 0.0 |
| **Disparate Impact (DI)** | 0.69 → 0.97 | ≥ 0.80 |
| **Average Odds Difference (AOD)** | Measured | ≈ 0.0 |

---

## 🔧 Methodology

```
Step 1: Create patient dataset (1,500 records)
    ↓
Step 2: Identify racial bias via cost proxy
    ↓
Step 3: Measure SPD & DI (Before)
    ↓
Step 4: Train Random Forest model
    ↓
Step 5: Analyze feature importance (SHAP-style)
    ↓
Step 6: Remove biased proxy + Apply Reweighing
    ↓
Step 7: Generate Risk Register + Audit Report
```

---

## 📋 Risk Register

| ID | Risk | Likelihood | Impact | Score |
|----|------|-----------|--------|-------|
| R01 | Racial bias in triage | High (4/5) | Critical (5/5) | 20/25 |
| R02 | Cost proxy misuse | High (4/5) | Critical (5/5) | 20/25 |
| R03 | Under-treatment bias | Medium (3/5) | Critical (5/5) | 15/25 |
| R04 | Legal/compliance risk | Medium (3/5) | High (4/5) | 12/25 |
| R05 | Model drift | Medium (3/5) | Medium (3/5) | 9/25 |

---

## 📈 Results

| Metric | Before | After |
|--------|--------|-------|
| DI | 0.69 ❌ | 0.97 ✅ |
| Racial Bias | Present | Removed |
| Compliance | ❌ Fail | ✅ Pass |

---

## 📜 Regulatory Framework

| Regulation | Requirement | Status |
|-----------|------------|--------|
| **Civil Rights Act §1557** | No racial discrimination in healthcare | ✅ Addressed |
| **EU AI Act** | High-risk AI — mandatory bias audit | ✅ Documented |
| **NIST AI RMF** | Map → Measure → Manage risk | ✅ Applied |

---

## 🛠️ Tech Stack

- **Python 3.8+** | **IBM AIF360** | **Scikit-learn** | **Pandas / NumPy**

---

## 🚀 How to Run

```bash
pip install aif360 pandas numpy scikit-learn
python healthcare_ai_risk.py
```

---

## 👤 Author

**Santosh Gaud** — AI Risk Specialist | Fairness Auditor

[![GitHub](https://img.shields.io/badge/GitHub-Saantosh--AI108-black?style=flat-square&logo=github)](https://github.com/Saantosh-AI108)
[![Email](https://img.shields.io/badge/Email-santoshgaudai108@gmail.com-red?style=flat-square&logo=gmail)](mailto:santoshgaudai108@gmail.com)

---
*Part of AI Risk Assessment Portfolio — github.com/Saantosh-AI108*
