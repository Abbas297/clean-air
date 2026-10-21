# Advisory Rule Evaluation & Success Criterion 4 Audit

## 1. Executive Summary & Plain Verdict

**Declared Success Criterion 4:** *"At +24h, advisory recall >= baseline recall, with precision no worse."*

**Plain Verdict: Success Criterion 4 is NOT MET at +24h.**

- On the held-out test split at **+24h**, the baseline rule (current PM2.5 ≥ 55.5 µg/m³) achieves a **Recall of 82.4%** and a **Precision of 82.4%**.
- **Rule (c) (`p50 >= 55.5`):** Achieves higher Precision (**88.0%** vs 82.4%) and cuts the False Alarm Rate nearly in half (**7.6%** vs 13.3%), but achieves a lower Recall (**75.0%** vs 82.4%). It fails the recall requirement.
- **Rule (b) (`p90 >= 55.5`):** Achieves near-perfect Recall (**99.0%** vs 82.4%), but suffers poor Precision (**53.5%** vs 82.4%) and an unacceptably high False Alarm Rate (**64.7%** vs 13.3%). It fails the precision requirement.
- **Earlier Horizons (+12h and +6h):** In contrast to +24h, **Rule (c) meets Success Criterion 4 at both +12h and +6h** (at +12h: Recall 80.5% vs 78.8%, Precision 84.8% vs 78.8%; at +6h: Recall 86.1% vs 80.6%, Precision 81.7% vs 80.6%).

---

## 2. Validation Split Rule Selection (+24h)

Evaluated on 15,706 validation hours (July 2, 2024 to June 30, 2025, containing the 2024–25 smog season):

| Decision Rule | Trigger Condition | Recall | Precision | False Positive Rate (FPR) | F1-Score | Operational Interpretation |
|---|---|---|---|---|---|---|
| **Baseline A** | Current PM2.5 ≥ 55.5 µg/m³ | 78.5% | 78.6% | 22.0% | 0.785 | Persistence benchmark |
| **Candidate (c)** | LightGBM $p_{50} \ge 55.5$ | 68.9% | 84.8% | 12.7% | 0.760 | Conservative; high precision, low false alarms |
| **Candidate (b)** | Conformal $p_{90} \ge 55.5$ | 98.9% | 60.9% | 65.2% | 0.754 | Sensitive; near-total recall, high false alarms |

### Selection Decision on Validation:
- **Candidate (c) ($p_{50} \ge 55.5$) is selected as the operational advisory rule.** While it misses some exceedances at +24h due to quantile regression shrinkage toward the climatological mean, its $F_1$ score is higher (0.760 vs 0.754) and its false alarm rate is dramatically lower (12.7% vs 65.2%). A 65% false-alarm rate under Rule (b) would trigger school closures on 2 out of every 3 clean days, destroying public trust.

---

## 3. Held-Out Test Evaluation Across Horizons (+6h, +12h, +24h)

Evaluated on identical held-out test rows ($N = 23,529$ to $23,547$, July 2025 to September 2026):

| Horizon | Decision Rule | N Samples | Recall | Precision | False Positive Rate (FPR) | F1-Score | Beats Baseline on Both? |
|---|---|---|---|---|---|---|---|
| **+6h** | Baseline A (Current $\ge 55.5$) | 23,547 | 80.6% | 80.6% | 14.6% | 0.806 | - |
| **+6h** | **Chosen Rule (c) ($p_{50} \ge 55.5$)** | 23,547 | **86.1%** | **81.7%** | **14.5%** | **0.839** | **YES** |
| *+6h* | *Candidate (b) ($p_{90} \ge 55.5$)* | 23,547 | *98.9%* | *57.7%* | *54.6%* | *0.729* | *NO (Precision trails)* |
| **+12h** | Baseline A (Current $\ge 55.5$) | 23,541 | 78.8% | 78.8% | 16.0% | 0.788 | - |
| **+12h** | **Chosen Rule (c) ($p_{50} \ge 55.5$)** | 23,541 | **80.5%** | **84.8%** | **10.8%** | **0.826** | **YES** |
| *+12h* | *Candidate (b) ($p_{90} \ge 55.5$)* | 23,541 | *99.1%* | *52.5%* | *67.2%* | *0.687* | *NO (Precision trails)* |
| **+24h** | Baseline A (Current $\ge 55.5$) | 23,529 | 82.4% | 82.4% | 13.3% | 0.824 | - |
| **+24h** | **Chosen Rule (c) ($p_{50} \ge 55.5$)** | 23,529 | **75.0%** | **88.0%** | **7.6%** | **0.810** | **NO (Recall trails)** |
| *+24h* | *Candidate (b) ($p_{90} \ge 55.5$)* | 23,529 | *99.0%* | *53.5%* | *64.6%* | *0.695* | *NO (Precision trails)* |

---

## 4. City-by-City Breakdown at +24h

| City | Rule | N Samples | Exceedance Hours | Recall | Precision | False Positive Rate (FPR) | Meets Criterion 4? |
|---|---|---|---|---|---|---|---|
| **Islamabad** | Baseline A | 4,861 | 795 | 70.8% | 70.8% | 5.7% | - |
| **Islamabad** | Rule (c) ($p_{50}$) | 4,861 | 795 | **71.7%** | **76.7%** | **4.3%** | **YES** |
| **Lahore** | Baseline A | 7,958 | 5,756 | 83.1% | 82.9% | 44.7% | - |
| **Lahore** | Rule (c) ($p_{50}$) | 7,958 | 5,756 | **70.8%** | **90.0%** | **20.6%** | **NO** |
| **Peshawar** | Baseline A | 10,710 | 3,537 | 83.9% | 84.0% | 7.9% | - |
| **Peshawar** | Rule (c) ($p_{50}$) | 10,710 | 3,537 | **82.6%** | **87.9%** | **5.6%** | **NO** |

### City Analysis at +24h:
- **Islamabad:** **Meets Criterion 4**. Rule (c) achieves Recall of 71.7% (vs 70.8% baseline) and Precision of 76.7% (vs 70.8% baseline), with lower false alarms (4.3% vs 5.7%).
- **Lahore:** Precision is substantially higher (90.0% vs 82.9%, FPR 20.6% vs 44.7%), but Recall is 70.8% (vs 83.1% baseline). Because baseline persistence benefits strongly from strong autocorrelation during prolonged multi-day smog episodes, $p_{50}$ point estimates are more cautious, leading to missed exceedance hours.
- **Peshawar:** Precision is higher (87.9% vs 84.0%), but Recall trails marginally (82.6% vs 83.9%). Recall is nearly parity (-1.3%), reflecting robust zero-shot transfer despite 0 training hours.

---

## 5. Methodological Integrity & Constraints
- Evaluated strictly using frozen LightGBM models in `data/gold/models/` and validation conformal adjustments.
- Evaluated strictly on the held-out test split (`2025-07-02` to `2026-09-30T23:59:59Z`) with 24-hour embargo.
- Zero models, baselines, split bounds, or pipeline code were modified.

*Generated automatically by `scripts/advisory_eval.py`.*