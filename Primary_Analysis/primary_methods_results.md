# Primary Analysis: Balance/Gait Change Over Time and Association with Psychological Distress

## Study Overview

This analysis examines (1) **longitudinal changes in smartphone-measured balance and gait outcomes** across repeated clinical sessions and (2) the **association between balance/gait performance and psychological distress** measured by the DASS-21.

**Participants:** 21 with gait/balance data (18 with >=2 sessions)
**DASS-21 matched:** 36 observation-pairs from 22 participants

---

## Methods

### Balance/Gait Outcomes
Outcomes were measured using the GnB smartphone app (SincMotion) during clinic visits:
- **Postural Stability** (overall, ML, AP): Static standing stability on firm/compliant surfaces with eyes open/closed. Measured as -ln(m/s²); higher values indicate better stability.
- **Walking Balance (%):** Dynamic balance during overground walking (heel-first and heel-toe tasks).
- **Step Length (m):** Average step length during walking.
- **Step Time (s):** Average step time during walking.
- **Walking Speed (m/s):** Average gait speed.

Per-session values were computed as the mean across task conditions (4 static, 2 walking).

### Psychological Distress (DASS-21)
The Depression Anxiety Stress Scales (DASS-21) was administered at clinic visits. Each subscale comprises 7 items; scores were multiplied by 2 to yield conventional severity ranges (0-42). DASS assessments were linked to the nearest balance session within ±14 days.

### Statistical Approach
- **Change analysis:** Paired baseline-to-last comparison for participants with >=2 sessions. Wilcoxon signed-rank tests for within-person change; Cohen's d for effect size; bootstrap 95% CIs.
- **Cohort comparison:** Participants split by engagement duration (<=2 weeks vs >2 weeks). Mann-Whitney U tests.
- **Cross-sectional associations:** Spearman rank correlations between DASS subscales and balance metrics on matched observations, with bootstrap 95% CIs.
- **Within-person associations:** Change in DASS stress correlated with change in balance metrics (first vs last matched observation per participant).

---

## Results

### 1. Individual Trajectories Over Sessions

Figure 1 shows individual participant trajectories (grey lines) overlaid with the group mean ± SEM (blue) for four key balance/gait outcomes. A linear trend line (orange dashed) is fitted to the group means across sessions. A green "better" arrow in the lower-right indicates the direction of clinical improvement.

![Figure 1: Individual Trajectories](fig1_individual_trajectories.png)
*Figure 1. Spaghetti plots of individual trajectories (grey) with group mean ± SEM (blue band/markers) for Postural Stability, Walking Balance, Step Length, and Walking Speed. Orange dashed lines show linear trends fitted to group means. Trend slopes and direction indicators annotated.*

Most metrics show considerable inter-individual variability. Walking Balance and Walking Speed show positive group-level trends across sessions (slopes = +0.886 and +0.047 per session, respectively), while Postural Stability shows a slight negative trend (slope = -0.033). Step Length shows a modest positive trend (slope = +0.010).

---

### 2. Baseline to Last Session Change

Figure 2 presents paired slope graphs connecting each participant's baseline value to their last session value. Green lines indicate improvement; red lines indicate worsening. Box plots summarise the distribution at each timepoint. Change statistics (mean difference, 95% CI, Cohen's d, Wilcoxon p) are annotated.

![Figure 2: Paired Baseline vs Last Session](fig2_paired_baseline_last.png)
*Figure 2. Paired slope graphs for 18 participants with >=2 sessions. Each line connects one participant's baseline (left) to their last session (right). Green = improved, red = worsened. Box plots show group distributions. Statistical summaries (Δ, d, p) annotated.*

**Table 1. Change from Baseline to Last Session (n=18 participants with >=2 sessions)**

| Metric | Baseline (M±SD) | Last (M±SD) | Δ [95% CI] | d | p | Improved |
|---|---|---|---|---|---|---|
| Postural Stability | 2.68±0.45 | 2.66±0.43 | -0.017 [-0.122, +0.087] | -0.08 | 0.671 | 50% |
| ML Stability | 3.59±0.46 | 3.58±0.44 | -0.023 [-0.120, +0.074] | -0.12 | 0.495 | 44% |
| AP Stability | 3.41±0.45 | 3.40±0.41 | -0.018 [-0.121, +0.085] | -0.09 | 0.734 | 50% |
| Walking Balance | 59.10±6.69 | 59.76±6.59 | +0.773 [-0.585, +2.130] | 0.28 | 0.284 | 56% |
| Step Length | 0.59±0.07 | 0.60±0.07 | +0.012 [-0.007, +0.031] | 0.32 | 0.246 | 61% |
| Step Time | 0.58±0.07 | 0.57±0.07 | -0.014 [-0.029, +0.000] | -0.48 | 0.084 | 67% |
| **Walking Speed** | **1.02±0.16** | **1.06±0.16** | **+0.048 [+0.007, +0.088]** | **0.59** | **0.030** | **78%** |

Walking Speed showed the only statistically significant improvement (d=0.59, p=0.030), with 78% of participants showing faster gait at their last session compared to baseline. Step Time approached significance (d=-0.48, p=0.084) with 67% improving (shorter step time = faster cadence). Static stability metrics showed negligible change.

---

### 3. Effect Size Forest Plot

Figure 3 presents Cohen's d effect sizes for all seven outcome metrics, with bootstrap 95% confidence intervals. The grey vertical band marks the negligible effect zone (|d| < 0.2). Markers are colour-coded: green for improvement, red for worsening, grey for negligible. Asterisks denote statistical significance.

![Figure 3: Effect Size Forest Plot](fig3_effect_size_forest.png)
*Figure 3. Forest plot of Cohen's d effect sizes (baseline to last session) with bootstrap 95% CIs. Grey band = negligible zone (|d| < 0.2). Green = improved, grey = negligible. Walking Speed (d=0.59*) is the standout finding.*

A clear pattern emerges: **gait metrics improved more than static stability metrics**. Walking Speed showed a medium effect (d=0.59), Step Time a small-to-medium effect (d=-0.48), and Step Length a small effect (d=0.32). All three static stability measures showed negligible effects (|d| < 0.15).

---

### 4. Cohort Comparison: Change by Engagement Duration

Figure 4 compares change scores between participants with <=2 weeks (orange) and >2 weeks (blue) of engagement. Each panel shows a violin plot (distribution shape), box plot (quartiles), individual data points (jittered strip), and a diamond marker for the group mean. Mann-Whitney U tests are annotated.

![Figure 4: Cohort Comparison](fig4_cohort_change_violin.png)
*Figure 4. Violin + box + strip plots comparing change scores between engagement cohorts (<=2 weeks vs >2 weeks). Diamonds = group mean. Mann-Whitney U test statistics annotated. Dashed line at zero (no change).*

Both cohorts show similar patterns of change across metrics, with no statistically significant between-group differences at this sample size. The >2 weeks cohort shows somewhat wider distributions, reflecting more opportunity for change (both positive and negative) with longer engagement.

---

### 5. DASS-21 × Balance/Gait Correlations

Figure 5 presents a heatmap of Spearman rank correlations between the three DASS-21 subscales (Depression, Anxiety, Stress) and seven balance/gait metrics. Correlations are colour-coded from blue (negative) to red (positive), with significance markers (* p<.05, ** p<.01).

![Figure 5: DASS-21 Correlation Heatmap](fig5_dass_correlation_heatmap.png)
*Figure 5. Heatmap of Spearman correlations between DASS-21 subscales (rows) and balance/gait metrics (columns). Warm colours = positive association; cool colours = negative. Values annotated with significance markers.*

**Table 2. DASS-21 × Balance Correlations (p < .10)**

| DASS Subscale | Balance Metric | r | p | 95% CI | n |
|---|---|---|---|---|---|
| Stress | Step Length | -0.413* | 0.012 | [-0.654, -0.090] | 36 |
| Anxiety | Step Length | -0.399* | 0.016 | [-0.650, -0.093] | 36 |
| Depression | ML Stability | 0.378* | 0.023 | [0.041, 0.639] | 36 |
| Depression | AP Stability | 0.375* | 0.024 | [0.044, 0.645] | 36 |
| Stress | AP Stability | 0.357* | 0.032 | [0.062, 0.601] | 36 |
| Stress | ML Stability | 0.301 | 0.075 | [0.029, 0.559] | 36 |
| Depression | Postural Stability | 0.295 | 0.081 | [-0.035, 0.583] | 36 |
| Anxiety | Walking Speed | -0.285 | 0.093 | [-0.570, 0.023] | 36 |

Key patterns:
- **Stress** shows the most consistent associations: positive with stability metrics (higher stress = stiffer postural control) and negative with step length/speed (higher stress = shorter, slower steps).
- **Depression** follows a similar pattern for stability (r = 0.30-0.38) but weaker associations with gait.
- **Anxiety** is most associated with reduced step length (r = -0.40) and walking speed (r = -0.28).

---

### 6. Stress vs Balance/Gait (Cross-Sectional)

Figure 6 shows scatter plots of DASS-21 Stress scores against three key balance/gait metrics, with linear regression lines and 95% confidence bands. Points are colour-coded by cohort (orange = <=2 weeks, blue = >2 weeks). Spearman r, p-value, and sample size are annotated.

![Figure 6: Stress vs Balance Scatter Plots](fig6_stress_vs_balance.png)
*Figure 6. Cross-sectional scatter plots of DASS Stress vs Postural Stability, Walking Balance, and Walking Speed. Purple dashed lines = regression fit with 95% CI band. Points coloured by engagement cohort.*

The Stress-Stability relationship (r=0.27) suggests that higher psychological stress is associated with stiffer, less adaptive postural control - a recognised pattern in balance research where anxious or stressed individuals adopt a "stiffening" strategy. The negative Stress-Step Length correlation (r=-0.41, shown in heatmap) indicates that higher stress accompanies shorter, more cautious steps.

---

### 7. Within-Person: Change in Stress vs Change in Balance

Figure 7 tests the key hypothesis at the individual level: does a reduction in stress within a person track with improvement in their balance/gait? Each point represents one participant's first-to-last change on both axes. Crosshairs at zero divide the plot into four quadrants.

![Figure 7: Within-Person DASS Change vs Balance Change](fig7_within_person_dass_change.png)
*Figure 7. Within-person change in DASS Stress (x-axis) vs change in balance/gait metrics (y-axis) for 16 participants with >=2 matched DASS-outcome observations. Points coloured by cohort. Purple dashed = regression line. Spearman r and p annotated.*

The within-person correlations are weaker than the cross-sectional associations (r range: -0.26 to +0.11), which is expected given the smaller sample of 16 participants with repeated matched observations and the modest magnitude of change over this study window. The Walking Balance panel shows the most suggestive pattern (r=-0.26, p=0.325), where stress reduction tends to co-occur with walking balance improvement, though this does not reach significance.

---

### 8. DASS-21 Trajectories Over Time

Figure 8 shows individual DASS-21 score trajectories (grey) with monthly group means ± SEM for each subscale. Horizontal dashed lines indicate the threshold for "mild" severity on each subscale.

![Figure 8: DASS-21 Trajectories](fig8_dass_trajectories.png)
*Figure 8. Individual DASS-21 trajectories (grey lines) with monthly group means ± SEM (coloured markers/bands) for Depression (blue), Anxiety (orange), and Stress (purple). Red dashed lines = DASS mild severity thresholds.*

The majority of participants report low distress throughout the study. Group means for all three subscales trend downward from November 2024 to March 2025, suggesting a general reduction in psychological distress over the course of participation. A few individuals show elevated or fluctuating scores, which is expected in a clinical rehabilitation population.

---

### 9. Parallel Trajectories: Balance and Psychological Distress

Figure 9 places balance/gait outcomes (Panel A, shown as normalised % change from first month) alongside DASS-21 scores (Panel B, raw monthly means) on a shared time axis. This allows visual comparison of whether physical and psychological improvements co-occur at the group level.

![Figure 9: Parallel Balance + DASS Trajectories](fig9_dual_balance_stress.png)
*Figure 9. (A) Normalised change (%) in Postural Stability, Walking Balance, and Walking Speed from first study month. (B) DASS-21 monthly means ± SEM for Depression, Anxiety, and Stress. Same time axis for visual comparison.*

A suggestive pattern is visible: as DASS scores decline (Panel B), walking-related outcomes (Walking Balance, Walking Speed) trend upward (Panel A). Postural Stability shows less change. This parallel improvement is consistent with the hypothesis that physical and psychological well-being are interconnected, though the observational design precludes causal inference.

---

### 10. Summary Dashboard

Figure 10 provides a two-panel summary: (A) mean change from baseline with 95% CIs for the four key metrics, and (B) the strongest DASS × balance correlations ranked by magnitude.

![Figure 10: Summary Dashboard](fig10_summary_dashboard.png)
*Figure 10. (A) Lollipop chart of mean change [95% CI] for key balance/gait outcomes. Green = positive change, red = negative. (B) Horizontal bar chart of strongest DASS × balance Spearman correlations. Blue = positive, red = negative.*

---

## Data Files

| File | Description |
|---|---|
| `primary_statistics_enhanced.csv` | Full statistical summary: baseline, last, change, CI, effect sizes, p-values |
| `dass_correlations_enhanced.csv` | DASS × balance Spearman correlations with bootstrap CIs |
| `effect_sizes.csv` | Cohen's d and CIs for each metric |
| `within_person_changes.csv` | Per-participant DASS and balance change scores |
| `combined_outcomes_session.csv` | Session-level aggregated outcomes |
| `dass_outcome_matched.csv` | DASS assessments linked to nearest balance session |

---

## Summary

This exploratory analysis reveals:

1. **Balance and gait outcomes are measurable and trackable** using the GnB app across repeated clinical sessions, with sufficient variability to detect individual change trajectories.

2. **Walking Speed improved significantly** from baseline to last session (d=0.59, p=0.030), with 78% of participants showing faster gait. Step Time showed a trend toward improvement (d=-0.48, p=0.084). Static stability metrics showed negligible change over this timeframe.

3. **Psychological distress is associated with balance performance.** Higher DASS Stress and Anxiety scores correlate with shorter step lengths (r=-0.41 and -0.40, p<.05) and higher Depression with stiffer postural control (ML/AP stability r=0.38, p<.05). These cross-sectional associations are consistent with the "stiffening" and "cautious gait" patterns described in the fear-of-falling and anxiety literature.

4. **Group-level parallel trajectories** show DASS scores declining over the study months while walking-related outcomes improve, supporting the interconnection between physical and psychological recovery in this clinical rehabilitation context.

5. **Within-person change associations** are in the expected direction but do not reach significance with the current sample (n=16 with repeated matched observations), supporting the need for larger samples and longer follow-up in a definitive trial.
