# Executive Summary: SME Churn Analysis & Retention Strategy

---

##  Executive Summary

### **Situation**
PowerCo is currently experiencing an annual customer churn rate of **9.7%** across its Small and Medium Enterprise (SME) customer base. Historically, the executive team operated under the hypothesis that this attrition was primarily driven by customer price sensitivity specifically, price hikes in off-peak and variable tariff structures causing dissatisfied clients to switch suppliers.

---

### **Complication**
Our rigorous exploratory data analysis and hypothesis testing revealed that **price sensitivity is NOT the primary driver of customer churn**. 

* **No Linear Correlation:** December-to-January price variations (`*_diff_dec_jan_*`) exhibit a near-zero linear correlation ($r \approx 0$) with churn status. Both retained and churned cohorts show virtually identical price-change distributions.
* **High-Value Risk:** Churned accounts carry an average tenure of **5.1 years**. PowerCo is not merely losing newly onboarded clients; it is losing established, high-volume, and high-margin accounts.
* **Imbalance Distortion:** A naive strategy or uncalibrated model predicting "no churn" yields ~90% accuracy due to class imbalance, disguising substantial revenue loss from uncaptured churners.

---

### **Question**
How can PowerCo accurately predict which high-value SME customers are at risk of churning, and what strategic retention mechanisms should be deployed to protect net margin without eroding profitability through blanket price cuts?

---

### **Recommended Solution**

1. **Deploy a Non-Linear Predictive Model:** 
   Implement the trained **Random Forest Classifier** with balanced class weights (`class_weight='balanced'`). The model effectively captures non-linear decision boundaries and identifies key churn drivers, including:
   * **Net Margin (`net_margin`)**
   * **Account Tenure (`tenure` / `months_activ`)**
   * **12-Month Consumption Baseline (`cons_12m`)**
   * **Contract Renewal Proximity (`months_renewal`)**

2. **Shift to Proactive, Targeted Interventions:** 
   Stop offering widespread, indiscriminate price discounts. Instead, deploy sales and retention teams to target the top **15–20% highest-risk accounts** 60 to 90 days prior to contract expiration (`months_renewal`).

3. **Value-Weighted Retention Perks:** 
   Offer high-margin, long-tenure accounts custom loyalty benefits, structured contract extension incentives, or simplified single-tier tariff options rather than simple rate cuts.

---

##  Key Findings Summary

| Metric / Category | Observed Finding | Operational / Strategic Impact ("So What?") |
| :--- | :--- | :--- |
| **Baseline Churn Rate** | **9.72%** across 14,606 SME accounts. | Class imbalance requires evaluating models via **ROC-AUC**, **Precision**, and **Recall** rather than raw accuracy. |
| **Price Delta Impact** | Dec–Jan price differences show negligible correlation ($r \approx -0.005$) with churn. | Blanket price reductions will fail to stop churn and will directly erode operational net margins. |
| **Primary Churn Drivers** | Churn is governed by profitability (`net_margin`), tenure, volume, and contract renewal timing. | Risk scores must be weighted by customer lifetime value (CLV) to prioritize high-volume "whale" accounts. |
