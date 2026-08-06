# Executive Summary: SME Churn Analysis & Retention Strategy

---

##  Executive Summary

### **Situation**
PowerCo is currently experiencing an annual customer churn rate of **9.7%** across its Small and Medium Enterprise (SME) customer base[cite: 4]. Historically, the executive team operated under the hypothesis that this attrition was primarily driven by customer price sensitivity—specifically, price hikes in off-peak and variable tariff structures causing dissatisfied clients to switch suppliers[cite: 4].

---

### **Complication**
Our rigorous exploratory data analysis and hypothesis testing revealed that **price sensitivity is NOT the primary driver of customer churn**[cite: 4]. 

* **No Linear Correlation:** December-to-January price variations (`*_diff_dec_jan_*`) exhibit a near-zero linear correlation ($r \approx 0$) with churn status[cite: 4]. Both retained and churned cohorts show virtually identical price-change distributions[cite: 4].
* **High-Value Risk:** Churned accounts carry an average tenure of **5.1 years**[cite: 4]. PowerCo is not merely losing newly onboarded clients; it is losing established, high-volume, and high-margin accounts[cite: 4].
* **Imbalance Distortion:** A naive strategy or uncalibrated model predicting "no churn" yields ~90% accuracy due to class imbalance, disguising substantial revenue loss from uncaptured churners[cite: 4].

---

### **Question**
How can PowerCo accurately predict which high-value SME customers are at risk of churning, and what strategic retention mechanisms should be deployed to protect net margin without eroding profitability through blanket price cuts?

---

### **Recommended Solution**

1. **Deploy a Non-Linear Predictive Model:** 
   Implement the trained **Random Forest Classifier** with balanced class weights (`class_weight='balanced'`)[cite: 3]. The model effectively captures non-linear decision boundaries and identifies key churn drivers, including:
   * **Net Margin (`net_margin`)**[cite: 3, 5]
   * **Account Tenure (`tenure` / `months_activ`)**[cite: 3, 5]
   * **12-Month Consumption Baseline (`cons_12m`)**[cite: 3, 5]
   * **Contract Renewal Proximity (`months_renewal`)**[cite: 3, 5]

2. **Shift to Proactive, Targeted Interventions:** 
   Stop offering widespread, indiscriminate price discounts[cite: 4]. Instead, deploy sales and retention teams to target the top **15–20% highest-risk accounts** 60 to 90 days prior to contract expiration (`months_renewal`)[cite: 3, 4, 5].

3. **Value-Weighted Retention Perks:** 
   Offer high-margin, long-tenure accounts custom loyalty benefits, structured contract extension incentives, or simplified single-tier tariff options rather than simple rate cuts[cite: 4].

---

##  Key Findings Summary

| Metric / Category | Observed Finding | Operational / Strategic Impact ("So What?") |
| :--- | :--- | :--- |
| **Baseline Churn Rate** | **9.72%** across 14,606 SME accounts[cite: 4]. | Class imbalance requires evaluating models via **ROC-AUC**, **Precision**, and **Recall** rather than raw accuracy[cite: 3, 4]. |
| **Price Delta Impact** | Dec–Jan price differences show negligible correlation ($r \approx -0.005$) with churn[cite: 4]. | Blanket price reductions will fail to stop churn and will directly erode operational net margins[cite: 4]. |
| **Primary Churn Drivers** | Churn is governed by profitability (`net_margin`), tenure, volume, and contract renewal timing[cite: 3, 5]. | Risk scores must be weighted by customer lifetime value (CLV) to prioritize high-volume "whale" accounts[cite: 4]. |
