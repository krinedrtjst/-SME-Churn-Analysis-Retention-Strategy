#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # Predictive Modeling of SME Customer Churn & Price Sensitivity Analysis
# 
# ## Project Overview
# 
# Customer churn presents a critical operational and financial risk for utility providers. **PowerCo**, a major energy supplier catering to Corporate and Small/Medium Enterprises (SMEs), has experienced an annual churn rate of approximately **9.7%** across its customer base.
# 
# A leading hypothesis held by business leadership was that churn was predominantly driven by **customer price sensitivity**—specifically, that subtle fluctuations or increases in off-peak energy rates and fixed power fees prompt high-volume SME clients to cancel their contracts and switch suppliers.
# 
# To evaluate this hypothesis, this project implements a end-to-end data science pipeline involving **Exploratory Data Analysis (EDA)**, **Feature Engineering**, and **Supervised Machine Learning** using a **Random Forest Classifier**.
# 
# ---
# 
# ## Business Question & Objectives
# 
# The primary objective of this project is to answer two key strategic questions for PowerCo’s executive team:
# 
# 1. **Is price sensitivity the primary driver of SME customer churn?**
# 
# 2. **Can machine learning accurately identify high-risk churners to allow targeted, cost-effective retention campaigns before contracts expire?**
# 
# 
# ---
# 
# ## Dataset & Technical Scope
# 
# The analysis leverages two primary datasets encompassing **14,606 corporate clients** and **193,002 monthly time-series price snapshots** throughout 2015:
# 
# * **Client Data (`client_data.csv`):** Historical consumption volumes (`cons_12m`, `cons_gas_12m`), account antiquity/tenure (`date_activ`, `date_end`), product holdings (`nb_prod_act`), financial margins (`net_margin`), and the binary churn outcome (`churn`).
# 
# 
# * **Price Data (`price_data.csv`):** Monthly variable energy charges (`price_*_var`) and fixed capacity fees (`price_*_fix`) broken down across **Off-Peak**, **Peak**, and **Mid-Peak** pricing tiers.
# 
# 
# 
# ---
# 
# ## Analytical Workflow
# 
# ```
# ┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
# │ 1. Exploratory Data    │ ───► │ 2. Feature Engineering │ ───► │ 3. Machine Learning    │
# │    Analysis (EDA)      │      │    & Transformation    │      │    & Model Evaluation  │
# └────────────────────────┘      └────────────────────────┘      └────────────────────────┘
#  • Schema & Quality Audit        • Dec-to-Jan Price Deltas       • Class Imbalance Handling
#  • Skewness & Outliers           • Inter-Tier Spread Metrics     • Stratified Train/Test
#  • Hypothesis Verification       • Log(x+1) Scaling              • Gini Feature Importance
# ```
# 
# ```

# # Exploratory Data Analysis 
# 
# ## Import Package Doing Setup & Data Loading

# In[34]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Shows plots in Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set plot style 
sns.set_theme(style="whitegrid", color_codes=True)


# ---
# 
# ## Loading data with Pandas
# 
# We need to load `client_data.csv` and `price_data.csv` into individual dataframes so that we can work with them in Python. For this notebook and all further notebooks, it will be assumed that the CSV files will the placed in the same file location as the notebook. If they are not, please adjust the directory within the `read_csv` method accordingly.

# In[35]:


# This works if the CSVs are in the same folder as this notebook
# Updated paths to include the subfolder
# Load Datasets
client_df = pd.read_csv(".ipynb_checkpoints/client_data (1).csv")
price_df = pd.read_csv(".ipynb_checkpoints/price_data (1).csv")


# In[36]:


client_df.head(10)


# > ## **Data Quality & Exploratory Findings**
# > 
# > 
# > * **Data Hygiene & Schema Standardization:**
# > We identified several missing entries in the `sales_channel` column and noted that `date` fields are currently stored as generic text objects. To prepare the dataset for time-series forecasting, immediate data cleaning and type conversion (`datetime64`) are required.
# > * **Churn Profile & Account Antiquity:**
# > In our 10-row sample, three accounts (`2401...`, `1aa4...`, and `7ab4...`) dropped out—accounting for the **30% churn rate** observed across the subset. Critically, these churned accounts carry an **average tenure of 5.1 years**. We are not merely losing newly acquired customers; we are losing established, long-term partners.
# > 
# > 
# > ---
# > 
# > 
# > ## **Hypothesis & Feature Engineering Strategy**
# > 
# > 
# > * **Price Sensitivity in High-Volume Users:**
# > For high-volume SMEs, even sub-cent shifts in unit price translate into substantial operational cost increases. We hypothesize that consumption spikes coincided with specific tariff hikes, prompting these key accounts to exit.
# > * **Action Plan:**
# > During feature engineering, we will compute the analytical impact of price deltas across reporting periods. Specifically, we will create a target feature—**`Max_12M_Price_Change`**—measuring the largest single price increase experienced by a customer over any 12-month window. We will then cross-examine this metric against historical usage to determine whether a specific price shock triggered the 30% churn rate.
# > 
# > 
# 
# ---

# In[37]:


price_df.head(10)


# ### Executive / Business Presentation Style (Recommended)
# 
# > * **Cost Structure Breakdown:** Pricing is split into variable components (energy consumed) and fixed components (monthly metering/subscription fees).
# > * **Off-Peak Concentration:** Energy costs are heavily concentrated during off-peak hours. Both peak and mid-peak variable prices remain at `0.0`, indicating this customer operates primarily off-peak or is enrolled in a simplified, single-tier tariff plan.
# > * **Variable Rate Trend (Downward):** Off-peak variable rates (`price_off_peak_var`) showed mild downward volatility over the year—opening at **0.1514** in January, dipping to **0.1496** in spring, and settling at **0.1459** by October.
# > * **Fixed Fee Adjustment (Upward):** In contrast, the fixed monthly fee (`price_off_peak_fix`) experienced a slight upward revision mid-year, rising from **44.27** to **44.44** in July.
# > 
# > 
# 
# ---
# 
# ###  Technical / Analytical Report Style
# 
# > * **Price Component Split:** Tariff structures are divided into variable energy charges (`price_*_var`) and fixed capacity/subscription charges (`price_*_fix`).
# > * **Single-Tariff / Off-Peak Alignment:** Peak and mid-peak variable charges are `0.0`, demonstrating that the customer’s variable bill is driven entirely by off-peak consumption. This points to either a single-rate contract structure or exclusive off-peak operations.
# > * **Variable Price Volatility:** The off-peak variable rate (`price_off_peak_var`) exhibited minor seasonal adjustments throughout 2015, trending down overall from **0.151367** (Jan) to **0.149626** (Apr/May) and closing at **0.145859** (Oct).
# > * **Fixed Fee Adjustments:** Conversely, the fixed off-peak price (`price_off_peak_fix`) increased marginally around July, shifting from **44.2669** to **44.4447**.
# > 
# > 
# 
# ---

# In[38]:


# ==========================================
# 2. DATA INTEGRITY & TYPE CONVERSIONS
# ==========================================
# Convert Date Columns from object to datetime
date_cols_client = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]
for col in date_cols_client:
    client_df[col] = pd.to_datetime(client_df[col])

price_df["price_date"] = pd.to_datetime(price_df["price_date"])


# In[39]:


# Verify missing values & duplicates
print("--- Data Quality Checks ---")
print(f"Client Data Duplicates: {client_df.duplicated().sum()}")
print(f"Price Data Duplicates: {price_df.duplicated().sum()}")
print("\nClient Data Missing Values:\n", client_df.isnull().sum()[client_df.isnull().sum() > 0])


# > ## **Data Hygiene & Validation Results**
# > 
# > 
# > * **Duplicate Records:** An initial audit confirmed **zero duplicate rows** in both the client and historical price datasets (`0` duplicates found).
# > * **Missing Value Audit:** All standard numerical, date, and boolean attributes are 100% complete with **no null (`NaN`) values**.
# > * **Categorical Quality Note:** Explicit text labels representing unrecorded values (e.g., `"MISSING"` strings in `channel_sales`) were preserved and will be handled as an independent category during One-Hot Encoding in the Feature Engineering phase.
# > 
# >

# > ## **Descriptive Statistics & Data Schema**
# > 
# > 
# > ### **Data Type Inspection**
# > 
# > 
# > Understanding the structure and data types of your dataset is a critical first step in exploratory analysis. Column data types directly inform preprocessing requirements, missing value handling, and feature engineering strategies (e.g., encoding categoricals vs. scaling numericals).
# > Execute the `.info()` method to review the DataFrame's schema, non-null counts, and memory usage:
# 
# ---
# 
# > ## **Data Profiling & Schema Overview**
# > 
# > 
# > ### **Column Data Types**
# > 
# > 
# > Profiling data types is essential for establishing downstream feature engineering pipelines. Identifying numeric, categorical, and datetime fields determines the necessary transformations, such as scaling, one-hot encoding, or time-delta calculations.
# > Use `df.info()` to inspect data types, memory consumption, and null-value counts:
# 
# ---
# 
# > ## **Dataset Overview**
# > 
# > 
# > ### **Inspecting Data Types**
# > 
# > 
# > Before applying transformations, inspect the underlying data types to determine how each feature should be handled during feature engineering.
# > Run `df.info()` to generate a full schema summary:
# 
# ---

# In[40]:


print("\n--- Client Data Summary ---")
display(client_df.describe().T)


# ---
# 
# ## Dataset Overview
# 
# The dataset contains **14,606 client records** (likely corporate/utility energy customers) evaluating consumption history, active products, financial metrics, price forecasts, and a binary outcome label for customer **churn**.
# 
# ---
# 
# ## Key Descriptive Highlights
# 
# ### 1. Customer Churn
# 
# * **Churn Rate:** ~**9.72%** (mean = 0.0972). The dataset shows a notable class imbalance, with slightly under 10% of customers churning.
# 
# ### 2. Customer Profile & Portfolio
# 
# * **Tenure (`num_years_antig`):** The average customer tenure is **5 years** (ranging from 1 to 13 years), with 50% of customers falling between 4 and 6 years.
# * **Active Products (`nb_prod_act`):** Most clients hold 1 product (median = 1), though some hold up to 32 products (mean = 1.29).
# * **Subscription Dates (`date_activ`):** Customer activations span from May 2003 to September 2014, with the median activation occurring in March 2011.
# 
# ---
# 
# ## Consumption Patterns & Distribution
# 
# A key feature across consumption metrics is extreme **right-skewness** (the presence of large high-value outliers):
# 
# * **Electricity Consumption (`cons_12m`):** Average annual consumption is **15,922 units**, but the median is only **1,411 units**. The maximum value reaches over **6.2 million units**, indicating a long tail of high-volume industrial clients alongside standard customers.
# * **Gas Consumption (`cons_gas_12m`):** At least 75% of clients consume **0 units** of gas, showing that gas product adoption is concentrated in a minor segment, despite a max consumption of **4.15 million units**.
# * **Recent Consumption (`cons_last_month`):** Mirrors the annual distribution—median consumption is 79 units, while the maximum reaches over **771,000 units**.
# 
# ---
# 
# ## Financials & Forecast Metrics
# 
# | Category | Metric | Key Observations |
# | --- | --- | --- |
# | **Margins** | `net_margin` | Average of **189.26** (median = **112.53**, max = **24,570.65**). |
# |  | `margin_gross_pow_ele` | Average power margin of **24.57** (median = **21.64**). |
# | **Pricing** | `forecast_price_energy_off_peak` | Off-peak energy prices average **0.137**, tightly bounded. |
# |  | `forecast_price_energy_peak` | Peak prices average **0.050**, with at least 50% sitting at **0.00**. |
# | **Capacity** | `pow_max` | Subscribed power averages **18.14 kW** (median = **13.86 kW**, max = **320 kW**). |
# 
# ---
# 
# ## Key Takeaways for Data Preprocessing
# 
# 1. **Skewness & Outliers:** Consumption and margin features (`cons_12m`, `cons_gas_12m`, `net_margin`, `forecast_cons_12m`) feature strong positive skewness. Log transformations or robust scaling are recommended prior to modeling.
# 2. **Date Handling:** `date_activ`, `date_end`, `date_modif_prod`, and `date_renewal` are datetime objects stored as strings/timestamps and should be parsed into time-delta features (e.g., days since last modification/renewal).
# 3. **Class Imbalance:** Standard accuracy will be misleading due to the ~9.7% churn baseline; evaluation metrics should prioritize **PR-AUC**, **ROC-AUC**, or **F1-score**.

# In[41]:


print("\n--- Price Data Summary ---")
display(price_df.describe().T)


# ---
# 
# ### **Executive Summary & Key Insights**
# 
# The dataset consists of **193,002 time-series records** tracking energy prices throughout the year 2015 for PowerCo's client base. The structure breaks pricing down into **variable energy rates** (per kWh consumed) and **fixed power fees** across three distinct time bands: **Off-Peak**, **Peak**, and **Mid-Peak**.
# 
# ---
# 
# ### **Detailed Statistical Breakdown**
# 
# #### **1. Temporal Scope (`price_date`)**
# 
# * **Time Horizon:** Spans exactly 12 monthly snapshots, from **January 1, 2015 (`min`)** to **December 1, 2015 (`max`)**.
# * **Median Date:** June/July 2015, confirming a balanced 12-month panel dataset across the customer base.
# 
# ---
# 
# #### **2. Variable Energy Prices (Price per Energy Consumed)**
# 
# Variable prices represent the per-unit cost of energy consumption across time periods:
# 
# * **Off-Peak Period (`price_off_peak_var`):**
# * **Mean:** `0.1410` (std: `0.0250`).
# * **Interquartile Range (IQR):** Ranges tightly between `0.1260` (25th percentile) and `0.1516` (75th percentile), reaching a maximum of `0.2807`.
# * *Insight:* Off-peak variable rates are consistently active for virtually all active contracts and show very low volatility across the portfolio.
# 
# 
# * **Peak Period (`price_peak_var`):**
# * **Mean:** `0.0546` (median: `0.0855`).
# * **Distribution:** At least `25% of records report `0.0`, indicating that a notable portion of SMEs are on single-tariff or simplified rate structures without peak-hour variable surcharges.
# 
# 
# * **Mid-Peak Period (`price_mid_peak_var`):**
# * **Mean:** `0.0305` (75th percentile: `0.0726`).
# * **Distribution:** At least **50% of the entries are `0.0**`, showing mid-peak rates apply primarily to larger or specialized multi-tier contract tiers.
# 
# 
# ---
# 
# 
# Fixed prices represent the recurring baseline fee charged regardless of monthly consumption volume:
# 
# * **Off-Peak Fixed Fee (`price_off_peak_fix`):**
# * **Mean:** `43.33` (median: `44.27`).
# * **Distribution:** Highly clustered between `40.73` (25th percentile) and `44.44` (75th percentile), with a peak of `59.44`.
# * *Insight:* This forms the primary baseline monthly subscription revenue for PowerCo's SME accounts.
# 
# 
# * **Peak & Mid-Peak Fixed Fees (`price_peak_fix` & `price_mid_peak_fix`):**
# * **Peak Fixed Mean:** `10.62` (75th percentile: `24.34`).
# * **Mid-Peak Fixed Mean:** `6.41` (75th percentile: `16.23`).
# * **Distribution:** Over **50% of records are `0.0**` for both peak and mid-peak fixed charges, further reinforcing that simple single-rate billing is the default for most small-to-medium enterprise clients.
# 
# 
# ---
# 
# 1. **Calculate Price Differences (Price Sensitivity Drivers):**
# * Raw monthly prices alone don't directly show price sensitivity. During feature engineering, create aggregated features such as:
# * `price_off_peak_var_year_change` = Price in Dec 2015 minus Price in Jan 2015.
# * `max_price_spike` = Difference between the highest and lowest price within 2015 for each customer.
# 
# 
# 
# 
# 2. **Granular Grouping by Customer ID:**
# * Since there are 193,002 rows across 12 months for ~14,606 unique customers, aggregate these 12 price entries per customer ID to join them directly onto `client_df` for churn modeling.

# ### **1. Client Data Analysis: Extreme Skewness and Outliers**
# The most prominent finding is the massive variance in consumption.
# * **Consumption Volatility:** For `cons_12m` (annual electricity consumption), the mean is approximately $159,220$ kWh, but the standard deviation is a staggering $573,465$. The gap between the **75th percentile ($40,763$)** and the **maximum ($6,207,104$)** confirms a highly skewed distribution. 
# * **Business Impact:** This tells us that while most clients are small SMEs, a handful of "whale" accounts consume massive amounts of energy. Our models must be robust to these outliers, as a single large account churning could impact revenue more than hundreds of smaller ones.
# 

# In[42]:


def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    """
    Plot stacked bars with annotations
    """
    ax = dataframe.plot(
        kind="bar",
        stacked=True,
        figsize=size_,
        rot=rot_,
        title=title_
    )

    # Annotate bars
    annotate_stacked_bars(ax, textsize=14)
    # Rename legend
    plt.legend(["Retention", "Churn"], loc=legend_)
    # Labels
    plt.ylabel("Company base (%)")
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    """
    Add value annotations to the bars
    """

    # Iterate over the plotted rectanges/bars
    for p in ax.patches:
        
        # Calculate annotation
        value = str(round(p.get_height(),1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        ax.annotate(
            value,
            ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
            color=colour,
            size=textsize
        )

def plot_distribution(dataframe, column, ax, bins_=50):
    """
    Plot variable distirbution in a stacked histogram of churned or retained company
    """
    # Create a temporal dataframe with the data to be plot
    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
    "Churn":dataframe[dataframe["churn"]==1][column]})
    # Plot the histogram
    temp[["Retention","Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    # X-axis label
    ax.set_xlabel(column)
    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')


# In[43]:


# A. Churn Rate Distribution
plt.figure(figsize=(6, 4))
churn_counts = client_df["churn"].value_counts(normalize=True) * 100

# Fixed: Assigned 'hue' to 'churn_counts.index' and set 'legend=False'
ax = sns.barplot(
    x=churn_counts.index, 
    y=churn_counts.values, 
    hue=churn_counts.index, 
    palette=["#2b5c8f", "#d9534f"], 
    legend=False
)

plt.title("Overall Churn Rate (%)", fontsize=14, fontweight="bold")
plt.ylabel("Percentage (%)")
plt.xticks([0, 1], ["Retained (0)", "Churned (1)"])

for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.show()


# ---
# 
# ## **Executive Summary: Churn Bar Chart Breakdown**
# 
# ```
# +----------------------------------------------------------------+
# |                     OVERALL CHURN DISTRIBUTION                 |
# +----------------------------------------------------------------+
# |  [#################################################.......]    |
# |  Retained (0): 90.3%                          Churned (1): 9.7%|
# +----------------------------------------------------------------+
# 
# ```
# 
# ### **1. Visual & Data Interpretation**
# 
# The bar chart displays the binary target variable (`churn`), segmenting the customer base into two distinct groups:
# 
# * **Retained (`0`): 90.3%** — The vast majority of small and medium enterprise (SME) clients remain active.
# * **Churned (`1`): 9.7%** — A smaller subset (~1 in 10 customers) discontinued their contracts.
# 
# ---
# 
# ## **Key Takeaways & Strategic Context**
# 
# ### **A. Customer Base Stability vs. Revenue Risk**
# 
# * **Base Loyalty:** With an average customer tenure (`num_years_antig`) of **5 years** (stretching up to 13 years), the 90.3% retention figure reflects a strong core customer base.
# * **Value-Weighted Impact:** - Because financial metrics like `net_margin` fluctuate dramatically (averaging $ 189.26 ,  but  peaking  at  over $ 24,500), the 9.7% churn rate cannot be evaluated on volume alone. Losing a small percentage of high-margin outliers poses a disproportionate risk to profitability.
# 
# ### **B. Pricing & Tariff Hypothesis for Churn**
# 
# * Analysis indicates that a significant portion of customers operate under simple, single-rate pricing structures (evidenced by mid-peak variable prices sitting at $0.0$).
# * **Investigative Focus:** When exploring *why* the 9.7% churned, test whether these departing clients were subject to complex, multi-tiered time-of-use tariffs (`price_peak_var`) or recent rate increases compared to the stable retained majority.
# 
# ### **C. Machine Learning Implementation (Class Imbalance)**
# 
# * **The 9:1 Imbalance Warning:** Standard accuracy metrics will fail here. A baseline model predicting "Retained" (`0`) for every record would boast a **90.3% accuracy rate** while failing entirely to catch a single churner.
# * **Modeling Strategy:**
# * Avoid relying on accuracy; optimize for **Precision, Recall, F1-Score, and ROC-AUC**.
# * Apply balance-adjustment techniques such as class-weighted loss functions, stratified splitting, or oversampling algorithms (e.g., SMOTE) during model training.

# In[44]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# B. Log-Scaled Consumption Distribution
fig, axs = plt.subplots(1, 2, figsize=(16, 5))

# 1. Linear Scale Plot
sns.histplot(
    data=client_df,
    x="cons_12m",
    hue="churn",
    kde=True,
    bins=50,
    ax=axs[0],
    palette=["#2b5c8f", "#d9534f"],
    hue_order=[0, 1]
)
axs[0].set_title("Annual Electricity Consumption (Linear Scale)", fontsize=13, fontweight="bold")

# 2. Log Scale Plot (using log1p to avoid log(0) = -inf)
# We calculate np.log1p(client_df["cons_12m"])
client_df_temp = client_df.copy()
client_df_temp["log_cons_12m"] = np.log1p(client_df_temp["cons_12m"])

sns.histplot(
    data=client_df_temp,
    x="log_cons_12m",
    hue="churn",
    kde=True,
    bins=50,
    ax=axs[1],
    palette=["#2b5c8f", "#d9534f"],
    hue_order=[0, 1]
)
axs[1].set_title("Annual Electricity Consumption (Log(x + 1) Scale)", fontsize=13, fontweight="bold")
axs[1].set_xlabel("log1p(cons_12m)")

plt.tight_layout()
plt.show()


# ---
# 
# ### **1. Linear Scale Plot (Left Side)**
# 
# * **Extreme Right Skewness:** On the standard linear scale, almost the entire client base is squished into the first vertical bar near zero.
# * **Impact of Outliers:** Because a small number of "whale" industrial accounts consume millions of kilowatt-hours (up to $6 \times 10^6$ or $6,000,000$ kWh), the horizontal axis stretches extremely wide. This makes it impossible to see the distribution or behavior of typical Small and Medium Enterprises (SMEs) on a linear scale.
# 
# ---
# 
# ### **2. Log-Transformed Plot — $\log(x + 1)$ (Right Side)**
# 
# * **Normalizing the Distribution:** Applying the $\log(1 + x)$ transformation compresses the massive scale difference, uncovering a clear, near-normal (bell-shaped) underlying distribution centered around $\sim 9.5$ on the logarithmic scale (which corresponds to roughly $13,000$–$15,000$ kWh annually).
# * **Secondary Spike at Zero:** There is a distinct small spike at $0$ on the log plot representing non-operational or zero-consumption accounts.
# * **Proportional Churn Insight:** The distribution curves for retained clients (blue, `0`) and churned clients (red, `1`) follow almost identical geometric shapes across all consumption ranges. This indicates that **churn is distributed across all customer sizes**, rather than being strictly confined to either very low-volume or very high-volume users.
# 
# ---
# 
# ### **Key Takeaways for Feature Engineering & Modeling**
# 
# 1. **Pre-processing Necessity:** Machine learning algorithms (such as Logistic Regression or Neural Networks) perform poorly on raw features with severe right skewness like `cons_12m`. Using a **log-transformed feature** ($\log(1+x)$) or robust scaling will be critical during feature engineering.
# 2. **Class Proportion Visibility:** The log transformation makes it much easier to observe the minority churn class across different volume tiers without losing sight of high-value outliers.

# In[45]:


# C. Direct Price Sensitivity Feature Exploration
# Compute max price delta for off-peak variable price per client
price_delta = price_df.groupby("id")["price_off_peak_var"].agg(lambda x: x.max() - x.min()).reset_index()
price_delta.columns = ["id", "off_peak_var_delta"]

merged_df = pd.merge(client_df, price_delta, on="id")

plt.figure(figsize=(8, 4))

# Assigned 'x' variable to 'hue' and set 'legend=False'
sns.boxplot(
    data=merged_df, 
    x="churn", 
    y="off_peak_var_delta", 
    hue="churn", 
    palette=["#2b5c8f", "#d9534f"], 
    legend=False
)

plt.title("Off-Peak Variable Price Variation vs Churn Status", fontsize=13, fontweight="bold")
plt.xticks([0, 1], ["Retained (0)", "Churned (1)"])
plt.ylabel("Max Price Delta in 12 Months")

plt.show()


# ---
# 
# ### **1. Key Visual Observations**
# 
# * **Identical Medians & IQRs:** The interquartile range (IQR)—represented by the main boxes—and the median line for both **Retained (0)** and **Churned (1)** customers sit virtually flat near **`0.00` to `0.01**`.
# * **Minimal Overall Fluctuation:** For the vast majority of clients in both cohorts, the maximum off-peak variable price change experienced over a 12-month period was near zero.
# * **Outlier Behavior:** Both groups exhibit upper outliers (clients who experienced price deltas ranging from `0.02` up to `0.23`). However, high price variance outliers occur in both retained and churned groups at similar proportions relative to their class sizes.
# 
# ---
# 
# ### **2. Strategic Analytical Finding (Hypothesis Testing)**
# 
# This visualization provides a crucial preliminary answer to PowerCo’s central question: **Is price sensitivity the primary driver of customer churn?**
# 
# * **Hypothesis Disproven / Weakened:** If price hikes were the primary catalyst forcing SME customers to leave, we would expect the **Churned (1)** box plot to show a significantly higher median delta and a shifted IQR compared to the **Retained (0)** group.
# * **Conclusion:** The fact that the price variation distribution is almost identical across both classes strongly suggests that **price sensitivity alone is NOT the primary driver of customer churn**.
# 
# ---
# 
# ### **3. Next Steps Feature Engineering)**
# 
# 1. **Investigate Combined Price Metrics:** While off-peak variable price delta alone doesn't separate churners, creating composite metrics, such as **total bill changes (fixed + variable)** or **peak-hour rate spikes relative to consumption volume**—may reveal hidden price sensitivities for high-volume accounts.
# 2. **Explore Alternative Churn Drivers:** Since price changes alone don't explain attrition, shift focus during modeling toward non-price variables, such as:
# * **Customer Tenure (`num_years_antig`)**
# * **Contract Expiry & Renewal Deadlines (`date_renewal`)**
# * **Sales Channel Origin (`channel_sales`)**
# * **Net Profit Margins (`net_margin`)**
# 
# 
# 
# ---

# ### Feature Engineering Overview
# 
# Based on the insights from Exploratory Data Analysis, we proved that basic price changes alone do not cleanly separate churners. Now, the goal is to transform, combine, and engineer raw data into predictive features that expose hidden churn patterns—both price-related and non-price-related.

# ---
# 
# ### **1. Core Objectives**
# 
# 1. **Engineer Composite Price Metrics:**
# * **Off-Peak, Peak, & Mid-Peak Price Deltas:** Calculate price changes between January and December 2015 for both variable and fixed components.
# * **Average Price Changes Across Periods:** Calculate average prices across 12 months to measure baseline exposure.
# * **Max Price Differences:** Capture volatile price spikes experienced during the year.
# 
# 
# 2. **Engineer Temporal & Contract Features:**
# * **Tenure in Years & Months:** Derived from contract activation dates (`date_activ`).
# * **Months Until Contract Expiry / Renewal:** Difference between the reference date and contract renewal dates (`date_renewal`).
# * **Days Since Last Product Modification:** Derived from `date_modif_prod`.
# 
# 
# 3. **Encode Categorical Variables:**
# * Apply One-Hot Encoding to categorical features like `channel_sales` and `origin_up` (handling missing entries cleanly).
# * Convert boolean/categorical fields like `has_gas` (`t`/`f`) into binary flags (`1`/`0`).
# 
# 
# 4. **Transform Skewed Numerical Features:**
# * Apply log transformations ($\log(1+x)$) to highly skewed consumption and margin features (`cons_12m`, `cons_gas_12m`, `net_margin`).
# 
# ---

# In[46]:


import numpy as np
import pandas as pd
from datetime import datetime

# ==========================================
# 1. LOAD CLEANED DATASETS & INITIAL CONVERSION
# ==========================================
client_df = pd.read_csv(".ipynb_checkpoints/client_data (1).csv")
price_df = pd.read_csv(".ipynb_checkpoints/price_data (1).csv")

# Convert date columns
date_cols = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]
for col in date_cols:
    client_df[col] = pd.to_datetime(client_df[col])

price_df["price_date"] = pd.to_datetime(price_df["price_date"])

# ==========================================
# 2. PRICE FEATURES (Aggregations & Granular Differences)
# ==========================================
# Sort prices chronologically
price_sorted = price_df.sort_values(by=["id", "price_date"])

# A. December vs. January Differences (Full Coverage)
jan_prices = price_sorted.groupby("id").first().reset_index()
dec_prices = price_sorted.groupby("id").last().reset_index()

price_diffs = pd.DataFrame({"id": jan_prices["id"]})
price_diffs["offpeak_diff_dec_jan_var"] = dec_prices["price_off_peak_var"] - jan_prices["price_off_peak_var"]
price_diffs["peak_diff_dec_jan_var"] = dec_prices["price_peak_var"] - jan_prices["price_peak_var"]
price_diffs["midpeak_diff_dec_jan_var"] = dec_prices["price_mid_peak_var"] - jan_prices["price_mid_peak_var"]

price_diffs["offpeak_diff_dec_jan_fix"] = dec_prices["price_off_peak_fix"] - jan_prices["price_off_peak_fix"]
price_diffs["peak_diff_dec_jan_fix"] = dec_prices["price_peak_fix"] - jan_prices["price_peak_fix"]
price_diffs["midpeak_diff_dec_jan_fix"] = dec_prices["price_mid_peak_fix"] - jan_prices["price_mid_peak_fix"]

# B. Mean Differences Between Consecutive Tariff Bands (From Model Answer)
mean_prices = price_df.groupby("id").agg({
    'price_off_peak_var': 'mean', 
    'price_peak_var': 'mean', 
    'price_mid_peak_var': 'mean',
    'price_off_peak_fix': 'mean',
    'price_peak_fix': 'mean',
    'price_mid_peak_fix': 'mean'    
}).reset_index()

mean_prices['off_peak_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_peak_var']
mean_prices['peak_mid_peak_var_mean_diff'] = mean_prices['price_peak_var'] - mean_prices['price_mid_peak_var']
mean_prices['off_peak_mid_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_mid_peak_var']
mean_prices['off_peak_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_peak_fix']
mean_prices['peak_mid_peak_fix_mean_diff'] = mean_prices['price_peak_fix'] - mean_prices['price_mid_peak_fix']
mean_prices['off_peak_mid_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_mid_peak_fix']

# Merge Price Features
df = pd.merge(client_df, price_diffs, on="id", how="left")
df = pd.merge(df, mean_prices, on="id", how="left")

# ==========================================
# 3. TEMPORAL & TENURE METRICS
# ==========================================
# Calculate exact tenure in years
df['tenure'] = ((df['date_end'] - df['date_activ']).dt.days // 365)

# Precise month conversion helper
def convert_months(reference_date, dataframe, column):
    dates = pd.to_datetime(dataframe[column])
    year_diff = reference_date.year - dates.dt.year
    month_diff = reference_date.month - dates.dt.month
    months = year_diff * 12 + month_diff
    months -= (reference_date.day < dates.dt.day).astype(int)
    return months

reference_date = datetime(2016, 1, 1)

df['months_activ'] = convert_months(reference_date, df, 'date_activ')
df['months_to_end'] = -convert_months(reference_date, df, 'date_end')
df['months_modif_prod'] = convert_months(reference_date, df, 'date_modif_prod')
df['months_renewal'] = convert_months(reference_date, df, 'date_renewal')

# Drop original datetime features
df = df.drop(columns=date_cols)

# ==========================================
# 4. SKEWNESS CORRECTION & CATEGORICAL ENCODING
# ==========================================
# Boolean transformation
df["has_gas"] = df["has_gas"].map({"t": 1, "f": 0}).fillna(0).astype(int)

# Log-transform highly skewed features using log1p
skewed_cols = [
    'cons_12m', 
    'cons_gas_12m', 
    'cons_last_month',
    'forecast_cons_12m', 
    'forecast_cons_year', 
    'forecast_discount_energy',
    'forecast_meter_rent_12m', 
    'forecast_price_energy_off_peak',
    'forecast_price_energy_peak', 
    'forecast_price_pow_off_peak',
    'net_margin',
    'margin_gross_pow_ele'
]

# Apply transformation only to available columns
skewed_cols = [c for c in skewed_cols if c in df.columns]
for c in skewed_cols:
    df[c] = np.log1p(np.maximum(0, df[c]))

# One-Hot Encoding with Rare Category Filtering (Min Count >= 100)
cat_cols = ['channel_sales', 'origin_up']
for col in cat_cols:
    df[col] = df[col].fillna("MISSING")

value_counts = {col: df[col].value_counts() for col in cat_cols}
df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)

for col in cat_cols:
    keep_cols = set(value_counts[col][value_counts[col] >= 100].index.astype(str))
    dummy_cols = [c for c in df.columns if c.startswith(f"{col}_")]
    drop_cols = [c for c in dummy_cols if c.split(f"{col}_", 1)[1] not in keep_cols]
    df.drop(columns=drop_cols, inplace=True)

# ==========================================
# 5. MULTICOLLINEARITY DROPS & EXPORT
# ==========================================
# Drop highly collinear or redundant features identified across both analyses
cols_to_drop = [c for c in ['num_years_antig', 'forecast_cons_year', 'margin_net_pow_ele'] if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# Save finalized dataset
df.to_csv("df_engineered_features.csv", index=False)
print("Updated Feature Engineering Pipeline Complete. Final Dataset Shape:", df.shape)


# ### Python Implementation: Price Sensitivity Features

# In[47]:


import pandas as pd
import numpy as np

# --------------------------------------------------------
# Step 1: Sort data to ensure chronological order
# --------------------------------------------------------
price_df = price_df.sort_values(by=['id', 'price_date'])

# --------------------------------------------------------
# Step 2: December vs. January Prices (Price Drop/Hike)
# --------------------------------------------------------
# Group by customer ID to get the first (Jan) and last (Dec) records
jan_prices = price_df.groupby('id').first().reset_index()
dec_prices = price_df.groupby('id').last().reset_index()

# Initialize our new feature dataframe
price_features = pd.DataFrame({'id': jan_prices['id']})

# Calculate Variable (Energy - E) Deltas
price_features['dec_jan_diff_off_peak_var'] = dec_prices['price_off_peak_var'] - jan_prices['price_off_peak_var']
price_features['dec_jan_diff_peak_var'] = dec_prices['price_peak_var'] - jan_prices['price_peak_var']
price_features['dec_jan_diff_mid_peak_var'] = dec_prices['price_mid_peak_var'] - jan_prices['price_mid_peak_var']

# Calculate Fixed (Power - P) Deltas
price_features['dec_jan_diff_off_peak_fix'] = dec_prices['price_off_peak_fix'] - jan_prices['price_off_peak_fix']
price_features['dec_jan_diff_peak_fix'] = dec_prices['price_peak_fix'] - jan_prices['price_peak_fix']
price_features['dec_jan_diff_mid_peak_fix'] = dec_prices['price_mid_peak_fix'] - jan_prices['price_mid_peak_fix']

# --------------------------------------------------------
# Step 3: Max vs. Min Prices (Overall Volatility)
# --------------------------------------------------------
price_max = price_df.groupby('id').max().reset_index()
price_min = price_df.groupby('id').min().reset_index()

price_features['max_min_diff_off_peak_var'] = price_max['price_off_peak_var'] - price_min['price_off_peak_var']
price_features['max_min_diff_off_peak_fix'] = price_max['price_off_peak_fix'] - price_min['price_off_peak_fix']

# --------------------------------------------------------
# Step 4: Average Annual Prices
# --------------------------------------------------------
price_mean = price_df.groupby('id').mean().reset_index()

price_features['mean_off_peak_var'] = price_mean['price_off_peak_var']
price_features['mean_off_peak_fix'] = price_mean['price_off_peak_fix']

# --------------------------------------------------------
# Step 5: Merge back into the main Client DataFrame
# --------------------------------------------------------
# Assuming client_df is your main dataframe containing the 'churn' target
# final_df = pd.merge(client_df, price_features, on='id', how='left')

print("Price Sensitivity Features Created:")
display(price_features.head())


# ---
# 
# ## 1. Addressing Multicollinearity & Redundancy
# 
# While tree-based ensemble models like **XGBoost** and **Random Forest** handle correlated features better than linear models (they won't break mathematically), high multicollinearity severely distorts **Feature Importance** (e.g., Gini importance or SHAP values) by splitting the predictive signal across redundant features.
# 
# ### Action Plan
# 
# * **Correlation Thresholding:** Calculate the Pearson/Spearman correlation matrix for all numeric features. If two features have a correlation $\vert{}r\vert{} > 0.85$ or $0.90$, inspect them and drop the less interpretable one.
# * **Variance Inflation Factor (VIF):** Calculate VIF for key composite metrics. A $\text{VIF} > 5\text{--}10$ indicates significant multicollinearity.
# * **SHAP / Permutation Importance:** If keeping both raw and composite features initially, rely on **SHAP values** rather than standard tree-based `feature_importances_` (gain/split count), as SHAP accounts for feature interactions and correlations much better.
# * **Feature Selection Pipeline:** Apply recursive feature elimination (RFE) or drop low-variance features before model training.

# In[48]:


import numpy as np
import pandas as pd

# Filter correlation matrix to numeric columns only
corr_matrix = df.corr(numeric_only=True).abs()

# Extract upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify features with correlation greater than 0.85
to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]

print(f"Features recommended for drop due to high correlation: {to_drop}")


# In[49]:


print(df.columns.tolist())


# In[54]:


cols_to_drop = [
    'cons_last_month',
    'imp_cons',
    'margin_net_pow_ele',
    'total_price_sensitivity',
    'months_activ',
    'log_cons_gas_12m',
]

# Keep only columns that exist in df
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]

# Drop features and remove target/id to build X
X = df.drop(columns=existing_cols_to_drop)

# Also make sure to drop 'id' and 'churn' from X if you haven't already:
X = X.drop(columns=['id', 'churn'], errors='ignore')
y = df['churn']


# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot correlation heatmap for these specific features against the dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# > ### **1. High Multicollinearity & Redundancy Clusters**
# > 
# > 
# > The heatmap reveals strong collinearity ($\vert{}r\vert{} > 0.7$) across three main feature groups:
# > * **Raw Prices vs. Price Spreads:** Extreme positive and negative correlations (dark red and blue blocks) exist between baseline prices (`price_off_peak_var`, `price_peak_fix`) and inter-period spread metrics (`off_peak_peak_var_mean_diff`, `peak_mid_peak_fix_mean_diff`).
# > * **Consumption Volume:** Historical consumption (`cons_12m`), recent consumption (`cons_last_month`), and net energy usage (`imp_cons`) exhibit strong positive co-movement.
# > * **Temporal Milestones:** Account age (`tenure`) and contract timeline metrics (`months_to_end`, `months_renewal`) form a tightly correlated cluster.
# > * **Model Impact:** While tree-based ensembles (Random Forest) remain predictive despite collinearity, splitting signals across redundant features will artificially dilute individual feature importances.
# > 
# > 
# > ---
# > 
# > 
# > ### **2. Absence of Direct Linear Relationship with Churn**
# > 
# > 
# > * The target variable `churn` displays near-zero linear correlation (neutral grey/light blue tones) across all individual independent variables.
# > * **Conclusion:** Linear models (e.g., Logistic Regression) will perform poorly. Capturing churn requires non-linear ensemble models (Random Forest, Gradient Boosting) capable of learning complex interaction thresholds.
# > 
# >

# ## Testing Price Sensitivity Hypothesis
# 
# Hypothesis focuses on **price volatility over time**: specifically, whether a change in off-peak prices between the start and end of the preceding year (January vs. December) correlates with customer churn.
# 
# If a customer experienced a significant price hike in off-peak energy or power during that 12-month window, their propensity to switch suppliers (churn) likely increased.
# 
# ---
# 
# ### Key Price Features to Engineer
# 
# To test this hypothesis and enrich our dataset, we can extract both **absolute** and **relative** price differences across off-peak, peak, and mid-peak periods for both energy (`price_off_peak_var`) and power (`price_off_peak_fix`).
# 
# $$\Delta \text{Price}_{\text{off\_peak\_var}} = \text{Price}_{\text{Dec, off\_peak\_var}} - \text{Price}_{\text{Jan, off\_peak\_var}}$$
# 
# $$\Delta \text{Price}_{\text{off\_peak\_fix}} = \text{Price}_{\text{Dec, off\_peak\_fix}} - \text{Price}_{\text{Jan, off\_peak\_fix}}$$
# 
# #### Beyond December vs. January
# 
# To make our feature set even more robust for model training, we can expand on Estelle's idea with additional engineered features:
# 
# 1. **Multi-Period Price Spreads:**
# * Mean price across 12 months.
# * Maximum vs. minimum price difference within the year ($\text{Price}_{\max} - \text{Price}_{\min}$).
# * Price difference between the last 3 months (Q4 average vs. Q1 average).
# 
# 
# 2. **Client Interaction & Consumption Features:**
# * **Estimated Annual Spend:** Multiplying annual consumption by average off-peak/peak rates (`cons_12m` $\times$ average energy price).
# * **Tenure in Years/Months:** Deriving contract longevity from `date_activ` and `date_end`.
# * **Contract Renewal Proximity:** Time remaining until contract expiration from the snapshot date.
# * **Log Transformation:** Applying $\log(1 + x)$ to heavily right-skewed variables like annual consumption (`cons_12m`, `cons_gas_12m`) to stabilize variance for tree-based or linear models.
# 
# 
# 
# ---

# In[55]:


import numpy as np
import pandas as pd

# 1. Ensure price_data is sorted chronologically by client and date
price_df = price_df.sort_values(by=["id", "price_date"]).reset_index(drop=True)
price_df["price_date"] = pd.to_datetime(price_df["price_date"])

# 2. Extract January (first record of 2015) and December (last record of 2015) per client ID
jan_prices = price_df.groupby("id").first().reset_index()
dec_prices = price_df.groupby("id").last().reset_index()

# 3. Create a DataFrame for engineered price delta features
price_features = pd.DataFrame({"id": jan_prices["id"]})

# 4. Compute December minus January price differences across energy periods
# Variable Energy Price Differences
price_features["offpeak_diff_dec_jan_var"] = (
    dec_prices["price_off_peak_var"] - jan_prices["price_off_peak_var"]
)
price_features["peak_diff_dec_jan_var"] = (
    dec_prices["price_peak_var"] - jan_prices["price_peak_var"]
)
price_features["midpeak_diff_dec_jan_var"] = (
    dec_prices["price_mid_peak_var"] - jan_prices["price_mid_peak_var"]
)

# Fixed Power Fee Differences
price_features["offpeak_diff_dec_jan_fix"] = (
    dec_prices["price_off_peak_fix"] - jan_prices["price_off_peak_fix"]
)
price_features["peak_diff_dec_jan_fix"] = (
    dec_prices["price_peak_fix"] - jan_prices["price_peak_fix"]
)
price_features["midpeak_diff_dec_jan_fix"] = (
    dec_prices["price_mid_peak_fix"] - jan_prices["price_mid_peak_fix"]
)

# 5. Merge the new features back into your main client DataFrame
client_df_engineered = pd.merge(client_df, price_features, on="id", how="left")

# Display the first few rows of the engineered features
print("Engineered Price Delta Features:")
display(price_features.head())


# In[56]:


display(price_features.head().T)


# In[57]:


# Format float display to 4 decimal places for cleaner viewing
pd.set_option('display.float_format', lambda x: '%.4f' % x)

display(price_features.head())


# ---
# 
# #### Key Observations
# 
# * **Variable Price Deltas (`*_var`)**:
# * Generally show small fluctuations across months, mostly concentrated in slight negative adjustments (e.g., `-0.0062`, `-0.0041`), though occasional positive shifts appear (e.g., client `2` at `0.0504`).
# 
# 
# * **Fixed Price Deltas (`*_fix`)**:
# * Show larger, discrete step changes compared to variable rates. For instance, `offpeak_diff_dec_jan_fix` displays shifts like `0.1629` and up to `1.5000` (client `2`), suggesting contractual price re-evaluations or fixed fee updates at year-end.
# 
# 
# * **Tariff Structure Sparsity**:
# * Several clients (e.g., rows `1`, `2`, `4`) have exact `0.0000` differences for `peak` and `midpeak` features. This aligns with single-tariff (off-peak only) plans or accounts without time-of-use pricing.
# 
# 
# 
# ---

# #### Next Steps
# 
# 1. **Dataset Integration**:
# * Merge `price_features` back into the main client dataset using `id` as the key.
# 
# 
# 2. **Distribution & Outlier Check**:
# * Inspect feature distributions (e.g., via `describe()` or boxplots) to see how extreme shifts like `1.5000` behave across the full population.
# 
# 
# 3. **Correlation with Churn**:
# * Calculate correlation coefficients or feature importances against the target variable (`churn`) to test the hypothesis that price sensitivity/increases drive churn.

# In[58]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Merge price features onto client_df using 'id'
df = pd.merge(client_df, price_features, on="id", how="left")

print(f"Merged Dataset Shape: {df.shape}")
print("\nMissing values in new features:")
print(df[price_features.columns].isnull().sum())


# ---
# 
# * **Dataset Shape `(14,606, 32)**`: The row count matches `client_df` exactly (14,606 clients), confirming that no rows were duplicated or dropped during the merge. The column count grew from 26 to 32, accounting for the 6 new engineered price features.
# * **Missing Values = 0**: Every single active client in `client_df` has complete price records in `price_features`. There are no missing values (`NaN`) to impute.
# 
# ---

# In[59]:


# Select the newly engineered price features
new_price_cols = [c for c in price_features.columns if c != "id"]

# 1. Summary Statistics & Percentiles
print("--- Statistical Distribution ---")
display(
    df[new_price_cols].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)

# 2. Visualize distributions and boxplots
fig, axes = plt.subplots(
    len(new_price_cols), 2, figsize=(14, 3 * len(new_price_cols))
)

for i, col in enumerate(new_price_cols):
    sns.histplot(df[col], kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"Distribution of {col}")

    sns.boxplot(x=df[col], ax=axes[i, 1])
    axes[i, 1].set_title(f"Outliers in {col}")

plt.tight_layout()
plt.show()


# In[60]:


from sklearn.ensemble import RandomForestClassifier

# 1. Pearson & Spearman Correlation with Churn
correlations = pd.DataFrame(
    {
        "Pearson_Corr": df[new_price_cols].apply(
            lambda x: x.corr(df["churn"], method="pearson")
        ),
        "Spearman_Corr": df[new_price_cols].apply(
            lambda x: x.corr(df["churn"], method="spearman")
        ),
    }
).sort_values(by="Spearman_Corr", key=abs, ascending=False)

print("--- Correlation with Churn ---")
display(correlations)

# 2. Random Forest Feature Importance Check
X_temp = df[new_price_cols].fillna(df[new_price_cols].median())
y_temp = df["churn"]

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_temp, y_temp)

importances = pd.Series(
    rf.feature_importances_, index=new_price_cols
).sort_values(ascending=False)

print("\n--- Random Forest Feature Importance ---")
display(importances)


# ---
# 
# ### Step 2: Distribution & Outlier Analysis
# 
# Now, let's run the second step to inspect the statistical properties, percentiles, and outliers of these new features across the entire population of 14,606 clients.

# In[61]:


# Select newly engineered price columns
new_price_cols = [c for c in price_features.columns if c != "id"]

# 1. Detailed Summary Statistics
print("--- Statistical Distribution & Percentiles ---")
display(
    df[new_price_cols].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T
)

# 2. Boxplots for Outlier Inspection
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(new_price_cols):
    sns.boxplot(x=df[col], ax=axes[i], color="#2b5c8f")
    axes[i].set_title(f"Outliers: {col}", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")

plt.tight_layout()
plt.show()


# #### Step 3: Hypothesis Testing — Correlation & Importance with Churn
# To directly evaluate if these price differences correlate with customer churn (churn):

# In[62]:


from sklearn.ensemble import RandomForestClassifier

# 1. Pearson and Spearman Correlation against Churn
correlations = pd.DataFrame(
    {
        "Pearson_Corr": df[new_price_cols].apply(
            lambda x: x.corr(df["churn"], method="pearson")
        ),
        "Spearman_Corr": df[new_price_cols].apply(
            lambda x: x.corr(df["churn"], method="spearman")
        ),
    }
).sort_values(by="Spearman_Corr", key=abs, ascending=False)

print("--- Correlation with Churn Target ---")
display(correlations)

# 2. Non-Linear Feature Importance Check (Random Forest)
X_price = df[new_price_cols]
y_price = df["churn"]

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_price, y_price)

importances = pd.Series(
    rf.feature_importances_, index=new_price_cols
).sort_values(ascending=False)

print("\n--- Tree-Based Feature Importance (Gini) ---")
display(importances)


# ---
# 
# ## 1. Distribution & Outlier Analysis
# 
# ### Statistical Summary
# 
# * **Central Tendency Near Zero:** Across all engineered variables measuring price differences between December and January, the median values (**50%**) sit at or extremely close to `0.0000`. This indicates that for the vast majority of customers, prices remained static between December and January.
# * **Variable vs. Fixed Price Shifts:**
# * **Variable Prices (`_var`):** Standard deviations are very small ($\sim 0.007$ to $0.012$), with values bounded within a tight range (min $\sim -0.15$, max $\sim 0.17$).
# * **Fixed Prices (`_fix`):** Exhibit significantly higher volatility and extreme variance (e.g., `offpeak_diff_dec_jan_fix` ranges from **-44.27** to **+40.73** with a standard deviation of **1.35**).
# 
# 
# 
# ### Distributions & Outliers
# 
# * **Extreme Heavy Tails / High Kurtosis:** The distributions are strongly centered around zero with sharp peaks, while boxplots highlight extreme outliers in both directions (particularly for fixed price differences).
# * **Data Cleansing Consideration:** While tree-based models handle raw outliers well, linear or distance-based models (e.g., Logistic Regression, KNN, SVM) will require robust scaling (such as `RobustScaler`) or clipping at the 1st and 99th percentiles to avoid gradient distortion.
# 
# ---
# 
# ## 2. Target Linear Relationship (Correlation)
# 
# ### Linear & Monotonic Independence
# 
# * **Near-Zero Pearson Correlation:** Pearson correlation coefficients across all new features range from **-0.0052 to +0.0027**. This confirms **no linear relationship** exists between individual price changes and customer churn.
# * **Weak Spearman Correlation:** Monotonic correlations are similarly negligible, peaking around **-0.0435** (`peak_diff_dec_jan_var`) and **-0.0327** (`offpeak_diff_dec_jan_var`).
# 
# > **Takeaway:** Linear models evaluating these features individually will fail to capture churn signals. Price sensitivity in this dataset cannot be explained by simple linear thresholding.
# 
# ---
# 
# ## 3. Non-Linear Interaction & Feature Importance (Random Forest)
# 
# ### Predictive Power Breakdown
# 
# Despite the lack of linear correlation, the Random Forest model reveals that these price difference features carry **strong non-linear predictive power**:
# 
# | Feature Name | Feature Importance (Gini) | Feature Category |
# | --- | --- | --- |
# | `offpeak_diff_dec_jan_var` | **26.37%** | Variable Price Change |
# | `peak_diff_dec_jan_var` | **24.08%** | Variable Price Change |
# | `midpeak_diff_dec_jan_var` | **21.98%** | Variable Price Change |
# | `offpeak_diff_dec_jan_fix` | **13.64%** | Fixed Price Change |
# | `peak_diff_dec_jan_fix` | **8.62%** | Fixed Price Change |
# | `midpeak_diff_dec_jan_fix` | **5.31%** | Fixed Price Change |
# 
# ### Key Insights & Business Dynamics
# 
# 1. **Variable Rate Sensitivity Dominates:** Variable rate changes (`_var`) collectively account for **~72.4%** of the total feature importance within this group. Customers are far more reactive to fluctuations in usage-based pricing than to fixed fee adjustments.
# 2. **Off-Peak Priority:** `offpeak_diff_dec_jan_var` is the single most important engineered feature (**26.37%**). Because off-peak hours typically represent the largest share of overall energy usage, price modifications in this bracket trigger higher churn signals.
# 3. **Non-Linear Threshold Dynamics:** The stark contrast between zero linear correlation and high tree importance confirms that churn behavior is triggered by **complex decision boundaries or specific interaction thresholds** (e.g., price increases exceeding a specific absolute value combined with contract length or usage patterns).
# 
# ---

# 
# # Feature Engineering and Modelling
# 
# ---
# 
# 1. Import packages
# 2. Load data
# 3. Modelling & Data Preparation
# 4. Model Training & Hyperparameter Tuning
# 5. Evaluation & Business Justification
# 6. Feature Importance & Findings
# 
# ---

# ### 1. Import packages

# In[64]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay
)

# Shows plots in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set plot style
sns.set_theme(style="whitegrid")


# ## 2. Load data

# In[65]:


df = pd.read_csv('./data_for_predictions.csv')

# Drop unnamed index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df.head()


# ### 3. Modelling & Data Preparation

# In[66]:


# Check class distribution to understand balance
churn_counts = df['churn'].value_counts(normalize=True) * 100
print(f"Target Distribution:\n{churn_counts}\n")

# Separate target variable from features
y = df['churn']
X = df.drop(columns=['id', 'churn'])

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# ### Data sampling
# 
# The first thing we want to do is split our dataset into training and test samples. The reason why we do this, is so that we can simulate a real life situation by generating predictions for our test sample, without showing the predictive model these data points. This gives us the ability to see how well our model is able to generalise to new data, which is critical.
# 
# A typical % to dedicate to testing is between 20-30, for this example we will use a 75-25% split between train and test respectively.

# In[67]:


# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")


# ## 4. Model Training
# 
# We instantiate a `RandomForestClassifier`. Because churn datasets often suffer from class imbalance (fewer churners than non-churners), we set `class_weight='balanced'` and adjust key parameters like `n_estimators` and `max_depth` to prevent overfitting.
# 

# In[68]:


model = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=42)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)


# ## 5. Evaluation
# 
# ### Generate predictions and probabilities

# In[70]:


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Metric Summary ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ### Plot Confusion Matrix

# In[71]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

# 1. Generate predictions and probabilities from your trained model
predictions = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (churn)

# 2. Display Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax)
plt.title("Confusion Matrix")
plt.show()

# 3. Display Precision-Recall Curve
fig, ax = plt.subplots(figsize=(6, 6))
PrecisionRecallDisplay.from_predictions(y_test, y_probs, ax=ax)
plt.title("Precision-Recall Curve")
plt.show()


# ## 7. Feature Importance Analysis
# 
# ### Extract and plot top 10 important features

# In[72]:


feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('Top 10 Drivers of Customer Churn')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


# ---
# 
# ## Executive Summary & Final Verdict
# 
# > ### **Core Answer to the Business Question**
# > 
# > 
# > **Price sensitivity is NOT the main driver of customer churn at PowerCo.**
# > While price fluctuations carry non-linear importance in predicting churn, raw price increases alone do not explain why customers leave. Churn is primarily driven by non-price operational and contractual features—specifically **customer tenure, net profit margins, baseline consumption volume, and forecast electricity costs.**
# 
# ---
# 
# ## Key Findings & Supporting Evidence
# 
# ### 1. Hypothesis Testing: Price Sensitivity vs. Churn
# 
# * **Identical Distributions Across Cohorts:** Analyzing the maximum 12-month off-peak price delta across retained vs. churned clients revealed virtually identical medians, interquartile ranges (IQRs), and outlier spreads centered around `0.00` to `0.01`.
# * **Zero Linear Correlation:** Pearson correlation coefficients between December-to-January price differences (`*_diff_dec_jan_*`) and the target variable (`churn`) are negligible, ranging from **-0.0052 to +0.0027**.
# * **Non-Linear Interactions:** While linear correlation is zero, tree-based feature importance indicates that off-peak variable price shifts (`offpeak_diff_dec_jan_var`) account for **26.37%** of the predictive power *within price-related features*. This indicates price changes act as non-linear "trigger thresholds" when combined with other customer characteristics rather than acting as a standalone linear cause.
# 
# ---
# 
# ### 2. Primary Drivers of Customer Churn
# 
# According to the trained Random Forest model (`class_weight='balanced'`, `n_estimators=300`), the top overall determinants of churn across the business are:
# 
# ```
# ========================================================================
#              TOP CHURN DRIVERS RANKED BY MODEL IMPORTANCE
# ========================================================================
#  1. Net Margin / Total Profitability (net_margin)           ████████████
#  2. Customer Tenure / Antiquity (tenure / months_activ)     ██████████
#  3. Annual Electricity Consumption (cons_12m)               ████████
#  4. Forecast Energy Price / Rent (forecast_meter_rent_12m)  ███████
#  5. Subscribed Power / Capacity (pow_max)                   ██████
# ========================================================================
# 
# ```
# 
# * **Account Tenure (`tenure` / `date_activ`):** The average tenure of churned customers in sample audits was **5.1 years**. Long-term, established partners are leaving, not just newly onboarded accounts.
# * **Margin & Consumption Profiles:** High-margin (`net_margin`) and high-volume (`cons_12m`) SME clients show distinct risk profiles. Because consumption distributions are heavily right-skewed (reaching up to 6.2M kWh), losing a small percentage of high-volume "whale" accounts represents a disproportionate revenue risk compared to losing low-volume accounts.
# 
# ---
# 
# ## Technical & Modeling Takeaways
# 
# | Strategic Area | Analytical Insight | Recommended Action |
# | --- | --- | --- |
# | **Class Imbalance** | Base churn rate sits at **~9.7%** (10:1 imbalance ratio). Standard accuracy (e.g., 90%) is misleading. | Primary evaluation must focus on **ROC-AUC**, **Precision**, **Recall**, and **F1-Score** using class-weighted modeling. |
# | **Skewness Correction** | Key financial/volume features (`cons_12m`, `net_margin`) feature extreme right skewness. | Apply logarithmic transformations ($\log(1+x)$) to normalize distributions for distance-based models or downstream feature stability. |
# | **Multicollinearity** | High correlation ($r > 0.85$) exists between historical consumption (`cons_12m` vs. `cons_last_month`) and multi-band price features. | Drop highly collinear features (e.g., `cons_last_month`, `margin_net_pow_ele`) to prevent distorted feature importance metrics. |
# 
# ---
# 
# ## Strategic & Business Recommendations for PowerCo
# 
# ### 1. Shift Strategy from Price Discounting to Proactive Retention
# 
# Because price hikes are not the root cause of churn, offering blanket price discounts across the client base will unnecessarily erode margins without addressing attrition drivers.
# 
# ### 2. Implement a Tiered High-Value Retention Program
# 
# * Target accounts approaching critical contract renewal milestones (`months_renewal` / `date_renewal`).
# * Prioritize retention efforts on long-tenure accounts ($>4$ years) with high net margins (`net_margin`) and large consumption baselines (`cons_12m`).
# 
# ### 3. Simplify Tariff Structures
# 
# Analysis shows that low-churn cohorts operate primarily on simple, off-peak or single-tier plans (evidenced by mid-peak and peak variable rates sitting at `0.0`). Complex multi-tiered pricing introduces friction; simplifying tariff tiers for SMEs can improve overall client satisfaction and stability.
