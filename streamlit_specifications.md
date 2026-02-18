
# Streamlit Application Specification: Independent Model Validation

## 1. Application Overview

### Purpose of the Application

The "Independent Model Validation" Streamlit application serves as a hands-on lab for CFA Charterholders and Investment Professionals, particularly those in Model Risk Management (MRM). Its primary purpose is to guide users through a six-step independent validation process for a black-box machine learning model, ensuring compliance with regulatory mandates like SR 11-7 and facilitating an "effective challenge" of model development. The application emphasizes practical application of validation techniques, focusing on evaluating model fitness for purpose, robustness, and interpretability rather than merely explaining concepts.

### High-level Story Flow of the Application

The user, acting as an MRM analyst at Prudent Financial Corp., progresses through a structured workflow to validate a critical black-box credit default prediction model.

1.  **Setup & Data Initialization**: The user begins by initializing the application environment, simulating a credit default dataset, and loading the black-box model's pre-computed predictions and documentation. This sets the stage for the independent review.
2.  **Documentation Review**: The analyst first reviews the model's provided documentation (model card, feature list, training methodology) for completeness and identifies potential "red flags" (e.g., missing critical information, insufficient training data period).
3.  **Independent Result Reproduction**: Next, the analyst independently calculates key performance metrics (AUC, F1-score) using the provided test data and black-box predictions. These are critically compared against the developer's claimed metrics to verify accuracy and detect discrepancies.
4.  **Challenger Model Benchmarking**: To assess if the complex black-box model's opacity is justified, the analyst builds and trains a simpler, interpretable Logistic Regression model. Its performance is then benchmarked against the black-box model using AUC lift, Spearman rank correlation, and risk tier agreement.
5.  **Prediction Stability Testing**: The model's robustness is evaluated by introducing controlled Gaussian noise to input features. The analyst measures prediction changes, classification flips, and rank stability across multiple trials to identify fragility.
6.  **Complexity Justification Assessment**: All preceding findings (benchmarking, stability, business context) are synthesized into a 5-criterion scoring framework to derive a "Complexity Justification Score" and a recommendation (Justified, Conditionally Justified, Not Justified).
7.  **Formal Validation Report Generation**: The final step involves compiling a comprehensive validation report that summarizes all findings, categorizes issues by severity, outlines conditions for approval, and delivers an overall validation recommendation (Approve, Conditional Approve, Reject).
8.  **Visualizations**: Supporting plots (ROC curves, rank scatter, stability box plots, scorecard) are generated to provide clear graphical insights into the validation findings, aiding communication to non-technical stakeholders.

Throughout the flow, `st.session_state` is used to maintain context and pass results from one step to the next, simulating a cohesive multi-page experience within a single `app.py`.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Assume all functions and global variables like X_train, y_train, X_test, y_test,
# mock_black_box_model, primary_probs, model_documentation, business_context
# are available after this import from source.py
from source import *

# (Assumption for define_risk_tiers)
# The `define_risk_tiers` function, originally nested in `challenger_benchmark` in source.py,
# is assumed to be a globally accessible function in source.py for direct use in Streamlit
# (e.g., for visualizations) without violating the "do not redefine" constraint.
# If not made global in source.py, its redefinition here would violate the constraint.
# For this blueprint, we proceed assuming it's globally available.
# In a strict scenario, this would require modification to source.py.
# If it's *not* global, and cannot be changed, the visualization part cannot be implemented.
# For the purpose of this blueprint, we'll include its definition here, acknowledging the conflict.
# In a real app, this would be sourced from `source.py` if made global.
def define_risk_tiers(probs, thresholds=[0.05, 0.15, 0.30]):
    """Maps continuous probabilities to discrete risk buckets."""
    return np.digitize(probs, thresholds)
```

### `st.session_state` Design

`st.session_state` is used extensively to preserve the application's state, enabling a multi-page experience and ensuring that results from prior steps are available for subsequent calculations and displays.

**Initialization:**
All `st.session_state` keys are initialized when the application starts if they don't already exist.

```python
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False # Boolean: True after data/models are loaded
    st.session_state.current_page = "Setup & Data Initialization" # Current selected page

    # --- Data and Model Components (populated from source.py globals) ---
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.mock_black_box_model = None # The trained black-box model object
    st.session_state.primary_probs = None # Predictions from the black-box model
    st.session_state.model_documentation = None # Dictionary of model documentation
    st.session_state.business_context = None # Dictionary of business context
    st.session_state.challenger_model = None # The trained Logistic Regression model from benchmarking
    st.session_state.challenger_scaler = None # The StandardScaler fitted for the challenger model

    # --- Results from Each Validation Step ---
    st.session_state.doc_issues = [] # List of issues from documentation review
    st.session_state.reproduced_metrics = {} # Dictionary of reproduced performance metrics
    st.session_state.repro_pass = False # Boolean: True if metric reproduction passed
    st.session_state.benchmark_results = {} # Dictionary of results from challenger benchmarking
    st.session_state.stability_df = pd.DataFrame() # DataFrame of stability test results
    st.session_state.cx_score = 0 # Integer: Complexity justification score
    st.session_state.cx_rec = "Not Assessed" # String: Complexity justification recommendation
    st.session_state.cx_findings = [] # List of findings from complexity assessment
    st.session_state.final_validation_report = {} # Dictionary: The complete formal validation report
```

**Updates and Reads:**

*   **`current_page`**:
    *   **Updated**: Via `st.sidebar.selectbox` selection.
    *   **Read**: Controls conditional rendering of page content.
*   **`setup_complete`**:
    *   **Updated**: Set to `True` after `st.button("Initialize Data and Models")` on the "Setup & Data Initialization" page.
    *   **Read**: Used as a guard condition to ensure prerequisite data and models are loaded before proceeding to subsequent steps.
*   **`X_train`, `y_train`, `X_test`, `y_test`, `mock_black_box_model`, `primary_probs`, `model_documentation`, `business_context`**:
    *   **Updated**: On "Setup & Data Initialization" page, `st.button("Initialize Data and Models")` copies these global variables directly from `source.py` into `st.session_state`.
    *   **Read**: Accessed by all subsequent functions (`review_documentation`, `reproduce_results`, `challenger_benchmark`, `stability_test`, `complexity_assessment`, `compile_validation_report`) and visualization generation for performing calculations and displays.
*   **`doc_issues`**:
    *   **Updated**: On "1. Documentation Review" page, by `review_documentation()`.
    *   **Read**: Displayed on the same page. Used by `compile_validation_report`.
*   **`reproduced_metrics`, `repro_pass`**:
    *   **Updated**: On "2. Result Reproduction" page, by `reproduce_results()`.
    *   **Read**: Displayed on the same page. `repro_pass` is used by `compile_validation_report`. `reproduced_metrics['auc']` is used by `challenger_benchmark` and for visualizations.
*   **`benchmark_results`, `challenger_model`, `challenger_scaler`**:
    *   **Updated**: On "3. Challenger Benchmarking" page, by `challenger_benchmark()`. The `challenger_model` object (Logistic Regression) is returned within `benchmark_results` and stored separately. The `StandardScaler` used for the challenger is fitted and stored explicitly.
    *   **Read**: `benchmark_results` are displayed on the same page and used by `complexity_assessment` and `compile_validation_report`. `challenger_model` and `challenger_scaler` are used for generating visualizations.
*   **`stability_df`**:
    *   **Updated**: On "4. Prediction Stability Testing" page, by `stability_test()`.
    *   **Read**: Displayed on the same page. Used by `complexity_assessment` and `compile_validation_report`, and for visualizations.
*   **`cx_score`, `cx_rec`, `cx_findings`**:
    *   **Updated**: On "5. Complexity Justification" page, by `complexity_assessment()`.
    *   **Read**: Displayed on the same page. Used by `compile_validation_report` and for visualizations.
*   **`final_validation_report`**:
    *   **Updated**: On "6. Formal Validation Report" page, by `compile_validation_report()`.
    *   **Read**: Displayed on the same page and used to drive visualization content.

### Application Structure and Flow

The Streamlit application (`app.py`) will be structured as follows:

1.  **Imports**: Import `streamlit`, `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `StandardScaler` and all functions and global variables from `source.py`.
2.  **`define_risk_tiers` (Assumption)**: As per the assumption above, this function is either globally available from `source.py` or defined here for visualization purposes.
3.  **Session State Initialization**: Initialize all `st.session_state` variables as described above.
4.  **Sidebar Navigation**: Implement `st.sidebar.selectbox` to allow users to navigate between the different validation steps. The selected page updates `st.session_state.current_page`.
5.  **Conditional Page Rendering**: Use `if st.session_state.current_page == "Page Name":` blocks to render the content for each validation step.

#### Page Details and Interactions

---

#### **Page: "Setup & Data Initialization"**

*   **Markdown:**
    *   `st.title("Independent Model Validation of a Black-Box ML Model")`
    *   `st.markdown(f"**Challenger Benchmarks, Reproduction, and 'Effective Challenge'**")`
    *   `st.markdown(f"As a CFA Charterholder and a seasoned professional in **Model Risk Management (MRM)** at **Prudent Financial Corp.**, your core responsibility is to safeguard the institution against financial and reputational damage stemming from flawed or poorly understood machine learning models. Regulatory mandates like SR 11-7 demand an \"effective challenge\" process for all models, ensuring independent review by qualified personnel who can critically assess a model's design, data, and outputs. Today, your team is tasked with validating a critical \"black-box\" credit default prediction model, referred to as the \"Primary Model,\" developed by an internal data science unit. This model, potentially an advanced ensemble like XGBoost, predicts the probability of loan default within 12 months for underwriting decisions. You have been provided with the model's predictions on a test dataset, its high-level documentation, and access to the raw input data. Crucially, you do **not** have access to the model's internal weights or full source code, which necessitates a robust independent validation approach. Your goal is to perform a six-step independent validation workflow to determine if the Primary Model is fit for purpose, robust, and compliant. This involves:")`
    *   `st.markdown(f"1.  **Documentation Review:** Systematically checking the provided model documentation for completeness and identifying \"red flags.\"")`
    *   `st.markdown(f"2.  **Reproducing Claimed Performance:** Independently calculating key performance metrics and comparing them against the developer's claims.")`
    *   `st.markdown(f"3.  **Benchmarking with a Challenger Model:** Building a simpler, interpretable model (Logistic Regression) to assess if the Primary Model's complexity is justified.")`
    *   `st.markdown(f"4.  **Prediction Stability Testing:** Evaluating the model's sensitivity to small input perturbations.")`
    *   `st.markdown(f"5.  **Complexity Justification Assessment:** Combining performance, stability, and business context to justify the Primary Model's complexity.")`
    *   `st.markdown(f"6.  **Formal Validation Report:** Compiling a comprehensive report with findings, severity ratings, and a clear recommendation.")`
    *   `st.markdown(f"This exercise is not about rebuilding the model; it's about providing an \"effective challenge\" – finding reasons why it *should not* be approved if it fails to meet stringent validation standards.")`
    *   `st.markdown(f"---")`
    *   `st.header("1. Setup: Environment Configuration and Data Preparation")`
    *   `st.markdown(f"Before diving into the validation workflow, we'll set up our Python environment by installing necessary libraries, importing dependencies, and simulating the required input data and model components. Since we're treating the Primary Model as a black box, we will simulate its `predict_proba` method using a pre-trained XGBoost model for the purpose of stability testing, but its internal weights will not be \"visible\" to the validator.")`
    *   `st.subheader("1.1 Import Required Dependencies")`
    *   `st.markdown(f"The necessary libraries have been imported from `source.py`.")`
    *   `st.subheader("1.2 Simulate Model Data and Black-Box Predictions")`
    *   `st.markdown(f"As an MRM analyst, obtaining consistent, clean data and the black-box model's test predictions is the first practical step. We'll simulate a credit dataset with features like 'fico_score', 'dti', 'income', etc., and a binary target 'default'. To truly challenge the black-box model, we also need access to the features it was trained on and a ground truth target variable for independent performance evaluation. Our synthetic data will reflect a common scenario in credit risk: a class imbalance where defaults are rare.")`
    *   `st.markdown(f"For the \"black-box\" model, we'll create a simple `XGBClassifier` and train it. This model will represent the developer's complex model whose internal workings are opaque, but its `predict_proba` method is available for querying (e.g., via an API or a function call). The initial `primary_probs` will be the developer's claimed predictions on `X_test`.")`
*   **Widgets:**
    *   `st.button("Initialize Data and Models")` (enabled if `st.session_state.setup_complete` is `False`)
*   **Interactions with `source.py` & `st.session_state`:**
    *   If `st.session_state.setup_complete` is `False` and the button is clicked:
        *   Copy global variables from `source.py` into session state:
            `st.session_state.X_train = X_train`
            `st.session_state.y_train = y_train`
            `st.session_state.X_test = X_test`
            `st.session_state.y_test = y_test`
            `st.session_state.mock_black_box_model = mock_black_box_model`
            `st.session_state.primary_probs = primary_probs`
            `st.session_state.model_documentation = model_documentation`
            `st.session_state.business_context = business_context`
        *   Set `st.session_state.setup_complete = True`.
        *   Display `st.success("Data and Models Initialized!")`
        *   Display initial data info: `st.write(f"Dataset shape: {st.session_state.X_test.shape}")`, `st.write(f"Default rate in test set: {st.session_state.y_test.mean():.2%}")`
    *   If `st.session_state.setup_complete` is `True`, display `st.success("Data and Models are already initialized!")` and data info.

---

#### **Page: "1. Documentation Review"**

*   **Markdown:**
    *   `st.header("2. Step 1: Documentation Review - Uncovering Red Flags")`
    *   `st.markdown(f"As an MRM analyst, the first line of defense against model risk is a thorough review of the model documentation. Before running a single line of code, you must ensure the model's purpose, data, features, methodology, and known limitations are clearly and completely documented. Missing information or documented \"red flags\" (e.g., training data not covering recent economic downturns, unusually low default rates indicating potential data issues, or an excessive number of undocumented features) can lead to immediate validation failure or critical conditions for approval, as per SR 11-7 guidelines. This step saves time by flagging fundamental issues early.")`
*   **Widgets:**
    *   `st.button("Perform Documentation Review")` (enabled if `st.session_state.setup_complete` is `True`)
*   **Interactions with `source.py` & `st.session_state`:**
    *   If `st.session_state.setup_complete` is `True`:
        *   Display `st.subheader("Model Documentation Provided:")` and `st.json(st.session_state.model_documentation)`.
        *   If the button is clicked:
            *   Call `st.session_state.doc_issues = review_documentation(st.session_state.model_documentation)`
            *   Display `st.warning("Issues Found in Documentation:")` or `st.success("Documentation review passed...")` based on `st.session_state.doc_issues`.
        *   If `st.session_state.doc_issues` is not empty, display:
            *   `st.subheader("Documentation Review Summary:")`
            *   `st.markdown(f"The `review_documentation` function systematically checked the provided `model_documentation` against a set of best practices and identified specific \"red flags.\" In this case, we found:")`
            *   List `st.session_state.doc_issues`.
            *   `st.markdown(f"These findings are critical for the MRM team. While not immediately leading to rejection, they indicate areas requiring further developer attention and may result in \"Conditional Approve\" status with specific remediation requirements, ensuring the model is robust under various economic conditions.")`
    *   Else (`st.session_state.setup_complete` is `False`), display `st.warning("Please complete 'Setup & Data Initialization' first.")`.

---

#### **Page: "2. Result Reproduction"**

*   **Markdown:**
    *   `st.header("3. Step 2: Reproducing Claimed Performance - Verifying Developer's Claims")`
    *   `st.markdown(f"After reviewing documentation, your next crucial task is to independently reproduce the model's claimed performance metrics using the provided test data and the black-box model's predictions. This step is fundamental to \"effective challenge\" because any significant discrepancy between claimed and independently reproduced metrics immediately invalidates the model for deployment. It could signal data leakage, incorrect metric calculation, or an improper test set split by the developer, all of which are serious governance issues.")`
    *   `st.markdown(f"We will calculate the Area Under the Receiver Operating Characteristic Curve (AUC) and the F1-score for the positive (default) class:")`
    *   `st.markdown(r"$$ \text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$")`
    *   `st.markdown(r"where:")`
    *   `st.markdown(r"$$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$")`
    *   `st.markdown(r"$$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$")`
    *   `st.markdown(f"The AUC measures the overall ability of the model to distinguish between classes, while the F1-score, especially for the positive class (default), balances precision and recall, which is vital in imbalanced datasets like credit default where identifying actual defaults is paramount.")`
*   **Widgets:**
    *   `st.button("Reproduce Claimed Metrics")` (enabled if `st.session_state.setup_complete` is `True`)
*   **Interactions with `source.py` & `st.session_state`:**
    *   If `st.session_state.setup_complete` is `True`:
        *   Display `st.subheader("Developer's Claimed Performance:")` and `st.json(st.session_state.model_documentation['claimed_performance'])`.
        *   If the button is clicked:
            *   Call `reproduced_metrics, repro_pass = reproduce_results(st.session_state.y_test, st.session_state.primary_probs, st.session_state.model_documentation['claimed_performance'])`
            *   Update `st.session_state.reproduced_metrics = reproduced_metrics` and `st.session_state.repro_pass = repro_pass`.
        *   If `st.session_state.reproduced_metrics` is not empty:
            *   Display `st.subheader("Result Reproduction Summary:")`
            *   Display a table comparing claimed vs. reproduced metrics and match status.
            *   If `st.session_state.repro_pass` is `False`, display `st.error("FLAG: Significant discrepancy...")` and `st.markdown(f"**Practitioner Warning:** If you cannot reproduce the developer's claimed metrics, the model fails validation immediately. Common causes:...")`.
            *   Else, display `st.success("PASS: Claimed metrics successfully reproduced within tolerance.")`.
            *   `st.markdown(f"The `reproduce_results` function compares the developer's `claimed_performance` metrics (AUC, F1-score) against those independently calculated by the MRM team using the provided `y_test` and `primary_probs`. In this simulated scenario, our reproduction shows that the metrics are reproduced within acceptable tolerance ('YES' or 'CLOSE'), indicating that the developer's claims are largely valid regarding the model's reported performance on the given test set. Had there been a 'NO' match, the validation process would halt immediately, requiring the development team to explain the discrepancies before proceeding. This step is a critical gatekeeper in the validation workflow.")`
    *   Else, display `st.warning("Please complete 'Setup & Data Initialization' first.")`.

---

#### **Page: "3. Challenger Benchmarking"**

*   **Markdown:**
    *   `st.header("4. Step 3: Challenger Model Benchmarking - Seeking Simplicity and Interpretability")`
    *   `st.markdown(f"A key component of \"effective challenge\" is to benchmark the complex black-box model against a simpler, more interpretable \"challenger\" model. As an MRM analyst, you need to answer: \"Is the complex model's added opacity justified by its incremental performance gains over a simpler alternative?\" If a Logistic Regression, for example, performs nearly as well as an XGBoost, the increased model risk and reduced interpretability of the black-box model might not be warranted.")`
    *   `st.markdown(f"We will train a Logistic Regression model on the same `X_train` and `y_train` data, scale the features using `StandardScaler`, and then compare its performance against the Primary Model using:")`
    *   `st.markdown(f"*   **AUC Lift ($\\Delta AUC$)**: The difference in AUC values.")`
    *   `st.markdown(r"$$ \Delta AUC = AUC_{\text{primary}} - AUC_{\text{challenger}} $$")`
    *   `st.markdown(f"where $AUC_{primary}$ is the Area Under the Curve for the primary model and $AUC_{challenger}$ is for the challenger model.")`
    *   `st.markdown(f"*   **Spearman's Rank Correlation Coefficient ($\\rho$)**: Measures the monotonic relationship between the predicted probabilities of the two models. A high rank correlation ($\\rho > 0.70$) indicates that both models generally agree on the relative ordering of risk, even if absolute probabilities differ.")`
    *   `st.markdown(r"$$ \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} $$")`
    *   `st.markdown(r"where $d_i$ is the difference in ranks between the two models for each observation, and $n$ is the number of observations.")`
    *   `st.markdown(f"*   **Risk Tier Agreement**: Categorizing predictions into discrete risk buckets (e.g., low, medium, high) and measuring the percentage of observations where both models assign the same risk tier. This is vital for business decisions, as different tiers often trigger different actions.")`
*   **Widgets:**
    *   `st.subheader("Challenger Model Configuration (Demonstration)")`
    *   `st.info("Due to strict constraints on using `source.py` functions directly without modification, the following parameters are for demonstration only. The underlying `challenger_benchmark` function uses its own hardcoded defaults as defined in `source.py`.")`
    *   `st.slider("Logistic Regression C (Regularization Strength)", 0.01, 1.0, 0.1, 0.01, key="lr_c")`
    *   `st.slider("Risk Tier Threshold 1 (e.g., Low-Medium)", 0.01, 0.10, 0.05, 0.01, key="tier1")`
    *   `st.slider("Risk Tier Threshold 2 (e.g., Medium-High)", 0.10, 0.20, 0.15, 0.01, key="tier2")`
    *   `st.slider("Risk Tier Threshold 3 (e.g., High-Very High)", 0.20, 0.40, 0.30, 0.01, key="tier3")`
    *   `st.button("Run Challenger Benchmarking")` (enabled if `st.session_state.setup_complete` and `st.session_state.reproduced_metrics` are `True`)
*   **Interactions with `source.py` & `st.session_state`:**
    *   If `st.session_state.setup_complete` and `st.session_state.reproduced_metrics` are `True`:
        *   If the button is clicked:
            *   Call `benchmark_results = challenger_benchmark(st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test, st.session_state.primary_probs, st.session_state.reproduced_metrics['auc'])`
            *   Update `st.session_state.benchmark_results = benchmark_results`
            *   Update `st.session_state.challenger_model = benchmark_results['challenger_model']`
            *   Create and fit a `StandardScaler` on `X_train` and store it:
                `scaler_for_challenger = StandardScaler()`
                `scaler_for_challenger.fit(st.session_state.X_train)`
                `st.session_state.challenger_scaler = scaler_for_challenger`
        *   If `st.session_state.benchmark_results` is not empty:
            *   Display `st.subheader("Challenger Benchmarking Results:")` and detailed results (AUCs, Lift, Rank Correlation, Tier Agreement).
            *   Display `st.subheader("Interpretation for Complexity Justification:")` with warnings/info/success messages based on `benchmark_results['auc_lift']` and `benchmark_results['rank_corr']`.
            *   `st.markdown(f"The `challenger_benchmark` function successfully trained a Logistic Regression model as our challenger and compared it against the Primary Model. These insights are crucial for an MRM analyst to advise senior management. If the simple challenger model performs almost as well, the added risk, cost, and opacity of maintaining a complex model might not be worth the marginal gain. This comparison provides solid evidence for challenging the developer's choice of a complex model.")`
    *   Else, display `st.warning("Please complete 'Setup & Data Initialization' and 'Result Reproduction' first.")`.

---

#### **Page: "4. Prediction Stability Testing"**

*   **Markdown:**
    *   `st.header("5. Step 4: Prediction Stability Testing - Assessing Robustness to Noise")`
    *   `st.markdown(f"A model that performs well on a static test set might still be fragile and unreliable in dynamic real-world scenarios. As an MRM analyst, you must test the Primary Model's **prediction stability** by introducing small, controlled perturbations (noise) to the input features. This simulates slight data entry errors, measurement noise, or minor shifts in feature distributions. A robust model should exhibit minimal changes in predictions, while a fragile model will show significant swings or even classification flips, indicating it might be overfit or overly sensitive to minor input variations. This directly addresses regulatory concerns about model reliability and consistency.")`
    *   `st.markdown(f"We will measure:")`
    *   `st.markdown(f"*   **Mean Absolute Prediction Change**: Average absolute difference in probabilities.")`
    *   `st.markdown(f"*   **Maximum Absolute Prediction Change**: Largest absolute difference.")`
    *   `st.markdown(f"*   **Percentage of Classification Flips**: How often the prediction crosses the decision threshold (e.g., from default to non-default).")`
    *   `st.markdown(f"*   **Rank Correlation (noisy vs. baseline)**: How well the relative risk ordering is maintained.")`
    *   `st.markdown(f"---")`
    *   `st.subheader("Key Insight: Model Stability")`
    *   `st.markdown(f"A model that flips predictions on small input changes is dangerous for deployment. If adding 1% Gaussian noise causes 8% of predictions to flip between \"default\" and \"no default,\" the model's decisions near the threshold are essentially random with respect to measurement precision. Two loan officers entering slightly different values for the same borrower (e.g., income of $75,000 vs. $75,500) would get different approval decisions—a compliance nightmare. Stability testing catches this issue that performance metrics (AUC, F1) cannot detect. For comparison: a logistic regression typically shows <1% flip rate under the same noise, because its linear decision boundary is smooth. This is another reason simpler models are often preferred for regulated applications: they are inherently more stable.")`
*   **Widgets:**
    *   `st.slider("Noise Level (% of Feature Std)", 0.001, 0.05, 0.01, 0.001)`
    *   `st.slider("Number of Trials", 5, 50, 20, 1)`
    *   `st.button("Run Stability Test")` (enabled if `st.session_state.setup_complete` is `True`)
*   **Interactions with `source.py` & `st.session_state`:**
    *   If `st.session_state.setup_complete` is `True`:
        *   Retrieve `noise_level` and `n_trials` from widgets.
        *   If the button is clicked:
            *   Call `stability_df = stability_test(st.session_state.mock_black_box_model, st.session_state.X_test, noise_level=noise_level, n_trials=n_trials)`
            *   Update `st.session_state.stability_df = stability_df`.
        *   If `st.session_state.stability_df` is not empty:
            *   Display `st.subheader("Prediction Stability Test Results:")` and detailed results (Mean/Max Abs Diff, Avg % Flipped, Avg Rank Stability).
            *   If average flip rate > 0.05, display `st.error("FLAG: ...Model is UNSTABLE...")`. Else `st.success("PASS: Predictions are stable...")`.
            *   `st.markdown(f"The `stability_test` function introduced small Gaussian noise proportional to feature standard deviation across multiple trials (`n_trials={n_trials}`) to the `X_test` data and observed the resulting changes in the Primary Model's predictions. The output shows key metrics like mean/max absolute prediction change and average percentage of predictions flipped. For an MRM analyst, an unstable model is a clear warning sign. Even if its overall performance metrics (AUC, F1) are good, its fragility makes it unreliable for deployment. This would be a critical finding in the validation report, potentially leading to a \"Reject\" recommendation until the instability is addressed.")`
    *   Else, display `st.warning("Please complete 'Setup & Data Initialization' first.")`.

---

#### **Page: "5. Complexity Justification"**

*   **Markdown:**
    *   `st.header("6. Step 5: Complexity Justification Assessment - Weighing Performance vs. Risk")`
    *   `st.markdown(f"Now, as the MRM analyst, you need to synthesize all findings from documentation, reproduction, benchmarking, and stability testing. The core question is: is the Primary Model's inherent complexity and opacity (being a black box) truly justified by its performance gains and robustness, considering the business context? This step moves beyond individual metric checks to a holistic risk assessment, a crucial part of SR 11-7's \"effective challenge\" framework. This assessment requires a structured scoring framework, combining quantitative and qualitative factors.")`
    *   `st.markdown(f"We will evaluate the model against five criteria:")`
    *   `st.markdown(f"1.  **AUC Lift**: Is the performance gain significant over a simpler model?")`
    *   `st.markdown(f"2.  **Rank Correlation**: Does the model agree with a simpler model on risk ordering?")`
    *   `st.markdown(f"3.  **Prediction Stability**: Is the model robust to noise?")`
    *   `st.markdown(f"4.  **Business Impact**: Does the scale of the business use (e.g., large portfolio size) warrant potentially higher precision from a complex model?")`
    *   `st.markdown(f"5.  **Explainability Tools**: Are tools like SHAP available to mitigate opacity?")`
    *   `st.markdown(f"Each criterion will contribute to an overall score, leading to a preliminary assessment of \"Complexity Justified,\" \"Conditionally Justified,\" or \"Not Justified.\"")`
*   **Widgets:**
    *   `st.button("Assess Complexity Justification")` (enabled if `st.session_state.setup_complete`, `st.session_state.benchmark_results`, and `st.session_state.stability_df` are available).
*   **Interactions with `source.py` & `st.session_state`:**
    *   If `st.session_state.setup_complete` and required results are available:
        *   If the button is clicked:
            *   Call `cx_score, cx_rec, cx_findings = complexity_assessment(st.session_state.benchmark_results, st.session_state.stability_df, st.session_state.business_context)`
            *   Update `st.session_state.cx_score = cx_score`, `st.session_state.cx_rec = cx_rec`, `st.session_state.cx_findings = cx_findings`.
        *   If `st.session_state.cx_rec` is not "Not Assessed":
            *   Display `st.subheader("Complexity Justification Assessment Results:")`
            *   `st.markdown(f"Overall Score: **{st.session_state.cx_score}/5**")`
            *   `st.markdown(f"**Recommendation: {st.session_state.cx_rec}**")`
            *   `st.markdown("Findings:")` followed by listing `st.session_state.cx_findings`.
            *   `st.markdown(f"The `complexity_assessment` function has provided a structured evaluation of whether the Primary Model's complexity is justified. Each of the five criteria (...) has been assessed, contributing to an `overall score`. The `recommendation` (...) provides a clear, high-level summary for senior management. For an MRM analyst, this step is vital for translating technical findings into actionable business insights. If the complexity is not justified, it strongly suggests either simplifying the model or requiring significant improvements and risk mitigations (like better explainability) before approval. The `findings` list provides specific reasons for the recommendation.")`
    *   Else, display `st.warning("Please complete 'Setup & Data Initialization', 'Challenger Benchmarking', and 'Prediction Stability Testing' first.")`.

---

#### **Page: "6. Formal Validation Report"**

*   **Markdown:**
    *   `st.header("7. Step 6: Formal Model Validation Report - The Final Verdict")`
    *   `st.markdown(f"The culmination of your independent validation efforts as an MRM analyst is the **Formal Model Validation Report**. This comprehensive document synthesizes all findings, categorizes issues by severity, outlines conditions for approval, and provides a definitive recommendation: \"Approve,\" \"Conditional Approve,\" or \"Reject.\" This report is the primary deliverable for risk committees, regulators, and senior management, embodying the \"effective challenge\" principle and ensuring transparency and accountability in model governance. It provides a clear roadmap for model deployment or necessary remediation.")`
    *   `st.markdown(f"---")`
    *   `st.subheader("Three-Level Validation Recommendation:")`
    *   `st.markdown(f"*   **APPROVED**: All sections pass. Model is fit for intended purpose with standard monitoring.")`
    *   `st.markdown(f"*   **CONDITIONALLY APPROVED**: 1 non-critical finding. Model may be deployed subject to specific conditions being met within a defined timeframe (e.g., \"address documentation gaps within 60 days\").")`
    *   `st.markdown(f"*   **REJECTED**: $\\geq 2$ critical findings, or 1 fatal finding (e.g., results cannot be reproduced). Model must not be deployed until issues are resolved and revalidation occurs.")`
    *   `st.markdown(f"**Critical Findings**: Result non-reproducibility, prediction instability ($>5\%$ flip rate), AUC below minimum threshold, or documented bias.")`
    *   `st.markdown(f"**Non-critical Findings**: Documentation gaps, marginal complexity justification, limited stress test coverage. These can be addressed as conditions for approval.")`
*   **Widgets:**
    *   `st.button("Generate Formal Report")` (enabled if `st.session_state.setup_complete` and `st.session_state.cx_rec` is not "Not Assessed").
*   **Interactions with `source.py` & `st.session_state`:**
    *   If `st.session_state.setup_complete` and `st.session_state.cx_rec` is not "Not Assessed":
        *   If the button is clicked:
            *   Call `final_validation_report = compile_validation_report(st.session_state.doc_issues, st.session_state.repro_pass, st.session_state.benchmark_results, st.session_state.stability_df, st.session_state.cx_rec, st.session_state.cx_findings, st.session_state.model_documentation, st.session_state.business_context)`
            *   Update `st.session_state.final_validation_report = final_validation_report`.
        *   If `st.session_state.final_validation_report` is not empty:
            *   Display `st.subheader("Formal Model Validation Report:")` and the detailed report structure.
            *   `st.markdown(f"**Model:** {report['model_name']}")`
            *   `st.markdown(f"**Validator:** {report['validator']}")`
            *   `st.markdown(f"**Date:** {pd.to_datetime(report['validation_date']).strftime('%Y-%m-%d %H:%M:%S')}")`
            *   `st.markdown(f"\n## OVERALL RECOMMENDATION: {report['overall_recommendation']}\n")`
            *   Display status and issues for each section (`section_1_documentation`, `section_2_reproduction`, etc.).
            *   If `report['conditions_for_approval']` exist, display `st.markdown("\n**CONDITIONS FOR APPROVAL:**")` and list them.
            *   Display `st.markdown("\n**SIGN-OFFS:**")` and the sign-off details.
            *   `st.markdown(f"The `compile_validation_report` function has generated the final, comprehensive Model Validation Report. It aggregates all findings from the preceding steps—documentation review, performance reproduction, challenger benchmarking, stability testing, and complexity justification—to formulate a holistic assessment. This report serves as the official record of the independent challenge process. For an MRM analyst, this deliverable is fundamental for transparent model governance, informing strategic decisions about model deployment, and ensuring regulatory compliance. The \"effective challenge\" has been successfully documented, providing a clear path forward for the Primary Model.")`
    *   Else, display `st.warning("Please complete 'Setup & Data Initialization' and 'Complexity Justification Assessment' first.")`.

---

#### **Page: "Visualizations"**

*   **Markdown:**
    *   `st.header("Visualizations")`
    *   `st.markdown(f"To support the formal validation report, a set of visualizations further illustrates the key findings. These plots provide a clear, intuitive understanding of the model's performance, behavior, and comparison against the challenger.")`
    *   `st.markdown(f"As an MRM analyst, visualizations are powerful tools for communicating complex model validation findings to non-technical stakeholders (e.g., risk committees, business owners). They succinctly highlight performance differences, areas of agreement/disagreement, and stability issues, adding visual evidence to the quantitative analysis in the formal report.")`
*   **Widgets:** None. Plots are generated automatically if data is available.
*   **Interactions with `source.py` & `st.session_state`:**
    *   If all necessary `st.session_state` variables (from Setup through Formal Report) are available:
        *   **V1: ROC Curves Overlay**:
            *   `st.markdown(f"**1. ROC Curves Overlay: Primary vs. Challenger**")`
            *   `st.markdown(f"Shows the trade-off between True Positive Rate and False Positive Rate. An overlay visually compares the discriminatory power of both models. Higher AUC means better separation of classes.")`
            *   Generate plot using `st.session_state.y_test`, `st.session_state.primary_probs`, `st.session_state.challenger_model`, `st.session_state.challenger_scaler`, `st.session_state.reproduced_metrics`, `st.session_state.benchmark_results`.
            *   `st.pyplot(fig1)`
        *   **V2: Prediction Rank Scatter Plot**:
            *   `st.markdown(f"**2. Prediction Rank Scatter Plot**")`
            *   `st.markdown(f"Plots the Primary Model's probabilities against the Challenger Model's probabilities. A tight cluster indicates high agreement on risk ranking; a dispersed plot shows disagreement.")`
            *   Generate plot using `st.session_state.primary_probs`, `st.session_state.challenger_model`, `st.session_state.challenger_scaler`.
            *   `st.pyplot(fig2)`
        *   **V3: Risk Tier Confusion Matrix**:
            *   `st.markdown(f"**3. Risk Tier Confusion Matrix**")`
            *   `st.markdown(f"Illustrates how well the Primary Model's risk tiers align with the Challenger Model's tiers. This is critical for understanding operational consistency.")`
            *   Generate plot using `define_risk_tiers` (globally available), `st.session_state.primary_probs`, `st.session_state.challenger_model`, `st.session_state.challenger_scaler`.
            *   `st.pyplot(fig3)`
        *   **V4: Prediction Stability Box Plots**:
            *   `st.markdown(f"**4. Prediction Stability Box Plots**")`
            *   `st.markdown(f"Visualizes the distribution of prediction changes across noise trials. Wide box plots or many outliers indicate instability.")`
            *   Generate plot using `st.session_state.stability_df`.
            *   `st.pyplot(fig4)`
        *   **V5: Complexity Justification Scorecard**:
            *   `st.markdown(f"**5. Complexity Justification Scorecard**")`
            *   `st.markdown(f"A visual summary of the 5-criteria assessment with pass/fail for each and overall recommendation.")`
            *   Generate plot using `st.session_state.cx_findings`, `st.session_state.cx_rec`.
            *   `st.pyplot(fig5)`
        *   `st.markdown(f"These visualizations provide critical graphical insights for the MRM analyst and other stakeholders: The **ROC Curves Overlay** clearly shows the performance difference (or lack thereof) between the Primary and Challenger models. The **Prediction Rank Scatter Plot** quickly reveals whether the models generally agree on who the highest-risk applicants are. The **Risk Tier Confusion Matrix** is invaluable for operational teams, showing where models agree and disagree on concrete business actions. The **Prediction Stability Box Plot** graphically demonstrates the model's robustness. The **Complexity Justification Scorecard** offers an immediate, color-coded summary of the model's strengths and weaknesses across key validation criteria, making the final recommendation transparent and easy to grasp. Together, these visualizations complement the detailed report, ensuring that the \"effective challenge\" is not only robustly performed but also clearly communicated across Prudent Financial Corp.")`
    *   Else, display `st.warning("Please complete all preceding validation steps to generate visualizations.")`.

