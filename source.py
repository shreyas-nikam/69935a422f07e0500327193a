import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix
from sklearn.datasets import make_classification
from scipy.stats import spearmanr
import xgboost as xgb
import datetime
from typing import Dict, List, Tuple, Any, Union


def define_risk_tiers(probs: np.ndarray, thresholds: List[float] = None) -> np.ndarray:
    """
    Maps continuous probabilities to discrete risk buckets.
    Args:
        probs (np.ndarray): Array of prediction probabilities.
        thresholds (List[float], optional): List of thresholds to define tiers.
                                           Defaults to [0.05, 0.15, 0.30].
    Returns:
        np.ndarray: Array of risk tier assignments.
    """
    if thresholds is None:
        thresholds = [0.05, 0.15, 0.30]
    return np.digitize(probs, thresholds)


def generate_data(
    n_samples: int = 2000,
    n_features: int = 20,
    n_informative: int = 10,
    n_redundant: int = 5,
    n_classes: int = 2,
    weights: List[float] = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Generates a realistic synthetic dataset for credit default prediction and splits it.

    Creates features with realistic distributions and relationships that mirror actual credit data:
    - FICO scores follow normal distribution (300-850 range)
    - Debt-to-income ratios with realistic constraints
    - Log-normal income distribution
    - Correlated loan amounts and credit metrics
    - Default probability driven by creditworthiness factors

    Args:
        n_samples (int): Number of samples (loan applications).
        n_features (int): Total number of features (must be >= 15 for named features).
        n_informative (int): Unused (kept for backward compatibility).
        n_redundant (int): Unused (kept for backward compatibility).
        n_classes (int): Number of classes (binary classification only).
        weights (List[float], optional): Target default rate [no_default, default]. Defaults to [0.9, 0.1].
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        X, y, X_train, X_test, y_train, y_test dataframes/series, and feature names.
    """
    if weights is None:
        weights = [0.9, 0.1]  # Simulate class imbalance

    np.random.seed(random_state)

    # Generate realistic credit-related features
    # 1. FICO Score: Normal distribution, mean=690, std=70, clipped to [300, 850]
    fico_score = np.clip(np.random.normal(690, 70, n_samples), 300, 850)

    # 2. Debt-to-Income Ratio: Beta distribution scaled to [5%, 60%]
    dti = np.random.beta(2, 5, n_samples) * 55 + 5  # Skewed toward lower DTI

    # 3. Annual Income: Log-normal distribution, median ~$60k
    income = np.random.lognormal(np.log(60000), 0.6, n_samples)
    income = np.clip(income, 15000, 500000)  # Realistic bounds

    # 4. Loan Amount: Correlated with income (typically 0.5-4x annual income)
    loan_to_income_ratio = np.random.uniform(0.5, 4.0, n_samples)
    loan_amount = income * loan_to_income_ratio + \
        np.random.normal(0, 5000, n_samples)
    loan_amount = np.clip(loan_amount, 1000, 500000)

    # 5. Employment Length: Exponential-like distribution (years)
    employment_length = np.clip(np.random.exponential(5, n_samples), 0, 40)

    # 6. Credit Utilization Ratio: Percentage of credit used
    credit_utilization = np.random.beta(2, 3, n_samples) * 100  # 0-100%

    # 7. Number of Open Credit Lines
    num_open_accounts = np.random.poisson(8, n_samples) + 2
    num_open_accounts = np.clip(num_open_accounts, 1, 30)

    # 8. Number of Delinquencies in Last 2 Years
    # Correlated with FICO (lower FICO = more delinquencies)
    delinq_prob = 1 / (1 + np.exp((fico_score - 650) / 50))
    num_delinquencies = np.random.binomial(3, delinq_prob)

    # 9. Total Credit Limit
    total_credit_limit = income * np.random.uniform(0.5, 3.0, n_samples)
    total_credit_limit = np.clip(total_credit_limit, 5000, 1000000)

    # 10. Months Since Last Delinquency (-1 if never, else 0-84 months)
    months_since_delinq = np.where(
        num_delinquencies > 0,
        np.random.exponential(24, n_samples).clip(0, 84),
        -1  # Never had delinquency
    )

    # 11. Revolving Balance
    revolving_balance = (credit_utilization / 100) * total_credit_limit

    # 12. Number of Mortgage Accounts
    num_mortgage_accounts = np.random.binomial(2, 0.35, n_samples)

    # 13. Inquiries in Last 6 Months (hard credit pulls)
    inquiries_last_6m = np.random.poisson(1.5, n_samples)
    inquiries_last_6m = np.clip(inquiries_last_6m, 0, 10)

    # 14. Age of Oldest Credit Line (months)
    oldest_credit_line_age = np.random.gamma(3, 40, n_samples)
    oldest_credit_line_age = np.clip(oldest_credit_line_age, 6, 600)

    # 15. Total Balance on Accounts
    total_balance = loan_amount * 0.8 + revolving_balance + \
        np.random.normal(0, 10000, n_samples)
    total_balance = np.clip(total_balance, 0, 1000000)

    # Build feature matrix
    feature_data = {
        'fico_score': fico_score,
        'dti': dti,
        'income': income,
        'loan_amount': loan_amount,
        'employment_length': employment_length,
        'credit_utilization': credit_utilization,
        'num_open_accounts': num_open_accounts,
        'num_delinquencies': num_delinquencies,
        'total_credit_limit': total_credit_limit,
        'months_since_delinq': months_since_delinq,
        'revolving_balance': revolving_balance,
        'num_mortgage_accounts': num_mortgage_accounts,
        'inquiries_last_6m': inquiries_last_6m,
        'oldest_credit_line_age': oldest_credit_line_age,
        'total_balance': total_balance,
    }

    # Add generic features if n_features > 15
    if n_features > 15:
        for i in range(n_features - 15):
            feature_data[f'feature_{i}'] = np.random.randn(n_samples)

    feature_names = list(feature_data.keys())
    X = pd.DataFrame(feature_data)

    # Generate realistic default target based on creditworthiness
    # Logistic formula: P(default) = 1 / (1 + exp(-z))
    # where z is a linear combination of features

    # Normalize key features for logistic calculation
    fico_norm = (fico_score - 690) / 70
    dti_norm = (dti - 30) / 15
    income_norm = (np.log(income) - np.log(60000)) / 0.6
    loan_to_income = loan_amount / income
    util_norm = (credit_utilization - 50) / 25

    # Calculate default probability (z-score)
    z = (
        # Base log-odds (low default rate)
        -2.0
        - 1.5 * fico_norm                       # FICO: strong negative effect
        + 0.8 * dti_norm                        # DTI: positive effect
        - 0.3 * income_norm                     # Income: weak negative effect
        # Loan-to-income: positive effect
        + 0.6 * np.log(loan_to_income)
        + 0.5 * util_norm                       # Credit utilization: positive effect
        + 0.4 * num_delinquencies               # Delinquencies: strong positive effect
        + 0.3 * inquiries_last_6m / 5           # Recent inquiries: positive effect
        # Credit history: negative effect
        - 0.2 * np.log(oldest_credit_line_age / 12 + 1)
        + np.random.normal(0, 0.5, n_samples)   # Random noise
    )

    # Convert to probability
    default_prob = 1 / (1 + np.exp(-z))

    # Adjust to match target default rate
    target_default_rate = weights[1]
    current_mean = default_prob.mean()
    # Shift probabilities to match target rate
    adjustment = np.log(target_default_rate / (1 - target_default_rate)
                        ) - np.log(current_mean / (1 - current_mean))
    default_prob = 1 / (1 + np.exp(-(z + adjustment)))

    # Generate binary outcomes
    y_raw = (np.random.random(n_samples) < default_prob).astype(int)
    y = pd.Series(y_raw, name='default')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"Dataset shape: {X.shape}")
    print(f"Default rate in training set: {y_train.mean():.2%}")
    print(f"Default rate in test set: {y_test.mean():.2%}")
    print(
        f"Mean FICO score: {fico_score.mean():.0f} (std: {fico_score.std():.0f})")
    print(f"Mean DTI: {dti.mean():.1f}% (std: {dti.std():.1f}%)")
    print(f"Median Income: ${np.median(income):,.0f}")

    return X, y, X_train, X_test, y_train, y_test, feature_names


def simulate_black_box_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
    n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.1,
    random_state: int = 42
) -> Tuple[xgb.XGBClassifier, np.ndarray]:
    """
    Simulates the developer's black-box model (XGBoost) by training it
    and generating predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum tree depth.
        learning_rate (float): Step size shrinkage to prevent overfitting.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[xgb.XGBClassifier, np.ndarray]:
        The trained XGBoost model and its predicted probabilities on the test set.
    """
    mock_black_box_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,  # Suppress future warning
        eval_metric='logloss',
        random_state=random_state,
        scale_pos_weight=(y_train == 0).sum() / (y_train ==
                                                 # Addresses class imbalance
                                                 1).sum()
    )
    mock_black_box_model.fit(X_train.to_numpy(), y_train.to_numpy())
    primary_probs = mock_black_box_model.predict_proba(X_test.to_numpy())[:, 1]
    return mock_black_box_model, primary_probs


def load_model_metadata(feature_names: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads or defines the model documentation and business context.
    In a real application, this might involve loading from configuration files
    (e.g., JSON, YAML).

    Args:
        feature_names (List[str]): List of feature names from the generated data.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
        A dictionary containing model documentation and another for business context.
    """
    model_documentation = {
        'model_purpose': 'Predict probability of loan default for underwriting',
        'target_variable': 'Binary default (0/1) within 12 months',
        'training_data': {
            'source': 'LendingClub historical loans',
            'period': '2012-2019',
            'n_samples': '100,000 loans',
            'default_rate': '5.2%',
            'class_balance_method': 'scale_pos_weight in XGBoost',
        },
        'features': {
            'count': len(feature_names),
            'list': feature_names,
            'missing_value_treatment': 'Median imputation',
            'feature_engineering': 'None (raw features only)',
        },
        'algorithm': {
            'type': 'XGBoost Gradient Boosted Trees',
            'hyperparameters': 'n_estimators=200, max_depth=4, lr=0.1',
            'tuning_method': '5-fold CV with AUC optimization',
        },
        'claimed_performance': {
            'auc': 0.82,
            'f1_default_class': 0.42,
            'test_set': '20% holdout, stratified',
        },
        'limitations_documented': [
            'Not validated for negative interest rate environments',
            'Training period does not include a recession',
            'Limited testing on minority groups'
        ],
    }

    business_context = {
        'portfolio_size_millions': 500,
        'shap_available': True,
        'regulatory_impact_critical': True
    }
    return model_documentation, business_context


def review_documentation(checklist: Dict[str, Any]) -> List[str]:
    """
    Reviews model documentation for completeness and red flags based on best practices.

    Args:
        checklist (Dict[str, Any]): Dictionary containing model documentation.

    Returns:
        List[str]: A list of issues and warnings found during the review.
    """
    issues = []

    required_sections = [
        'model_purpose', 'target_variable', 'training_data',
        'features', 'algorithm', 'claimed_performance',
        'limitations_documented'
    ]
    for section in required_sections:
        if section not in checklist:
            issues.append(f"MISSING: '{section}' not documented.")

    train_data_period = checklist.get('training_data', {}).get('period', '')
    # This check is simplified for synthetic data; in reality, a more sophisticated date parser would be used.
    if ('2020' not in train_data_period and 'recession' not in train_data_period.lower()) and \
       not any('recession' in lim.lower() for lim in checklist.get('limitations_documented', [])):
        issues.append(
            "WARNING: Training period does not cover a recession, and this is not explicitly documented as a limitation.")

    default_rate_str = checklist.get('training_data', {}).get(
        'default_rate', '0').replace('%', '')
    try:
        default_rate = float(default_rate_str)
        if default_rate < 3.0:  # Threshold for concern, e.g., < 3%
            issues.append(
                f"WARNING: Very low default rate ({default_rate}%) may cause class imbalance issues or data quality concerns.")
    except ValueError:
        issues.append(
            "WARNING: Default rate in documentation is not a valid number.")

    feature_count = checklist.get('features', {}).get('count', 0)
    if feature_count > 50:  # Arbitrary threshold for "too many features"
        issues.append(
            f"WARNING: High dimensionality ({feature_count} features) may increase overfitting risk or reduce interpretability.")

    num_limitations = len(checklist.get('limitations_documented', []))
    if num_limitations < 2:
        issues.append(
            f"WARNING: Only {num_limitations} limitations documented. Developer may be overconfident or analysis is incomplete.")

    print("\n--- DOCUMENTATION REVIEW ---")
    print("=" * 55)
    print(
        f"Sections present: {sum(1 for s in required_sections if s in checklist)}/{len(required_sections)}")
    print(f"Issues found: {len(issues)}")
    for issue in issues:
        print(f" - {issue}")
    print("-" * 55)

    return issues


def reproduce_results(y_test: pd.Series, primary_probs: np.ndarray, claimed_metrics: Dict[str, float], threshold: float = 0.5) -> Tuple[Dict[str, float], bool]:
    """
    Independently reproduces claimed performance metrics by re-calculating them
    on the validator's test set using the primary model's predictions.

    Args:
        y_test (pd.Series): True labels of the test set.
        primary_probs (np.ndarray): Predicted probabilities from the primary model.
        claimed_metrics (Dict[str, float]): Dictionary of metrics claimed by the developer.
        threshold (float): Classification threshold for converting probabilities to binary predictions.

    Returns:
        Tuple[Dict[str, float], bool]:
        A dictionary of reproduced metrics and a boolean indicating if all metrics matched.
    """
    y_pred = (primary_probs > threshold).astype(int)

    reproduced_metrics = {
        'auc': {
            'claimed_value': claimed_metrics.get('auc', 'N/A'),
            'reproduced_value': roc_auc_score(y_test, primary_probs),
            'match': None
        },
        'f1_default_class': {
            'claimed_value': claimed_metrics.get('f1_default_class', 'N/A'),
            'reproduced_value': f1_score(y_test, y_pred, pos_label=1),
            'match': None
        }
    }

    print("\n--- RESULT REPRODUCTION ---")
    print("=" * 55)
    print(f"{'Metric':<25s} {'Claimed':>10s} {'Reproduced':>10s} {'Match':>8s}")
    print("-" * 55)

    all_match = True
    for metric, claimed_val in claimed_metrics.items():
        # Skip non-numeric values (like 'test_set' metadata)
        if not isinstance(claimed_val, (int, float)):
            continue

        if metric in reproduced_metrics:
            repro_val = reproduced_metrics[metric]['reproduced_value']
            diff = abs(repro_val - claimed_val)
            match_status = 'YES' if diff < 0.01 else (
                'CLOSE' if diff < 0.03 else 'NO')
            reproduced_metrics[metric]['match'] = match_status
            print(
                f"{metric:<25s} {claimed_val:>10.4f} {repro_val:>10.4f} {match_status:>8s}")
            if match_status == 'NO':
                all_match = False
        else:
            print(f"{metric:<25s} {claimed_val:>10.4f} {'N/A':>10s} {'N/A':>8s}")
            all_match = False

    print("-" * 55)
    if not all_match:
        print("\nFLAG: Significant discrepancy between claimed and reproduced metrics. INVESTIGATE DATA PROCESSING OR TEST SET DEFINITION DIFFERENCES. Validation MUST PAUSE.")
    else:
        print("\nPASS: Claimed metrics successfully reproduced within tolerance.")

    return reproduced_metrics, all_match


def challenger_benchmark(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    primary_probs: np.ndarray, primary_auc: float, thresholds: List[float] = None
) -> Dict[str, Any]:
    """
    Builds a simpler challenger model (Logistic Regression) and compares its performance
    and risk ranking to the primary model. This helps assess if the complexity of the
    primary model is justified.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        primary_probs (np.ndarray): Predicted probabilities from the primary model on the test set.
        primary_auc (float): AUC score of the primary model.
        thresholds (List[float], optional): Thresholds for defining risk tiers.
                                           Defaults to [0.05, 0.15, 0.30].

    Returns:
        Dict[str, Any]:
        A dictionary containing challenger model metrics, comparison results,
        and the trained challenger model and scaler.
    """
    if thresholds is None:
        thresholds = [0.05, 0.15, 0.30]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    challenger_model = LogisticRegression(
        class_weight='balanced',  # Crucial for imbalanced datasets
        max_iter=1000,
        C=0.1,  # Regularization strength
        random_state=42
    )
    challenger_model.fit(X_train_scaled, y_train)
    challenger_probs = challenger_model.predict_proba(X_test_scaled)[:, 1]
    challenger_auc = roc_auc_score(y_test, challenger_probs)

    rank_corr, p_value = spearmanr(primary_probs, challenger_probs)

    primary_tiers = define_risk_tiers(primary_probs, thresholds)
    challenger_tiers = define_risk_tiers(challenger_probs, thresholds)
    tier_agreement = (primary_tiers == challenger_tiers).mean()

    auc_lift = primary_auc - challenger_auc

    print("\n--- CHALLENGER MODEL BENCHMARKING ---")
    print("=" * 55)
    print(f"Primary Model (XGBoost) AUC:   {primary_auc:.4f}")
    print(f"Challenger (Logistic Reg) AUC: {challenger_auc:.4f}")
    print(f"AUC Lift from Complexity:      {auc_lift:+.4f}")
    print(f"Rank Correlation (Spearman):   {rank_corr:.4f} (p={p_value:.2e})")
    print(f"Risk Tier Agreement:           {tier_agreement:.1%}")
    print("-" * 55)

    complexity_justified = False
    if auc_lift < 0.02:  # Negligible lift
        print("\nFLAG: AUC lift < 2% (negligible). Complexity may NOT be justified. Consider simpler model.")
        complexity_justified = False
    elif 0.02 <= auc_lift < 0.05:  # Moderate lift
        print(
            f"\nNOTE: Moderate lift ({auc_lift:.1%}). Complexity may be justified if business impact warrants it.")
        complexity_justified = True  # Tentatively true, subject to other criteria
    else:  # Substantial lift
        print(
            f"\nPASS: Substantial lift ({auc_lift:.1%}). Complexity is justified by performance gain.")
        complexity_justified = True

    if rank_corr < 0.70:
        print("\nFLAG: Low rank correlation (<0.70). Models disagree substantially on risk ranking. Investigate which model's rankings better predict defaults.")
        complexity_justified = False  # A significant concern
    elif rank_corr < 0.85:
        print("\nNOTE: Moderate rank correlation. Some disagreement on risk ordering, but overall trend similar.")
    else:
        print("\nPASS: High rank correlation. Models largely agree on risk ordering.")

    return {
        'primary_auc': primary_auc,
        'challenger_auc': challenger_auc,
        'auc_lift': auc_lift,
        'rank_corr': rank_corr,
        'tier_agreement': tier_agreement,
        'complexity_justified': complexity_justified,
        'challenger_model': challenger_model,
        'scaler': scaler  # Return the fitted scaler for consistent scaling in plots
    }


def stability_test(mock_black_box_model: xgb.XGBClassifier, X_test: pd.DataFrame, noise_level: float = 0.01, n_trials: int = 10) -> pd.DataFrame:
    """
    Tests the stability of model predictions by adding small random Gaussian noise
    to inputs and observing changes in predictions.

    Args:
        mock_black_box_model (xgb.XGBClassifier): The trained black-box model.
        X_test (pd.DataFrame): Test features.
        noise_level (float): The standard deviation of the noise as a proportion of feature standard deviation.
        n_trials (int): Number of noise injection trials.

    Returns:
        pd.DataFrame: A DataFrame containing stability metrics for each trial.
    """
    baseline_probs = mock_black_box_model.predict_proba(X_test.to_numpy())[
        :, 1]
    baseline_preds = (baseline_probs > 0.5).astype(int)

    stability_scores = []
    feature_std = X_test.std(axis=0)

    for trial in range(n_trials):
        noise = np.random.randn(*X_test.shape) * \
            (feature_std.to_numpy() * noise_level)
        X_noisy = X_test + noise
        X_noisy = X_noisy.clip(lower=X_test.min().to_numpy(), upper=X_test.max(
        ).to_numpy())  # Simple clipping to keep values within reasonable range

        noisy_probs = mock_black_box_model.predict_proba(X_noisy.to_numpy())[
            :, 1]
        noisy_preds = (noisy_probs > 0.5).astype(int)

        abs_diff = np.abs(noisy_probs - baseline_probs)

        # Handle cases where all probabilities are the same (spearmanr returns NaN)
        if len(np.unique(baseline_probs)) > 1 and len(np.unique(noisy_probs)) > 1:
            rank_change, _ = spearmanr(baseline_probs, noisy_probs)
        else:
            rank_change = 1.0 if np.all(baseline_probs == noisy_probs) else 0.0

        pct_flipped = (baseline_preds != noisy_preds).mean()

        stability_scores.append({
            'trial': trial,
            'mean_abs_diff': abs_diff.mean(),
            'max_abs_diff': abs_diff.max(),
            'pct_flipped': pct_flipped,
            'rank_correlation': rank_change,
        })

    stability_df = pd.DataFrame(stability_scores)
    mean_pct_flipped = stability_df['pct_flipped'].mean()

    print(f"\n--- PREDICTION STABILITY TEST ({n_trials} trials) ---")
    print("=" * 55)
    print(f"Noise level: {noise_level*100:.1f}% of feature std")
    print(
        f"Mean Absolute Prediction Change: {stability_df['mean_abs_diff'].mean():.4f}")
    print(
        f"Max Absolute Prediction Change:  {stability_df['max_abs_diff'].mean():.4f}")
    print(f"Average % Predictions Flipped:   {mean_pct_flipped:.2%}")
    print(
        f"Average Rank Stability (Spearman): {stability_df['rank_correlation'].mean():.4f}")
    print("-" * 55)

    if mean_pct_flipped > 0.05:  # Threshold for instability: >5% predictions flip
        print(f"\nFLAG: Average {mean_pct_flipped:.2%} predictions flip with {noise_level*100:.1f}% noise. Model is UNSTABLE near decision boundary, posing reliability risks.")
    else:
        print(
            f"\nPASS: Predictions are stable under small perturbations (average {mean_pct_flipped:.2%} flips).")

    return stability_df


def complexity_assessment(
    benchmark_results: Dict[str, Any],
    stability_results: pd.DataFrame,
    business_context: Dict[str, Any]
) -> Tuple[int, str, List[str]]:
    """
    Combines benchmarking and stability results with business context to assess
    whether the complexity of the primary model is justified.

    Args:
        benchmark_results (Dict[str, Any]): Results from the challenger model benchmarking.
        stability_results (pd.DataFrame): Results from the prediction stability test.
        business_context (Dict[str, Any]): Business context information.

    Returns:
        Tuple[int, str, List[str]]:
        An overall score, a recommendation string, and a list of detailed findings.
    """
    score = 0
    max_score = 5
    findings = []

    auc_lift_threshold_pass = 0.02

    if benchmark_results['auc_lift'] > auc_lift_threshold_pass:
        score += 1
        findings.append(
            f"PASS: AUC lift ({benchmark_results['auc_lift']:.1%}) > {auc_lift_threshold_pass:.1%} justifies complexity.")
    elif benchmark_results['auc_lift'] > 0:
        findings.append(
            f"NOTE: AUC lift ({benchmark_results['auc_lift']:.1%}) is marginal. Complexity may be justified with conditions.")
    else:
        findings.append(
            f"FAIL: AUC lift ({benchmark_results['auc_lift']:.1%}) does not justify complexity.")

    rank_corr_threshold = 0.70
    if benchmark_results['rank_corr'] > rank_corr_threshold:
        score += 1
        findings.append(
            f"PASS: High rank correlation ({benchmark_results['rank_corr']:.2f}) with simpler model.")
    else:
        findings.append(
            f"FAIL: Low rank correlation ({benchmark_results['rank_corr']:.2f}) suggests different signal than challenger.")

    avg_flip_rate = stability_results['pct_flipped'].mean()
    stability_threshold_pass = 0.05
    if avg_flip_rate < stability_threshold_pass:
        score += 1
        findings.append(
            f"PASS: Flip rate ({avg_flip_rate:.1%}) < {stability_threshold_pass:.1%} threshold. Model is stable.")
    else:
        findings.append(
            f"FAIL: Flip rate ({avg_flip_rate:.1%}) exceeds {stability_threshold_pass:.1%} threshold. Model is unstable.")

    portfolio_size_threshold = 100  # millions
    if business_context.get('portfolio_size_millions', 0) > portfolio_size_threshold:
        score += 1
        findings.append(
            "PASS: Large portfolio justifies potential precision gains from complex model.")
    else:
        findings.append(
            "NOTE: Small portfolio; marginal gains may not matter for complexity justification.")

    if business_context.get('shap_available', False):
        score += 1
        findings.append(
            "PASS: SHAP explanations available for transparency, mitigating black-box risk.")
    else:
        findings.append(
            "FAIL: No explainability tools available. Opacity is a risk for a complex model.")

    recommendation = ""
    if score >= 4:
        recommendation = "COMPLEXITY JUSTIFIED"
    elif score >= 2:
        recommendation = "CONDITIONALLY JUSTIFIED (address failures)"
    else:
        recommendation = "COMPLEXITY NOT JUSTIFIED - use simpler model"

    print("\n--- COMPLEXITY JUSTIFICATION ASSESSMENT ---")
    print("=" * 55)
    print(f"Overall Score: {score}/{max_score}")
    print("\nFindings:")
    for f in findings:
        print(f" - {f}")
    print(f"\nRECOMMENDATION: {recommendation}")
    print("-" * 55)

    return score, recommendation, findings


def compile_validation_report(
    doc_issues: List[str],
    repro_pass: bool,
    benchmark_results: Dict[str, Any],
    stability_results: pd.DataFrame,
    cx_rec: str,
    cx_findings: List[str],
    model_documentation: Dict[str, Any],
    business_context: Dict[str, Any],
    cx_score: int  # Added cx_score to signature
) -> Dict[str, Any]:
    """
    Compiles and prints a formal model validation report with an overall recommendation.

    Args:
        doc_issues (List[str]): Issues found during documentation review.
        repro_pass (bool): Whether claimed metrics were reproduced successfully.
        benchmark_results (Dict[str, Any]): Results from the challenger model benchmarking.
        stability_results (pd.DataFrame): Results from the prediction stability test.
        cx_rec (str): Complexity justification recommendation.
        cx_findings (List[str]): Detailed findings from complexity assessment.
        model_documentation (Dict[str, Any]): The full model documentation.
        business_context (Dict[str, Any]): The business context.
        cx_score (int): The numerical score from the complexity assessment.

    Returns:
        Dict[str, Any]: A dictionary representing the complete validation report.
    """
    critical_failures = 0
    conditions_for_approval = []

    # --- Assess Critical Failures ---
    if not repro_pass:
        critical_failures += 1
        conditions_for_approval.append(
            "CRITICAL: Developer must explain and resolve discrepancies in claimed vs. reproduced metrics.")

    if stability_results['pct_flipped'].mean() > 0.05:
        critical_failures += 1
        conditions_for_approval.append(
            "CRITICAL: Address model instability to small input perturbations (prediction flip rate > 5%).")

    if cx_rec == "COMPLEXITY NOT JUSTIFIED" and business_context.get('regulatory_impact_critical', False):
        critical_failures += 1
        conditions_for_approval.append(
            "CRITICAL: Re-evaluate model choice as complexity is NOT justified for a critical model. Consider simpler alternatives or significant improvements.")

    if any("MISSING" in issue for issue in doc_issues):
        critical_failures += 1
        conditions_for_approval.append(
            "CRITICAL: Address missing required sections in model documentation.")

    # --- Determine Overall Recommendation ---
    overall_recommendation = ""
    # Allowing one minor doc issue if all else good
    if critical_failures == 0 and len(doc_issues) <= 1 and cx_rec == "COMPLEXITY JUSTIFIED":
        overall_recommendation = 'APPROVED'
    elif critical_failures == 0 and len(doc_issues) > 0:
        overall_recommendation = 'CONDITIONALLY APPROVED'
    elif critical_failures == 0 and cx_rec == "CONDITIONALLY JUSTIFIED (address failures)":
        overall_recommendation = 'CONDITIONALLY APPROVED'
    else:
        overall_recommendation = 'REJECTED'

    # --- Formulate Conditions for Approval (non-critical issues) ---
    if overall_recommendation in ['CONDITIONALLY APPROVED', 'APPROVED']:
        # Add conditions based on documentation issues
        for issue in doc_issues:
            if "MISSING" not in issue and "CRITICAL" not in issue and "WARNING" in issue:  # Non-critical warnings
                conditions_for_approval.append(
                    f"Developer must address documentation issue: '{issue.replace('WARNING: ', '')}'.")

        # Add conditions for complexity justification if not fully justified
        if cx_rec == "CONDITIONALLY JUSTIFIED (address failures)":
            conditions_for_approval.append(
                "Developer must demonstrate why complexity is justified, e.g., through further analysis of incremental value or enhanced explainability.")
        if not business_context.get('shap_available', False):
            conditions_for_approval.append(
                "Implement SHAP explanations for all decline decisions to improve model transparency.")

        # General best practices for monitoring
        conditions_for_approval.append(
            "Deploy with monthly AUC monitoring and drift detection.")
        conditions_for_approval.append(
            "Retrain trigger: AUC drops >5% or default rate exceeds 8%.")
        conditions_for_approval.append("Annual revalidation required.")

    # --- Construct the Report Dictionary ---
    report = {
        'model_name': 'XGBoost Credit Default v1.0',
        'validator': 'Model Risk Management Team',
        'validation_date': datetime.datetime.now().isoformat(),
        'overall_recommendation': overall_recommendation,
        'critical_failures_count': critical_failures,

        'section_1_documentation': {
            'status': 'PASS' if len(doc_issues) <= 1 else ('CONDITIONAL' if critical_failures == 0 else 'FAIL'),
            'issues': doc_issues,
        },
        'section_2_reproduction': {
            'status': 'PASS' if repro_pass else 'FAIL',
            'note': 'All claimed metrics reproduced within tolerance' if repro_pass else 'DISCREPANCY DETECTED',
        },
        'section_3_challenger': {
            'status': 'PASS' if benchmark_results['complexity_justified'] else 'CONDITIONAL',
            'auc_lift': benchmark_results['auc_lift'],
            'rank_correlation': benchmark_results['rank_corr'],
            'tier_agreement': benchmark_results['tier_agreement'],
        },
        'section_4_stability': {
            'status': 'PASS' if stability_results['pct_flipped'].mean() < 0.05 else 'FAIL',
            'avg_flip_rate': stability_results['pct_flipped'].mean(),
            'rank_correlation_avg': stability_results['rank_correlation'].mean(),
        },
        'section_5_complexity': {
            'status': cx_rec,
            'score': f"{cx_score}/5",  # Use cx_score directly
            'findings': cx_findings
        },
        'conditions_for_approval': conditions_for_approval if overall_recommendation in ['CONDITIONALLY APPROVED', 'APPROVED'] else [],
        'sign_off': {
            'Validator Lead': 'Jane Doe, CFA - Feb 2026',
            'MRM Director': 'John Smith - Feb 2026',
            'Risk Committee': 'Prudent Financial Risk Committee - March 2026',
        }
    }

    # --- Print the Formal Report ---
    print("\n" + "=" * 60)
    print("INDEPENDENT MODEL VALIDATION REPORT".center(60))
    print("=" * 60)
    print(f"Model: {report['model_name']}")
    print(f"Validator: {report['validator']}")
    print(
        f"Date: {datetime.datetime.strptime(report['validation_date'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOVERALL RECOMMENDATION: {report['overall_recommendation']}\n")

    print(
        f"SECTION 1: DOCUMENTATION REVIEW: {report['section_1_documentation']['status']}")
    for issue in report['section_1_documentation']['issues']:
        print(f" - {issue}")

    print(
        f"\nSECTION 2: INDEPENDENT REPRODUCTION: {report['section_2_reproduction']['status']}")
    print(f" - Note: {report['section_2_reproduction']['note']}")

    print(
        f"\nSECTION 3: CHALLENGER MODEL BENCHMARKING: {report['section_3_challenger']['status']}")
    print(f" - AUC Lift: {report['section_3_challenger']['auc_lift']:.4f}")
    print(
        f" - Rank Correlation: {report['section_3_challenger']['rank_correlation']:.4f}")
    print(
        f" - Tier Agreement: {report['section_3_challenger']['tier_agreement']:.1%}")

    print(
        f"\nSECTION 4: PREDICTION STABILITY TESTING: {report['section_4_stability']['status']}")
    print(
        f" - Average Prediction Flip Rate: {report['section_4_stability']['avg_flip_rate']:.2%}")
    print(
        f" - Average Rank Correlation (Noisy vs Baseline): {report['section_4_stability']['rank_correlation_avg']:.4f}")

    print(
        f"\nSECTION 5: COMPLEXITY JUSTIFICATION: {report['section_5_complexity']['status']}")
    print(f" - Score: {report['section_5_complexity']['score']}")
    for finding in report['section_5_complexity']['findings']:
        print(f" - {finding}")

    if report['conditions_for_approval']:
        print("\nCONDITIONS FOR APPROVAL:")
        for cond in report['conditions_for_approval']:
            print(f" - {cond}")

    print("\nSIGN-OFFS:")
    for role, signature in report['sign_off'].items():
        print(f" {role}: {signature}")

    print("\n" + "=" * 60)

    return report


def generate_validation_plots(
    y_test: pd.Series,
    primary_probs: np.ndarray,
    benchmark_results: Dict[str, Any],
    reproduced_metrics: Dict[str, float],
    stability_results: pd.DataFrame,
    cx_rec: str,
    cx_findings: List[str],
    X_test: pd.DataFrame,
    risk_tier_thresholds: List[float] = None
) -> None:
    """
    Generates and displays (or returns) the various validation plots.

    Args:
        y_test (pd.Series): True labels of the test set.
        primary_probs (np.ndarray): Predicted probabilities from the primary model.
        benchmark_results (Dict[str, Any]): Results from the challenger model benchmarking.
        reproduced_metrics (Dict[str, float]): Metrics reproduced by the validator.
        stability_results (pd.DataFrame): Results from the prediction stability test.
        cx_rec (str): Complexity justification recommendation string.
        cx_findings (List[str]): Detailed findings from complexity assessment.
        X_test (pd.DataFrame): Test features (needed for challenger model prediction after scaling).
        risk_tier_thresholds (List[float], optional): Thresholds for defining risk tiers.
                                                      Defaults to [0.05, 0.15, 0.30].
    """
    if risk_tier_thresholds is None:
        risk_tier_thresholds = [0.05, 0.15, 0.30]

    # 1. ROC Curves Overlay
    plt.figure(figsize=(8, 6))
    fpr_primary, tpr_primary, _ = roc_curve(y_test, primary_probs)
    # Use the scaler returned from challenger_benchmark
    challenger_probs = benchmark_results['challenger_model'].predict_proba(
        benchmark_results['scaler'].transform(X_test)
    )[:, 1]
    fpr_challenger, tpr_challenger, _ = roc_curve(y_test, challenger_probs)

    plt.plot(fpr_primary, tpr_primary,
             label=f'Primary Model (AUC = {reproduced_metrics["auc"]:.3f})', color='darkorange')
    plt.plot(fpr_challenger, tpr_challenger,
             label=f'Challenger Model (AUC = {benchmark_results["challenger_auc"]:.3f})', color='navy')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: Primary vs. Challenger Model')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Prediction Rank Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(primary_probs, challenger_probs, alpha=0.5, s=10)
    plt.xlabel('Primary Model Probabilities')
    plt.ylabel('Challenger Model Probabilities')
    plt.title('Prediction Rank Scatter Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Risk Tier Confusion Matrix
    primary_tiers = define_risk_tiers(primary_probs, risk_tier_thresholds)
    challenger_tiers = define_risk_tiers(
        challenger_probs, risk_tier_thresholds)
    tier_cm = confusion_matrix(primary_tiers, challenger_tiers)

    plt.figure(figsize=(7, 6))
    sns.heatmap(tier_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Tier {i}' for i in range(
                    len(risk_tier_thresholds) + 1)],
                yticklabels=[f'Tier {i}' for i in range(len(risk_tier_thresholds) + 1)])
    plt.xlabel('Challenger Model Risk Tiers')
    plt.ylabel('Primary Model Risk Tiers')
    plt.title('Risk Tier Agreement (Confusion Matrix)')
    plt.tight_layout()
    plt.show()

    # 4. Prediction Stability Box Plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=stability_results['mean_abs_diff'], palette='viridis')
    plt.ylabel('Mean Absolute Prediction Change Across Trials')
    plt.title('Distribution of Model Prediction Changes Under Noise')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5. Complexity Justification Scorecard
    criteria_labels = [
        "AUC Lift > 2%",
        "Rank Correlation > 0.7",
        "Stability (Flip Rate < 5%)",
        "Business Impact > $100M",
        "Explainability Tools (SHAP) Available"
    ]
    scorecard_status = []
    for finding in cx_findings:
        if "PASS:" in finding:
            scorecard_status.append("PASS")
        elif "NOTE:" in finding:
            scorecard_status.append("CONDITIONAL")
        elif "FAIL:" in finding:
            scorecard_status.append("FAIL")
        else:
            scorecard_status.append("N/A")

    plt.figure(figsize=(10, 6))

    # Map status strings to integers for coloring: FAIL=0, CONDITIONAL=1, PASS=2
    status_mapping_for_heatmap = {'FAIL': 0,
                                  'CONDITIONAL': 1, 'PASS': 2, 'N/A': 0}
    numeric_scores_for_heatmap = [
        status_mapping_for_heatmap.get(s, 0) for s in scorecard_status]

    # Create a DataFrame suitable for heatmap (1 row, N columns)
    df_for_heatmap = pd.DataFrame(
        [numeric_scores_for_heatmap], columns=criteria_labels)

    # Use a diverging color map (e.g., RdYlGn) for 3 states
    cmap_custom = plt.cm.get_cmap('RdYlGn', 3)

    ax = sns.heatmap(df_for_heatmap, annot=True, fmt='s',  # Use 's' to allow custom text in loop
                     cmap=cmap_custom, cbar=False, linewidths=0.5, linecolor='black',
                     yticklabels=False, xticklabels=df_for_heatmap.columns,
                     annot_kws={"va": "center", "fontsize": 10})

    # Iterate over the annotation texts and set them to the actual status strings
    for i, text in enumerate(ax.texts):
        text.set_text(scorecard_status[i])

    # Adjust x-axis tick labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.title(f'Complexity Justification Scorecard (Overall: {cx_rec})')
    plt.tight_layout()
    plt.show()


def run_model_validation(
    n_samples: int = 2000,
    n_features: int = 20,
    n_informative: int = 10,
    n_redundant: int = 5,
    test_size: float = 0.3,
    random_state: int = 42,
    black_box_n_estimators: int = 200,
    black_box_max_depth: int = 4,
    black_box_learning_rate: float = 0.1,
    noise_level: float = 0.01,
    n_stability_trials: int = 20,
    risk_tier_thresholds: List[float] = None,  # For challenger and plotting
    show_plots: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the entire model validation pipeline, from data generation to
    report compilation and optional plotting.

    Args:
        n_samples (int): Number of samples for synthetic data generation.
        n_features (int): Number of features for synthetic data generation.
        n_informative (int): Number of informative features.
        n_redundant (int): Number of redundant features.
        test_size (float): Proportion of data for the test set.
        random_state (int): Seed for reproducibility.
        black_box_n_estimators (int): n_estimators for the mock black-box XGBoost model.
        black_box_max_depth (int): max_depth for the mock black-box XGBoost model.
        black_box_learning_rate (float): learning_rate for the mock black-box XGBoost model.
        noise_level (float): Level of noise for stability testing.
        n_stability_trials (int): Number of trials for stability testing.
        risk_tier_thresholds (List[float], optional): Thresholds for defining risk tiers.
                                                      Defaults to [0.05, 0.15, 0.30].
        show_plots (bool): If True, generates and displays validation plots.

    Returns:
        Dict[str, Any]: The complete final validation report dictionary.
    """
    if risk_tier_thresholds is None:
        risk_tier_thresholds = [0.05, 0.15, 0.30]

    # 1. Data Generation
    X, y, X_train, X_test, y_train, y_test, feature_names = generate_data(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, test_size=test_size, random_state=random_state
    )

    # 2. Simulate Black-Box Model
    mock_black_box_model, primary_probs = simulate_black_box_model(
        X_train, y_train, X_test,
        n_estimators=black_box_n_estimators, max_depth=black_box_max_depth,
        learning_rate=black_box_learning_rate, random_state=random_state
    )

    # 3. Define Documentation and Business Context
    model_documentation, business_context = load_model_metadata(feature_names)

    # 4. Review Documentation
    doc_issues = review_documentation(model_documentation)

    # 5. Reproduce Results
    claimed_performance = model_documentation['claimed_performance']
    reproduced_metrics, repro_pass = reproduce_results(
        y_test, primary_probs, claimed_performance)

    # 6. Challenger Model Benchmarking
    benchmark_results = challenger_benchmark(
        X_train, y_train, X_test, y_test, primary_probs, reproduced_metrics[
            'auc']['reproduced_value'],
        thresholds=risk_tier_thresholds
    )

    # 7. Stability Test
    stability_results = stability_test(
        mock_black_box_model, X_test, noise_level=noise_level, n_trials=n_stability_trials
    )

    # 8. Complexity Assessment
    cx_score, cx_rec, cx_findings = complexity_assessment(
        benchmark_results, stability_results, business_context)

    # 9. Compile Final Validation Report
    final_report = compile_validation_report(
        doc_issues, repro_pass, benchmark_results, stability_results,
        cx_rec, cx_findings, model_documentation, business_context, cx_score
    )

    # 10. Generate Plots (optional)
    if show_plots:
        generate_validation_plots(
            y_test, primary_probs, benchmark_results, reproduced_metrics,
            stability_results, cx_rec, cx_findings, X_test, risk_tier_thresholds
        )

    return final_report


# ============================================================================
# Wrapper functions for Streamlit app compatibility
# ============================================================================

def generate_synthetic_credit_data(n_samples=2000, n_features=20, test_size=0.3, random_state=42):
    """
    Wrapper for generate_data() that returns only the split datasets.
    Compatible with the Streamlit app.py interface.
    """
    X, y, X_train, X_test, y_train, y_test, feature_names = generate_data(
        n_samples=n_samples,
        n_features=n_features,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, y_train, X_test, y_test


def train_primary_model_black_box(X_train, y_train, X_test, n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42):
    """
    Wrapper for simulate_black_box_model() that trains and returns the model and predictions.
    Compatible with the Streamlit app.py interface.
    """
    model, probs = simulate_black_box_model(
        X_train, y_train, X_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    return model, probs


def create_challenger_model(X_train, y_train, X_test, random_state=42):
    """
    Creates a simple challenger model (Logistic Regression) for benchmarking.
    Compatible with the Streamlit app.py interface.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    challenger_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=0.1,
        random_state=random_state
    )
    challenger_model.fit(X_train_scaled, y_train)
    challenger_probs = challenger_model.predict_proba(X_test_scaled)[:, 1]
    
    return challenger_model, challenger_probs


def evaluate_model_performance(y_test, probs, threshold=0.5):
    """
    Evaluates model performance using AUC and F1 score.
    Compatible with the Streamlit app.py interface.
    
    Args:
        y_test: True labels
        probs: Predicted probabilities
        threshold: Decision threshold for binary classification
        
    Returns:
        Tuple of (auc, f1_score)
    """
    auc = roc_auc_score(y_test, probs)
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_test, preds)
    return auc, f1


def compute_spearman_rank_correlation(probs1, probs2):
    """
    Computes Spearman rank correlation between two sets of predictions.
    Compatible with the Streamlit app.py interface.
    """
    corr, _ = spearmanr(probs1, probs2)
    return corr


def run_stability_test(X_test, y_test, base_probs, model, noise_level=0.01, n_trials=50, threshold=0.5):
    """
    Wrapper for stability_test() that includes additional metrics for the Streamlit app.
    Compatible with the Streamlit app.py interface.
    
    Args:
        X_test: Test features DataFrame
        y_test: Test labels
        base_probs: Baseline probabilities
        model: Trained model
        noise_level: Standard deviation of noise as proportion of feature std
        n_trials: Number of noise injection trials
        threshold: Decision threshold for binary classification
        
    Returns:
        Dict with stability metrics including trial_metrics and flip_rate
    """
    # Run the stability test
    stability_df = stability_test(model, X_test, noise_level=noise_level, n_trials=n_trials)
    
    # Calculate flip rate using the threshold
    base_preds = (base_probs >= threshold).astype(int)
    
    flip_rates = []
    feature_std = X_test.std(axis=0)
    
    for trial in range(n_trials):
        noise = np.random.randn(*X_test.shape) * (feature_std.to_numpy() * noise_level)
        X_noisy = X_test + noise
        X_noisy = X_noisy.clip(lower=X_test.min().to_numpy(), upper=X_test.max().to_numpy())
        
        noisy_probs = model.predict_proba(X_noisy.to_numpy())[:, 1]
        noisy_preds = (noisy_probs >= threshold).astype(int)
        
        flip_rate = (base_preds != noisy_preds).mean()
        flip_rates.append(flip_rate)
    
    avg_flip_rate = np.mean(flip_rates)
    
    return {
        'trial_metrics': stability_df,
        'flip_rate': avg_flip_rate
    }


if __name__ == "__main__":
    print("Running full model validation pipeline demonstration...")
    # Example usage:
    # This will run the entire validation process and print the report
    # and show all plots if show_plots is True.
    final_validation_report_output = run_model_validation(show_plots=True)
    print("\nValidation pipeline completed. Final report returned.")
    # You can further process final_validation_report_output here,
    # e.g., save it to a JSON file.
    # import json
    # with open('validation_report.json', 'w') as f:
    #     json.dump(final_validation_report_output, f, indent=4)
