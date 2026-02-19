import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NOTE: This lab intentionally keeps "business logic" in source.py
# (data generation, evaluation, stability simulation, etc.)
from source import (
    generate_synthetic_credit_data,
    train_primary_model_black_box,
    create_challenger_model,
    evaluate_model_performance,
    compute_spearman_rank_correlation,
    define_risk_tiers,
    run_stability_test,
)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="QuLab: Lab 38: Model Validation Exercise", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()

st.title("QuLab: Lab 38: Model Validation Exercise")
st.caption("Audience: CFA charterholders, portfolio managers, risk analysts • Goal: decision-useful model validation, not coding.")
st.divider()

# -----------------------------
# Session state
# -----------------------------


def _init_state():
    defaults = {
        "setup_complete": False,
        "nav": "Setup & Validation Package",
        "X_train": None,
        "y_train": None,
        "X_test": None,
        "y_test": None,
        "primary_model": None,
        "challenger_model": None,
        "primary_probs": None,
        "challenger_probs": None,
        "doc_issues": [],
        "repro_results": None,
        "benchmark_results": None,
        "stability_results": None,
        "complexity_assessment": None,
        "validation_report": None,
        # Policy settings (lab defaults; users can change)
        "policy_profile": "Standard (Lab)",
        # classification cutoff used for F1 etc. (explicitly shown)
        "policy_threshold": 0.50,
        "policy_auc_tol_close": 0.01,    # reproduction tolerance: close
        "policy_auc_tol_fail": 0.03,     # reproduction tolerance: fail
        "policy_f1_tol_close": 0.01,
        "policy_f1_tol_fail": 0.03,
        "policy_min_auc_lift": 0.02,     # complexity justification
        "policy_min_rank_corr": 0.70,
        "policy_min_tier_agree": 0.80,
        "policy_max_flip_rate": 0.05,
        "tier_thresholds": [0.05, 0.15, 0.30],  # risk tier boundaries
        "portfolio_size_mm": 500,        # business materiality example
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# -----------------------------
# Sidebar: Navigation + Policy
# -----------------------------
PAGES = [
    "Setup & Validation Package",
    "1) Documentation Review (Gate 1)",
    "2) Result Reproduction (Gate 2)",
    "3) Challenger Benchmarking",
    "4) Prediction Stability Testing",
    "5) Complexity Justification",
    "6) Formal Validation Report",
    "Visualizations",
]

st.sidebar.subheader("Navigate")
st.session_state.nav = st.sidebar.radio(
    "Section", PAGES, index=PAGES.index(st.session_state.nav))

with st.sidebar.expander("Validation policy settings (lab)", expanded=False):
    st.markdown(
        """
These settings exist to prevent “numbers without governance.”  
In real institutions, they come from your **MRM policy** (tolerance, materiality, cutoffs).
        """.strip()
    )

    profile = st.selectbox(
        "Policy profile",
        ["Conservative", "Standard (Lab)", "Aggressive"],
        index=["Conservative", "Standard (Lab)", "Aggressive"].index(
            st.session_state.policy_profile),
        help="Pre-fills tolerances and cutoffs. You can still override anything below."
    )
    st.session_state.policy_profile = profile

    if profile == "Conservative":
        st.session_state.policy_auc_tol_close = 0.005
        st.session_state.policy_auc_tol_fail = 0.015
        st.session_state.policy_f1_tol_close = 0.005
        st.session_state.policy_f1_tol_fail = 0.015
        st.session_state.policy_min_auc_lift = 0.03
        st.session_state.policy_min_rank_corr = 0.80
        st.session_state.policy_min_tier_agree = 0.90
        st.session_state.policy_max_flip_rate = 0.03
    elif profile == "Aggressive":
        st.session_state.policy_auc_tol_close = 0.02
        st.session_state.policy_auc_tol_fail = 0.05
        st.session_state.policy_f1_tol_close = 0.02
        st.session_state.policy_f1_tol_fail = 0.05
        st.session_state.policy_min_auc_lift = 0.01
        st.session_state.policy_min_rank_corr = 0.60
        st.session_state.policy_min_tier_agree = 0.70
        st.session_state.policy_max_flip_rate = 0.08
    else:
        # Standard (Lab)
        st.session_state.policy_auc_tol_close = 0.01
        st.session_state.policy_auc_tol_fail = 0.03
        st.session_state.policy_f1_tol_close = 0.01
        st.session_state.policy_f1_tol_fail = 0.03
        st.session_state.policy_min_auc_lift = 0.02
        st.session_state.policy_min_rank_corr = 0.70
        st.session_state.policy_min_tier_agree = 0.80
        st.session_state.policy_max_flip_rate = 0.05

    st.markdown("**Operational cutoffs used in this lab**")
    st.session_state.policy_threshold = st.slider(
        "Decision cutoff (used for F1 / confusion matrices)",
        min_value=0.05, max_value=0.95, value=float(st.session_state.policy_threshold), step=0.01,
        help="Credit decisions rarely use 0.50 by default. Choose a cutoff that matches an underwriting rule (e.g., PD>8% decline)."
    )
    st.session_state.portfolio_size_mm = st.number_input(
        "Portfolio size (USD mm) for materiality lens",
        min_value=10, max_value=5000, value=int(st.session_state.portfolio_size_mm), step=10,
        help="Used for the 'business impact' criterion in complexity justification."
    )

    st.markdown("**Reproduction tolerances**")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.policy_auc_tol_close = st.number_input("AUC: close tolerance", value=float(
            st.session_state.policy_auc_tol_close), step=0.001, format="%.3f")
        st.session_state.policy_f1_tol_close = st.number_input("F1: close tolerance", value=float(
            st.session_state.policy_f1_tol_close), step=0.001, format="%.3f")
    with c2:
        st.session_state.policy_auc_tol_fail = st.number_input("AUC: fail tolerance", value=float(
            st.session_state.policy_auc_tol_fail), step=0.001, format="%.3f")
        st.session_state.policy_f1_tol_fail = st.number_input("F1: fail tolerance", value=float(
            st.session_state.policy_f1_tol_fail), step=0.001, format="%.3f")

    st.markdown("**Complexity justification cutoffs**")
    c3, c4 = st.columns(2)
    with c3:
        st.session_state.policy_min_auc_lift = st.number_input("Minimum AUC lift required", value=float(
            st.session_state.policy_min_auc_lift), step=0.005, format="%.3f")
        st.session_state.policy_min_rank_corr = st.number_input("Minimum rank correlation (ρ)", value=float(
            st.session_state.policy_min_rank_corr), step=0.05, format="%.2f")
    with c4:
        st.session_state.policy_min_tier_agree = st.number_input("Minimum tier agreement", value=float(
            st.session_state.policy_min_tier_agree), step=0.05, format="%.2f")
        st.session_state.policy_max_flip_rate = st.number_input("Maximum flip rate allowed", value=float(
            st.session_state.policy_max_flip_rate), step=0.01, format="%.2f")

    st.markdown("**Risk tier thresholds**")
    st.session_state.tier_thresholds = st.text_input(
        "Tier boundaries (comma-separated PD thresholds)",
        value=",".join([str(x) for x in st.session_state.tier_thresholds]),
        help="Defines Tier 0/1/2/3 segmentation. These are policy choices; label tiers with actions in the report."
    )
    try:
        st.session_state.tier_thresholds = [float(
            x.strip()) for x in st.session_state.tier_thresholds.split(",") if x.strip() != ""]
    except Exception:
        st.warning(
            "Could not parse tier thresholds. Using previous valid values.")

st.sidebar.divider()

# -----------------------------
# Shared helpers (pedagogical)
# -----------------------------
SEVERITY = {
    "Training period does not include recessionary stress scenarios.": "Critical",
    "Limited testing on minority demographic subgroups.": "Critical",
    "No explicit model explainability strategy documented.": "Material",
    "Feature engineering steps lack clear governance approvals.": "Material",
}


def severity_badge(level: str) -> str:
    if level == "Fatal":
        return "🛑 Fatal"
    if level == "Critical":
        return "🔴 Critical"
    if level == "Material":
        return "🟠 Material"
    return "🟡 Minor"


def gate_banner(ok: bool, pass_text: str, fail_text: str):
    if ok:
        st.success(pass_text)
    else:
        st.error(fail_text)


def checkpoint(question: str, options: list, correct: str, explanation: str):
    st.markdown("#### Checkpoint (1 minute)")
    ans = st.radio(question, options, index=None, horizontal=False)
    if ans is None:
        st.info("Answer to unlock the explanation.")
        return
    if ans == correct:
        st.success(f"Correct — {explanation}")
    else:
        st.warning(f"Not quite — {explanation}")


# -----------------------------
# Setup & Validation Package
# -----------------------------
if st.session_state.nav == "Setup & Validation Package":
    st.header("Setup: What the Validator Receives (Data, Scores, Documentation)")
    st.markdown(
        """
**Learning goal:** treat validation as a governance workflow: you validate a **model package** (data + score file + documentation), not “cool algorithms.”  

**What this lab simulates:**  
- A credit default prediction setting with rare defaults (class imbalance).  
- A “primary” model that you treat as black-box (you validate outputs, not internals).  
- A simpler “challenger” benchmark used for effective challenge.
        """.strip()
    )

    with st.expander("Assumptions & limits (read before running)", expanded=False):
        st.info(
            """
- Data are **synthetic**. In production, you must validate representativeness, missingness, and drift vs. your portfolio.
- The “black-box” stance in this lab means: you focus on **scores + documentation**, not the training pipeline.
- All numeric cutoffs come from the sidebar **policy settings** (lab defaults).
            """.strip()
        )

    if st.button("Load validation package (data + scores + documentation)"):
        X_train, y_train, X_test, y_test = generate_synthetic_credit_data()
        primary_model, primary_probs = train_primary_model_black_box(
            X_train, y_train, X_test)
        challenger_model, challenger_probs = create_challenger_model(
            X_train, y_train, X_test)

        st.session_state.X_train, st.session_state.y_train = X_train, y_train
        st.session_state.X_test, st.session_state.y_test = X_test, y_test
        st.session_state.primary_model, st.session_state.primary_probs = primary_model, primary_probs
        st.session_state.challenger_model, st.session_state.challenger_probs = challenger_model, challenger_probs
        st.session_state.setup_complete = True

    if st.session_state.setup_complete:
        st.subheader("Package summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Train rows", st.session_state.X_train.shape[0])
        with c2:
            st.metric("Test rows", st.session_state.X_test.shape[0])
        with c3:
            st.metric("Test default rate",
                      f"{float(np.mean(st.session_state.y_test))*100:.2f}%")

        st.caption(
            "Decision translation: Low default rate means accuracy is not informative — focus on ranking, cutoffs, and cost asymmetry."
        )

        checkpoint(
            "If the package is incomplete (missing score file or documentation), what should a validator do?",
            ["Proceed using best effort", "Stop and request missing artifacts",
                "Only check AUC and move on"],
            "Stop and request missing artifacts",
            "Validation starts with package integrity. You cannot defend decisions without the artifacts that produced them."
        )
    else:
        st.warning("Load the validation package to begin.")

# -----------------------------
# 1) Documentation Review (Gate 1)
# -----------------------------
elif st.session_state.nav == "1) Documentation Review (Gate 1)":
    st.header("1) Documentation Review (Gate 1): Is the model package approve-able?")
    st.markdown(
        """
**Learning goal:** documentation is a control — it determines whether the model can be governed, audited, and defended.  
A validator asks: **Is the model sufficiently documented to be used in a regulated decision?** before asking “Is it accurate?”
        """.strip()
    )

    if not st.session_state.setup_complete:
        st.warning("Run setup first.")
    else:
        # Simulated model documentation package
        with st.expander("Model card (simulated validation package)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    "**Purpose**: Predict probability of credit default for retail applicants.")
                st.markdown("**Target**: 12-month default event (binary).")
                st.markdown(
                    "**Training period**: 2012–2019 (no recessionary stress window).")
            with c2:
                st.markdown(
                    f"**Training sample size**: {st.session_state.X_train.shape[0]:,}")
                st.markdown(
                    f"**Observed default rate (train)**: {float(np.mean(st.session_state.y_train))*100:.2f}%")
                st.markdown(
                    "**Deployment use**: underwriting + pricing tiering.")

            st.markdown("**Known limitations (as disclosed)**")
            st.markdown(
                "- Training period does not include recessionary stress scenarios.")
            st.markdown("- Limited testing on minority demographic subgroups.")
            st.markdown(
                "- No explicit model explainability strategy documented.")
            st.markdown(
                "- Feature engineering steps lack clear governance approvals.")

        st.markdown("### Findings (with severity)")
        issues = [
            "Training period does not include recessionary stress scenarios.",
            "Limited testing on minority demographic subgroups.",
            "No explicit model explainability strategy documented.",
            "Feature engineering steps lack clear governance approvals.",
        ]
        st.session_state.doc_issues = issues

        # Show as a table for audit-readiness
        df_issues = pd.DataFrame({
            "Issue": issues,
            "Severity": [severity_badge(SEVERITY.get(i, "Minor")) for i in issues],
            "Why this matters (decision relevance)": [
                "Downturn behavior drives PD calibration + capital planning risk.",
                "Incomplete fair lending evidence can block approval or require conditions.",
                "You may not be able to justify outcomes or support adverse action explanations.",
                "Weak governance evidence increases model risk and audit exposure.",
            ],
            "Evidence to clear": [
                "Add stress-period data or demonstrate robust stress performance.",
                "Subgroup performance + disparate impact review (policy-aligned).",
                "Provide explainability plan: what, when, and how it is governed.",
                "Provide approvals, lineage, and change control artifacts.",
            ]
        })
        st.dataframe(df_issues, use_container_width=True)

        gate_banner(
            ok=True,
            pass_text="Gate 1 completed: Documentation reviewed and issues categorized.",
            fail_text="Gate 1 not completed."
        )

        checkpoint(
            "A strong AUC can compensate for missing documentation in a regulated underwriting model.",
            ["True", "False"],
            "False",
            "Documentation is a governance requirement. Without it, you cannot defend decisions, monitor risk, or satisfy audit requirements."
        )

# -----------------------------
# 2) Result Reproduction (Gate 2)
# -----------------------------
elif st.session_state.nav == "2) Result Reproduction (Gate 2)":
    st.header("2) Result Reproduction (Gate 2): Do we trust the reported results?")
    st.markdown(
        """
**Learning goal:** reproduction tests **package integrity** (same data, same scores, same metric definitions).  
If reproduction fails, do **not** proceed to deeper analysis — reconcile versions first.
        """.strip()
    )

    if not st.session_state.setup_complete:
        st.warning("Run setup first.")
    else:
        with st.expander("Why the cutoff matters (plain English)", expanded=False):
            st.info(
                f"""
This lab computes F1 using a **decision cutoff** of **{st.session_state.policy_threshold:.2f}** (from sidebar).  
In credit, that cutoff should represent an underwriting rule (e.g., “PD > 8% ⇒ decline” or “PD > 5% ⇒ manual review”).
                """.strip()
            )

        st.markdown("We will calculate the Area Under the Receiver Operating Characteristic Curve (AUC) and the F1-score for the positive (default) class:")
        # --- DO NOT REMOVE: formulae from original markdown blocks ---
        st.markdown(
            r"""
$$
\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
""")
        st.markdown(r"where:")
        st.markdown(
            r"""
$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$
""")
        st.markdown(
            r"""
$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$
""")
        # --- end preserved formula blocks ---

        st.markdown("### Reproduce developer-claimed metrics")
        st.caption(
            "Decision translation: A reproduction failure is a versioning / integrity problem, not a 'model performance' debate.")

        # Simulated dev-claimed results (as in original concept)
        claimed_auc = 0.82
        claimed_f1 = 0.45

        if st.button("Run reproduction check"):
            auc_val, f1_val = evaluate_model_performance(
                st.session_state.y_test,
                st.session_state.primary_probs,
                threshold=st.session_state.policy_threshold
            )

            # Apply tolerance policy (close / fail)
            def _status(claim, actual, tol_close, tol_fail):
                diff = abs(claim - actual)
                if diff <= tol_close:
                    return "PASS"
                if diff <= tol_fail:
                    return "CLOSE (investigate rounding / dataset version)"
                return "FAIL (stop — reconcile artifacts)"

            auc_status = _status(
                claimed_auc, auc_val, st.session_state.policy_auc_tol_close, st.session_state.policy_auc_tol_fail)
            f1_status = _status(
                claimed_f1, f1_val, st.session_state.policy_f1_tol_close, st.session_state.policy_f1_tol_fail)

            repro = pd.DataFrame({
                "Metric": ["AUC", "F1"],
                "Claimed": [claimed_auc, claimed_f1],
                "Reproduced": [float(auc_val), float(f1_val)],
                "Abs Diff": [abs(claimed_auc - auc_val), abs(claimed_f1 - f1_val)],
                "Status (policy-based)": [auc_status, f1_status],
            })
            st.session_state.repro_results = repro

        if st.session_state.repro_results is not None:
            st.dataframe(st.session_state.repro_results,
                         use_container_width=True)

            any_fail = any(
                "FAIL" in s for s in st.session_state.repro_results["Status (policy-based)"].tolist())
            gate_banner(
                ok=not any_fail,
                pass_text="Gate 2 PASS: results reproduced within policy tolerance. You may proceed.",
                fail_text="Gate 2 FAIL: stop here. Request dataset/score version reconciliation before proceeding."
            )

            checkpoint(
                "If reproduction fails, what's the best next step?",
                ["Tighten the threshold until it matches", "Proceed to challenger benchmarking anyway",
                    "Reconcile versions of data, labels, and score file"],
                "Reconcile versions of data, labels, and score file",
                "Reproduction failures are usually definition/version mismatches (split, labels, score precision) and must be fixed before interpretation."
            )

# -----------------------------
# 3) Challenger Benchmarking
# -----------------------------
elif st.session_state.nav == "3) Challenger Benchmarking":
    st.header(
        "3) Challenger Benchmarking: Does the black box add value vs a simpler model?")
    st.markdown(
        """
**Learning goal:** complexity must earn its governance burden.  
You benchmark the primary model against a simpler challenger on the **same test set** and interpret the results operationally (ranking + tiers).
        """.strip()
    )

    if not st.session_state.setup_complete:
        st.warning("Run setup first.")
    else:
        st.markdown(
            "We will evaluate the Challenger Model against the Primary Model using:")
        st.markdown(
            f"*   **AUC Lift ($\\Delta AUC$)**: The difference in AUC values.")
        # --- DO NOT REMOVE: formula block from original markdown ---
        st.markdown(
            r"""
$$
\Delta AUC = AUC_{\text{primary}} - AUC_{\text{challenger}}
$$
""")
        # --- end preserved formula block ---
        st.markdown(
            f"where $AUC_{{primary}}$ is the primary model AUC and $AUC_{{challenger}}$ is the challenger model AUC.")
        st.markdown(
            "*   **Rank Correlation ($\\rho$)**: Spearman correlation between risk rankings (do they order borrowers similarly?).")
        # --- DO NOT REMOVE: Spearman formula from original markdown ---
        st.markdown(
            r"""
$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$
""")
        # --- end preserved formula block ---
        st.markdown(
            "*   **Risk Tier Agreement**: Percent of applicants assigned to the same risk tier (operational consistency).")

        with st.expander("Tier semantics (make the tiers decision-relevant)", expanded=False):
            st.info(
                """
A tier is only meaningful if it maps to an action. Example mapping (you can adapt):  
- Tier 0: auto-approve  
- Tier 1: approve / standard pricing  
- Tier 2: manual review / tighter terms  
- Tier 3: decline  
                """.strip()
            )

        if st.button("Run benchmark comparison"):
            primary_auc, primary_f1 = evaluate_model_performance(
                st.session_state.y_test, st.session_state.primary_probs, threshold=st.session_state.policy_threshold
            )
            challenger_auc, challenger_f1 = evaluate_model_performance(
                st.session_state.y_test, st.session_state.challenger_probs, threshold=st.session_state.policy_threshold
            )

            auc_lift = float(primary_auc - challenger_auc)
            rho = float(compute_spearman_rank_correlation(
                st.session_state.primary_probs, st.session_state.challenger_probs))

            tiers_primary = define_risk_tiers(
                st.session_state.primary_probs, thresholds=st.session_state.tier_thresholds)
            tiers_chall = define_risk_tiers(
                st.session_state.challenger_probs, thresholds=st.session_state.tier_thresholds)
            tier_agreement = float(np.mean(tiers_primary == tiers_chall))

            bench = {
                "Primary AUC": float(primary_auc),
                "Challenger AUC": float(challenger_auc),
                "AUC Lift (Primary - Challenger)": auc_lift,
                "Rank Corr (Spearman ρ)": rho,
                "Tier Agreement": tier_agreement,
                "Primary F1 (cutoff)": float(primary_f1),
                "Challenger F1 (cutoff)": float(challenger_f1),
                "Cutoff used": float(st.session_state.policy_threshold),
            }
            st.session_state.benchmark_results = bench

        if st.session_state.benchmark_results is not None:
            st.subheader("Benchmark Results")

            # Display key metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Primary AUC", f"{st.session_state.benchmark_results['Primary AUC']:.3f}")
                st.metric(
                    "Primary F1", f"{st.session_state.benchmark_results['Primary F1 (cutoff)']:.3f}")
            with col2:
                st.metric(
                    "Challenger AUC", f"{st.session_state.benchmark_results['Challenger AUC']:.3f}")
                st.metric(
                    "Challenger F1", f"{st.session_state.benchmark_results['Challenger F1 (cutoff)']:.3f}")
            with col3:
                st.metric(
                    "AUC Lift", f"{st.session_state.benchmark_results['AUC Lift (Primary - Challenger)']:.3f}")
                st.metric(
                    "Cutoff Used", f"{st.session_state.benchmark_results['Cutoff used']:.2f}")

            # Display comparison metrics in a table
            comparison_df = pd.DataFrame([
                {"Metric": "Rank Correlation (Spearman ρ)",
                 "Value": f"{st.session_state.benchmark_results['Rank Corr (Spearman ρ)']:.3f}"},
                {"Metric": "Risk Tier Agreement",
                    "Value": f"{st.session_state.benchmark_results['Tier Agreement']:.1%}"},
            ])
            st.dataframe(comparison_df, use_container_width=True,
                         hide_index=True)

            st.markdown("### Decision translation (how to act on these)")
            auc_lift = st.session_state.benchmark_results["AUC Lift (Primary - Challenger)"]
            rho = st.session_state.benchmark_results["Rank Corr (Spearman ρ)"]
            tier_agree = st.session_state.benchmark_results["Tier Agreement"]

            st.write(
                f"- If **AUC lift** increases, you may justify complexity *if* stability and governance are acceptable.")
            st.write(
                f"- If **rank correlation** is very high, the simpler model orders borrowers similarly (complexity may not be worth it).")
            st.write(f"- If **tier agreement** is low, staffing and outcomes will change materially (manual review queue, decline rates, pricing tiers).")

            # Policy-based interpretation
            pass_lift = auc_lift >= st.session_state.policy_min_auc_lift
            pass_rho = rho >= st.session_state.policy_min_rank_corr
            pass_tier = tier_agree >= st.session_state.policy_min_tier_agree

            st.markdown("### Guardrails (policy-based)")
            st.write(
                f"- Minimum AUC lift required: **{st.session_state.policy_min_auc_lift:.3f}** → {'PASS' if pass_lift else 'FAIL'}")
            st.write(
                f"- Minimum rank correlation: **{st.session_state.policy_min_rank_corr:.2f}** → {'PASS' if pass_rho else 'FAIL'}")
            st.write(
                f"- Minimum tier agreement: **{st.session_state.policy_min_tier_agree:.2f}** → {'PASS' if pass_tier else 'FAIL'}")

            checkpoint(
                "High AUC lift alone is sufficient to approve a complex model.",
                ["True", "False"],
                "False",
                "Lift is not enough. You also need stability, documentation, explainability, and decision-threshold performance."
            )

# -----------------------------
# 4) Prediction Stability Testing
# -----------------------------
elif st.session_state.nav == "4) Prediction Stability Testing":
    st.header(
        "4) Prediction Stability: Do small input changes create unstable decisions?")
    st.markdown(
        """
**Learning goal:** robustness is a governance requirement.  
You test whether small, realistic perturbations (data entry, rounding, minor reporting differences) cause **score volatility** or **decision flips**.
        """.strip()
    )

    if not st.session_state.setup_complete:
        st.warning("Run setup first.")
    else:
        st.markdown(
            """
A model can look accurate but behave inconsistently near decision cutoffs.  
Stability testing estimates how often “small changes” cause different outcomes.
            """.strip()
        )

        noise_level = st.slider(
            "Noise level (% of feature standard deviation)",
            min_value=0.0, max_value=0.10, value=0.01, step=0.005,
            help="This is a proxy for small measurement noise. In production you would define this based on data quality evidence."
        )
        n_trials = st.slider("Number of perturbation trials",
                             min_value=10, max_value=200, value=50, step=10)

        with st.expander("Translate noise into business units (intuition)", expanded=False):
            st.info(
                """
This lab expresses noise as a fraction of feature variability.  
To make it decision-relevant, translate into units:  
- If a feature’s std is 15 percentage points, then 1% std ≈ 0.15 percentage points.  
- If a FICO-like variable had std ≈ 70, then 1% std ≈ 0.7 points.  
                """.strip()
            )

        if st.button("Run stability test"):
            results = run_stability_test(
                X_test=st.session_state.X_test,
                y_test=st.session_state.y_test,
                base_probs=st.session_state.primary_probs,
                model=st.session_state.primary_model,
                noise_level=noise_level,
                n_trials=n_trials,
                threshold=st.session_state.policy_threshold,
            )
            st.session_state.stability_results = results

        if st.session_state.stability_results is not None:
            st.subheader("Stability summary")
            st.dataframe(st.session_state.stability_results["trial_metrics"].describe(
            ), use_container_width=True)

            flip_rate = float(st.session_state.stability_results["flip_rate"])
            st.metric("Decision flip rate", f"{flip_rate*100:.2f}%")

            st.caption(
                "Decision translation: Higher flip rate means inconsistent underwriting/pricing around the cutoff, increasing operational and compliance risk."
            )

            st.markdown("### Guardrails (policy-based)")
            st.write(
                f"Maximum flip rate allowed: **{st.session_state.policy_max_flip_rate:.2%}** → {'PASS' if flip_rate <= st.session_state.policy_max_flip_rate else 'FAIL'}")

            checkpoint(
                "Which borrowers typically drive flip-rate risk?",
                ["Far from the cutoff", "Near the cutoff",
                    "Only the highest PD tail"],
                "Near the cutoff",
                "Small perturbations matter most where the score is close to the decision rule boundary."
            )

# -----------------------------
# 5) Complexity Justification
# -----------------------------
elif st.session_state.nav == "5) Complexity Justification":
    st.header(
        "5) Complexity Justification: Is complexity worth the governance burden?")
    st.markdown(
        """
**Learning goal:** synthesize evidence into an audit-defensible judgment.  
Complexity is justified only if incremental value is material **and** governance risks are controlled.
        """.strip()
    )

    if not st.session_state.setup_complete:
        st.warning("Run setup first.")
    else:
        if st.session_state.benchmark_results is None or st.session_state.stability_results is None:
            st.warning(
                "Complete benchmarking and stability sections first (evidence needed).")
        else:
            # Criteria evaluation (policy-based)
            auc_lift = float(
                st.session_state.benchmark_results["AUC Lift (Primary - Challenger)"])
            rho = float(
                st.session_state.benchmark_results["Rank Corr (Spearman ρ)"])
            tier_agree = float(
                st.session_state.benchmark_results["Tier Agreement"])
            flip_rate = float(st.session_state.stability_results["flip_rate"])

            # Materiality lens (simple lab heuristic)
            business_material = st.session_state.portfolio_size_mm >= 100

            # Transparency: in this lab, treat as missing unless explicitly documented
            has_explainability_plan = any(
                "explainability" in s.lower() for s in st.session_state.doc_issues)

            # Scorecard
            findings = []

            perf_ok = auc_lift >= st.session_state.policy_min_auc_lift
            findings.append(("Performance lift", "PASS" if perf_ok else "FAIL",
                            f"AUC lift={auc_lift:.3f} vs policy min {st.session_state.policy_min_auc_lift:.3f}"))

            agreement_ok = (rho >= st.session_state.policy_min_rank_corr) and (
                tier_agree >= st.session_state.policy_min_tier_agree)
            findings.append(("Operational consistency", "PASS" if agreement_ok else "NOTE",
                            f"ρ={rho:.2f}, tier agreement={tier_agree:.2%}"))

            stability_ok = flip_rate <= st.session_state.policy_max_flip_rate
            findings.append(("Stability near cutoff", "PASS" if stability_ok else "FAIL",
                            f"Flip rate={flip_rate:.2%} vs max {st.session_state.policy_max_flip_rate:.2%}"))

            business_ok = business_material
            findings.append(("Business materiality", "PASS" if business_ok else "NOTE",
                            f"Portfolio={st.session_state.portfolio_size_mm}mm (lab lens)"))

            explain_ok = not has_explainability_plan  # doc issue exists means missing
            findings.append(("Explainability governance", "FAIL" if not explain_ok else "PASS",
                            "Explainability strategy must be documented and governed."))

            score = sum(1 for _, status, _ in findings if status == "PASS")

            recommendation = "Conditionally Approved"
            if any(status == "FAIL" for _, status, _ in findings):
                recommendation = "Rejected (governance risk not controlled)"
            elif score >= 4:
                recommendation = "Approved (subject to standard monitoring)"

            st.session_state.complexity_assessment = {
                "score_pass_count": score,
                "recommendation": recommendation,
                "findings": findings,
            }

            st.subheader("Complexity scorecard (evidence-based)")
            df = pd.DataFrame(findings, columns=[
                              "Criterion", "Status", "Evidence / Rationale"])
            st.dataframe(df, use_container_width=True)

            st.markdown("### Decision translation")
            st.write("- If the model is **Rejected**, reduce complexity or bring additional evidence (stability, documentation, explainability, subgroup testing).")
            st.write(
                "- If **Conditionally Approved**, you may use it only under explicit conditions and monitoring until gaps are remediated.")
            st.write(
                "- If **Approved**, proceed with monitoring and periodic revalidation according to policy.")

            checkpoint(
                "A 'conditional approval' means the model is safe to use without changes.",
                ["True", "False"],
                "False",
                "Conditional approval means the model can be used only if specific remediation and monitoring conditions are met."
            )

# -----------------------------
# 6) Formal Validation Report
# -----------------------------
elif st.session_state.nav == "6) Formal Validation Report":
    st.header("6) Formal Validation Report: audit-ready decision package")
    st.markdown(
        """
**Learning goal:** translate analysis into an audit artifact: section statuses, findings, conditions, and recommendation.  
Every number must have a traceable definition and assumption.
        """.strip()
    )

    if not st.session_state.setup_complete:
        st.warning("Run setup first.")
    else:
        if st.session_state.complexity_assessment is None:
            st.warning("Complete complexity justification first.")
        else:
            if st.button("Generate formal validation report"):
                # Build a structured report from existing artifacts
                report = {
                    "Model": "Credit Default PD (Primary)",
                    "Use": "Underwriting + pricing tiering",
                    "Cutoff used (for F1 / decisions)": float(st.session_state.policy_threshold),
                    "Tier thresholds": st.session_state.tier_thresholds,
                    "Policy profile": st.session_state.policy_profile,
                    "Sections": {
                        "Documentation Review (Gate 1)": "Completed",
                        "Result Reproduction (Gate 2)": "Completed" if st.session_state.repro_results is not None else "Not run",
                        "Challenger Benchmarking": "Completed" if st.session_state.benchmark_results is not None else "Not run",
                        "Stability Testing": "Completed" if st.session_state.stability_results is not None else "Not run",
                        "Complexity Justification": "Completed",
                    },
                    "Key findings": st.session_state.complexity_assessment["findings"],
                    "Recommendation": st.session_state.complexity_assessment["recommendation"],
                    "Conditions (if conditional)": [
                        "Provide documented explainability strategy (what is produced, how it is governed).",
                        "Add subgroup / fair lending evidence consistent with policy.",
                        "Implement monitoring with explicit triggers (PD drift, approval rate shift, tier migration).",
                    ],
                    "Not covered in this lab (would be required in production)": [
                        "Fair lending / disparate impact testing on protected classes (policy-aligned).",
                        "Out-of-time stress testing and macro sensitivity analysis.",
                        "Data lineage and full production control testing.",
                    ],
                }
                st.session_state.validation_report = report

            if st.session_state.validation_report is not None:
                st.subheader("Executive summary")

                report = st.session_state.validation_report

                # Model Information
                st.markdown("#### Model Information")
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.write(f"**Model:** {report['Model']}")
                    st.write(f"**Use:** {report['Use']}")
                with info_col2:
                    st.write(
                        f"**Decision Cutoff:** {report['Cutoff used (for F1 / decisions)']:.2f}")
                    st.write(f"**Policy Profile:** {report['Policy profile']}")

                # Section Status
                st.markdown("#### Validation Sections Status")
                sections_df = pd.DataFrame([
                    {"Section": k, "Status": v}
                    for k, v in report['Sections'].items()
                ])
                st.dataframe(
                    sections_df, use_container_width=True, hide_index=True)

                # Key Findings
                st.markdown("#### Key Findings")
                findings_df = pd.DataFrame(
                    report['Key findings'],
                    columns=["Criterion", "Status", "Evidence / Rationale"]
                )
                st.dataframe(
                    findings_df, use_container_width=True, hide_index=True)

                # Recommendation
                st.markdown("#### Recommendation")
                if "Rejected" in report['Recommendation']:
                    st.error(f"🚫 {report['Recommendation']}")
                elif "Conditionally" in report['Recommendation']:
                    st.warning(f"⚠️ {report['Recommendation']}")
                else:
                    st.success(f"✅ {report['Recommendation']}")

                # Conditions
                st.markdown("#### Conditions (if conditional)")
                for condition in report['Conditions (if conditional)']:
                    st.write(f"- {condition}")

                # Limitations
                with st.expander("Not covered in this lab (would be required in production)"):
                    for item in report['Not covered in this lab (would be required in production)']:
                        st.write(f"- {item}")

                st.markdown("### Guardrails against misinterpretation")
                st.warning(
                    """
- Do not interpret AUC as a deployment decision by itself. Tie outcomes to underwriting cutoffs and tier actions.
- Treat all cutoffs as **policy choices**; document why a threshold exists and what changes when it moves.
- Ensure every claim has an evidence artifact: metric definition, dataset version, score file version, and assumptions.
                    """.strip()
                )

# -----------------------------
# Visualizations
# -----------------------------
elif st.session_state.nav == "Visualizations":
    st.header("Visualizations: communicate governance evidence")
    st.markdown(
        """
**Learning goal:** each plot must answer a specific validation question and include a decision translation.  
Use charts to reduce ambiguity, not to decorate results.
        """.strip()
    )

    if not st.session_state.setup_complete:
        st.warning("Run setup first.")
    else:
        if st.session_state.benchmark_results is None:
            st.info("Run benchmarking to populate comparison visuals.")
        if st.session_state.stability_results is None:
            st.info("Run stability test to populate stability visuals.")

        # ROC overlay
        st.subheader("ROC comparison (ranking power)")
        st.caption(
            "How to read: higher curve/AUC = better discrimination. Decision translation: tiny lifts may not justify complexity.")
        from sklearn.metrics import roc_curve, auc

        fpr_p, tpr_p, _ = roc_curve(
            st.session_state.y_test, st.session_state.primary_probs)
        fpr_c, tpr_c, _ = roc_curve(
            st.session_state.y_test, st.session_state.challenger_probs)
        auc_p = auc(fpr_p, tpr_p)
        auc_c = auc(fpr_c, tpr_c)

        fig, ax = plt.subplots()
        ax.plot(fpr_p, tpr_p, label=f"Primary (AUC={auc_p:.3f})")
        ax.plot(fpr_c, tpr_c, label=f"Challenger (AUC={auc_c:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        # Probability scatter
        st.subheader("Score agreement: Primary vs Challenger (scatter)")
        st.caption(
            "How to read: tight diagonal = similar scoring. Decision translation: dispersion implies different ordering/segmentation.")
        fig2, ax2 = plt.subplots()
        ax2.scatter(st.session_state.primary_probs,
                    st.session_state.challenger_probs, alpha=0.35)
        ax2.set_xlabel("Primary PD")
        ax2.set_ylabel("Challenger PD")
        st.pyplot(fig2, clear_figure=True)

        # Tier confusion matrix
        st.subheader("Risk tier agreement (operational buckets)")
        st.caption(
            "How to read: off-diagonal mass = different tier actions (approve/review/decline).")
        tiers_p = define_risk_tiers(
            st.session_state.primary_probs, thresholds=st.session_state.tier_thresholds)
        tiers_c = define_risk_tiers(
            st.session_state.challenger_probs, thresholds=st.session_state.tier_thresholds)
        cm = pd.crosstab(tiers_p, tiers_c, rownames=[
                         "Primary tier"], colnames=["Challenger tier"])
        st.dataframe(cm, use_container_width=True)

        # Stability chart (if available)
        if st.session_state.stability_results is not None:
            st.subheader(
                "Stability distribution (trial-level score perturbation)")
            st.caption(
                "How to read: wider distribution/outliers = instability. Decision translation: instability near cutoffs increases governance risk.")
            trial_df = st.session_state.stability_results["trial_metrics"]
            fig3, ax3 = plt.subplots()
            ax3.boxplot(trial_df["mean_abs_diff"].values)
            ax3.set_ylabel("Mean absolute PD change (trial)")
            st.pyplot(fig3, clear_figure=True)


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
