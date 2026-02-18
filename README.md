# QuLab: Lab 38: Model Validation Exercise

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Lab 38: Model Validation Exercise** is a Streamlit-powered interactive laboratory project designed for professionals in Model Risk Management (MRM). It simulates a critical scenario where an MRM analyst at "Prudent Financial Corp." is tasked with validating a black-box credit default prediction model ("Primary Model") developed by an internal data science unit.

The application guides the user through a rigorous, six-step independent model validation workflow, adhering to regulatory mandates like SR 11-7, which demand an "effective challenge" process. Users will critically assess the model's design, data, and outputs without access to its internal weights or full source code. The ultimate goal is to determine if the Primary Model is fit for purpose, robust, and compliant for underwriting decisions, culminating in a formal validation report and a clear recommendation.

This lab project emphasizes providing an "effective challenge"—identifying reasons why a model *should not* be approved if it fails to meet stringent validation standards.

## Features

The application provides an interactive workflow covering essential model validation steps:

1.  **Setup & Data Initialization**:
    *   Configures the environment, installs dependencies, and simulates a credit dataset.
    *   Initializes the "black-box" Primary Model and its claimed predictions, along with initial model documentation and business context.
2.  **Documentation Review**:
    *   Systematically checks provided model documentation for completeness and identifies "red flags" (e.g., missing information, unaddressed limitations).
3.  **Result Reproduction**:
    *   Independently calculates key performance metrics (e.g., AUC, F1-score) using the provided test data and black-box predictions.
    *   Compares these independently reproduced metrics against the developer's claims to ensure consistency and prevent governance issues.
4.  **Challenger Benchmarking**:
    *   Trains a simpler, interpretable Logistic Regression model (the "Challenger Model") on the same data.
    *   Compares the Primary Model's performance against the Challenger Model using metrics like AUC Lift, Spearman's Rank Correlation, and Risk Tier Agreement to justify the Primary Model's complexity.
5.  **Prediction Stability Testing**:
    *   Assesses the Primary Model's robustness by introducing small, controlled perturbations (noise) to input features.
    *   Measures prediction changes, maximum absolute changes, and classification flips to identify fragility or overfitting.
6.  **Complexity Justification Assessment**:
    *   Synthesizes findings from documentation, reproduction, benchmarking, and stability tests.
    *   Evaluates the model against five criteria (AUC Lift, Rank Correlation, Prediction Stability, Business Impact, Explainability Tools) to determine if its complexity is justified relative to its performance and risks.
7.  **Formal Validation Report**:
    *   Compiles a comprehensive validation report summarizing all findings, categorizing issues by severity, and outlining conditions for approval.
    *   Provides a definitive recommendation: "Approve," "Conditional Approve," or "Reject."
8.  **Visualizations**:
    *   Offers insightful plots such as ROC Curves Overlay, Prediction Rank Scatter Plots, Risk Tier Confusion Matrices, and Prediction Stability Box Plots to visually support the validation findings.
    *   Includes a Complexity Justification Scorecard for a quick visual summary.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

Ensure you have Python 3.8+ installed.

*   [Python Installation Guide](https://www.python.org/downloads/)

### Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository_url>
    cd <repository_name> # e.g., QuLab-ModelValidation
    ```
    *(Note: Replace `<repository_url>` and `<repository_name>` with your actual repository details if hosted on GitHub/GitLab)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    The application uses `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `xgboost`.
    Create a `requirements.txt` file in your project root with the following content:

    ```
    streamlit==1.33.0 # or newer compatible version
    pandas==2.2.2 # or newer compatible version
    numpy==1.26.4 # or newer compatible version
    matplotlib==3.8.4 # or newer compatible version
    seaborn==0.13.2 # or newer compatible version
    scikit-learn==1.4.2 # or newer compatible version
    xgboost==2.0.3 # or newer compatible version
    ```

    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure `source.py` is in the same directory:**
    The `streamlit_app.py` (or `app.py`) relies on functions defined in `source.py`. Make sure this file is present in the same directory as your main Streamlit script.

## Usage

1.  **Run the Streamlit application:**
    Open your terminal or command prompt, navigate to the directory containing your `streamlit_app.py` (or `app.py`) and `source.py` files, and run:
    ```bash
    streamlit run streamlit_app.py
    ```
    (Replace `streamlit_app.py` with the actual name of your main Streamlit script if different.)

2.  **Access the application:**
    Your web browser should automatically open to the Streamlit application (usually at `http://localhost:8501`). If not, copy the URL provided in your terminal and paste it into your browser.

3.  **Navigate the Validation Steps:**
    *   Use the **sidebar dropdown menu** on the left to navigate through the sequential validation steps.
    *   Start with "Setup & Data Initialization" to get all data and models ready.
    *   Proceed through the steps in order (Documentation Review, Result Reproduction, etc.) as the results from earlier steps are prerequisites for later ones.
    *   The "Visualizations" page provides graphical summaries of the validation findings.

## Project Structure

```
.
├── streamlit_app.py     # Main Streamlit application script
├── source.py            # Business logic and helper functions for validation steps
└── requirements.txt     # List of Python dependencies
```

*   `streamlit_app.py`: Contains the Streamlit UI, session state management, page navigation, and orchestrates the calls to the validation functions.
*   `source.py`: Encapsulates the core logic for each validation step, such as `review_documentation`, `reproduce_results`, `challenger_benchmark`, `stability_test`, `complexity_assessment`, and `compile_validation_report`. This separation allows for cleaner code and easier testing of business logic.

## Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/) (for interactive web application development)
*   **Language**: [Python](https://www.python.org/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (for Challenger Model, preprocessing, metrics), [XGBoost](https://xgboost.readthedocs.io/) (simulated Primary Model)
*   **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

## Contributing

This project is primarily a lab exercise. However, if you have suggestions for improvements, bug fixes, or new features, feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file (if available, otherwise consider it implied for educational content) for details.

## Contact

For questions or feedback, please reach out via the QuantUniversity platform or related channels.
