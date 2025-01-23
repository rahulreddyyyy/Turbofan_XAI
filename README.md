Remaining Useful Life (RUL) Prediction of Turbofan Engines Using Explainable AI

Authors: Abhay Raj Yadav, Mohd Adnan Hasan, Rahul Reddy

Introduction: This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. By leveraging Random Forest regression and integrating advanced explainability techniques like LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (Shapley Additive Explanations), this framework combines predictive accuracy with interpretability. The proposed approach ensures that maintenance decisions are transparent, reliable, and actionable, making it particularly relevant for high-risk industries like aerospace.

Features RUL Prediction:

Utilizes Random Forest for modeling engine degradation patterns. Handles both simple (FD001, FD002) and complex (FD003, FD004) scenarios of the C-MAPSS dataset. Explainability:

LIME: Explains individual predictions by approximating the model locally using interpretable surrogates. SHAP: Provides global and local feature contributions, helping identify critical factors influencing predictions. Scalability and Robustness:

Ensures scalability with Random Forest’s ensemble nature. Supports parallel processing for faster training on large datasets. Key Insights:

Identifies significant sensors and operational conditions affecting engine degradation. Highlights actionable insights for maintenance strategies. Dataset The project uses the NASA C-MAPSS dataset, which simulates turbofan engine performance under various operating conditions and failure modes. The dataset includes four subsets:

FD001: Single operating condition, one failure mode. FD002: Multiple operating conditions, one failure mode. FD003: Single operating condition, two failure modes. FD004: Multiple operating conditions, two failure modes. Key data points include:

Operational settings (3 continuous variables). Sensor measurements (21 continuous variables). Engine cycles (used for RUL calculation).

Methodology

Preprocessing RUL Calculation: Ground truth computed as the difference between maximum and current engine cycles. Feature Selection: Removal of redundant sensors (e.g., Sensors 22 and 23). Normalization: Min-Max scaling applied to all features. Data Splitting: 80-20 train-test split, ensuring robust evaluation.

Model Training Algorithm: Random Forest Regression. Hyperparameters: Number of Trees: 100. Maximum Depth: Tuned per subset. Random State: 42 (ensures reproducibility).

Explainability LIME: Highlights feature importance for specific predictions. SHAP: Quantifies global and local contributions of features to model predictions.

Results:

Model Performance Dataset MAE MSE R² FD001 24.72 1121.75 0.6560 FD002 23.32 994.86 0.3504 FD003 31.09 1892.59 -0.1045 FD004 31.97 1900.73 0.3606 Simpler Datasets (FD001, FD002): The model performs well with high R² values and low error rates. Complex Datasets (FD003, FD004): Performance declines due to overlapping failure mechanisms. Explainability Insights

LIME: Identifies critical sensors like Sensor 11 and Sensor 15 in simpler datasets. SHAP: Highlights Sensor 2 (pressure ratios) and Sensor 7 (fuel flow) as globally important features.

Installation and Usage Prerequisites Python 3.7 or later. Libraries: numpy, pandas, scikit-learn, lime, shap, matplotlib.

Steps to Run Clone the repository: git clone https://github.com/rahulreddyyyy/Turbofan_XAI.git

Future Work: Explore advanced models like Gradient Boosting Machines and hybrid deep learning architectures. Address challenges in complex datasets (FD003, FD004) using feature engineering or alternative assembly methods. Investigate real-time RUL prediction for live operational scenarios.

References NASA C-MAPSS Dataset: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq/about_data
