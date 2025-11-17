
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Linear Regression Hyperparameter Dashboard")

# -----------------------------
# Step 1: Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Step 2: Select Target Column
    # -----------------------------
    target_column = st.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # -----------------------------
    # Step 3: Select Model & Hyperparameters
    # -----------------------------
    st.sidebar.title("Linear Regression Models")

    model_type = st.sidebar.selectbox(
        'Select Model',
        ('Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net')
    )

    fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True)

    if model_type == "Linear Regression":
        normalize = st.sidebar.checkbox("Normalize", value=False)

    elif model_type == "Ridge Regression":
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
        solver = st.sidebar.selectbox(
            "Solver",
            ('auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga')
        )

    elif model_type == "Lasso Regression":
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
        max_iter = st.sidebar.number_input("Max Iterations", 100, 5000, 1000)

    elif model_type == "Elastic Net":
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
        l1_ratio = st.sidebar.slider("L1 Ratio", 0.0, 1.0, 0.5)
        max_iter = st.sidebar.number_input("Max Iterations", 100, 5000, 1000)

    # -----------------------------
    # Step 4: Split Data & Run Model
    # -----------------------------
    if st.sidebar.button("Run Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "Linear Regression":
            model = LinearRegression(fit_intercept=fit_intercept)

        elif model_type == "Ridge Regression":
            model = Ridge(alpha=alpha, solver=solver, fit_intercept=fit_intercept)

        elif model_type == "Lasso Regression":
            model = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept)

        elif model_type == "Elastic Net":
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                               max_iter=max_iter, fit_intercept=fit_intercept)

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # -----------------------------
        # Step 5: Plot & Metrics
        # -----------------------------
        fig = plt.figure(figsize=(6,5))
        if X.shape[1] == 1:  # 1 feature -> 2D plot
            ax = fig.add_subplot(111)
            ax.scatter(X_train, y_train, color="blue", alpha=0.5, label="Train Actual")
            ax.scatter(X_train, y_train_pred, color="green", alpha=0.7, label="Train Predicted")
            ax.scatter(X_test, y_test, color="orange", alpha=0.5, label="Test Actual")
            ax.scatter(X_test, y_test_pred, color="red", alpha=0.7, label="Test Predicted")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Target")
            ax.legend()

        elif X.shape[1] == 2:  # 2 features -> 3D plot
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_train[:,0], X_train[:,1], y_train, color="blue", alpha=0.5, label="Train Actual")
            ax.scatter(X_train[:,0], X_train[:,1], y_train_pred, color="green", alpha=0.7, label="Train Predicted")
            ax.scatter(X_test[:,0], X_test[:,1], y_test, color="orange", alpha=0.5, label="Test Actual")
            ax.scatter(X_test[:,0], X_test[:,1], y_test_pred, color="red", alpha=0.7, label="Test Predicted")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Target")
            ax.legend()

        else:  # >2 features -> Predicted vs Actual
            ax = fig.add_subplot(111)
            ax.scatter(y_train, y_train_pred, color="green", alpha=0.7, label="Train")
            ax.scatter(y_test, y_test_pred, color="red", alpha=0.7, label="Test")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            ax.legend()

        st.pyplot(fig)

        st.subheader("📊 Model Performance")

        # Train metrics
        st.write("**Train MSE:**", round(mean_squared_error(y_train, y_train_pred), 2))
        st.write("**Train R² Score:**", round(r2_score(y_train, y_train_pred), 2))

        # Test metrics
        st.write("**Test MSE:**", round(mean_squared_error(y_test, y_test_pred), 2))
        st.write("**Test R² Score:**", round(r2_score(y_test, y_test_pred), 2))

        # Optional: Check fit
        if r2_score(y_train, y_train_pred) > 0.9 and r2_score(y_test, y_test_pred) > 0.8:
            st.success("✅ Model is performing well (Good Fit)")
        elif r2_score(y_train, y_train_pred) > 0.9 and r2_score(y_test, y_test_pred) < 0.7:
            st.warning("⚠️ Model may be Overfitting")
        elif r2_score(y_train, y_train_pred) < 0.7:
            st.warning("⚠️ Model may be Underfitting")
else:
    st.info("Please upload a CSV file to get started.")
