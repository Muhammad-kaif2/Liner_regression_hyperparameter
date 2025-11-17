import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.sidebar.title("Linear Regression Models")

model_type = st.sidebar.selectbox(
    'Select Model',
    ('Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net')
)

fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True)

# Hyperparameter settings based on selected model
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


# Generate synthetic dataset
X, y = make_regression(
    n_samples=300,
    n_features=1,
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
if st.sidebar.button("Run Model"):

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

    y_pred = model.predict(X_test)

    # Plot graph
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue", alpha=0.5)
    ax.plot(X_test, y_pred, color="red")
    plt.xlabel("Feature")
    plt.ylabel("Target")

    st.pyplot(fig)

    st.subheader("📊 Model Performance")
    st.write("**Mean Squared Error:**", round(mean_squared_error(y_test, y_pred), 2))
    st.write("**R² Score:**", round(r2_score(y_test, y_pred), 2))
