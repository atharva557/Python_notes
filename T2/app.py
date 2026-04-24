import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

st.title("AutoML Dashboard")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.shape)
    st.dataframe(df.head())

    st.header('1. Data Preprocessing')
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        st.write("Missing values detected")
        missing_choice = st.radio("Select method to handle missing values",
                                  ["Drop Rows", "Fill with Mean/Mode"])
        
        if missing_choice == "Drop Rows":
            df = df.dropna()
        elif missing_choice == "Fill with Mean/Mode":
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Identify and encode object/category columns
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if object_cols:
        label_encoder = LabelEncoder()
        for col in object_cols:
            df[col] = label_encoder.fit_transform(df[col].astype(str).astype('category').cat.codes)
    
    st.write("Cleaned DataFrame:")
    st.dataframe(df)

    st.subheader('Correlation Heatmap')
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.header('2. Model Training')
    
    # Target column selection (must be non-empty column)
    available_columns = df.columns.tolist()
    target_column = st.selectbox("Select Target Column (y)", available_columns)
    
    # Feature columns X
    X_columns = [col for col in available_columns if col != target_column]
    
    # Algorithm selection
    algorithm_options = ['Linear Regression', 'Polynomial Regression', 'kNN', 'Random Forest', 'SVM']
    algorithm = st.selectbox("Select Algorithm", algorithm_options)
    
    # Train button
    train_model = st.button("Train Model")
    
    # Import inside button click handler for cleaner scikit-learn imports
    if train_model:
        # Import at execution time
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge as PolynomialRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier as RandomForestRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVC
        from sklearn.svm import SVR
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import accuracy_score, r2_score
        
        import matplotlib.pyplot as plt
        
        X = df[X_columns]
        y = df[target_column]
        
        # 80/20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Prepare model for Polynomial Regression first
        if algorithm == 'Polynomial Regression':
            poly_features = PolynomialFeatures(degree=2)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            st.session_state['poly'] = poly_features
            
            y_pred_poly = model.predict(X_test_poly)
            score = model.score(X_test_poly, y_test)
            metric_name = "R2 Score"
            
            st.write(f"Polynomial Regression trained.")
            st.write(f"{metric_name}: {score:.4f}")
            
            # Create scatter plot for Polynomial Regression
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred_poly, alpha=0.7, edgecolors='k')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_title('Actual vs Predicted')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            st.pyplot(fig)
            
            # Save model to session state
            st.session_state['model'] = model
            st.session_state['model_type'] = algorithm
            st.session_state['feature_columns'] = X_columns
            
        else:
            # For all other algorithms - standard training
            model = None
            
            if algorithm == 'Linear Regression':
                model = LinearRegression()
                test_set = X_test
                y_test_set = y_test
            else:
                if algorithm == 'kNN':
                    model = KNeighborsClassifier()
                elif algorithm == 'Random Forest':
                    model = RandomForestClassifier(n_estimators=100)
                elif algorithm == 'SVM':
                    model = SVC()
            
            if model is not None:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                metric_name = "R2 Score" if hasattr(model, 'coef_') else "Accuracy Score"
                
                st.write(f"{model.__class__.__name__} trained.")
                st.write(f"{metric_name}: {score:.4f}")
                
                # Create scatter plot
                y_pred = model.predict(X_test)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_title('Actual vs Predicted')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                st.pyplot(fig)
                
                # Save model to session state
                st.session_state['model'] = model
                st.session_state['model_type'] = algorithm
                st.session_state['feature_columns'] = X_columns
            
    st.header('3. Make a Prediction')
    
    if 'model' in st.session_state:
        X_dynamic = {}
        for col in df.columns:
            if col == target_column:
                continue
            st.write(f"Feature: {col}")
            val = st.number_input(f"Enter {col} value", min_value=0.0, max_value=1000.0, step=0.1)
            X_dynamic[col] = val
        
        predict_button = st.button("Predict")
        
        if predict_button:
            # Convert inputs to 2D numpy array
            import numpy as np
            X_input = np.array([list(X_dynamic.values())])
            
            # Get model info from session state
            model_info = st.session_state
            
            # Check if Polynomial Regression was chosen
            if st.session_state.model_type == 'Polynomial Regression':
                poly_features = st.session_state['poly']
                X_input_transformed = poly_features.transform(X_input)
                
                # Get original feature names after polynomial expansion
                polynomial_features = 'PolynomialFeatures'
                st.write(f"Transformed features shape: {X_input_transformed.shape}")
                st.write(f"Transformed feature names: {poly_features.get_feature_names_out(X.columns)}")
            
            # Make prediction
            prediction = st.session_state['model'].predict(X_input_transformed if st.session_state['model_type'] == 'Polynomial Regression' else X_input)
            
            st.success(f"Prediction: {prediction[0]}")
            
