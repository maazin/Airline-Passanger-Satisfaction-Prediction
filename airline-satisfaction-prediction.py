import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

# Simulate airline passenger satisfaction dataset
def generate_synthetic_dataset(n_samples=100000):
    """
    Generate a synthetic dataset for airline passenger satisfaction prediction.
    
    Columns:
    - Flight Distance
    - Seat Comfort
    - Inflight Entertainment
    - Food and Drink Quality
    - Staff Service
    - Departure Delay
    - Arrival Delay
    - Customer Type (Loyal/Disloyal)
    - Travel Type (Business/Personal)
    - Age
    - Gender
    - Satisfaction (Target Variable)
    """
    data = {
        'flight_distance': np.random.normal(1500, 500, n_samples),
        'seat_comfort': np.random.randint(1, 6, n_samples),
        'inflight_entertainment': np.random.randint(1, 6, n_samples),
        'food_quality': np.random.randint(1, 6, n_samples),
        'staff_service': np.random.randint(1, 6, n_samples),
        'departure_delay': np.random.normal(15, 30, n_samples),
        'arrival_delay': np.random.normal(15, 30, n_samples),
        'customer_type': np.random.choice(['Loyal', 'Disloyal'], n_samples),
        'travel_type': np.random.choice(['Business', 'Personal'], n_samples),
        'age': np.random.normal(40, 15, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic satisfaction based on features
    satisfaction_score = (
        df['seat_comfort'] * 0.2 + 
        df['inflight_entertainment'] * 0.2 + 
        df['food_quality'] * 0.2 + 
        df['staff_service'] * 0.2 - 
        np.abs(df['departure_delay']) * 0.001 - 
        np.abs(df['arrival_delay']) * 0.001
    )
    
    df['satisfaction'] = (satisfaction_score > satisfaction_score.median()).astype(int)
    
    return df

# Generate dataset
df = generate_synthetic_dataset()

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    plt.figure(figsize=(15, 10))
    
    # Satisfaction Distribution
    plt.subplot(2, 2, 1)
    df['satisfaction'].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%')
    plt.title('Satisfaction Distribution')
    
    # Satisfaction by Customer Type
    plt.subplot(2, 2, 2)
    sns.barplot(x='customer_type', y='satisfaction', data=df)
    plt.title('Satisfaction by Customer Type')
    
    # Satisfaction by Travel Type
    plt.subplot(2, 2, 3)
    sns.barplot(x='travel_type', y='satisfaction', data=df)
    plt.title('Satisfaction by Travel Type')
    
    # Correlation Heatmap
    plt.subplot(2, 2, 4)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()

# Preprocessing
def preprocess_data(df):
    # Label Encoding for Categorical Variables
    le = LabelEncoder()
    df['customer_type_encoded'] = le.fit_transform(df['customer_type'])
    df['travel_type_encoded'] = le.fit_transform(df['travel_type'])
    df['gender_encoded'] = le.fit_transform(df['gender'])
    
    # Select features
    features = [
        'flight_distance', 'seat_comfort', 'inflight_entertainment', 
        'food_quality', 'staff_service', 'departure_delay', 
        'arrival_delay', 'customer_type_encoded', 'travel_type_encoded', 
        'age', 'gender_encoded'
    ]
    
    X = df[features]
    y = df['satisfaction']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Model
def train_random_forest(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Model Evaluation
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return rf_classifier

# Feature Importance
def plot_feature_importance(model, feature_names):
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    plt.figure(figsize=(10, 6))
    feature_importances.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Importances in Passenger Satisfaction')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

# Main Execution
print("Performing Exploratory Data Analysis...")
perform_eda(df)

print("\nPreprocessing Data...")
X_train, X_test, y_train, y_test = preprocess_data(df)

print("\nTraining Random Forest Classifier...")
rf_model = train_random_forest(X_train, X_test, y_train, y_test)

print("\nAnalyzing Feature Importance...")
feature_names = [
    'flight_distance', 'seat_comfort', 'inflight_entertainment', 
    'food_quality', 'staff_service', 'departure_delay', 
    'arrival_delay', 'customer_type', 'travel_type', 
    'age', 'gender'
]
plot_feature_importance(rf_model, feature_names)

print("\nProject Complete! 🛫 Airline Passenger Satisfaction Prediction Successful.")
