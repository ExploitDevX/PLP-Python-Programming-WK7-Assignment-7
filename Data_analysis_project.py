import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    print("=== Task 1: Loading and Exploring Data ===")
    
    # Load dataset (using Iris as example)
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        print("Dataset loaded successfully (Iris dataset).\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Display first few rows
    print("First 5 rows:")
    print(df.head())
    
    # Check data structure
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Clean data (no missing values in Iris, but showing the code)
    df = df.dropna()  # Would remove rows with missing values if any existed
    print("\nData cleaned (missing values handled).")
    
    return df

# Task 2: Basic Data Analysis
def perform_analysis(df):
    print("\n=== Task 2: Basic Data Analysis ===")
    
    # Basic statistics
    print("\nBasic statistics for numerical columns:")
    print(df.describe())
    
    # Group by species and calculate means
    if 'species' in df.columns:
        print("\nMean measurements by species:")
        print(df.groupby('species').mean())
    
    # Additional findings
    print("\nAdditional observations:")
    print("- Setosa species generally has smaller measurements")
    print("- Petal dimensions show clearer separation between species than sepal dimensions")

# Task 3: Data Visualization
def create_visualizations(df):
    print("\n=== Task 3: Data Visualization ===")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Line chart (simulated time series)
    plt.subplot(2, 2, 1)
    df_sample = df.sample(10).sort_index()
    for feature in df.columns[:-1]:  # Exclude species column
        plt.plot(df_sample.index, df_sample[feature], label=feature)
    plt.title('Sample Measurements (Simulated Time Series)')
    plt.xlabel('Sample Index')
    plt.ylabel('Measurement (cm)')
    plt.legend()
    
    # 2. Bar chart (average measurements by species)
    plt.subplot(2, 2, 2)
    df.groupby('species').mean().plot(kind='bar')
    plt.title('Average Measurements by Species')
    plt.ylabel('Measurement (cm)')
    plt.xticks(rotation=0)
    
    # 3. Histogram (sepal length distribution)
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='sepal length (cm)', hue='species', kde=True)
    plt.title('Sepal Length Distribution by Species')
    
    # 4. Scatter plot (sepal length vs petal length)
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
    plt.title('Sepal Length vs Petal Length')
    
    plt.tight_layout()
    plt.savefig('visualizations.png')
    print("\nVisualizations saved as 'visualizations.png'")

# Main execution
if __name__ == "__main__":
    # Load and explore data
    iris_df = load_and_explore_data()
    
    if iris_df is not None:
        # Perform analysis
        perform_analysis(iris_df)
        
        # Create visualizations
        create_visualizations(iris_df)
        
        print("\n=== Analysis Complete ===")
    else:
        print("Analysis could not be completed due to data loading issues.")
