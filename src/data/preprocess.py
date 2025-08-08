import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_breast_cancer, make_classification


def generate_synthetic_data(n_samples=1000, n_features=20, random_state=42):
    """Generate synthetic classification dataset"""
    # Ensure n_informative + n_redundant <= n_features
    n_informative = min(10, n_features - 2)
    n_redundant = min(5, n_features - n_informative)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,  # Add this for stability
        random_state=random_state
    )

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    return df


def load_wine_data():
    """Load and return wine dataset as DataFrame"""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df


def load_breast_cancer_data():
    """Load and return breast cancer dataset as DataFrame"""
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split DataFrame into train and test sets"""
    return train_test_split(df, test_size=test_size, random_state=random_state)


def preprocess_data(input_path, output_train_path, output_test_path, test_size=0.2):
    """Preprocess data from input path and save to output paths"""
    df = pd.read_csv(input_path)
    train_df, test_df = split_data(df, test_size=test_size)

    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)

    return train_df, test_df
