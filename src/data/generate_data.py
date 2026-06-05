import pandas as pd
import numpy as np
from pathlib import Path


def generate_churn_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic telco customer churn data with realistic patterns.

    Features:
    - tenure: months as customer (0-72)
    - monthly_charges: monthly bill amount ($20-$120)
    - total_charges: total amount charged over tenure
    - contract: Month-to-month, One year, or Two year
    - churn: 0 (no churn) or 1 (churn)
    """
    rng = np.random.default_rng(random_state)

    # Generate tenure (months as customer)
    tenure = rng.integers(0, 73, n_samples)

    # Generate contract types with realistic distribution
    contract_probs = [0.55, 0.25, 0.20]  # month-to-month more common
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=contract_probs
    )

    # Generate monthly charges with slight variation by contract type
    base_charges = rng.uniform(20, 120, n_samples)
    contract_discount = np.where(contracts == "Two year", -10,
                                 np.where(contracts == "One year", -5, 0))
    monthly_charges = np.clip(base_charges + contract_discount, 20, 120)

    # Calculate total charges based on tenure and monthly charges
    # Add some noise to make it realistic
    total_charges = tenure * monthly_charges + rng.normal(0, 100, n_samples)
    total_charges = np.clip(total_charges, 0, None)

    # Generate churn with realistic correlations:
    # - Higher monthly charges → higher churn risk
    # - Longer contracts → lower churn risk
    # - Longer tenure → lower churn risk
    churn_prob = np.zeros(n_samples)

    # Base churn probability
    churn_prob += 0.3

    # Monthly charges impact (higher charges = higher churn)
    churn_prob += (monthly_charges - 70) / 200

    # Contract type impact
    churn_prob += np.where(contracts == "Month-to-month", 0.25,
                          np.where(contracts == "One year", -0.1, -0.2))

    # Tenure impact (longer tenure = lower churn)
    churn_prob -= tenure / 200

    # Clip probabilities to valid range
    churn_prob = np.clip(churn_prob, 0, 1)

    # Generate binary churn outcome
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "contract": contracts,
        "churn": churn,
    })

    return df


def save_data(df: pd.DataFrame, output_path: str = "data/raw/churn.csv"):
    """Save generated data to CSV file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Generated {len(df)} samples")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"Contract distribution:\n{df['contract'].value_counts()}")
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    df = generate_churn_data(n_samples=5000)
    save_data(df)
