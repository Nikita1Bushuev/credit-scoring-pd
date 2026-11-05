import numpy as np
import pandas as pd

data = pd.DataFrame(
    {
        "LIMIT_BAL": np.random.randint(10000, 500000, 100),
        "SEX": np.random.choice([1, 2], 100),
        "EDUCATION": np.random.choice([1, 2, 3, 4], 100),
        "MARRIAGE": np.random.choice([1, 2, 3], 100),
        "AGE": np.random.randint(21, 60, 100),
        "default": np.random.choice([0, 1], 100),
    }
)
data.to_csv("sample_data.csv", index=False)
