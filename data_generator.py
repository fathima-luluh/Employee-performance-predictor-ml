import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = pd.DataFrame({
    'age': np.random.randint(22, 60, n),
    'experience': np.random.randint(1, 20, n),
    'department': np.random.choice(['HR', 'IT', 'Sales'], n),
    'salary': np.random.randint(20000, 100000, n),
    'training_hours': np.random.randint(10, 100, n),
    'projects': np.random.randint(1, 10, n),
    'performance_score': np.random.randint(1, 5, n)
})

def label(row):
    if row['performance_score'] >= 4:
        return "High"
    elif row['performance_score'] == 3:
        return "Medium"
    else:
        return "Low"

data['performance'] = data.apply(label, axis=1)

data.to_csv("data/employee_data.csv", index=False)

print("✅ Dataset created successfully!")