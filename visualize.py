import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/employee_data.csv")

sns.countplot(x='performance', data=df)
plt.savefig("images/performance_distribution.png")
plt.show()