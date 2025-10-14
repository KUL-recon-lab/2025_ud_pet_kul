import pandas as pd
import numpy as np
from pathlib import Path

df_all_data = pd.read_csv(Path("../data_stats/") / "data_stats.csv")

df_screened = pd.read_csv("mips_screening_cc.csv")

# In [7]: print(df_screened.columns)
# Index(['case', 'cat', 'cat_name'], dtype='object')

# count how many cases we have per "cat_name", print a nice summary
print(df_screened["cat_name"].value_counts())
