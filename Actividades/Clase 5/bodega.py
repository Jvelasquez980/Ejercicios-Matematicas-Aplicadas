import pandas as pd 
import numpy as np
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)
X = wine.data[["alcohol", "malic_acid", "ash"]]
df_wine = X.copy()

