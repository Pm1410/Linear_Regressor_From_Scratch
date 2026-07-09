import pandas as pd 
import numpy as np

df=pd.read_csv("data/raw/raw.csv")


df["Returns"] = df["Close"]/df["Close"].shift(1) - 1 
# print(df.head(10))

ret=pd.DataFrame()
ret["Returns"]=df["Returns"]
ret["1_day_ago"]=df["Returns"].shift(1)
ret["2_day_ago"]=df["Returns"].shift(2)
ret["3_day_ago"]=df["Returns"].shift(3)
ret["4_day_ago"]=df["Returns"].shift(4)
ret["5_day_ago"]=df["Returns"].shift(5)
ret.dropna(inplace=True)

# print(ret.head(10))
print(len(ret)*8//10)
X=ret.drop(columns=["Returns"])
y=ret["Returns"]

X_train=X[:len(ret)*8//10]
X_test=X[len(ret)*8//10:]

y_train=y[:len(ret)*8//10]
y_test=y[len(ret)*8//10:]

np.save("data/train_data/X_train.npy", X_train.values)
np.save("data/train_data/X_test.npy", X_test.values)

np.save("data/train_data/y_train.npy", y_train.values)
np.save("data/train_data/y_test.npy", y_test.values)