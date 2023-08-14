import numpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DShap import DShap

# 读取数据集
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]
data = pd.read_csv(data_url, names=column_names, na_values=" ?", skipinitialspace=True)

# 数据预处理
data.dropna(inplace=True)  # 删除含有缺失值的行
label_encoder = LabelEncoder()
data["income"] = label_encoder.fit_transform(data["income"])  # 将目标变量编码为数值
data = pd.get_dummies(data,
                      columns=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                               "native-country"])

# 划分数据集
X_raw = data.drop("income", axis=1)
y_raw = data["income"]
X_raw = X_raw[:5000]
y_raw = y_raw[:5000]
train_size = int(5000 * 0.8)

row_to_remove = X_raw.iloc[0]
X, y = X_raw[:train_size], y_raw[:train_size]
X_test, y_test = X_raw[train_size:], y_raw[train_size:]
print(X.iloc[1:])
print(X.shape, y.shape, X_test.shape, y_test.shape)
X = numpy.array(X)
y = numpy.array(y)
X_test = numpy.array(X_test)
y_test = numpy.array(y_test)
model = 'logistic'
problem = 'classification'
num_test = 200
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test,
              sources=None,
              sample_weight=None,
              model_family=model,
              metric='accuracy',
              overwrite=True,
              directory=directory, seed=0)

dshap.run(100, 0.1)

dshap.merge_results()

dshap.performance_plots([dshap.vals_tmc, dshap.vals_g, dshap.vals_loo], num_plot_markers=20,
                        sources=dshap.sources)
