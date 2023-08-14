from DShap import DShap
from shap_utils import *

MEM_DIR = './'
problem, model = 'classification', 'logistic'
hidden_units = []  # Empty list in the case of logistic regression.
train_size = 100
d, difficulty = 50, 1
num_classes = 2
tol = 0.03
target_accuracy = 0.7
important_dims = 5
# 这里是在返回分类器
clf = return_model(model, solver='liblinear', hidden_units=tuple(hidden_units))
_param = 1.0
for _ in range(100):
    # 生成一个具有多变量正太分布的随机样本集合,mean代表在所有维度上均为0的均值向量,d是维度,cov是多变量正态分布的协方差矩阵,表示在所有维度上协方差为1的单位矩阵.是的生成的随机样本之间是相互独立的,size是随机样本的数量,得到数量的维度是(10000,50)
    X_raw = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d),
                                          size=train_size + 5000)

    # 这里的y_raw就是根据特定任务生成的对应的标签
    _, y_raw, _, _ = label_generator(
        problem, X_raw, param=_param, difficulty=difficulty, important=important_dims)
    # 使用sklearn中的模型工具来对数据进行训练
    clf.fit(X_raw[:train_size], y_raw[:train_size])
    # 得到测试的精确度
    test_acc = clf.score(X_raw[train_size:], y_raw[train_size:])
    if test_acc > target_accuracy:
        break
    # 使得训练得到的模型能够大于目标的正确率,如果不大于的话就对这个参数进行更新,我觉得这里是未来保证生成的数据能用
    _param *= 1.1
print('Performance using the whole training set = {0:.2f}'.format(test_acc))
print(X_raw.shape, y_raw.shape)

X, y = X_raw[:train_size], y_raw[:train_size]
X_test, y_test = X_raw[train_size:], y_raw[train_size:]
model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test,
              sources=None,
              sample_weight=None,
              model_family=model,
              metric='accuracy',
              overwrite=True,
              directory=directory, seed=0)
dshap.run(100, 0.1, g_run=False)

X, y = X_raw[:100], y_raw[:100]
X_test, y_test = X_raw[100:], y_raw[100:]
model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test, model_family=model, metric='accuracy',
              directory=directory, seed=1)
dshap.run(100, 0.1)

X, y = X_raw[:100], y_raw[:100]
X_test, y_test = X_raw[100:], y_raw[100:]
model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test, model_family=model, metric='accuracy',
              directory=directory, seed=2)
dshap.run(100, 0.1)

dshap.merge_results()

dshap.performance_plots([dshap.vals_tmc, dshap.vals_g, dshap.vals_loo], num_plot_markers=20,
                        sources=dshap.sources)
