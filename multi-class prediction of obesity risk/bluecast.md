# BlueCast
- discussion: https://www.kaggle.com/competitions/playground-series-s4e2/discussion/472471
- notebook: https://www.kaggle.com/code/thomasmeiner/ps4e2-eda-feature-engineering-modelling/notebook?scriptVersionId=161250535
- documentation: https://github.com/ThomasMeissnerDS/BlueCast?tab=readme-ov-file#general-usage
- bluecast:an eda library


## 画图相关工具使用
```
explanatory analysis:（画图）
from bluecast.eda.analyse import (
    bi_variate_plots,#每个因素与target的小提琴图
    univariate_plots,#每个因素的柱状图和箱型图
    plot_count_pairs,#每个因素的不同值的train、test数据含量对比柱状图
    correlation_heatmap,#各个因素各自之间的相关性
    correlation_to_target,#每个因素与target的相关性
    plot_pca,#看出target的各个值的分离程度
    plot_tsne,#看出target的各个值的分离程度
    plot_pca_cumulative_variance,
    plot_theil_u_heatmap,
    check_unique_values,
    plot_null_percentage,#null值的分布图
    mutual_info_to_target,#不同因素对target值得信息增益mutual information score
    plot_pie_chart,#target的不同值的占比，画个饼图
)
```
```
from bluecast.blueprints.cast import BlueCast
automl=BlueCast(
    class_problem='multiclass',用各种模型得到拟合效果最好的模型
    target_column=target
)
automl.fit(train_data,target_col=target)
最后画出一张图，用不同的颜色，
从上到下排序不同因素对target的影响程度
```
## 检测是否发生数据泄露
```
from bluecast.eda.data_leakage_checks import (
    detect_categorical_leakage,detect_leakage_via_correlation
)
```

## 训练参数设置
```
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from sklearn.model_selection import RepeatedStratifiedKFold
```
### XgboostTuneParamsConfig
```
xgboost_param_config = XgboostTuneParamsConfig()
xgboost_param_config.steps_max = 1000 #epochs数量：过多则浪费计算资源，过拟合风险；过少可能欠拟合
xgboost_param_config.max_depth_max = 7 #树的深度：过深有过拟合风险
```
### TrainingConfig

```
# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.global_random_state = 600 #全局随机状态：保证实验可重复
train_config.hypertuning_cv_folds = 1 #交叉验证的分割数量：一般是5，增加计算成本，同时减少过拟合风险
train_config.hyperparameter_tuning_rounds = 500 #超参数调优的轮数：500意味着将尝试500种不同的超参数组合
train_config.hyperparameter_tuning_max_runtime_secs = 60 * 60 * 2 #最大运行时间为2小时
#train_config.enable_grid_search_fine_tuning = True #启用精细调优，模型将在更细致的参数网格上进行搜索
train_config.use_full_data_for_final_model = True 
#使用全部数据训练：最终模型可能会增加过拟合的风险，因为模型可能会过度学习训练数据中的噪声。
#如果设置为False，模型可能在一定程度上避免过拟合。
#如果最终模型只在部分数据上训练，那么模型的稳定性可能会受到影响，因为它可能没有充分学习到数据的整体分布。
train_config.precise_cv_tuning = True #启用精确的交叉验证调优可能意味着在调优过程中会更加精确地评估模型性能。
train_config.gridsearch_nb_parameters_per_grid = 5 #在搜索最佳超参数组合时，每个超参数要尝试多少个不同的值。
#train_config.cat_encoding_via_ml_algorithm = True 
# 在机器学习中，分类变量（也称为类别特征或名义特征）是那些代表不同类别或组的数据，比如性别（男、女）、颜色（红、蓝、绿）或者国家（美国、中国、英国）。这些类别数据通常是非数值型的，比如文字或者标签。
# 然而，大多数机器学习模型只能处理数值型数据。为了让这些模型能够理解和使用分类变量，我们需要将这些类别数据转换成数值型数据。这个过程就叫做“编码”。
# 启用机器学习算法进行分类变量编码，就像是给每个类别分配一个唯一的数字代码。这样，模型就可以用这些数字来代替原来的类别标签，从而进行数学计算和学习。
# 有几种常见的编码方法，比如：
# 独热编码（One-Hot Encoding）：这种方法会给每个类别创建一个新的二进制列（特征），如果原始数据中的类别对应这个特征，则该列的值为1，否则为0。例如，如果有三种颜色，那么每种颜色都会有一个对应的二进制特征，模型就会有三个新的特征来表示颜色。
# 标签编码（Label Encoding）：这种方法会将每个类别映射到一个整数值。例如，颜色“红”可能被编码为1，“蓝”被编码为2，“绿”被编码为3。这种方法在某些情况下可能会导致模型错误地认为数值之间有大小关系，比如认为2比1大，这在颜色编码中是不正确的。
# 目标编码（Target Encoding）：这种方法使用目标变量的平均值来表示类别。例如，如果某个类别与高目标变量值相关联，那么这个类别就会被赋予较高的数值。这种方法可以保留类别特征的有用信息，但也需要谨慎使用，以避免过拟合。
# 启用机器学习算法进行编码，就是选择一种或多种这些方法，将分类变量转换成模型可以理解的数值型数据。这样做可以让模型更好地学习和预测，同时也有助于提高模型的准确性和泛化能力。
#train_config.calculate_shap_values = False

skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1987)
```
