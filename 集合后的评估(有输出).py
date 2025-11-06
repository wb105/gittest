import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 导入必要的库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

# 特征处理和转换
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
# --- 新增集成相关库 ---
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# --- 1. 读取数据 ---
# 请将路径替换为您实际的文件路径
train_data_path = 'train.csv' # 例如: r'C:\Users\YourName\Desktop\train.csv'
test_data_path = 'test.csv'   # 例如: r'C:\Users\YourName\Desktop\test.csv'

try:
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    print("数据读取成功。")
except FileNotFoundError as e:
    print(f"错误：找不到文件。请检查路径是否正确。{e}")
    exit(1)
except Exception as e:
    print(f"读取数据时发生错误: {e}")
    exit(1)

# 删除无关变量
drop_cols = ['Over18','StandardHours','EmployeeNumber']
train_data = train_data.drop(columns=[col for col in drop_cols if col in train_data.columns])
test_data = test_data.drop(columns=[col for col in drop_cols if col in test_data.columns])

X_train_full = train_data.drop('Attrition', axis=1)
X_test_full = test_data.drop('Attrition', axis=1)
y_train_full = train_data['Attrition']
y_test_full = test_data['Attrition']

# --- 2. 特征工程 (保持优化) ---

def create_features(df):
    """创建新特征"""
    df = df.copy()
    # 确保参与运算的列是数值型
    numeric_cols_for_ops = ['YearsAtCompany', 'TotalWorkingYears', 'YearsInCurrentRole', 'NumCompaniesWorked',
                            'JobSatisfaction', 'EnvironmentSatisfaction', 'Age', 'DistanceFromHome', 'MonthlyIncome']
    for col in numeric_cols_for_ops:
         if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce') # 转换为数值型，非数值设为NaN

    # 1. 工作年限与公司年限的比率 (加1避免除零)
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        df['YearsAtCompanyRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    # 2. 当前职位在公司工作年限与总工作年限的比率
    if 'YearsInCurrentRole' in df.columns and 'TotalWorkingYears' in df.columns:
        df['YearsInCurrentRoleRatio'] = df['YearsInCurrentRole'] / (df['TotalWorkingYears'] + 1)
    # 3. 晋升次数与在公司工作年限的比率 (加1避免除零)
    if 'NumCompaniesWorked' in df.columns and 'YearsAtCompany' in df.columns:
        df['PromotionRatio'] = df['NumCompaniesWorked'] / (df['YearsAtCompany'] + 1)
    # 4. 工作满意度 * 环境满意度
    if 'JobSatisfaction' in df.columns and 'EnvironmentSatisfaction' in df.columns:
        df['SatisfactionInteraction'] = df['JobSatisfaction'] * df['EnvironmentSatisfaction']
    # 5. 年龄与工作年限的比率
    if 'Age' in df.columns and 'TotalWorkingYears' in df.columns:
        df['AgeToWorkingYearsRatio'] = df['Age'] / (df['TotalWorkingYears'] + 1)
    # 6. 月收入与在公司年限的比率 (作为潜在收入增长指标)
    if 'MonthlyIncome' in df.columns and 'YearsAtCompany' in df.columns:
        df['IncomeGrowthPotential'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    # 7. 距离家的距离与环境满意度交互
    if 'DistanceFromHome' in df.columns and 'EnvironmentSatisfaction' in df.columns:
        df['DistHome_Satisfaction'] = df['DistanceFromHome'] * df['EnvironmentSatisfaction']

    return df

def encode_categorical_target(df_train_with_target, df_train_features, df_test_features, target_col, categorical_cols):
    """对类别特征进行目标编码，并处理未知类别"""
    df_train_encoded = df_train_features.copy()
    df_test_encoded = df_test_features.copy()

    for col in categorical_cols:
        if col not in df_train_encoded.columns or col not in df_test_encoded.columns:
            print(f"警告: 列 '{col}' 在训练集或测试集的特征中不存在，跳过编码。")
            continue

        # 计算训练集（包含目标变量）的目标编码
        # 使用较小的平滑因子，减少对低频类别的平滑
        global_mean = df_train_with_target[target_col].mean()
        agg = df_train_with_target.groupby(col)[target_col].agg(['count', 'mean'])
        smoothing_factor = 5 # 减小平滑因子
        smooth_mean = (agg['count'] * agg['mean'] + smoothing_factor * global_mean) / (agg['count'] + smoothing_factor)
        encoding_map = smooth_mean.to_dict()

        # 应用到训练集特征 (使用map)
        df_train_encoded[col] = df_train_encoded[col].map(encoding_map).fillna(global_mean)

        # 应用到测试集特征 (处理未知类别)
        df_test_encoded[col] = df_test_encoded[col].map(encoding_map).fillna(global_mean)

    return df_train_encoded, df_test_encoded

def handle_skewness(df_train, df_test, numeric_cols):
    """处理数值特征的偏态"""
    df_train_skewed = df_train.copy()
    df_test_skewed = df_test.copy()

    skewed_features = []
    for col in numeric_cols:
        if col not in df_train_skewed.columns: # 确保列存在
            continue
        # 确保列是数值型
        df_train_skewed[col] = pd.to_numeric(df_train_skewed[col], errors='coerce')
        skewness = df_train_skewed[col].skew()
        if abs(skewness) > 0.5:  # 设定偏态阈值
            # 对训练集进行变换
            df_train_skewed[col] = np.log1p(df_train_skewed[col].clip(lower=0))
            # 对测试集也应用相同的变换
            df_test_skewed[col] = np.log1p(df_test_skewed[col].clip(lower=0))
            skewed_features.append(col)
    print(f"进行对数变换的特征: {skewed_features}")
    return df_train_skewed, df_test_skewed, skewed_features

def select_features_rf(X_train, y_train, X_test, n_features_to_select=100):
    """使用随机森林选择特征"""
    # 确保X_train, y_train, X_test都是数值型
    X_train_rf = X_train.select_dtypes(include=[np.number])
    X_test_rf = X_test.select_dtypes(include=[np.number])
    # 确保列一致
    common_cols = X_train_rf.columns.intersection(X_test_rf.columns)
    if len(common_cols) == 0:
        print("错误：训练集和测试集没有共同的数值型特征。")
        return X_train_rf, X_test_rf, X_train_rf.columns
    X_train_rf = X_train_rf[common_cols]
    X_test_rf = X_test_rf[common_cols]

    n_features_to_select = min(n_features_to_select, len(common_cols), len(X_train_rf))
    if n_features_to_select <= 0:
        print("错误：特征选择数量无效。")
        return X_train_rf, X_test_rf, common_cols

    selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector.fit(X_train_rf, y_train)

    # 获取特征重要性
    feature_importances = selector.feature_importances_
    feature_names = X_train_rf.columns

    # 获取重要性最高的特征索引
    top_indices = np.argsort(feature_importances)[::-1][:n_features_to_select]

    selected_features = feature_names[top_indices]

    return X_train_rf[selected_features], X_test_rf[selected_features], selected_features

# 1. 创建新特征
X_train_full_with_features = create_features(X_train_full)
X_test_full_with_features = create_features(X_test_full)

# 2. 分离类别特征和数值特征
nominal_features = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']

# 3. 目标编码 (使用包含目标的原始训练集来计算映射)
X_train_encoded_with_features, X_test_encoded_with_features = encode_categorical_target(train_data, X_train_full_with_features, X_test_full_with_features, 'Attrition', nominal_features)

# 4. 更新特征列表 (此时类别特征已被编码为数值)
all_features_after_encoding_with = X_train_encoded_with_features.columns.tolist()

# 5. 处理偏态 (现在所有特征都应是数值型)
X_train_skewed_with, X_test_skewed_with, skewed_features_list_with = handle_skewness(X_train_encoded_with_features, X_test_encoded_with_features, all_features_after_encoding_with)

# --- 评估特征有效性 ---
# --- A. 准备不包含新特征的数据集 (仅包含原始特征和经过编码/变换的特征) ---
# 重新处理不添加新特征的版本
X_train_full_without_features = X_train_full # 不调用 create_features
X_test_full_without_features = X_test_full

# 目标编码
X_train_encoded_without_features, X_test_encoded_without_features = encode_categorical_target(train_data, X_train_full_without_features, X_test_full_without_features, 'Attrition', nominal_features)

# 更新特征列表
all_features_after_encoding_without = X_train_encoded_without_features.columns.tolist()

# 处理偏态
X_train_skewed_without, X_test_skewed_without, skewed_features_list_without = handle_skewness(X_train_encoded_without_features, X_test_encoded_without_features, all_features_after_encoding_without)

# --- B. 特征选择 (分别对有/无新特征的数据集进行) ---
try:
    # 对包含新特征的数据集进行特征选择
    X_train_selected_with, X_test_selected_with, selected_feature_names_with = select_features_rf(X_train_skewed_with, y_train_full, X_test_skewed_with, n_features_to_select=40) # 选择40个特征
    print(f"特征选择完成 (包含新特征)，选择的特征数量: {len(selected_feature_names_with)}")
    print(f"选择的特征 (包含新特征): {list(selected_feature_names_with)}")
    # 确定哪些新特征被选中
    original_feature_names = set(X_train_encoded_without_features.columns) # 原始特征名集合
    selected_new_features = [f for f in selected_feature_names_with if f not in original_feature_names]
    print(f"被选中的新特征: {selected_new_features}")
    
    # 对不包含新特征的数据集进行特征选择
    X_train_selected_without, X_test_selected_without, selected_feature_names_without = select_features_rf(X_train_skewed_without, y_train_full, X_test_skewed_without, n_features_to_select=40) # 选择40个特征
    print(f"特征选择完成 (不包含新特征)，选择的特征数量: {len(selected_feature_names_without)}")
    print(f"选择的特征 (不包含新特征): {list(selected_feature_names_without)}")

except Exception as e:
    print(f"特征选择过程中发生错误: {e}")
    # 如果特征选择失败，回退到使用所有处理后的特征
    X_train_selected_with = X_train_skewed_with
    X_test_selected_with = X_test_skewed_with
    selected_feature_names_with = X_train_skewed_with.columns
    X_train_selected_without = X_train_skewed_without
    X_test_selected_without = X_test_skewed_without
    selected_feature_names_without = X_train_skewed_without.columns
    print(f"使用所有特征 (包含新特征): {len(selected_feature_names_with)}")
    print(f"使用所有特征 (不包含新特征): {len(selected_feature_names_without)}")


# 7. 标准化 (使用更稳健的RobustScaler) - 对两个数据集分别进行
X_train_scaled_df_with = X_train_selected_with.select_dtypes(include=[np.number])
X_test_scaled_df_with = X_test_selected_with.select_dtypes(include=[np.number])

X_train_scaled_df_without = X_train_selected_without.select_dtypes(include=[np.number])
X_test_scaled_df_without = X_test_selected_without.select_dtypes(include=[np.number])

scaler_with = RobustScaler()
X_train_scaled_with = pd.DataFrame(
    scaler_with.fit_transform(X_train_scaled_df_with),
    columns=X_train_scaled_df_with.columns,
    index=X_train_scaled_df_with.index
)
X_test_scaled_with = pd.DataFrame(
    scaler_with.transform(X_test_scaled_df_with),
    columns=X_test_scaled_df_with.columns,
    index=X_test_scaled_df_with.index
)

scaler_without = RobustScaler()
X_train_scaled_without = pd.DataFrame(
    scaler_without.fit_transform(X_train_scaled_df_without),
    columns=X_train_scaled_df_without.columns,
    index=X_train_scaled_df_without.index
)
X_test_scaled_without = pd.DataFrame(
    scaler_without.transform(X_test_scaled_df_without),
    columns=X_test_scaled_df_without.columns,
    index=X_test_scaled_df_without.index
)

# 8. 计算 scale_pos_weight 用于处理类别不平衡
# 计算少数类 (Attrition=1) 和多数类 (Attrition=0) 的样本数量
neg, pos = np.bincount(y_train_full)
# scale_pos_weight = 负例数量 / 正例数量
scale_pos_weight = neg / pos
print(f"计算得到的 scale_pos_weight: {scale_pos_weight:.2f}")

# --- 3. 模型训练与评估 (比较有无新特征) ---
# 定义一个基础模型配置
def get_base_xgb_model():
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

# --- 模型A: 使用包含新特征的数据集 ---
model_with_features = get_base_xgb_model()
model_with_features.fit(X_train_scaled_with, y_train_full)

y_pred_proba_with = model_with_features.predict_proba(X_test_scaled_with)[:, 1]
y_pred_with = (y_pred_proba_with >= 0.5).astype(int)

# 评估模型A
test_auc_with = roc_auc_score(y_test_full, y_pred_proba_with)
test_acc_with = accuracy_score(y_test_full, y_pred_with)
test_f1_with = f1_score(y_test_full, y_pred_with)

# 过拟合评估A
y_train_pred_proba_with = model_with_features.predict_proba(X_train_scaled_with)[:, 1]
y_train_pred_with = (y_train_pred_proba_with >= 0.5).astype(int)
train_auc_with = roc_auc_score(y_train_full, y_train_pred_proba_with)
train_acc_with = accuracy_score(y_train_full, y_train_pred_with)
train_f1_with = f1_score(y_train_full, y_train_pred_with)

gap_auc_with = train_auc_with - test_auc_with
gap_acc_with = train_acc_with - test_acc_with
gap_f1_with = train_f1_with - test_f1_with

print("\n--- 模型性能比较 ---")
print("\n模型A (包含新特征):")
print(f"Train ROC AUC: {train_auc_with:.4f} | Test ROC AUC: {test_auc_with:.4f} | Gap(AUC): {gap_auc_with:+.4f}")
print(f"Train Accuracy: {train_acc_with:.4f} | Test Accuracy: {test_acc_with:.4f} | Gap(ACC): {gap_acc_with:+.4f}")
print(f"Train F1: {train_f1_with:.4f} | Test F1: {test_f1_with:.4f} | Gap(F1): {gap_f1_with:+.4f}")
print(f"Test AUC: {test_auc_with:.4f}, Test Accuracy: {test_acc_with:.4f}, Test F1: {test_f1_with:.4f}")

# --- 模型B: 使用不包含新特征的数据集 ---
model_without_features = get_base_xgb_model()
model_without_features.fit(X_train_scaled_without, y_train_full)

y_pred_proba_without = model_without_features.predict_proba(X_test_scaled_without)[:, 1]
y_pred_without = (y_pred_proba_without >= 0.5).astype(int)

# 评估模型B
test_auc_without = roc_auc_score(y_test_full, y_pred_proba_without)
test_acc_without = accuracy_score(y_test_full, y_pred_without)
test_f1_without = f1_score(y_test_full, y_pred_without)

# 过拟合评估B
y_train_pred_proba_without = model_without_features.predict_proba(X_train_scaled_without)[:, 1]
y_train_pred_without = (y_train_pred_proba_without >= 0.5).astype(int)
train_auc_without = roc_auc_score(y_train_full, y_train_pred_proba_without)
train_acc_without = accuracy_score(y_train_full, y_train_pred_without)
train_f1_without = f1_score(y_train_full, y_train_pred_without)

gap_auc_without = train_auc_without - test_auc_without
gap_acc_without = train_acc_without - test_acc_without
gap_f1_without = train_f1_without - test_f1_without

print("\n模型B (不包含新特征):")
print(f"Train ROC AUC: {train_auc_without:.4f} | Test ROC AUC: {test_auc_without:.4f} | Gap(AUC): {gap_auc_without:+.4f}")
print(f"Train Accuracy: {train_acc_without:.4f} | Test Accuracy: {test_acc_without:.4f} | Gap(ACC): {gap_acc_without:+.4f}")
print(f"Train F1: {train_f1_without:.4f} | Test F1: {test_f1_without:.4f} | Gap(F1): {gap_f1_without:+.4f}")
print(f"Test AUC: {test_auc_without:.4f}, Test Accuracy: {test_acc_without:.4f}, Test F1: {test_f1_without:.4f}")

# --- 比较结论 ---
print("\n--- 特征有效性评估 ---")
print(f"包含新特征的模型测试集 AUC: {test_auc_with:.4f}")
print(f"不包含新特征的模型测试集 AUC: {test_auc_without:.4f}")
if test_auc_with > test_auc_without:
    print("结论: 添加新特征后，测试集AUC提升，新特征可能有效。")
elif test_auc_with < test_auc_without:
    print("结论: 添加新特征后，测试集AUC下降，新特征可能无效或有害。")
else:
    print("结论: 添加新特征后，测试集AUC无变化，新特征可能没有提供额外信息。")

print(f"包含新特征的模型测试集 F1: {test_f1_with:.4f}")
print(f"不包含新特征的模型测试集 F1: {test_f1_without:.4f}")
if test_f1_with > test_f1_without:
    print("结论: 添加新特征后，测试集F1提升，新特征可能有效。")
elif test_f1_with < test_f1_without:
    print("结论: 添加新特征后，测试集F1下降，新特征可能无效或有害。")
else:
    print("结论: 添加新特征后，测试集F1无变化，新特征可能没有提供额外信息。")

print(f"包含新特征的模型 AUC Gap: {gap_auc_with:+.4f}")
print(f"不包含新特征的模型 AUC Gap: {gap_auc_without:+.4f}")
if abs(gap_auc_with) < abs(gap_auc_without):
    print("结论: 添加新特征后，AUC过拟合程度（Gap）减小，新特征可能有助于泛化。")
elif abs(gap_auc_with) > abs(gap_auc_without):
    print("结论: 添加新特征后，AUC过拟合程度（Gap）增大，新特征可能加剧了过拟合。")
else:
    print("结论: 添加新特征后，AUC过拟合程度（Gap）无变化。")

# --- C. 特征重要性分析 (使用包含新特征的模型) ---
print("\n--- 特征重要性分析 (包含新特征的模型) ---")
# 为了方便查看，我们使用特征选择后的特征名
feature_names = X_train_scaled_with.columns
importances = model_with_features.feature_importances_

# 创建特征名和重要性的DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("前10个最重要的特征:")
print(importance_df.head(30))

# 检查新特征的重要性
new_feature_names_set = set(['YearsAtCompanyRatio', 'YearsInCurrentRoleRatio', 'PromotionRatio', 'SatisfactionInteraction', 'AgeToWorkingYearsRatio', 'IncomeGrowthPotential', 'DistHome_Satisfaction'])
new_feature_importances = importance_df[importance_df['feature'].isin(new_feature_names_set)]
if not new_feature_importances.empty:
    print("\n新特征的重要性:")
    print(new_feature_importances)
else:
    print("\n在模型训练的特征中未找到新特征。")


'''
--- 模型性能比较 ---

模型A (包含新特征):
Train ROC AUC: 0.9548 | Test ROC AUC: 0.8468 | Gap(AUC): +0.1080
Train Accuracy: 0.8855 | Test Accuracy: 0.8200 | Gap(ACC): +0.0655
Train F1: 0.7123 | Test F1: 0.5191 | Gap(F1): +0.1932
Test AUC: 0.8468, Test Accuracy: 0.8200, Test F1: 0.5191

模型B (不包含新特征):
Train ROC AUC: 0.9491 | Test ROC AUC: 0.8484 | Gap(AUC): +0.1007
Train Accuracy: 0.8782 | Test Accuracy: 0.8286 | Gap(ACC): +0.0496
Train F1: 0.6955 | Test F1: 0.5385 | Gap(F1): +0.1570
Test AUC: 0.8484, Test Accuracy: 0.8286, Test F1: 0.5385

--- 特征有效性评估 ---
包含新特征的模型测试集 AUC: 0.8468
不包含新特征的模型测试集 AUC: 0.8484
结论: 添加新特征后，测试集AUC下降，新特征可能无效或有害。
包含新特征的模型测试集 F1: 0.5191
不包含新特征的模型测试集 F1: 0.5385
结论: 添加新特征后，测试集F1下降，新特征可能无效或有害。
包含新特征的模型 AUC Gap: +0.1080
不包含新特征的模型 AUC Gap: +0.1007
结论: 添加新特征后，AUC过拟合程度（Gap）增大，新特征可能加剧了过拟合。
'''
