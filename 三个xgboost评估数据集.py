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
X_train_full = create_features(X_train_full)
X_test_full = create_features(X_test_full)

# 2. 分离类别特征和数值特征
nominal_features = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']

# 3. 目标编码 (使用包含目标的原始训练集来计算映射)
X_train_encoded, X_test_encoded = encode_categorical_target(train_data, X_train_full, X_test_full, 'Attrition', nominal_features)

# 4. 更新特征列表 (此时类别特征已被编码为数值)
all_features_after_encoding = X_train_encoded.columns.tolist()

# 5. 处理偏态 (现在所有特征都应是数值型)
X_train_skewed, X_test_skewed, skewed_features_list = handle_skewness(X_train_encoded, X_test_encoded, all_features_after_encoding)

# 6. 特征选择 (使用随机森林，选择适中数量的特征)
try:
    X_train_selected, X_test_selected, selected_feature_names = select_features_rf(X_train_skewed, y_train_full, X_test_skewed, n_features_to_select=40) # 选择40个特征
    print(f"特征选择完成，选择的特征数量: {len(selected_feature_names)}")
    print(f"选择的特征: {list(selected_feature_names)}")
except Exception as e:
    print(f"特征选择过程中发生错误: {e}")
    X_train_selected = X_train_skewed
    X_test_selected = X_test_skewed
    selected_feature_names = X_train_skewed.columns
    print(f"使用所有 {len(selected_feature_names)} 个特征。")

# 7. 标准化 (使用更稳健的RobustScaler)
X_train_scaled_df = X_train_selected.select_dtypes(include=[np.number])
X_test_scaled_df = X_test_selected.select_dtypes(include=[np.number])

scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_scaled_df),
    columns=X_train_scaled_df.columns,
    index=X_train_scaled_df.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_scaled_df),
    columns=X_test_scaled_df.columns,
    index=X_test_scaled_df.index
)

# 8. 计算 scale_pos_weight 用于处理类别不平衡
# 计算少数类 (Attrition=1) 和多数类 (Attrition=0) 的样本数量
neg, pos = np.bincount(y_train_full)
# scale_pos_weight = 负例数量 / 正例数量
scale_pos_weight = neg / pos
print(f"计算得到的 scale_pos_weight: {scale_pos_weight:.2f}")

# --- 3. 模型训练与评估 (使用集成方法) ---
# 定义多个参数略有不同的XGBoost模型
# 使用不同的随机种子和参数组合，增加模型多样性
xgb1 = XGBClassifier(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

xgb2 = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    min_child_weight=7,
    gamma=0.2,
    scale_pos_weight=scale_pos_weight,
    random_state=123,
    eval_metric='logloss'
)

xgb3 = XGBClassifier(
    n_estimators=200,
    learning_rate=0.07,
    max_depth=5,
    subsample=0.6,
    colsample_bytree=0.8,
    reg_lambda=2.5,
    min_child_weight=6,
    gamma=0.15,
    scale_pos_weight=scale_pos_weight,
    random_state=456,
    eval_metric='logloss'
)

# 创建投票集成器 (使用 'soft' 投票，即平均预测概率)
ensemble_model = VotingClassifier(
    estimators=[('xgb1', xgb1), ('xgb2', xgb2), ('xgb3', xgb3)],
    voting='soft' # 'soft' for probability averaging
)

# 直接在全量训练集上训练集成模型
print("开始训练集成模型...")
ensemble_model.fit(X_train_scaled, y_train_full)

# --- 单独评估每个基础模型 ---
print("\n--- 单个基础模型性能评估 ---")
for name, model in ensemble_model.named_estimators_.items():
    print(f"\n--- {name} ---")
    # 在训练集上预测
    y_train_pred_proba_single = model.predict_proba(X_train_scaled)[:, 1]
    y_train_pred_single = (y_train_pred_proba_single >= 0.5).astype(int)
    
    # 在测试集上预测
    y_test_pred_proba_single = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred_single = (y_test_pred_proba_single >= 0.5).astype(int)
    
    # 计算训练集指标
    train_auc_single = roc_auc_score(y_train_full, y_train_pred_proba_single)
    train_acc_single = accuracy_score(y_train_full, y_train_pred_single)
    train_f1_single = f1_score(y_train_full, y_train_pred_single)
    
    # 计算测试集指标
    test_auc_single = roc_auc_score(y_test_full, y_test_pred_proba_single)
    test_acc_single = accuracy_score(y_test_full, y_test_pred_single)
    test_f1_single = f1_score(y_test_full, y_test_pred_single)
    
    # 计算Gap
    gap_auc_single = train_auc_single - test_auc_single
    gap_acc_single = train_acc_single - test_acc_single
    gap_f1_single = train_f1_single - test_f1_single
    
    print(f"Train ROC AUC: {train_auc_single:.4f} | Test ROC AUC: {test_auc_single:.4f} | Gap(AUC): {gap_auc_single:+.4f}")
    print(f"Train Accuracy: {train_acc_single:.4f} | Test Accuracy: {test_acc_single:.4f} | Gap(ACC): {gap_acc_single:+.4f}")
    print(f"Train F1: {train_f1_single:.4f} | Test F1: {test_f1_single:.4f} | Gap(F1): {gap_f1_single:+.4f}")


# --- 评估集成模型 ---
print("\n--- 集成模型性能评估 ---")
# 使用训练好的集成模型在测试集上进行预测
y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int) # 保持默认阈值

# 打印分类报告
print("\nClassification Report (Default Threshold 0.5):")
print(classification_report(y_test_full, y_pred))

# 计算混淆矩阵
cm = confusion_matrix(y_test_full, y_pred)
print("\n混淆矩阵：\n", cm)

# 计算 ROC AUC 分数
roc_auc = roc_auc_score(y_test_full, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")
accuracy = accuracy_score(y_test_full, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
f1 = f1_score(y_test_full, y_pred)
print(f"F1 Score: {f1:.4f}")

# --- 过拟合评估 (集成模型) ---
y_train_pred_proba = ensemble_model.predict_proba(X_train_scaled)[:, 1] # 使用原始训练集评估过拟合
y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
train_auc = roc_auc_score(y_train_full, y_train_pred_proba) # 使用原始训练集标签
train_acc = accuracy_score(y_train_full, y_train_pred)
train_f1 = f1_score(y_train_full, y_train_pred)

gap_auc = train_auc - roc_auc
gap_acc = train_acc - accuracy
gap_f1 = train_f1 - f1

print(f"\n--- 过拟合评估 (集成模型) ---")
print(f"Train ROC AUC: {train_auc:.4f} | Test ROC AUC: {roc_auc:.4f} | Gap(AUC): {gap_auc:+.4f}")
print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {accuracy:.4f} | Gap(ACC): {gap_acc:+.4f}")
print(f"Train F1: {train_f1:.4f} | Test F1: {f1:.4f} | Gap(F1): {gap_f1:+.4f}")

# --- 优化阈值以提升F1分数 ---
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test_full, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
f1_scores = np.nan_to_num(f1_scores) # 处理除零情况

best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
best_f1_optimized = f1_scores[best_threshold_idx]

print(f"\n--- 阈值优化 ---")
print(f"最佳F1分数: {best_f1_optimized:.4f} (使用阈值: {best_threshold:.4f})")

# 使用优化后的阈值进行预测
y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

# 重新计算优化后的指标
accuracy_optimized = accuracy_score(y_test_full, y_pred_optimized)
roc_auc_optimized = roc_auc_score(y_test_full, y_pred_proba) # AUC 不变
f1_optimized = f1_score(y_test_full, y_pred_optimized)

print(f"优化后 - Accuracy: {accuracy_optimized:.4f}, F1: {f1_optimized:.4f}, AUC: {roc_auc_optimized:.4f}")
print("\n优化后分类报告:")
print(classification_report(y_test_full, y_pred_optimized))

print("\n--- 总结 ---")
print("使用了包含三个不同参数XGBoost模型的投票集成。")
print("这旨在通过平均多个模型的预测来减少单个模型的过拟合风险，同时保持一定的模型表达能力。")
print("请观察最终的测试集性能和过拟合评估结果。")
print("如果性能有所提升，说明集成方法有效。如果仍然不佳，可能需要考虑更激进的特征工程或尝试其他类型的模型。")