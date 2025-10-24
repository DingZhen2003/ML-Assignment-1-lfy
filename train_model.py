# train_model.py
import numpy as np
import pandas as pd
import shap
import pickle

# train_model.py（节选修改部分）
def main():
    df = pd.read_csv('train.csv')

    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    num_cols = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    all_cols = cat_cols + num_cols

    X_raw = df[all_cols].copy()
    y = df['age'].values

    # === 新增：构造非线性特征 ===
    X_raw['balance_log'] = np.log1p(np.clip(X_raw['balance'], 0, None))
    X_raw['duration_log'] = np.log1p(np.clip(X_raw['duration'], 0, None))
    X_raw['campaign_sqrt'] = np.sqrt(np.clip(X_raw['campaign'], 0, None))
    X_raw['previous_log'] = np.log1p(np.clip(X_raw['previous'], 0, None))
    X_raw['balance_duration'] = X_raw['balance'] * X_raw['duration']
    X_raw['duration_per_campaign'] = X_raw['duration'] / (X_raw['campaign'] + 1)

    # 更新特征列表
    extended_num_cols = num_cols + [
        'balance_log', 'duration_log', 'campaign_sqrt',
        'previous_log', 'balance_duration', 'duration_per_campaign'
    ]

    # === 1. 手动预处理：One-Hot + 标准化（使用 extended_num_cols）===
    cat_encoders = {}
    encoded_features = []
    feature_names = []

    for col in cat_cols:
        unique_vals = sorted(X_raw[col].dropna().unique())
        cat_encoders[col] = {v: i for i, v in enumerate(unique_vals)}
        labels = X_raw[col].map(cat_encoders[col]).fillna(-1).astype(int)
        for idx in cat_encoders[col].values():
            encoded_features.append((labels == idx).astype(int).values)
            feature_names.append(f"{col}_{idx}")

    for col in extended_num_cols:
        encoded_features.append(X_raw[col].values.astype(float))
        feature_names.append(col)

    X = np.column_stack(encoded_features)
    num_start = len(feature_names) - len(extended_num_cols)
    num_means = np.nanmean(X[:, num_start:], axis=0)
    num_stds = np.nanstd(X[:, num_start:], axis=0)
    num_stds[num_stds == 0] = 1
    X[:, num_start:] = (X[:, num_start:] - num_means) / num_stds
    X = np.nan_to_num(X)

    # ... 后续训练、SHAP、特征选择、保存逻辑保持不变 ...

    # === 2. 梯度下降 + 早停 ===
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    lr = 0.003
    max_epochs = 30000
    patience = 50          # 容忍多少轮无显著改进
    min_delta = 3e-4       # 最小改进阈值
    best_loss = float('inf')
    patience_counter = 0
    losses = []

    print("开始训练（带早停）...")
    for epoch in range(max_epochs):
        y_pred = X @ w + b
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)

        # 早停检查
        if loss < best_loss - min_delta:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"🛑 早停触发！在 epoch {epoch} 停止训练。最后 loss: {loss:.6f}")
            break

        # 梯度更新
        dw = (2.0 / n) * (X.T @ error)
        db = (2.0 / n) * np.sum(error)
        w -= lr * dw
        b -= lr * db

        # 可选：每 500 轮打印一次
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    print(f"✅ 第一轮训练完成，最终 loss: {loss:.6f}")

    # === 3. SHAP 分析 ===
    def model_predict(X_in):
        return X_in @ w + b

    # === 3. SHAP 分析：使用 LinearExplainer（推荐）===
    explainer = shap.LinearExplainer((w, b), X)
    shap_vals = explainer(X)
    shap_imp = np.mean(np.abs(shap_vals.values), axis=0)

    # === 4. 特征选择（保留 top 80%）===
    threshold = np.percentile(shap_imp, 20)
    selected_idx = np.where(shap_imp >= threshold)[0]
    X_sel = X[:, selected_idx]

    # === 5. 用筛选后特征重新训练（同样加早停）===
    w2 = np.zeros(len(selected_idx))
    b2 = 0.0
    best_loss2 = float('inf')
    patience_counter2 = 0
    print("开始第二轮训练（特征筛选后）...")

    for epoch in range(max_epochs):
        y_pred = X_sel @ w2 + b2
        error = y_pred - y
        loss = np.mean(error ** 2)

        if loss < best_loss2 - min_delta:
            best_loss2 = loss
            patience_counter2 = 0
        else:
            patience_counter2 += 1

        if patience_counter2 >= patience:
            print(f"🛑 第二轮早停！epoch {epoch}, loss: {loss:.6f}")
            break

        dw = (2.0 / n) * (X_sel.T @ error)
        db = (2.0 / n) * np.sum(error)
        w2 -= lr * dw
        b2 -= lr * db

    print(f"✅ 第二轮训练完成，最终 loss: {loss:.6f}")

    # === 6. 保存模型 ===
    artifacts = {
        'weights': w2,
        'bias': b2,
        'cat_encoders': cat_encoders,
        'num_means': num_means,
        'num_stds': num_stds,
        'selected_idx': selected_idx,
        'num_numerical': len(num_cols),
        'cat_cols': cat_cols,
        'num_cols': num_cols
    }

    with open('model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

    print("✅ 模型已保存至 model.pkl")

if __name__ == '__main__':
    main()