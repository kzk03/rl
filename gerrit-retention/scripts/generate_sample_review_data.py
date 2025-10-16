"""
評価用のサンプルレビューデータを生成
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# パラメータ
n_reviewers = 50
n_reviews_per_reviewer = 20
start_date = datetime(2019, 1, 1)
end_date = datetime(2022, 12, 31)

data = []

for reviewer_id in range(n_reviewers):
    reviewer_email = f"reviewer{reviewer_id}@example.com"

    # レビュアーの活動期間をランダムに設定
    first_activity = start_date + timedelta(days=np.random.randint(0, 365))

    for review_id in range(n_reviews_per_reviewer):
        # レビュー日時
        days_offset = np.random.randint(0, (end_date - first_activity).days)
        request_time = first_activity + timedelta(days=days_offset)

        data.append({
            'reviewer_email': reviewer_email,
            'request_time': request_time,
            'project': f'project_{np.random.randint(1, 6)}',
        })

df = pd.DataFrame(data)
df = df.sort_values('request_time').reset_index(drop=True)

output_path = 'data/sample_reviews.csv'
df.to_csv(output_path, index=False)
print(f"サンプルデータを生成しました: {output_path}")
print(f"  件数: {len(df)}")
print(f"  期間: {df['request_time'].min()} ～ {df['request_time'].max()}")
print(f"  レビュアー数: {df['reviewer_email'].nunique()}")
