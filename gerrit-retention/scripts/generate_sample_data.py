#!/usr/bin/env python3
"""
リアルなサンプルデータ生成スクリプト
実際のGerritプロジェクトのような構造のデータを生成
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


def generate_realistic_sample_data():
    """リアルなサンプルデータを生成"""
    
    # 開発者データ
    developers = [
        {
            "developer_id": "john.doe@example.com",
            "name": "John Doe",
            "join_date": "2023-01-15",
            "expertise_areas": ["Python", "Machine Learning", "API Design"],
            "experience_level": 0.85,
            "collaboration_score": 0.78,
            "satisfaction_level": 0.72,
            "stress_accumulation": 0.35,
            "total_reviews": 156,
            "avg_response_time": 4.2,
            "retention_probability": 0.88
        },
        {
            "developer_id": "jane.smith@example.com", 
            "name": "Jane Smith",
            "join_date": "2022-08-20",
            "expertise_areas": ["Frontend", "React", "TypeScript"],
            "experience_level": 0.92,
            "collaboration_score": 0.65,
            "satisfaction_level": 0.45,
            "stress_accumulation": 0.78,
            "total_reviews": 203,
            "avg_response_time": 8.7,
            "retention_probability": 0.42
        },
        {
            "developer_id": "bob.wilson@example.com",
            "name": "Bob Wilson", 
            "join_date": "2023-03-10",
            "expertise_areas": ["Backend", "Database", "DevOps"],
            "experience_level": 0.68,
            "collaboration_score": 0.89,
            "satisfaction_level": 0.81,
            "stress_accumulation": 0.28,
            "total_reviews": 89,
            "avg_response_time": 3.1,
            "retention_probability": 0.91
        },
        {
            "developer_id": "alice.chen@example.com",
            "name": "Alice Chen",
            "join_date": "2023-06-01", 
            "expertise_areas": ["Mobile", "Flutter", "iOS"],
            "experience_level": 0.75,
            "collaboration_score": 0.82,
            "satisfaction_level": 0.69,
            "stress_accumulation": 0.52,
            "total_reviews": 124,
            "avg_response_time": 5.8,
            "retention_probability": 0.73
        },
        {
            "developer_id": "mike.brown@example.com",
            "name": "Mike Brown",
            "join_date": "2022-11-15",
            "expertise_areas": ["Security", "Infrastructure", "Monitoring"],
            "experience_level": 0.94,
            "collaboration_score": 0.71,
            "satisfaction_level": 0.88,
            "stress_accumulation": 0.15,
            "total_reviews": 267,
            "avg_response_time": 2.9,
            "retention_probability": 0.96
        }
    ]
    
    # レビューデータ
    reviews = []
    for i in range(50):
        review = {
            "review_id": f"review_{i+1:03d}",
            "change_id": f"change_{i+1:03d}",
            "author": random.choice([d["developer_id"] for d in developers]),
            "reviewer": random.choice([d["developer_id"] for d in developers]),
            "created_date": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
            "status": random.choice(["MERGED", "ABANDONED", "NEW", "DRAFT"]),
            "score": random.choice([-2, -1, 0, 1, 2]),
            "complexity": random.uniform(0.1, 1.0),
            "files_changed": random.randint(1, 15),
            "lines_added": random.randint(5, 500),
            "lines_deleted": random.randint(0, 200),
            "response_time_hours": random.uniform(0.5, 48.0),
            "project": random.choice(["core-api", "frontend-app", "mobile-client", "data-pipeline"])
        }
        reviews.append(review)
    
    # 特徴量データ
    features = []
    for dev in developers:
        feature = {
            "developer_id": dev["developer_id"],
            "expertise_level": dev["experience_level"],
            "collaboration_quality": dev["collaboration_score"], 
            "satisfaction_level": dev["satisfaction_level"],
            "stress_accumulation": dev["stress_accumulation"],
            "review_load": dev["total_reviews"],
            "response_time_avg": dev["avg_response_time"],
            "specialization_score": random.uniform(0.6, 0.95),
            "workload_balance": random.uniform(0.4, 0.9),
            "social_integration": random.uniform(0.5, 0.85),
            "growth_trajectory": random.uniform(0.3, 0.8),
            "burnout_risk": dev["stress_accumulation"],
            "retention_prediction": dev["retention_probability"]
        }
        features.append(feature)
    
    return developers, reviews, features

def save_sample_data():
    """サンプルデータを保存"""
    print("🔄 リアルなサンプルデータを生成中...")
    
    developers, reviews, features = generate_realistic_sample_data()
    
    # データディレクトリを作成
    data_dir = Path("data/processed/unified")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # データを保存
    with open(data_dir / "all_developers.json", "w", encoding="utf-8") as f:
        json.dump(developers, f, indent=2, ensure_ascii=False)
    
    with open(data_dir / "all_reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    
    with open(data_dir / "all_features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    
    print(f"✅ サンプルデータを生成しました:")
    print(f"   📊 開発者: {len(developers)}名")
    print(f"   📝 レビュー: {len(reviews)}件") 
    print(f"   🎯 特徴量: {len(features)}セット")
    print(f"   📁 保存先: {data_dir}")
    
    return True

if __name__ == "__main__":
    save_sample_data()