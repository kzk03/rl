#!/usr/bin/env python3
"""
簡単なAPIサーバー
"""

import json
import pickle
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


# モデルクラス定義
class SimpleRetentionModel:
    def __init__(self):
        self.trained_at = ""
        self.model_version = "1.0.0"
        self.feature_count = 0
    
    def predict(self, features):
        expertise = features.get('expertise_level', 0.5)
        collaboration = features.get('collaboration_quality', 0.5)
        satisfaction = features.get('satisfaction_level', 0.5)
        stress = features.get('stress_accumulation', 0.5)
        
        retention_prob = (
            expertise * 0.3 + 
            collaboration * 0.3 + 
            satisfaction * 0.3 - 
            stress * 0.1
        )
        
        return max(0.0, min(1.0, retention_prob))

class SimpleStressModel:
    def __init__(self):
        self.trained_at = ""
        self.model_version = "1.0.0"
        self.developer_count = 0
        self.review_count = 0
    
    def predict_stress(self, developer_data, context_data=None):
        base_stress = developer_data.get('stress_level', 0.5)
        expertise = developer_data.get('expertise_level', 0.5)
        collaboration = developer_data.get('collaboration_quality', 0.5)
        
        activity = developer_data.get('activity_pattern', {})
        commits_per_week = activity.get('commits_per_week', 10)
        reviews_per_week = activity.get('reviews_per_week', 10)
        
        workload_stress = min(1.0, (commits_per_week + reviews_per_week) / 30.0)
        
        total_stress = (
            base_stress * 0.4 +
            workload_stress * 0.3 +
            (1.0 - collaboration) * 0.2 +
            (1.0 - expertise) * 0.1
        )
        
        return max(0.0, min(1.0, total_stress))


class SimpleAPIHandler(BaseHTTPRequestHandler):
    def __init__(self, models, *args, **kwargs):
        self.models = models
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/':
            self.send_json({
                "message": "Gerrit開発者定着予測システム",
                "version": "1.0.0",
                "status": "running",
                "models_loaded": list(self.models.keys()),
                "timestamp": datetime.now().isoformat()
            })
        
        elif path == '/health':
            self.send_json({
                "status": "healthy",
                "models": {name: "loaded" for name in self.models.keys()},
                "timestamp": datetime.now().isoformat()
            })
        
        elif path == '/test':
            # テスト予測
            results = {}
            
            if 'retention' in self.models:
                try:
                    prob = self.models['retention'].predict({
                        'expertise_level': 0.7,
                        'collaboration_quality': 0.8,
                        'satisfaction_level': 0.7,
                        'stress_accumulation': 0.4
                    })
                    results['retention_prediction'] = prob
                except Exception as e:
                    results['retention_error'] = str(e)
            
            if 'stress' in self.models:
                try:
                    stress = self.models['stress'].predict_stress({
                        'stress_level': 0.4,
                        'expertise_level': 0.7,
                        'collaboration_quality': 0.8,
                        'activity_pattern': {'commits_per_week': 10}
                    })
                    results['stress_prediction'] = stress
                except Exception as e:
                    results['stress_error'] = str(e)
            
            self.send_json(results)
        
        else:
            self.send_error(404, "Not Found")
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
    
    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}")

def load_models():
    """モデルを読み込み"""
    models = {}
    models_dir = Path('models')
    
    # 定着予測モデル
    retention_path = models_dir / 'retention_model.pkl'
    if retention_path.exists():
        with open(retention_path, 'rb') as f:
            models['retention'] = pickle.load(f)
        print(f"✓ 定着予測モデル読み込み完了")
    
    # ストレス分析モデル
    stress_path = models_dir / 'stress_model.pkl'
    if stress_path.exists():
        with open(stress_path, 'rb') as f:
            models['stress'] = pickle.load(f)
        print(f"✓ ストレス分析モデル読み込み完了")
    
    return models

def main():
    print("🚀 Gerrit開発者定着予測システム - 簡易APIサーバー")
    
    # モデル読み込み
    models = load_models()
    print(f"📊 読み込み済みモデル: {list(models.keys())}")
    
    # ハンドラーファクトリー
    def handler_factory(*args, **kwargs):
        return SimpleAPIHandler(models, *args, **kwargs)
    
    # サーバー起動
    port = 8081
    server = HTTPServer(('0.0.0.0', port), handler_factory)
    
    print(f"🌐 サーバー起動: http://localhost:{port}")
    print("📋 利用可能なエンドポイント:")
    print("   GET /        - システム情報")
    print("   GET /health  - ヘルスチェック")
    print("   GET /test    - テスト予測")
    print("⏹️  停止: Ctrl+C")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 サーバーを停止しています...")
        server.shutdown()

if __name__ == "__main__":
    main()