#!/usr/bin/env python3
"""
Gerrit開発者定着予測システム - APIサーバー
"""

import asyncio
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import get_logger

# モデルクラスをインポート
try:
    from retention_training.train_retention_model import SimpleRetentionModel
    from stress_training.train_stress_model import SimpleStressModel
except ImportError:
    # モデルクラスが見つからない場合のダミークラス
    class SimpleRetentionModel:
        pass
    class SimpleStressModel:
        pass

# FastAPIをインポート（利用可能な場合）
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # FastAPIが利用できない場合のダミークラス
    class BaseModel:
        pass

logger = get_logger(__name__)

class DeveloperProfile(BaseModel):
    """開発者プロファイル"""
    email: str
    name: str
    expertise_level: float
    activity_pattern: Dict[str, Any]
    collaboration_quality: float
    stress_level: float
    project: str

class ReviewRequest(BaseModel):
    """レビューリクエスト"""
    change_id: str
    author_email: str
    complexity_score: float
    technical_domain: str
    urgency_level: str
    project: str

class PredictionRequest(BaseModel):
    """予測リクエスト"""
    developer: DeveloperProfile
    review: Optional[ReviewRequest] = None

class APIServer:
    """APIサーバークラス"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.models = {}
        self.load_models()
        
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Gerrit開発者定着予測システム",
                description="開発者の定着予測とストレス分析、レビュー推薦を行うAPIサーバー",
                version="1.0.0"
            )
            self.setup_routes()
        else:
            logger.warning("FastAPIが利用できません。簡易サーバーモードで実行します。")
            self.app = None
    
    def load_models(self):
        """モデルを読み込み"""
        models_dir = Path('models')
        
        # 定着予測モデル
        retention_model_path = models_dir / 'retention_model.pkl'
        if retention_model_path.exists():
            with open(retention_model_path, 'rb') as f:
                self.models['retention'] = pickle.load(f)
            logger.info(f"定着予測モデルを読み込みました: {retention_model_path}")
        
        # ストレス分析モデル
        stress_model_path = models_dir / 'stress_model.pkl'
        if stress_model_path.exists():
            with open(stress_model_path, 'rb') as f:
                self.models['stress'] = pickle.load(f)
            logger.info(f"ストレス分析モデルを読み込みました: {stress_model_path}")
        
        # PPOエージェント
        ppo_agent_path = models_dir / 'ppo_agent.zip'
        if ppo_agent_path.exists():
            try:
                with open(ppo_agent_path, 'rb') as f:
                    self.models['ppo'] = pickle.load(f)
                logger.info(f"PPOエージェントを読み込みました: {ppo_agent_path}")
            except Exception as e:
                logger.warning(f"PPOエージェントの読み込みに失敗しました: {e}")
                logger.info("PPOエージェントなしで続行します")
    
    def setup_routes(self):
        """APIルートを設定"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Gerrit開発者定着予測システム",
                "version": "1.0.0",
                "status": "running",
                "models_loaded": list(self.models.keys()),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "models": {name: "loaded" for name in self.models.keys()},
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/predict/retention")
        async def predict_retention(request: PredictionRequest):
            """開発者の定着確率を予測"""
            if 'retention' not in self.models:
                raise HTTPException(status_code=503, detail="定着予測モデルが利用できません")
            
            try:
                # 開発者データを特徴量に変換
                features = {
                    'expertise_level': request.developer.expertise_level,
                    'collaboration_quality': request.developer.collaboration_quality,
                    'satisfaction_level': 0.7,  # デフォルト値
                    'stress_accumulation': request.developer.stress_level
                }
                
                # 予測実行
                retention_prob = self.models['retention'].predict(features)
                
                return {
                    "developer_email": request.developer.email,
                    "retention_probability": retention_prob,
                    "risk_level": "high" if retention_prob < 0.3 else "medium" if retention_prob < 0.7 else "low",
                    "timestamp": datetime.now().isoformat()
                }
            
            except Exception as e:
                logger.error(f"定着予測エラー: {e}")
                raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")
        
        @self.app.post("/predict/stress")
        async def predict_stress(request: PredictionRequest):
            """開発者のストレスレベルを予測"""
            if 'stress' not in self.models:
                raise HTTPException(status_code=503, detail="ストレス分析モデルが利用できません")
            
            try:
                # 開発者データを変換
                developer_data = {
                    'stress_level': request.developer.stress_level,
                    'expertise_level': request.developer.expertise_level,
                    'collaboration_quality': request.developer.collaboration_quality,
                    'activity_pattern': request.developer.activity_pattern
                }
                
                # ストレス予測
                stress_level = self.models['stress'].predict_stress(developer_data)
                boiling_point = self.models['stress'].predict_boiling_point(developer_data)
                
                return {
                    "developer_email": request.developer.email,
                    "current_stress_level": stress_level,
                    "boiling_point_days": boiling_point,
                    "stress_category": "critical" if stress_level > 0.8 else "high" if stress_level > 0.6 else "medium" if stress_level > 0.4 else "low",
                    "timestamp": datetime.now().isoformat()
                }
            
            except Exception as e:
                logger.error(f"ストレス予測エラー: {e}")
                raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")
        
        @self.app.post("/recommend/review")
        async def recommend_review_action(request: PredictionRequest):
            """レビュー行動を推薦"""
            if 'ppo' not in self.models:
                raise HTTPException(status_code=503, detail="PPOエージェントが利用できません")
            
            if not request.review:
                raise HTTPException(status_code=400, detail="レビュー情報が必要です")
            
            try:
                # 状態を構築
                state = {
                    'reviewer_expertise': request.developer.expertise_level,
                    'review_complexity': request.review.complexity_score,
                    'reviewer_workload': request.developer.stress_level,
                    'urgency_level': 0.8 if request.review.urgency_level == "high" else 0.5
                }
                
                # 行動予測
                action = self.models['ppo'].predict_action(state)
                probabilities = self.models['ppo'].get_action_probabilities(state)
                
                action_names = ["decline", "accept", "defer"]
                
                return {
                    "developer_email": request.developer.email,
                    "change_id": request.review.change_id,
                    "recommended_action": action_names[action],
                    "action_probabilities": {
                        "decline": probabilities[0],
                        "accept": probabilities[1],
                        "defer": probabilities[2]
                    },
                    "confidence": max(probabilities),
                    "timestamp": datetime.now().isoformat()
                }
            
            except Exception as e:
                logger.error(f"レビュー推薦エラー: {e}")
                raise HTTPException(status_code=500, detail=f"推薦エラー: {str(e)}")
    
    def run_simple_server(self):
        """簡易サーバーを実行（FastAPI未使用）"""
        logger.info("簡易サーバーモードで実行中...")
        
        # 簡単なテスト実行
        test_developer = {
            'email': 'test@example.com',
            'expertise_level': 0.7,
            'collaboration_quality': 0.8,
            'stress_level': 0.4,
            'activity_pattern': {'commits_per_week': 10}
        }
        
        try:
            if 'retention' in self.models:
                retention_prob = self.models['retention'].predict({
                    'expertise_level': 0.7,
                    'collaboration_quality': 0.8,
                    'satisfaction_level': 0.7,
                    'stress_accumulation': 0.4
                })
                logger.info(f"テスト定着予測: {retention_prob:.3f}")
        except Exception as e:
            logger.error(f"定着予測テストエラー: {e}")
        
        try:
            if 'stress' in self.models:
                stress_level = self.models['stress'].predict_stress(test_developer)
                logger.info(f"テストストレス予測: {stress_level:.3f}")
        except Exception as e:
            logger.error(f"ストレス予測テストエラー: {e}")
        
        try:
            if 'ppo' in self.models:
                test_state = {
                    'reviewer_expertise': 0.7,
                    'review_complexity': 0.5,
                    'reviewer_workload': 0.4,
                    'urgency_level': 0.6
                }
                action = self.models['ppo'].predict_action(test_state)
                logger.info(f"テストレビュー推薦: {['decline', 'accept', 'defer'][action]}")
        except Exception as e:
            logger.error(f"レビュー推薦テストエラー: {e}")
        
        logger.info("簡易サーバーテスト完了")
        
        # 簡易HTTPサーバーを起動
        logger.info("簡易HTTPサーバーを起動中...")
        self.run_simple_http_server()
    
    def run_simple_http_server(self):
        """簡易HTTPサーバーを実行"""
        import http.server
        import json
        import socketserver
        from urllib.parse import parse_qs, urlparse
        
        class SimpleHandler(http.server.BaseHTTPRequestHandler):
            def __init__(self, api_server, *args, **kwargs):
                self.api_server = api_server
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {
                        "message": "Gerrit開発者定着予測システム",
                        "version": "1.0.0",
                        "status": "running",
                        "models_loaded": list(self.api_server.models.keys()),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
                
                elif self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {
                        "status": "healthy",
                        "models": {name: "loaded" for name in self.api_server.models.keys()},
                        "timestamp": datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
                
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'Not Found')
            
            def log_message(self, format, *args):
                # ログを無効化
                pass
        
        # ハンドラーをラップしてapi_serverを渡す
        def handler_factory(*args, **kwargs):
            return SimpleHandler(self, *args, **kwargs)
        
        port = 8080
        with socketserver.TCPServer(("", port), handler_factory) as httpd:
            logger.info(f"簡易HTTPサーバーがポート {port} で起動しました")
            logger.info("利用可能なエンドポイント:")
            logger.info("  GET / - システム情報")
            logger.info("  GET /health - ヘルスチェック")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                logger.info("サーバーを停止しています...")
                httpd.shutdown()
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """サーバーを実行"""
        if self.app and FASTAPI_AVAILABLE:
            logger.info(f"APIサーバーを開始: http://{host}:{port}")
            uvicorn.run(self.app, host=host, port=port)
        else:
            self.run_simple_server()

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerrit開発者定着予測システム - APIサーバー')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='ホストアドレス')
    parser.add_argument('--port', type=int, default=8080, help='ポート番号')
    parser.add_argument('--environment', type=str, default='development', help='実行環境')
    
    args = parser.parse_args()
    
    logger.info("APIサーバー開始")
    
    server = APIServer()
    
    # コマンドライン引数または環境変数からポート設定を取得
    port = args.port or int(os.getenv('PORT', 8080))
    host = args.host or os.getenv('HOST', '0.0.0.0')
    
    server.run(host=host, port=port)

if __name__ == "__main__":
    main()