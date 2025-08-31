"""
Gerrit APIクライアント

Gerrit REST APIとの接続、認証、データ取得を担当するクライアントです。
レート制限、リトライ、エラーハンドリング機能を含みます。
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin

import requests
from gerrit_retention.utils.config_manager import get_config_manager
from gerrit_retention.utils.logger import get_logger
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

logger = get_logger(__name__)


class GerritAPIError(Exception):
    """Gerrit API関連エラー"""
    pass


class GerritRateLimitError(GerritAPIError):
    """レート制限エラー"""
    pass


class GerritAuthenticationError(GerritAPIError):
    """認証エラー"""
    pass


class GerritClient:
    """
    Gerrit APIクライアント
    
    Gerrit REST APIとの通信を管理し、レート制限、リトライ、
    エラーハンドリングを提供します。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Gerritクライアントを初期化
        
        Args:
            config: 設定辞書（オプション）
        """
        self.config_manager = get_config_manager()
        self.config = config or self._load_config()
        
        # 基本設定
        self.base_url = self.config.get("url", "").rstrip("/")
        self.username = self.config.get("auth", {}).get("username", "")
        self.password = self.config.get("auth", {}).get("password", "")
        
        # データ抽出設定
        extraction_config = self.config.get("data_extraction", {})
        self.batch_size = extraction_config.get("batch_size", 1000)
        self.rate_limit_delay = extraction_config.get("rate_limit_delay", 1.0)
        self.max_retries = extraction_config.get("max_retries", 3)
        self.timeout = extraction_config.get("timeout", 30)
        
        # HTTPセッションを初期化
        self.session = self._create_session()
        
        # レート制限管理
        self.last_request_time = 0.0
        self.request_count = 0
        
        logger.info(f"Gerritクライアントを初期化: {self.base_url}")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定を読み込み"""
        return self.config_manager.get("gerrit", {})
    
    def _create_session(self) -> requests.Session:
        """HTTPセッションを作成"""
        session = requests.Session()
        
        # 認証を設定
        if self.username and self.password:
            session.auth = HTTPBasicAuth(self.username, self.password)
        
        # リトライ戦略を設定
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # デフォルトヘッダーを設定
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "gerrit-retention-client/0.1.0"
        })
        
        return session
    
    def _wait_for_rate_limit(self) -> None:
        """レート制限を考慮して待機"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"レート制限のため {sleep_time:.2f}秒 待機中...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        HTTP リクエストを実行
        
        Args:
            method: HTTPメソッド
            endpoint: APIエンドポイント
            **kwargs: requests.requestに渡す追加引数
            
        Returns:
            requests.Response: レスポンス
            
        Raises:
            GerritAPIError: API関連エラー
        """
        # レート制限を適用
        self._wait_for_rate_limit()
        
        # URLを構築
        url = urljoin(self.base_url, endpoint)
        
        # タイムアウトを設定
        kwargs.setdefault("timeout", self.timeout)
        
        try:
            logger.debug(f"{method} {url}")
            response = self.session.request(method, url, **kwargs)
            
            # ステータスコードをチェック
            if response.status_code == 401:
                raise GerritAuthenticationError("認証に失敗しました")
            elif response.status_code == 429:
                raise GerritRateLimitError("レート制限に達しました")
            elif response.status_code >= 400:
                raise GerritAPIError(f"API エラー: {response.status_code} - {response.text}")
            
            return response
            
        except requests.exceptions.Timeout:
            raise GerritAPIError(f"リクエストがタイムアウトしました: {url}")
        except requests.exceptions.ConnectionError:
            raise GerritAPIError(f"接続エラーが発生しました: {url}")
        except requests.exceptions.RequestException as e:
            raise GerritAPIError(f"リクエストエラー: {e}")
    
    def _parse_gerrit_response(self, response: requests.Response) -> Any:
        """
        Gerritレスポンスを解析
        
        Gerrit APIは通常、JSONレスポンスの前に ")]}'" プレフィックスを付けます。
        
        Args:
            response: HTTPレスポンス
            
        Returns:
            解析されたJSONデータ
        """
        text = response.text
        
        # Gerritの標準的なJSONプレフィックスを削除
        if text.startswith(")]}'"):
            text = text[4:]
        
        try:
            return json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as e:
            logger.error(f"JSONの解析に失敗しました: {e}")
            logger.debug(f"レスポンステキスト: {text[:500]}...")
            raise GerritAPIError(f"無効なJSONレスポンス: {e}")
    
    def test_connection(self) -> bool:
        """
        Gerrit接続をテスト
        
        Returns:
            bool: 接続が成功したかどうか
        """
        try:
            logger.info("Gerrit接続をテスト中...")
            
            # バージョン情報を取得してテスト
            response = self._make_request("GET", "/config/server/version")
            version_data = self._parse_gerrit_response(response)
            
            logger.info(f"Gerrit接続成功 - バージョン: {version_data}")
            return True
            
        except Exception as e:
            logger.error(f"Gerrit接続テストに失敗: {e}")
            return False
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """
        プロジェクト一覧を取得
        
        Returns:
            List[Dict[str, Any]]: プロジェクト情報のリスト
        """
        try:
            logger.info("プロジェクト一覧を取得中...")
            
            response = self._make_request("GET", "/projects/")
            projects_data = self._parse_gerrit_response(response)
            
            # 辞書形式からリスト形式に変換
            projects = []
            for project_name, project_info in projects_data.items():
                project_info["name"] = project_name
                projects.append(project_info)
            
            logger.info(f"{len(projects)}個のプロジェクトを取得しました")
            return projects
            
        except Exception as e:
            logger.error(f"プロジェクト一覧の取得に失敗: {e}")
            raise
    
    def get_changes(
        self,
        project: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        start: int = 0
    ) -> List[Dict[str, Any]]:
        """
        変更（Change）一覧を取得
        
        Args:
            project: プロジェクト名
            start_date: 開始日時
            end_date: 終了日時
            limit: 取得件数制限
            start: 開始オフセット
            
        Returns:
            List[Dict[str, Any]]: 変更情報のリスト
        """
        try:
            logger.info(f"プロジェクト '{project}' の変更一覧を取得中...")
            
            # クエリパラメータを構築
            query_parts = [f"project:{project}"]
            
            if start_date:
                query_parts.append(f"after:{start_date.strftime('%Y-%m-%d')}")
            
            if end_date:
                query_parts.append(f"before:{end_date.strftime('%Y-%m-%d')}")
            
            query = " AND ".join(query_parts)
            
            params = {
                "q": query,
                "o": ["DETAILED_ACCOUNTS", "ALL_REVISIONS", "ALL_FILES", "MESSAGES"],
                "start": start
            }
            
            if limit:
                params["n"] = min(limit, self.batch_size)
            else:
                params["n"] = self.batch_size
            
            # リクエストを実行
            response = self._make_request("GET", "/changes/", params=params)
            changes_data = self._parse_gerrit_response(response)
            
            logger.info(f"{len(changes_data)}個の変更を取得しました")
            return changes_data
            
        except Exception as e:
            logger.error(f"変更一覧の取得に失敗: {e}")
            raise
    
    def get_change_detail(self, change_id: str) -> Dict[str, Any]:
        """
        変更の詳細情報を取得
        
        Args:
            change_id: 変更ID
            
        Returns:
            Dict[str, Any]: 変更の詳細情報
        """
        try:
            logger.debug(f"変更 '{change_id}' の詳細を取得中...")
            
            params = {
                "o": [
                    "DETAILED_ACCOUNTS",
                    "ALL_REVISIONS", 
                    "ALL_FILES",
                    "MESSAGES",
                    "DETAILED_LABELS"
                ]
            }
            
            response = self._make_request("GET", f"/changes/{change_id}", params=params)
            change_data = self._parse_gerrit_response(response)
            
            return change_data
            
        except Exception as e:
            logger.error(f"変更詳細の取得に失敗: {e}")
            raise
    
    def get_change_reviews(self, change_id: str) -> List[Dict[str, Any]]:
        """
        変更のレビュー一覧を取得
        
        Args:
            change_id: 変更ID
            
        Returns:
            List[Dict[str, Any]]: レビュー情報のリスト
        """
        try:
            logger.debug(f"変更 '{change_id}' のレビュー一覧を取得中...")
            
            response = self._make_request("GET", f"/changes/{change_id}/messages")
            reviews_data = self._parse_gerrit_response(response)
            
            return reviews_data
            
        except Exception as e:
            logger.error(f"レビュー一覧の取得に失敗: {e}")
            raise
    
    def get_account_info(self, account_id: str) -> Dict[str, Any]:
        """
        アカウント情報を取得
        
        Args:
            account_id: アカウントID
            
        Returns:
            Dict[str, Any]: アカウント情報
        """
        try:
            logger.debug(f"アカウント '{account_id}' の情報を取得中...")
            
            response = self._make_request("GET", f"/accounts/{account_id}")
            account_data = self._parse_gerrit_response(response)
            
            return account_data
            
        except Exception as e:
            logger.error(f"アカウント情報の取得に失敗: {e}")
            raise
    
    def get_changes_batch(
        self,
        project: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        変更一覧をバッチで取得
        
        Args:
            project: プロジェクト名
            start_date: 開始日時
            end_date: 終了日時
            batch_callback: バッチ処理コールバック関数
            
        Returns:
            List[Dict[str, Any]]: 全変更情報のリスト
        """
        all_changes = []
        start = 0
        
        logger.info(f"プロジェクト '{project}' の変更をバッチで取得開始...")
        
        while True:
            try:
                changes = self.get_changes(
                    project=project,
                    start_date=start_date,
                    end_date=end_date,
                    start=start
                )
                
                if not changes:
                    break
                
                all_changes.extend(changes)
                
                # バッチコールバックを実行
                if batch_callback:
                    batch_callback(changes, len(all_changes))
                
                # 次のバッチへ
                start += len(changes)
                
                # バッチサイズより少ない場合は最後のバッチ
                if len(changes) < self.batch_size:
                    break
                
                logger.info(f"累計 {len(all_changes)}個の変更を取得済み...")
                
            except Exception as e:
                logger.error(f"バッチ取得中にエラーが発生: {e}")
                raise
        
        logger.info(f"バッチ取得完了: 合計 {len(all_changes)}個の変更")
        return all_changes
    
    def close(self) -> None:
        """クライアントを終了"""
        if self.session:
            self.session.close()
        
        logger.info(f"Gerritクライアントを終了 (総リクエスト数: {self.request_count})")


# ユーティリティ関数

def create_gerrit_client(config: Optional[Dict[str, Any]] = None) -> GerritClient:
    """
    Gerritクライアントを作成
    
    Args:
        config: 設定辞書（オプション）
        
    Returns:
        GerritClient: Gerritクライアントインスタンス
    """
    return GerritClient(config)


def test_gerrit_connection(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Gerrit接続をテスト
    
    Args:
        config: 設定辞書（オプション）
        
    Returns:
        bool: 接続が成功したかどうか
    """
    client = create_gerrit_client(config)
    try:
        return client.test_connection()
    finally:
        client.close()