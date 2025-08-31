"""
レビュー特徴量エンジニアリング

Gerrit特化のレビュー・Change特徴量を抽出・計算するモジュール。
Change複雑度、規模、技術領域、レビュースコアの特徴量化を含む。
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReviewFeatures:
    """レビュー特徴量データクラス"""
    change_id: str
    
    # Change基本情報
    change_size: int
    files_changed_count: int
    lines_added: int
    lines_deleted: int
    change_complexity: float
    
    # 技術領域特徴量
    technical_domain: str
    programming_languages: List[str]
    file_types: Dict[str, int]
    directory_depth: float
    
    # レビュー特徴量
    review_scores: List[int]
    review_score_distribution: Dict[int, int]
    avg_review_score: float
    review_consensus: float
    review_count: int
    
    # レビュー品質特徴量
    review_thoroughness: float
    review_response_time_avg: float
    review_effort_total: float
    constructive_feedback_ratio: float
    
    # Change特性
    change_urgency: float
    change_risk_level: float
    change_type: str  # 'feature', 'bugfix', 'refactor', 'docs'
    change_scope: str  # 'local', 'module', 'system'
    
    # 協力関係特徴量
    reviewer_diversity: float
    cross_team_review: bool
    expert_involvement: float


class ReviewFeatureExtractor:
    """レビュー特徴量抽出器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.complexity_weights = config.get('complexity_weights', {
            'file_count': 0.3,
            'line_changes': 0.4,
            'directory_spread': 0.2,
            'language_diversity': 0.1
        })
        self.language_extensions = self._initialize_language_extensions()
        self.risk_keywords = config.get('risk_keywords', [
            'security', 'auth', 'password', 'token', 'critical', 'urgent',
            'hotfix', 'emergency', 'production', 'database', 'migration'
        ])
        
    def _initialize_language_extensions(self) -> Dict[str, str]:
        """ファイル拡張子とプログラミング言語のマッピング"""
        return {
            '.py': 'Python',
            '.java': 'Java',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.sh': 'Shell',
            '.bash': 'Shell',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.json': 'JSON',
            '.xml': 'XML',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sql': 'SQL',
            '.md': 'Markdown',
            '.dockerfile': 'Docker',
            '.gradle': 'Gradle',
            '.maven': 'Maven'
        }
    
    def extract_features(self, 
                        change_data: Dict[str, Any],
                        reviews_data: List[Dict[str, Any]],
                        context_date: datetime) -> ReviewFeatures:
        """
        レビュー・Change特徴量を抽出
        
        Args:
            change_data: Change データ
            reviews_data: 関連するReview データのリスト
            context_date: 特徴量計算の基準日時
            
        Returns:
            ReviewFeatures: 抽出された特徴量
        """
        change_id = change_data.get('change_id', '')
        logger.info(f"レビュー特徴量を抽出中: {change_id}")
        
        # Change関連のレビューをフィルタリング
        change_reviews = [r for r in reviews_data 
                         if r.get('change_id') == change_id]
        
        # 各カテゴリの特徴量を計算
        basic_features = self._extract_basic_change_features(change_data)
        technical_features = self._extract_technical_features(change_data)
        review_score_features = self._extract_review_score_features(change_reviews)
        review_quality_features = self._extract_review_quality_features(change_reviews)
        change_characteristics = self._extract_change_characteristics(change_data)
        collaboration_features = self._extract_collaboration_features(
            change_data, change_reviews
        )
        
        return ReviewFeatures(
            change_id=change_id,
            **basic_features,
            **technical_features,
            **review_score_features,
            **review_quality_features,
            **change_characteristics,
            **collaboration_features
        )
    
    def _extract_basic_change_features(self, 
                                     change_data: Dict[str, Any]) -> Dict[str, Any]:
        """基本的なChange特徴量を抽出"""
        
        files_changed = change_data.get('files_changed', [])
        lines_added = change_data.get('lines_added', 0)
        lines_deleted = change_data.get('lines_deleted', 0)
        
        # Change規模の計算
        change_size = lines_added + lines_deleted
        files_changed_count = len(files_changed) if isinstance(files_changed, list) else 0
        
        # Change複雑度の計算
        change_complexity = self._calculate_change_complexity(
            files_changed_count, change_size, files_changed
        )
        
        return {
            'change_size': change_size,
            'files_changed_count': files_changed_count,
            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'change_complexity': change_complexity
        }
    
    def _calculate_change_complexity(self, 
                                   files_count: int,
                                   total_lines: int,
                                   files_changed: List[str]) -> float:
        """Change複雑度を計算"""
        
        # ファイル数による複雑度
        file_complexity = min(files_count / 10.0, 1.0)  # 10ファイル以上で最大
        
        # 行数による複雑度
        line_complexity = min(total_lines / 1000.0, 1.0)  # 1000行以上で最大
        
        # ディレクトリ分散による複雑度
        if files_changed:
            directories = set()
            for file_path in files_changed:
                if '/' in file_path:
                    directories.add('/'.join(file_path.split('/')[:-1]))
            directory_complexity = min(len(directories) / 5.0, 1.0)  # 5ディレクトリ以上で最大
        else:
            directory_complexity = 0.0
        
        # 言語多様性による複雑度
        languages = set()
        for file_path in files_changed:
            for ext, lang in self.language_extensions.items():
                if file_path.endswith(ext):
                    languages.add(lang)
                    break
        language_complexity = min(len(languages) / 3.0, 1.0)  # 3言語以上で最大
        
        # 重み付き合計
        weights = self.complexity_weights
        total_complexity = (
            weights['file_count'] * file_complexity +
            weights['line_changes'] * line_complexity +
            weights['directory_spread'] * directory_complexity +
            weights['language_diversity'] * language_complexity
        )
        
        return total_complexity
    
    def _extract_technical_features(self, 
                                   change_data: Dict[str, Any]) -> Dict[str, Any]:
        """技術領域特徴量を抽出"""
        
        files_changed = change_data.get('files_changed', [])
        technical_domain = change_data.get('technical_domain', 'unknown')
        
        # プログラミング言語の抽出
        programming_languages = self._extract_programming_languages(files_changed)
        
        # ファイルタイプの分布
        file_types = self._extract_file_types(files_changed)
        
        # ディレクトリ深度の計算
        directory_depth = self._calculate_directory_depth(files_changed)
        
        return {
            'technical_domain': technical_domain,
            'programming_languages': programming_languages,
            'file_types': file_types,
            'directory_depth': directory_depth
        }
    
    def _extract_programming_languages(self, 
                                     files_changed: List[str]) -> List[str]:
        """プログラミング言語を抽出"""
        languages = set()
        
        for file_path in files_changed:
            for ext, lang in self.language_extensions.items():
                if file_path.endswith(ext):
                    languages.add(lang)
                    break
        
        return list(languages)
    
    def _extract_file_types(self, 
                          files_changed: List[str]) -> Dict[str, int]:
        """ファイルタイプの分布を計算"""
        file_types = defaultdict(int)
        
        for file_path in files_changed:
            if '.' in file_path:
                extension = '.' + file_path.split('.')[-1].lower()
                file_types[extension] += 1
            else:
                file_types['no_extension'] += 1
        
        return dict(file_types)
    
    def _calculate_directory_depth(self, 
                                 files_changed: List[str]) -> float:
        """ディレクトリ深度の平均を計算"""
        if not files_changed:
            return 0.0
        
        depths = []
        for file_path in files_changed:
            depth = file_path.count('/')
            depths.append(depth)
        
        return np.mean(depths)
    
    def _extract_review_score_features(self, 
                                     reviews_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """レビュースコア特徴量を抽出"""
        
        if not reviews_data:
            return {
                'review_scores': [],
                'review_score_distribution': {},
                'avg_review_score': 0.0,
                'review_consensus': 0.0,
                'review_count': 0
            }
        
        # レビュースコアを収集
        review_scores = [r.get('score', 0) for r in reviews_data]
        
        # スコア分布を計算
        score_distribution = Counter(review_scores)
        
        # 平均スコア
        avg_review_score = np.mean(review_scores)
        
        # レビューコンセンサス（スコアの一致度）
        review_consensus = self._calculate_review_consensus(review_scores)
        
        return {
            'review_scores': review_scores,
            'review_score_distribution': dict(score_distribution),
            'avg_review_score': avg_review_score,
            'review_consensus': review_consensus,
            'review_count': len(reviews_data)
        }
    
    def _calculate_review_consensus(self, 
                                  review_scores: List[int]) -> float:
        """レビューコンセンサス（一致度）を計算"""
        if len(review_scores) <= 1:
            return 1.0
        
        # スコアの分散を使用（低い分散 = 高いコンセンサス）
        variance = np.var(review_scores)
        
        # 分散を0-1の範囲のコンセンサススコアに変換
        # Gerritスコア範囲（-2から+2）での最大分散は4
        max_variance = 4.0
        consensus = 1.0 - min(variance / max_variance, 1.0)
        
        return consensus
    
    def _extract_review_quality_features(self, 
                                       reviews_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """レビュー品質特徴量を抽出"""
        
        if not reviews_data:
            return {
                'review_thoroughness': 0.0,
                'review_response_time_avg': 0.0,
                'review_effort_total': 0.0,
                'constructive_feedback_ratio': 0.0
            }
        
        # レビューの詳細度
        review_thoroughness = self._calculate_review_thoroughness(reviews_data)
        
        # 平均応答時間
        response_times = [r.get('response_time_hours', 24) for r in reviews_data]
        review_response_time_avg = np.mean(response_times)
        
        # 総レビュー労力
        efforts = [r.get('review_effort_estimated', 1.0) for r in reviews_data]
        review_effort_total = sum(efforts)
        
        # 建設的フィードバックの割合
        constructive_feedback_ratio = self._calculate_constructive_feedback_ratio(
            reviews_data
        )
        
        return {
            'review_thoroughness': review_thoroughness,
            'review_response_time_avg': review_response_time_avg,
            'review_effort_total': review_effort_total,
            'constructive_feedback_ratio': constructive_feedback_ratio
        }
    
    def _calculate_review_thoroughness(self, 
                                     reviews_data: List[Dict[str, Any]]) -> float:
        """レビューの詳細度を計算"""
        thoroughness_scores = []
        
        for review in reviews_data:
            message = review.get('message', '')
            effort = review.get('review_effort_estimated', 1.0)
            
            # メッセージの長さによる詳細度
            message_score = min(len(message) / 200.0, 1.0)  # 200文字以上で最大
            
            # 推定労力による詳細度
            effort_score = min(effort / 3.0, 1.0)  # 3時間以上で最大
            
            # 総合詳細度
            thoroughness = (message_score + effort_score) / 2.0
            thoroughness_scores.append(thoroughness)
        
        return np.mean(thoroughness_scores)
    
    def _calculate_constructive_feedback_ratio(self, 
                                             reviews_data: List[Dict[str, Any]]) -> float:
        """建設的フィードバックの割合を計算"""
        if not reviews_data:
            return 0.0
        
        constructive_count = 0
        
        for review in reviews_data:
            score = review.get('score', 0)
            message = review.get('message', '')
            
            # 建設的フィードバックの判定
            is_constructive = (
                score >= 0 or  # 非負のスコア
                len(message) > 50 or  # 詳細なコメント
                any(keyword in message.lower() for keyword in [
                    'suggest', 'consider', 'recommend', 'improve', 'better'
                ])
            )
            
            if is_constructive:
                constructive_count += 1
        
        return constructive_count / len(reviews_data)
    
    def _extract_change_characteristics(self, 
                                      change_data: Dict[str, Any]) -> Dict[str, Any]:
        """Change特性を抽出"""
        
        # Change緊急度の計算
        change_urgency = self._calculate_change_urgency(change_data)
        
        # Changeリスクレベルの計算
        change_risk_level = self._calculate_change_risk_level(change_data)
        
        # Changeタイプの推定
        change_type = self._estimate_change_type(change_data)
        
        # Changeスコープの推定
        change_scope = self._estimate_change_scope(change_data)
        
        return {
            'change_urgency': change_urgency,
            'change_risk_level': change_risk_level,
            'change_type': change_type,
            'change_scope': change_scope
        }
    
    def _calculate_change_urgency(self, 
                                change_data: Dict[str, Any]) -> float:
        """Change緊急度を計算"""
        urgency_indicators = []
        
        # サブジェクトからの緊急度判定
        subject = change_data.get('subject', '').lower()
        urgent_keywords = ['urgent', 'critical', 'hotfix', 'emergency', 'asap']
        
        if any(keyword in subject for keyword in urgent_keywords):
            urgency_indicators.append(1.0)
        
        # ブランチ名からの緊急度判定
        branch = change_data.get('branch', '').lower()
        if 'hotfix' in branch or 'emergency' in branch:
            urgency_indicators.append(0.8)
        
        # 作成から更新までの時間（短い = 緊急）
        created = change_data.get('created', '')
        updated = change_data.get('updated', '')
        
        if created and updated:
            try:
                created_dt = datetime.fromisoformat(created)
                updated_dt = datetime.fromisoformat(updated)
                time_diff = (updated_dt - created_dt).total_seconds() / 3600  # 時間
                
                # 24時間以内の更新は緊急度が高い
                if time_diff < 24:
                    urgency_indicators.append(0.6)
            except:
                pass
        
        return np.mean(urgency_indicators) if urgency_indicators else 0.0
    
    def _calculate_change_risk_level(self, 
                                   change_data: Dict[str, Any]) -> float:
        """Changeリスクレベルを計算"""
        risk_indicators = []
        
        # サブジェクトからのリスク判定
        subject = change_data.get('subject', '').lower()
        
        if any(keyword in subject for keyword in self.risk_keywords):
            risk_indicators.append(0.8)
        
        # ファイル変更からのリスク判定
        files_changed = change_data.get('files_changed', [])
        
        # 重要なファイルの変更
        critical_patterns = [
            r'.*config.*', r'.*setting.*', r'.*auth.*', r'.*security.*',
            r'.*database.*', r'.*migration.*', r'.*deploy.*', r'.*prod.*'
        ]
        
        for file_path in files_changed:
            if any(re.match(pattern, file_path.lower()) for pattern in critical_patterns):
                risk_indicators.append(0.6)
                break
        
        # 大規模変更のリスク
        change_size = change_data.get('lines_added', 0) + change_data.get('lines_deleted', 0)
        if change_size > 500:  # 500行以上の変更
            risk_indicators.append(0.4)
        
        return np.mean(risk_indicators) if risk_indicators else 0.0
    
    def _estimate_change_type(self, 
                            change_data: Dict[str, Any]) -> str:
        """Changeタイプを推定"""
        subject = change_data.get('subject', '').lower()
        
        # キーワードベースの分類
        if any(keyword in subject for keyword in ['fix', 'bug', 'issue', 'error']):
            return 'bugfix'
        elif any(keyword in subject for keyword in ['refactor', 'cleanup', 'optimize']):
            return 'refactor'
        elif any(keyword in subject for keyword in ['doc', 'readme', 'comment']):
            return 'docs'
        elif any(keyword in subject for keyword in ['test', 'spec', 'unit']):
            return 'test'
        elif any(keyword in subject for keyword in ['add', 'new', 'implement', 'feature']):
            return 'feature'
        else:
            return 'other'
    
    def _estimate_change_scope(self, 
                             change_data: Dict[str, Any]) -> str:
        """Changeスコープを推定"""
        files_changed = change_data.get('files_changed', [])
        
        if not files_changed:
            return 'unknown'
        
        # ディレクトリの分散度で判定
        directories = set()
        for file_path in files_changed:
            if '/' in file_path:
                directories.add('/'.join(file_path.split('/')[:-1]))
        
        if len(directories) >= 5:
            return 'system'  # システム全体
        elif len(directories) >= 2:
            return 'module'  # モジュール間
        else:
            return 'local'   # ローカル
    
    def _extract_collaboration_features(self, 
                                      change_data: Dict[str, Any],
                                      reviews_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """協力関係特徴量を抽出"""
        
        # レビュワーの多様性
        reviewer_diversity = self._calculate_reviewer_diversity(reviews_data)
        
        # チーム横断レビュー
        cross_team_review = self._detect_cross_team_review(
            change_data, reviews_data
        )
        
        # エキスパートの関与度
        expert_involvement = self._calculate_expert_involvement(reviews_data)
        
        return {
            'reviewer_diversity': reviewer_diversity,
            'cross_team_review': cross_team_review,
            'expert_involvement': expert_involvement
        }
    
    def _calculate_reviewer_diversity(self, 
                                    reviews_data: List[Dict[str, Any]]) -> float:
        """レビュワーの多様性を計算"""
        if not reviews_data:
            return 0.0
        
        reviewers = set(r.get('reviewer_email', '') for r in reviews_data)
        
        # レビュワー数を正規化（5人以上で最大値）
        return min(len(reviewers) / 5.0, 1.0)
    
    def _detect_cross_team_review(self, 
                                change_data: Dict[str, Any],
                                reviews_data: List[Dict[str, Any]]) -> bool:
        """チーム横断レビューを検出"""
        # 簡易的な実装：異なるドメインのメールアドレスがあるかチェック
        author_email = change_data.get('author', '')
        author_domain = author_email.split('@')[-1] if '@' in author_email else ''
        
        for review in reviews_data:
            reviewer_email = review.get('reviewer_email', '')
            reviewer_domain = reviewer_email.split('@')[-1] if '@' in reviewer_email else ''
            
            # 異なるドメインまたは異なるサブドメイン
            if (reviewer_domain != author_domain and 
                reviewer_domain and author_domain):
                return True
        
        return False
    
    def _calculate_expert_involvement(self, 
                                    reviews_data: List[Dict[str, Any]]) -> float:
        """エキスパートの関与度を計算"""
        if not reviews_data:
            return 0.0
        
        expert_indicators = []
        
        for review in reviews_data:
            # エキスパートの指標
            score = abs(review.get('score', 0))
            effort = review.get('review_effort_estimated', 1.0)
            message_length = len(review.get('message', ''))
            
            # エキスパート度の計算
            expert_score = (
                (score / 2.0) * 0.4 +  # 高いスコア
                min(effort / 3.0, 1.0) * 0.3 +  # 高い労力
                min(message_length / 200.0, 1.0) * 0.3  # 詳細なコメント
            )
            
            expert_indicators.append(expert_score)
        
        return np.mean(expert_indicators)


def create_review_feature_extractor(config_path: str) -> ReviewFeatureExtractor:
    """
    設定ファイルからレビュー特徴量抽出器を作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        ReviewFeatureExtractor: 設定済みの抽出器
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    feature_config = config.get('review_features', {})
    return ReviewFeatureExtractor(feature_config)


def batch_extract_review_features(changes_data: List[Dict[str, Any]],
                                 reviews_data: List[Dict[str, Any]],
                                 extractor: ReviewFeatureExtractor,
                                 context_date: datetime) -> List[ReviewFeatures]:
    """
    複数のChangeに対してバッチで特徴量を抽出
    
    Args:
        changes_data: Change データのリスト
        reviews_data: Review データのリスト
        extractor: 特徴量抽出器
        context_date: 基準日時
        
    Returns:
        List[ReviewFeatures]: 抽出された特徴量のリスト
    """
    features_list = []
    
    for change_data in changes_data:
        try:
            features = extractor.extract_features(
                change_data, reviews_data, context_date
            )
            features_list.append(features)
        except Exception as e:
            logger.error(f"特徴量抽出エラー (Change ID: {change_data.get('change_id')}): {e}")
            continue
    
    return features_list


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_change = {
        'change_id': 'change1',
        'author': 'dev1@example.com',
        'subject': 'Fix critical security vulnerability in auth module',
        'created': '2023-01-15T10:00:00',
        'updated': '2023-01-15T18:00:00',
        'project': 'project-a',
        'branch': 'hotfix/security-fix',
        'technical_domain': 'security',
        'files_changed': [
            'src/auth/authentication.py',
            'src/auth/authorization.py',
            'tests/auth/test_auth.py',
            'docs/security.md'
        ],
        'lines_added': 150,
        'lines_deleted': 75
    }
    
    sample_reviews = [
        {
            'change_id': 'change1',
            'reviewer_email': 'security-expert@example.com',
            'timestamp': '2023-01-15T14:00:00',
            'score': 2,
            'message': 'Excellent fix! The implementation properly addresses the vulnerability. I suggest adding additional input validation in the authorization module as well.',
            'response_time_hours': 4.0,
            'review_effort_estimated': 2.5
        },
        {
            'change_id': 'change1',
            'reviewer_email': 'dev2@company.com',
            'timestamp': '2023-01-15T16:00:00',
            'score': 1,
            'message': 'LGTM with minor suggestions.',
            'response_time_hours': 6.0,
            'review_effort_estimated': 1.0
        }
    ]
    
    # 特徴量抽出器のテスト
    config = {
        'complexity_weights': {
            'file_count': 0.3,
            'line_changes': 0.4,
            'directory_spread': 0.2,
            'language_diversity': 0.1
        },
        'risk_keywords': [
            'security', 'auth', 'critical', 'urgent', 'hotfix'
        ]
    }
    
    extractor = ReviewFeatureExtractor(config)
    features = extractor.extract_features(
        sample_change,
        sample_reviews,
        datetime(2023, 2, 1)
    )
    
    print(f"レビュー特徴量抽出完了: {features.change_id}")
    print(f"Change複雑度: {features.change_complexity:.3f}")
    print(f"平均レビュースコア: {features.avg_review_score:.3f}")
    print(f"レビューコンセンサス: {features.review_consensus:.3f}")
    print(f"Changeタイプ: {features.change_type}")
    print(f"リスクレベル: {features.change_risk_level:.3f}")
    print(f"緊急度: {features.change_urgency:.3f}")