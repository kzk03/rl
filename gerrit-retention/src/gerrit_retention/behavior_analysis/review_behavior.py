"""
レビュー行動分析器

開発者のレビュー受諾行動を分析し、専門性マッチ度、ワークロード、
関係性要因を統合して受諾確率を計算する。
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ReviewDecision(Enum):
    """レビュー決定の種類"""
    ACCEPT = "accept"
    DECLINE = "decline"
    DEFER = "defer"


@dataclass
class ReviewRequest:
    """レビュー依頼データ"""
    change_id: str
    author_email: str
    reviewer_email: str
    subject: str
    files_changed: List[str]
    lines_added: int
    lines_deleted: int
    complexity_score: float
    technical_domain: str
    created_at: datetime
    deadline: Optional[datetime] = None
    urgency_level: float = 0.5
    estimated_effort_hours: float = 1.0


@dataclass
class DeveloperProfile:
    """開発者プロファイル"""
    email: str
    name: str
    expertise_areas: Dict[str, float]  # 技術領域 -> 専門度
    current_workload: float
    stress_level: float
    collaboration_history: Dict[str, float]  # 相手 -> 関係性スコア
    review_history: List[Dict[str, Any]]
    avg_response_time_hours: float
    acceptance_rate: float
    last_updated: datetime


@dataclass
class BehaviorAnalysisResult:
    """行動分析結果"""
    acceptance_probability: float
    expertise_match_score: float
    workload_impact_score: float
    relationship_score: float
    confidence_level: float
    risk_factors: List[str]
    recommendations: List[str]


class ReviewBehaviorAnalyzer:
    """レビュー行動分析器
    
    開発者のレビュー受諾行動を分析し、専門性マッチ度、ワークロード、
    関係性要因を統合して受諾確率を計算する。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定辞書
                - expertise_weight: 専門性重み (default: 0.4)
                - workload_weight: ワークロード重み (default: 0.3)
                - relationship_weight: 関係性重み (default: 0.3)
                - workload_threshold: ワークロード閾値 (default: 0.8)
                - expertise_threshold: 専門性閾値 (default: 0.6)
        """
        self.config = config
        self.expertise_weight = config.get('expertise_weight', 0.4)
        self.workload_weight = config.get('workload_weight', 0.3)
        self.relationship_weight = config.get('relationship_weight', 0.3)
        self.workload_threshold = config.get('workload_threshold', 0.8)
        self.expertise_threshold = config.get('expertise_threshold', 0.6)
        
        logger.info(f"ReviewBehaviorAnalyzer initialized with weights: "
                   f"expertise={self.expertise_weight}, "
                   f"workload={self.workload_weight}, "
                   f"relationship={self.relationship_weight}")
    
    def analyze_review_behavior(
        self,
        review_request: ReviewRequest,
        developer_profile: DeveloperProfile
    ) -> BehaviorAnalysisResult:
        """レビュー行動を分析して受諾確率を計算
        
        Args:
            review_request: レビュー依頼
            developer_profile: 開発者プロファイル
            
        Returns:
            BehaviorAnalysisResult: 分析結果
        """
        try:
            # 専門性マッチ度を計算
            expertise_score = self._calculate_expertise_match(
                review_request, developer_profile
            )
            
            # ワークロード影響度を計算
            workload_score = self._calculate_workload_impact(
                review_request, developer_profile
            )
            
            # 関係性スコアを計算
            relationship_score = self._calculate_relationship_score(
                review_request, developer_profile
            )
            
            # 統合受諾確率を計算
            acceptance_probability = self._calculate_integrated_probability(
                expertise_score, workload_score, relationship_score
            )
            
            # 信頼度を計算
            confidence_level = self._calculate_confidence_level(
                developer_profile, review_request
            )
            
            # リスク要因を特定
            risk_factors = self._identify_risk_factors(
                expertise_score, workload_score, relationship_score,
                developer_profile, review_request
            )
            
            # 推奨事項を生成
            recommendations = self._generate_recommendations(
                expertise_score, workload_score, relationship_score,
                developer_profile, review_request
            )
            
            return BehaviorAnalysisResult(
                acceptance_probability=acceptance_probability,
                expertise_match_score=expertise_score,
                workload_impact_score=workload_score,
                relationship_score=relationship_score,
                confidence_level=confidence_level,
                risk_factors=risk_factors,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"レビュー行動分析中にエラーが発生: {e}")
            # フォールバック値を返す
            return BehaviorAnalysisResult(
                acceptance_probability=0.5,
                expertise_match_score=0.5,
                workload_impact_score=0.5,
                relationship_score=0.5,
                confidence_level=0.3,
                risk_factors=["分析エラー"],
                recommendations=["手動確認が必要"]
            )
    
    def _calculate_expertise_match(
        self,
        review_request: ReviewRequest,
        developer_profile: DeveloperProfile
    ) -> float:
        """専門性マッチ度を計算
        
        Args:
            review_request: レビュー依頼
            developer_profile: 開発者プロファイル
            
        Returns:
            float: 専門性マッチ度 (0.0-1.0)
        """
        # 技術領域の専門性を確認
        domain_expertise = developer_profile.expertise_areas.get(
            review_request.technical_domain, 0.0
        )
        
        # ファイルパスベースの専門性を計算
        file_expertise_scores = []
        for file_path in review_request.files_changed:
            # ファイル拡張子から技術領域を推定
            file_domain = self._extract_domain_from_file(file_path)
            file_score = developer_profile.expertise_areas.get(file_domain, 0.0)
            file_expertise_scores.append(file_score)
        
        avg_file_expertise = np.mean(file_expertise_scores) if file_expertise_scores else 0.0
        
        # 複雑度による調整
        complexity_factor = min(1.0, review_request.complexity_score)
        
        # 統合専門性スコア
        expertise_score = (
            0.6 * domain_expertise +
            0.4 * avg_file_expertise
        ) * (1.0 - 0.2 * complexity_factor)  # 複雑度が高いほど専門性要求が高い
        
        return max(0.0, min(1.0, expertise_score))
    
    def _calculate_workload_impact(
        self,
        review_request: ReviewRequest,
        developer_profile: DeveloperProfile
    ) -> float:
        """ワークロード影響度を計算
        
        Args:
            review_request: レビュー依頼
            developer_profile: 開発者プロファイル
            
        Returns:
            float: ワークロード影響度 (0.0-1.0, 高いほど負荷が少ない)
        """
        # 現在のワークロード状況
        current_load = developer_profile.current_workload
        
        # 推定レビュー時間
        estimated_hours = review_request.estimated_effort_hours
        
        # 緊急度による調整
        urgency_factor = review_request.urgency_level
        
        # ストレスレベルによる調整
        stress_factor = developer_profile.stress_level
        
        # ワークロード影響度を計算（低いほど受諾しやすい）
        workload_impact = 1.0 - (
            0.4 * current_load +
            0.3 * min(1.0, estimated_hours / 8.0) +  # 8時間を基準に正規化
            0.2 * urgency_factor +
            0.1 * stress_factor
        )
        
        return max(0.0, min(1.0, workload_impact))
    
    def _calculate_relationship_score(
        self,
        review_request: ReviewRequest,
        developer_profile: DeveloperProfile
    ) -> float:
        """関係性スコアを計算
        
        Args:
            review_request: レビュー依頼
            developer_profile: 開発者プロファイル
            
        Returns:
            float: 関係性スコア (0.0-1.0)
        """
        # 過去の協力関係を確認
        author_email = review_request.author_email
        collaboration_score = developer_profile.collaboration_history.get(
            author_email, 0.5  # デフォルト値
        )
        
        # 過去のレビュー履歴から関係性を分析
        review_history_score = self._analyze_review_history_with_author(
            developer_profile.review_history, author_email
        )
        
        # 統合関係性スコア
        relationship_score = (
            0.7 * collaboration_score +
            0.3 * review_history_score
        )
        
        return max(0.0, min(1.0, relationship_score))
    
    def _calculate_integrated_probability(
        self,
        expertise_score: float,
        workload_score: float,
        relationship_score: float
    ) -> float:
        """統合受諾確率を計算
        
        Args:
            expertise_score: 専門性スコア
            workload_score: ワークロードスコア
            relationship_score: 関係性スコア
            
        Returns:
            float: 統合受諾確率 (0.0-1.0)
        """
        # 重み付き平均
        probability = (
            self.expertise_weight * expertise_score +
            self.workload_weight * workload_score +
            self.relationship_weight * relationship_score
        )
        
        # シグモイド関数で調整（より現実的な確率分布）
        adjusted_probability = 1.0 / (1.0 + np.exp(-5.0 * (probability - 0.5)))
        
        return max(0.0, min(1.0, adjusted_probability))
    
    def _calculate_confidence_level(
        self,
        developer_profile: DeveloperProfile,
        review_request: ReviewRequest
    ) -> float:
        """予測の信頼度を計算
        
        Args:
            developer_profile: 開発者プロファイル
            review_request: レビュー依頼
            
        Returns:
            float: 信頼度 (0.0-1.0)
        """
        # レビュー履歴の豊富さ
        history_richness = min(1.0, len(developer_profile.review_history) / 50.0)
        
        # プロファイルの新しさ
        days_since_update = (datetime.now() - developer_profile.last_updated).days
        freshness = max(0.0, 1.0 - days_since_update / 30.0)  # 30日で信頼度0
        
        # 専門領域の一致度
        domain_familiarity = developer_profile.expertise_areas.get(
            review_request.technical_domain, 0.0
        )
        
        confidence = (
            0.4 * history_richness +
            0.3 * freshness +
            0.3 * domain_familiarity
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _identify_risk_factors(
        self,
        expertise_score: float,
        workload_score: float,
        relationship_score: float,
        developer_profile: DeveloperProfile,
        review_request: ReviewRequest
    ) -> List[str]:
        """リスク要因を特定
        
        Returns:
            List[str]: リスク要因のリスト
        """
        risk_factors = []
        
        if expertise_score < self.expertise_threshold:
            risk_factors.append("専門性不足")
        
        if workload_score < 0.3:
            risk_factors.append("高ワークロード")
        
        if relationship_score < 0.4:
            risk_factors.append("協力関係不足")
        
        if developer_profile.stress_level > 0.7:
            risk_factors.append("高ストレス状態")
        
        if review_request.complexity_score > 0.8:
            risk_factors.append("高複雑度レビュー")
        
        if review_request.urgency_level > 0.8:
            risk_factors.append("高緊急度")
        
        return risk_factors
    
    def _generate_recommendations(
        self,
        expertise_score: float,
        workload_score: float,
        relationship_score: float,
        developer_profile: DeveloperProfile,
        review_request: ReviewRequest
    ) -> List[str]:
        """推奨事項を生成
        
        Returns:
            List[str]: 推奨事項のリスト
        """
        recommendations = []
        
        if expertise_score < self.expertise_threshold:
            recommendations.append("より専門性の高いレビュワーを検討")
        
        if workload_score < 0.3:
            recommendations.append("レビュー時期の調整を検討")
        
        if relationship_score < 0.4:
            recommendations.append("事前のコミュニケーションを推奨")
        
        if review_request.complexity_score > 0.8:
            recommendations.append("レビューを複数人で分担することを検討")
        
        if developer_profile.stress_level > 0.7:
            recommendations.append("開発者の負荷軽減を優先")
        
        return recommendations
    
    def _extract_domain_from_file(self, file_path: str) -> str:
        """ファイルパスから技術領域を抽出
        
        Args:
            file_path: ファイルパス
            
        Returns:
            str: 技術領域
        """
        # 簡単な拡張子ベースの領域推定
        if file_path.endswith(('.py', '.pyx')):
            return 'python'
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            return 'javascript'
        elif file_path.endswith(('.java', '.kt')):
            return 'java'
        elif file_path.endswith(('.cpp', '.cc', '.c', '.h')):
            return 'cpp'
        elif file_path.endswith(('.go',)):
            return 'go'
        elif file_path.endswith(('.rs',)):
            return 'rust'
        elif file_path.endswith(('.sql',)):
            return 'database'
        elif file_path.endswith(('.yaml', '.yml', '.json')):
            return 'config'
        elif file_path.endswith(('.md', '.rst', '.txt')):
            return 'documentation'
        else:
            return 'general'
    
    def _analyze_review_history_with_author(
        self,
        review_history: List[Dict[str, Any]],
        author_email: str
    ) -> float:
        """特定の作者との過去のレビュー履歴を分析
        
        Args:
            review_history: レビュー履歴
            author_email: 作者のメールアドレス
            
        Returns:
            float: 履歴ベースの関係性スコア (0.0-1.0)
        """
        if not review_history:
            return 0.5  # デフォルト値
        
        # 該当作者とのレビュー履歴を抽出
        author_reviews = [
            review for review in review_history
            if review.get('author_email') == author_email
        ]
        
        if not author_reviews:
            return 0.5  # 履歴なし
        
        # 受諾率を計算
        accepted_count = sum(
            1 for review in author_reviews
            if review.get('decision') == ReviewDecision.ACCEPT.value
        )
        
        acceptance_rate = accepted_count / len(author_reviews)
        
        # 最近の履歴により重みを付ける
        recent_reviews = [
            review for review in author_reviews
            if (datetime.now() - review.get('timestamp', datetime.min)).days <= 90
        ]
        
        if recent_reviews:
            recent_acceptance_rate = sum(
                1 for review in recent_reviews
                if review.get('decision') == ReviewDecision.ACCEPT.value
            ) / len(recent_reviews)
            
            # 最近の履歴により重みを付ける
            score = 0.7 * recent_acceptance_rate + 0.3 * acceptance_rate
        else:
            score = acceptance_rate
        
        return max(0.0, min(1.0, score))


def create_sample_review_request() -> ReviewRequest:
    """サンプルのレビュー依頼を作成（テスト用）"""
    return ReviewRequest(
        change_id="I1234567890abcdef",
        author_email="author@example.com",
        reviewer_email="reviewer@example.com",
        subject="Fix bug in authentication module",
        files_changed=["src/auth/login.py", "tests/test_auth.py"],
        lines_added=25,
        lines_deleted=10,
        complexity_score=0.6,
        technical_domain="python",
        created_at=datetime.now(),
        urgency_level=0.7,
        estimated_effort_hours=2.0
    )


def create_sample_developer_profile() -> DeveloperProfile:
    """サンプルの開発者プロファイルを作成（テスト用）"""
    return DeveloperProfile(
        email="reviewer@example.com",
        name="Sample Reviewer",
        expertise_areas={
            "python": 0.8,
            "javascript": 0.6,
            "database": 0.4
        },
        current_workload=0.6,
        stress_level=0.4,
        collaboration_history={
            "author@example.com": 0.7,
            "other@example.com": 0.5
        },
        review_history=[
            {
                "author_email": "author@example.com",
                "decision": "accept",
                "timestamp": datetime.now() - timedelta(days=10)
            }
        ],
        avg_response_time_hours=4.5,
        acceptance_rate=0.75,
        last_updated=datetime.now() - timedelta(days=1)
    )


if __name__ == "__main__":
    # テスト実行
    config = {
        'expertise_weight': 0.4,
        'workload_weight': 0.3,
        'relationship_weight': 0.3,
        'workload_threshold': 0.8,
        'expertise_threshold': 0.6
    }
    
    analyzer = ReviewBehaviorAnalyzer(config)
    
    # サンプルデータでテスト
    review_request = create_sample_review_request()
    developer_profile = create_sample_developer_profile()
    
    result = analyzer.analyze_review_behavior(review_request, developer_profile)
    
    print(f"受諾確率: {result.acceptance_probability:.3f}")
    print(f"専門性スコア: {result.expertise_match_score:.3f}")
    print(f"ワークロードスコア: {result.workload_impact_score:.3f}")
    print(f"関係性スコア: {result.relationship_score:.3f}")
    print(f"信頼度: {result.confidence_level:.3f}")
    print(f"リスク要因: {result.risk_factors}")
    print(f"推奨事項: {result.recommendations}")