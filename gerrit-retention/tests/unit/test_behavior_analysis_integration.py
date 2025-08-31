"""
行動分析システム統合テスト

レビュー行動分析、類似度計算、好み分析の統合テストを実行する。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from gerrit_retention.behavior_analysis import (
    ChangeInfo,
    DeveloperProfile,
    PreferenceAnalyzer,
    ReviewBehaviorAnalyzer,
    ReviewDecision,
    ReviewHistoryEntry,
    ReviewRequest,
    SimilarityCalculator,
)


class TestBehaviorAnalysisIntegration:
    """行動分析システム統合テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        # ReviewBehaviorAnalyzer設定
        self.review_config = {
            'expertise_weight': 0.4,
            'workload_weight': 0.3,
            'relationship_weight': 0.3,
            'workload_threshold': 0.8,
            'expertise_threshold': 0.6
        }
        self.review_analyzer = ReviewBehaviorAnalyzer(self.review_config)
        
        # SimilarityCalculator設定
        self.similarity_config = {
            'file_path_weight': 0.3,
            'tech_stack_weight': 0.25,
            'complexity_weight': 0.15,
            'functional_weight': 0.2,
            'domain_weight': 0.1
        }
        self.similarity_calculator = SimilarityCalculator(self.similarity_config)
        
        # PreferenceAnalyzer設定
        self.preference_config = {
            'min_history_size': 5,  # テスト用に小さく設定
            'preference_decay_days': 90,
            'tolerance_window_days': 30,
            'confidence_threshold': 0.7
        }
        self.preference_analyzer = PreferenceAnalyzer(self.preference_config)
    
    def test_review_behavior_analyzer_basic_functionality(self):
        """レビュー行動分析器の基本機能テスト"""
        # テストデータ作成
        review_request = ReviewRequest(
            change_id="I1234567890abcdef",
            author_email="author@example.com",
            reviewer_email="reviewer@example.com",
            subject="Fix authentication bug",
            files_changed=["src/auth/login.py", "tests/test_auth.py"],
            lines_added=25,
            lines_deleted=10,
            complexity_score=0.6,
            technical_domain="python",
            created_at=datetime.now(),
            urgency_level=0.7,
            estimated_effort_hours=2.0
        )
        
        developer_profile = DeveloperProfile(
            email="reviewer@example.com",
            name="Test Reviewer",
            expertise_areas={"python": 0.8, "javascript": 0.6},
            current_workload=0.6,
            stress_level=0.4,
            collaboration_history={"author@example.com": 0.7},
            review_history=[],
            avg_response_time_hours=4.5,
            acceptance_rate=0.75,
            last_updated=datetime.now()
        )
        
        # 分析実行
        result = self.review_analyzer.analyze_review_behavior(
            review_request, developer_profile
        )
        
        # 結果検証
        assert 0.0 <= result.acceptance_probability <= 1.0
        assert 0.0 <= result.expertise_match_score <= 1.0
        assert 0.0 <= result.workload_impact_score <= 1.0
        assert 0.0 <= result.relationship_score <= 1.0
        assert 0.0 <= result.confidence_level <= 1.0
        assert isinstance(result.risk_factors, list)
        assert isinstance(result.recommendations, list)
    
    def test_similarity_calculator_basic_functionality(self):
        """類似度計算器の基本機能テスト"""
        # テストデータ作成
        change1 = ChangeInfo(
            change_id="I1234567890abcdef",
            files_changed=["src/auth/login.py", "src/auth/models.py"],
            lines_added=45,
            lines_deleted=12,
            complexity_score=0.7,
            technical_domains=["python", "database"],
            functional_areas=["authentication", "testing"],
            author_email="dev1@example.com",
            created_at="2024-01-15T10:30:00Z"
        )
        
        change2 = ChangeInfo(
            change_id="I2345678901bcdefg",
            files_changed=["src/auth/logout.py", "src/auth/models.py"],
            lines_added=30,
            lines_deleted=8,
            complexity_score=0.6,
            technical_domains=["python", "database"],
            functional_areas=["authentication", "testing"],
            author_email="dev2@example.com",
            created_at="2024-01-16T14:20:00Z"
        )
        
        # 類似度計算
        result = self.similarity_calculator.calculate_similarity(change1, change2)
        
        # 結果検証
        assert 0.0 <= result.overall_similarity <= 1.0
        assert 0.0 <= result.file_path_similarity <= 1.0
        assert 0.0 <= result.technical_stack_similarity <= 1.0
        assert 0.0 <= result.complexity_similarity <= 1.0
        assert 0.0 <= result.functional_area_similarity <= 1.0
        assert 0.0 <= result.domain_similarity <= 1.0
        assert isinstance(result.detailed_breakdown, dict)
        
        # 高い類似度が期待される（同じ技術ドメインと機能領域）
        assert result.technical_stack_similarity > 0.5
        assert result.functional_area_similarity > 0.5
    
    def test_preference_analyzer_basic_functionality(self):
        """好み分析器の基本機能テスト"""
        # テスト用レビュー履歴作成
        review_history = []
        base_time = datetime.now() - timedelta(days=60)
        
        for i in range(10):
            entry = ReviewHistoryEntry(
                change_id=f"I{i:010d}",
                author_email=f"author{i % 3}@example.com",
                decision=ReviewDecision.ACCEPT if i % 3 != 0 else ReviewDecision.DECLINE,
                timestamp=base_time + timedelta(days=i * 6),
                technical_domains=["python", "javascript"][i % 2:i % 2 + 1],
                functional_areas=["api", "testing"][i % 2:i % 2 + 1],
                complexity_score=0.3 + (i % 5) * 0.1,
                change_size=50 + i * 20,
                response_time_hours=2.0 + (i % 3),
                relationship_score=0.4 + (i % 4) * 0.1,
                context_features={
                    'stress_level': 0.3 + (i % 3) * 0.1,
                    'workload': 0.4 + (i % 4) * 0.1
                }
            )
            review_history.append(entry)
        
        # 好み分析実行
        result = self.preference_analyzer.analyze_preferences(
            "test@example.com", review_history
        )
        
        # 結果検証
        assert result.preference_profile.developer_email == "test@example.com"
        assert 0.0 <= result.preference_profile.overall_acceptance_rate <= 1.0
        assert 0.0 <= result.preference_profile.confidence_score <= 1.0
        assert isinstance(result.preference_profile.technical_domain_preferences, dict)
        assert isinstance(result.preference_profile.functional_area_preferences, dict)
        
        assert result.tolerance_limit.consecutive_non_preferred_limit >= 1
        assert 0.0 <= result.tolerance_limit.stress_threshold <= 1.0
        assert 0.0 <= result.tolerance_limit.workload_threshold <= 1.0
        
        assert isinstance(result.current_tolerance_status, dict)
        assert isinstance(result.risk_assessment, dict)
        assert isinstance(result.recommendations, list)
    
    def test_integration_workflow(self):
        """統合ワークフローテスト"""
        # 1. 開発者の過去のレビュー履歴から好みを分析
        review_history = self._create_sample_review_history()
        preference_result = self.preference_analyzer.analyze_preferences(
            "developer@example.com", review_history
        )
        
        # 2. 新しいレビュー依頼の類似度を計算
        current_change = ChangeInfo(
            change_id="I_new_change",
            files_changed=["src/api/users.py", "tests/test_users.py"],
            lines_added=35,
            lines_deleted=5,
            complexity_score=0.5,
            technical_domains=["python"],
            functional_areas=["api", "testing"],
            author_email="requester@example.com",
            created_at="2024-01-20T09:00:00Z"
        )
        
        # 過去のChangeとの類似度を計算
        past_change = ChangeInfo(
            change_id="I_past_change",
            files_changed=["src/api/auth.py", "tests/test_auth.py"],
            lines_added=40,
            lines_deleted=8,
            complexity_score=0.6,
            technical_domains=["python"],
            functional_areas=["api", "testing"],
            author_email="other@example.com",
            created_at="2024-01-10T14:00:00Z"
        )
        
        similarity_result = self.similarity_calculator.calculate_similarity(
            current_change, past_change
        )
        
        # 3. レビュー行動分析で受諾確率を予測
        review_request = ReviewRequest(
            change_id=current_change.change_id,
            author_email=current_change.author_email,
            reviewer_email="developer@example.com",
            subject="Add user management API",
            files_changed=current_change.files_changed,
            lines_added=current_change.lines_added,
            lines_deleted=current_change.lines_deleted,
            complexity_score=current_change.complexity_score,
            technical_domain=current_change.technical_domains[0],
            created_at=datetime.now(),
            urgency_level=0.5,
            estimated_effort_hours=1.5
        )
        
        developer_profile = DeveloperProfile(
            email="developer@example.com",
            name="Test Developer",
            expertise_areas=preference_result.preference_profile.technical_domain_preferences,
            current_workload=0.5,
            stress_level=0.3,
            collaboration_history=preference_result.preference_profile.author_preferences,
            review_history=[],
            avg_response_time_hours=3.0,
            acceptance_rate=preference_result.preference_profile.overall_acceptance_rate,
            last_updated=datetime.now()
        )
        
        behavior_result = self.review_analyzer.analyze_review_behavior(
            review_request, developer_profile
        )
        
        # 統合結果の検証
        assert preference_result.preference_profile.confidence_score >= 0.0
        assert similarity_result.overall_similarity >= 0.0
        assert behavior_result.acceptance_probability >= 0.0
        
        # 高い類似度と好みの一致により、高い受諾確率が期待される
        if (similarity_result.overall_similarity > 0.7 and 
            preference_result.preference_profile.overall_acceptance_rate > 0.6):
            assert behavior_result.acceptance_probability > 0.5
    
    def _create_sample_review_history(self) -> list:
        """サンプルレビュー履歴を作成"""
        history = []
        base_time = datetime.now() - timedelta(days=90)
        
        for i in range(15):
            entry = ReviewHistoryEntry(
                change_id=f"I{i:010d}",
                author_email=f"author{i % 4}@example.com",
                decision=ReviewDecision.ACCEPT if i % 4 != 3 else ReviewDecision.DECLINE,
                timestamp=base_time + timedelta(days=i * 6),
                technical_domains=["python", "javascript", "java"][i % 3:i % 3 + 1],
                functional_areas=["api", "testing", "frontend"][i % 3:i % 3 + 1],
                complexity_score=0.2 + (i % 8) * 0.1,
                change_size=30 + i * 15,
                response_time_hours=1.0 + (i % 6),
                relationship_score=0.3 + (i % 7) * 0.1,
                context_features={
                    'stress_level': 0.2 + (i % 5) * 0.1,
                    'workload': 0.3 + (i % 6) * 0.1
                }
            )
            history.append(entry)
        
        return history


if __name__ == "__main__":
    # 簡単なテスト実行
    test_instance = TestBehaviorAnalysisIntegration()
    test_instance.setup_method()
    
    print("レビュー行動分析器テスト実行中...")
    test_instance.test_review_behavior_analyzer_basic_functionality()
    print("✓ レビュー行動分析器テスト完了")
    
    print("類似度計算器テスト実行中...")
    test_instance.test_similarity_calculator_basic_functionality()
    print("✓ 類似度計算器テスト完了")
    
    print("好み分析器テスト実行中...")
    test_instance.test_preference_analyzer_basic_functionality()
    print("✓ 好み分析器テスト完了")
    
    print("統合ワークフローテスト実行中...")
    test_instance.test_integration_workflow()
    print("✓ 統合ワークフローテスト完了")
    
    print("すべてのテストが正常に完了しました！")