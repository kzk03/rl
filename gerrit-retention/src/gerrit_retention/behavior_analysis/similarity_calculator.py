"""
類似度計算システム

ファイルパス、技術スタック、Change複雑度による類似度計算と
機能領域・ドメイン類似度の算出を行う。
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChangeInfo:
    """Change情報"""
    change_id: str
    files_changed: List[str]
    lines_added: int
    lines_deleted: int
    complexity_score: float
    technical_domains: List[str]
    functional_areas: List[str]
    author_email: str
    created_at: str


@dataclass
class SimilarityResult:
    """類似度計算結果"""
    overall_similarity: float
    file_path_similarity: float
    technical_stack_similarity: float
    complexity_similarity: float
    functional_area_similarity: float
    domain_similarity: float
    detailed_breakdown: Dict[str, Any]


class SimilarityCalculator:
    """類似度計算システム
    
    ファイルパス、技術スタック、Change複雑度による類似度計算と
    機能領域・ドメイン類似度の算出を行う。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定辞書
                - file_path_weight: ファイルパス重み (default: 0.3)
                - tech_stack_weight: 技術スタック重み (default: 0.25)
                - complexity_weight: 複雑度重み (default: 0.15)
                - functional_weight: 機能領域重み (default: 0.2)
                - domain_weight: ドメイン重み (default: 0.1)
        """
        self.config = config
        self.file_path_weight = config.get('file_path_weight', 0.3)
        self.tech_stack_weight = config.get('tech_stack_weight', 0.25)
        self.complexity_weight = config.get('complexity_weight', 0.15)
        self.functional_weight = config.get('functional_weight', 0.2)
        self.domain_weight = config.get('domain_weight', 0.1)
        
        # 技術スタック分類辞書
        self.tech_stack_mapping = self._initialize_tech_stack_mapping()
        
        # 機能領域分類辞書
        self.functional_area_mapping = self._initialize_functional_area_mapping()
        
        logger.info(f"SimilarityCalculator initialized with weights: "
                   f"file_path={self.file_path_weight}, "
                   f"tech_stack={self.tech_stack_weight}, "
                   f"complexity={self.complexity_weight}, "
                   f"functional={self.functional_weight}, "
                   f"domain={self.domain_weight}")
    
    def calculate_similarity(
        self,
        change1: ChangeInfo,
        change2: ChangeInfo
    ) -> SimilarityResult:
        """2つのChangeの類似度を計算
        
        Args:
            change1: 比較対象Change1
            change2: 比較対象Change2
            
        Returns:
            SimilarityResult: 類似度計算結果
        """
        try:
            # ファイルパス類似度を計算
            file_path_sim = self._calculate_file_path_similarity(
                change1.files_changed, change2.files_changed
            )
            
            # 技術スタック類似度を計算
            tech_stack_sim = self._calculate_technical_stack_similarity(
                change1.technical_domains, change2.technical_domains
            )
            
            # 複雑度類似度を計算
            complexity_sim = self._calculate_complexity_similarity(
                change1.complexity_score, change2.complexity_score
            )
            
            # 機能領域類似度を計算
            functional_sim = self._calculate_functional_area_similarity(
                change1.functional_areas, change2.functional_areas
            )
            
            # ドメイン類似度を計算
            domain_sim = self._calculate_domain_similarity(
                change1.technical_domains, change2.technical_domains
            )
            
            # 総合類似度を計算
            overall_sim = (
                self.file_path_weight * file_path_sim +
                self.tech_stack_weight * tech_stack_sim +
                self.complexity_weight * complexity_sim +
                self.functional_weight * functional_sim +
                self.domain_weight * domain_sim
            )
            
            # 詳細な分析結果
            detailed_breakdown = {
                'file_overlap_ratio': self._calculate_file_overlap_ratio(
                    change1.files_changed, change2.files_changed
                ),
                'directory_similarity': self._calculate_directory_similarity(
                    change1.files_changed, change2.files_changed
                ),
                'size_similarity': self._calculate_size_similarity(
                    change1.lines_added + change1.lines_deleted,
                    change2.lines_added + change2.lines_deleted
                ),
                'common_tech_domains': list(
                    set(change1.technical_domains) & set(change2.technical_domains)
                ),
                'common_functional_areas': list(
                    set(change1.functional_areas) & set(change2.functional_areas)
                )
            }
            
            return SimilarityResult(
                overall_similarity=overall_sim,
                file_path_similarity=file_path_sim,
                technical_stack_similarity=tech_stack_sim,
                complexity_similarity=complexity_sim,
                functional_area_similarity=functional_sim,
                domain_similarity=domain_sim,
                detailed_breakdown=detailed_breakdown
            )
            
        except Exception as e:
            logger.error(f"類似度計算中にエラーが発生: {e}")
            # フォールバック値を返す
            return SimilarityResult(
                overall_similarity=0.0,
                file_path_similarity=0.0,
                technical_stack_similarity=0.0,
                complexity_similarity=0.0,
                functional_area_similarity=0.0,
                domain_similarity=0.0,
                detailed_breakdown={}
            )
    
    def calculate_batch_similarity(
        self,
        target_change: ChangeInfo,
        candidate_changes: List[ChangeInfo],
        top_k: int = 10
    ) -> List[Tuple[ChangeInfo, SimilarityResult]]:
        """複数のChangeとの類似度を一括計算
        
        Args:
            target_change: 対象Change
            candidate_changes: 候補Changeリスト
            top_k: 上位k件を返す
            
        Returns:
            List[Tuple[ChangeInfo, SimilarityResult]]: 類似度順のChangeと結果のペア
        """
        similarities = []
        
        for candidate in candidate_changes:
            if candidate.change_id == target_change.change_id:
                continue  # 自分自身は除外
            
            similarity = self.calculate_similarity(target_change, candidate)
            similarities.append((candidate, similarity))
        
        # 類似度順にソート
        similarities.sort(key=lambda x: x[1].overall_similarity, reverse=True)
        
        return similarities[:top_k]
    
    def _calculate_file_path_similarity(
        self,
        files1: List[str],
        files2: List[str]
    ) -> float:
        """ファイルパス類似度を計算
        
        Args:
            files1: ファイルパスリスト1
            files2: ファイルパスリスト2
            
        Returns:
            float: ファイルパス類似度 (0.0-1.0)
        """
        if not files1 or not files2:
            return 0.0
        
        # 直接的なファイル重複
        files1_set = set(files1)
        files2_set = set(files2)
        direct_overlap = len(files1_set & files2_set)
        
        if direct_overlap > 0:
            # 直接重複がある場合は高い類似度
            jaccard_similarity = direct_overlap / len(files1_set | files2_set)
            return min(1.0, jaccard_similarity * 2.0)  # 重複を重視
        
        # ディレクトリレベルの類似度
        dir_similarity = self._calculate_directory_similarity(files1, files2)
        
        # ファイル名の類似度
        filename_similarity = self._calculate_filename_similarity(files1, files2)
        
        # 拡張子の類似度
        extension_similarity = self._calculate_extension_similarity(files1, files2)
        
        # 統合類似度
        path_similarity = (
            0.5 * dir_similarity +
            0.3 * filename_similarity +
            0.2 * extension_similarity
        )
        
        return max(0.0, min(1.0, path_similarity))
    
    def _calculate_technical_stack_similarity(
        self,
        domains1: List[str],
        domains2: List[str]
    ) -> float:
        """技術スタック類似度を計算
        
        Args:
            domains1: 技術ドメインリスト1
            domains2: 技術ドメインリスト2
            
        Returns:
            float: 技術スタック類似度 (0.0-1.0)
        """
        if not domains1 or not domains2:
            return 0.0
        
        domains1_set = set(domains1)
        domains2_set = set(domains2)
        
        # Jaccard類似度
        intersection = len(domains1_set & domains2_set)
        union = len(domains1_set | domains2_set)
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        
        # 技術スタックの関連性を考慮
        related_similarity = self._calculate_related_tech_similarity(
            domains1, domains2
        )
        
        # 統合類似度
        tech_similarity = 0.7 * jaccard_sim + 0.3 * related_similarity
        
        return max(0.0, min(1.0, tech_similarity))
    
    def _calculate_complexity_similarity(
        self,
        complexity1: float,
        complexity2: float
    ) -> float:
        """複雑度類似度を計算
        
        Args:
            complexity1: 複雑度1
            complexity2: 複雑度2
            
        Returns:
            float: 複雑度類似度 (0.0-1.0)
        """
        # 複雑度の差を類似度に変換
        complexity_diff = abs(complexity1 - complexity2)
        
        # ガウシアン類似度（差が小さいほど類似度が高い）
        similarity = np.exp(-2.0 * complexity_diff)
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_functional_area_similarity(
        self,
        areas1: List[str],
        areas2: List[str]
    ) -> float:
        """機能領域類似度を計算
        
        Args:
            areas1: 機能領域リスト1
            areas2: 機能領域リスト2
            
        Returns:
            float: 機能領域類似度 (0.0-1.0)
        """
        if not areas1 or not areas2:
            return 0.0
        
        areas1_set = set(areas1)
        areas2_set = set(areas2)
        
        # Jaccard類似度
        intersection = len(areas1_set & areas2_set)
        union = len(areas1_set | areas2_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_domain_similarity(
        self,
        domains1: List[str],
        domains2: List[str]
    ) -> float:
        """ドメイン類似度を計算
        
        Args:
            domains1: ドメインリスト1
            domains2: ドメインリスト2
            
        Returns:
            float: ドメイン類似度 (0.0-1.0)
        """
        if not domains1 or not domains2:
            return 0.0
        
        # コサイン類似度を使用
        domain_vector1 = self._create_domain_vector(domains1)
        domain_vector2 = self._create_domain_vector(domains2)
        
        return self._cosine_similarity(domain_vector1, domain_vector2)
    
    def _calculate_directory_similarity(
        self,
        files1: List[str],
        files2: List[str]
    ) -> float:
        """ディレクトリ類似度を計算"""
        dirs1 = set(str(Path(f).parent) for f in files1)
        dirs2 = set(str(Path(f).parent) for f in files2)
        
        if not dirs1 or not dirs2:
            return 0.0
        
        intersection = len(dirs1 & dirs2)
        union = len(dirs1 | dirs2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_filename_similarity(
        self,
        files1: List[str],
        files2: List[str]
    ) -> float:
        """ファイル名類似度を計算"""
        names1 = [Path(f).stem for f in files1]
        names2 = [Path(f).stem for f in files2]
        
        max_similarity = 0.0
        for name1 in names1:
            for name2 in names2:
                similarity = SequenceMatcher(None, name1, name2).ratio()
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_extension_similarity(
        self,
        files1: List[str],
        files2: List[str]
    ) -> float:
        """拡張子類似度を計算"""
        exts1 = set(Path(f).suffix for f in files1)
        exts2 = set(Path(f).suffix for f in files2)
        
        if not exts1 or not exts2:
            return 0.0
        
        intersection = len(exts1 & exts2)
        union = len(exts1 | exts2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_file_overlap_ratio(
        self,
        files1: List[str],
        files2: List[str]
    ) -> float:
        """ファイル重複率を計算"""
        files1_set = set(files1)
        files2_set = set(files2)
        
        if not files1_set or not files2_set:
            return 0.0
        
        intersection = len(files1_set & files2_set)
        min_size = min(len(files1_set), len(files2_set))
        
        return intersection / min_size if min_size > 0 else 0.0
    
    def _calculate_size_similarity(self, size1: int, size2: int) -> float:
        """サイズ類似度を計算"""
        if size1 == 0 and size2 == 0:
            return 1.0
        
        max_size = max(size1, size2)
        min_size = min(size1, size2)
        
        if max_size == 0:
            return 1.0
        
        return min_size / max_size
    
    def _calculate_related_tech_similarity(
        self,
        domains1: List[str],
        domains2: List[str]
    ) -> float:
        """関連技術の類似度を計算"""
        # 技術スタック間の関連性マップ
        tech_relations = {
            'python': ['django', 'flask', 'fastapi', 'pytest'],
            'javascript': ['react', 'vue', 'angular', 'nodejs'],
            'java': ['spring', 'maven', 'gradle', 'junit'],
            'cpp': ['cmake', 'boost', 'qt'],
            'database': ['sql', 'postgresql', 'mysql', 'mongodb']
        }
        
        related_score = 0.0
        total_comparisons = 0
        
        for domain1 in domains1:
            for domain2 in domains2:
                total_comparisons += 1
                
                # 直接一致
                if domain1 == domain2:
                    related_score += 1.0
                    continue
                
                # 関連技術チェック
                related1 = tech_relations.get(domain1, [])
                related2 = tech_relations.get(domain2, [])
                
                if domain2 in related1 or domain1 in related2:
                    related_score += 0.7
                elif set(related1) & set(related2):
                    related_score += 0.3
        
        return related_score / total_comparisons if total_comparisons > 0 else 0.0
    
    def _create_domain_vector(self, domains: List[str]) -> np.ndarray:
        """ドメインベクトルを作成"""
        all_domains = list(self.tech_stack_mapping.keys())
        vector = np.zeros(len(all_domains))
        
        for i, domain in enumerate(all_domains):
            if domain in domains:
                vector[i] = 1.0
        
        return vector
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """コサイン類似度を計算"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _initialize_tech_stack_mapping(self) -> Dict[str, List[str]]:
        """技術スタック分類マッピングを初期化"""
        return {
            'python': ['.py', '.pyx', '.pyi'],
            'javascript': ['.js', '.ts', '.jsx', '.tsx'],
            'java': ['.java', '.kt', '.scala'],
            'cpp': ['.cpp', '.cc', '.c', '.h', '.hpp'],
            'go': ['.go'],
            'rust': ['.rs'],
            'database': ['.sql'],
            'config': ['.yaml', '.yml', '.json', '.toml', '.ini'],
            'documentation': ['.md', '.rst', '.txt'],
            'web': ['.html', '.css', '.scss', '.less'],
            'shell': ['.sh', '.bash', '.zsh'],
            'docker': ['Dockerfile', '.dockerignore'],
            'build': ['Makefile', 'CMakeLists.txt', 'build.gradle', 'pom.xml']
        }
    
    def _initialize_functional_area_mapping(self) -> Dict[str, List[str]]:
        """機能領域分類マッピングを初期化"""
        return {
            'authentication': ['auth', 'login', 'oauth', 'jwt', 'session'],
            'database': ['db', 'model', 'migration', 'schema', 'query'],
            'api': ['api', 'rest', 'graphql', 'endpoint', 'controller'],
            'frontend': ['ui', 'component', 'view', 'template', 'style'],
            'testing': ['test', 'spec', 'mock', 'fixture', 'e2e'],
            'configuration': ['config', 'setting', 'env', 'properties'],
            'security': ['security', 'crypto', 'hash', 'encrypt', 'ssl'],
            'logging': ['log', 'audit', 'monitor', 'trace'],
            'deployment': ['deploy', 'docker', 'k8s', 'ci', 'cd'],
            'documentation': ['doc', 'readme', 'guide', 'manual']
        }
    
    def extract_technical_domains_from_files(
        self,
        files: List[str]
    ) -> List[str]:
        """ファイルリストから技術ドメインを抽出"""
        domains = set()
        
        for file_path in files:
            file_ext = Path(file_path).suffix.lower()
            
            for domain, extensions in self.tech_stack_mapping.items():
                if file_ext in extensions or any(ext in file_path for ext in extensions):
                    domains.add(domain)
        
        return list(domains)
    
    def extract_functional_areas_from_files(
        self,
        files: List[str]
    ) -> List[str]:
        """ファイルリストから機能領域を抽出"""
        areas = set()
        
        for file_path in files:
            file_path_lower = file_path.lower()
            
            for area, keywords in self.functional_area_mapping.items():
                if any(keyword in file_path_lower for keyword in keywords):
                    areas.add(area)
        
        return list(areas)


def create_sample_change_info() -> ChangeInfo:
    """サンプルのChange情報を作成（テスト用）"""
    return ChangeInfo(
        change_id="I1234567890abcdef",
        files_changed=[
            "src/auth/login.py",
            "src/auth/models.py",
            "tests/test_auth.py"
        ],
        lines_added=45,
        lines_deleted=12,
        complexity_score=0.7,
        technical_domains=["python", "database"],
        functional_areas=["authentication", "testing"],
        author_email="developer@example.com",
        created_at="2024-01-15T10:30:00Z"
    )


if __name__ == "__main__":
    # テスト実行
    config = {
        'file_path_weight': 0.3,
        'tech_stack_weight': 0.25,
        'complexity_weight': 0.15,
        'functional_weight': 0.2,
        'domain_weight': 0.1
    }
    
    calculator = SimilarityCalculator(config)
    
    # サンプルデータでテスト
    change1 = create_sample_change_info()
    change2 = ChangeInfo(
        change_id="I2345678901bcdefg",
        files_changed=[
            "src/auth/logout.py",
            "src/auth/models.py",
            "tests/test_logout.py"
        ],
        lines_added=30,
        lines_deleted=8,
        complexity_score=0.6,
        technical_domains=["python", "database"],
        functional_areas=["authentication", "testing"],
        author_email="another@example.com",
        created_at="2024-01-16T14:20:00Z"
    )
    
    result = calculator.calculate_similarity(change1, change2)
    
    print(f"総合類似度: {result.overall_similarity:.3f}")
    print(f"ファイルパス類似度: {result.file_path_similarity:.3f}")
    print(f"技術スタック類似度: {result.technical_stack_similarity:.3f}")
    print(f"複雑度類似度: {result.complexity_similarity:.3f}")
    print(f"機能領域類似度: {result.functional_area_similarity:.3f}")
    print(f"ドメイン類似度: {result.domain_similarity:.3f}")
    print(f"詳細分析: {result.detailed_breakdown}")