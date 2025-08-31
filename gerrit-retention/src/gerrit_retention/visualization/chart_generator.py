"""
チャート・レーダー生成システム

開発者の専門性と経験度を可視化するためのチャート生成機能を提供する。
技術領域別レーダーチャート、ファイル・ディレクトリ経験度マップを生成する。
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)


class ChartGenerator:
    """チャート・レーダー生成器
    
    開発者の専門性と経験度を可視化するチャートを生成する。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初期化
        
        Args:
            config: 可視化設定
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'outputs/visualizations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日本語フォント設定
        plt.rcParams['font.family'] = config.get('font_family', 'DejaVu Sans')
        plt.rcParams['figure.figsize'] = config.get('figure_size', (10, 8))
        
        # カラーパレット設定
        self.color_palette = config.get('color_palette', 'Set2')
        
    def generate_expertise_radar_chart(self, 
                                     developer_data: Dict[str, Any], 
                                     developer_email: str) -> str:
        """技術領域別専門性レーダーチャートを生成
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたレーダーチャートファイルのパス
        """
        try:
            # 開発者の専門性データを取得
            expertise_data = self._extract_expertise_data(developer_data, developer_email)
            
            if not expertise_data:
                logger.warning(f"開発者 {developer_email} の専門性データが見つかりません")
                return ""
            
            # レーダーチャートを作成
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 技術領域のラベル
            categories = list(expertise_data.keys())
            values = list(expertise_data.values())
            
            # 角度を計算
            angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
            angles += angles[:1]  # 円を閉じるため
            values += values[:1]  # 円を閉じるため
            
            # レーダーチャートを描画
            ax.plot(angles, values, 'o-', linewidth=2, label=developer_email)
            ax.fill(angles, values, alpha=0.25)
            
            # ラベルを設定
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            
            # グリッドを設定
            ax.grid(True)
            ax.set_title(f'技術領域別専門性レーダーチャート - {developer_email}', 
                        size=16, pad=20)
            
            # 凡例を追加
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            # ファイル保存
            filename = f"expertise_radar_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"専門性レーダーチャートを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"専門性レーダーチャート生成エラー: {e}")
            return ""
    
    def generate_file_experience_map(self, 
                                   developer_data: Dict[str, Any], 
                                   developer_email: str) -> str:
        """ファイル・ディレクトリ経験度マップを生成
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成された経験度マップファイルのパス
        """
        try:
            # ファイル経験度データを取得
            file_experience = self._extract_file_experience_data(developer_data, developer_email)
            
            if not file_experience:
                logger.warning(f"開発者 {developer_email} のファイル経験度データが見つかりません")
                return ""
            
            # 階層構造を作成
            hierarchy_data = self._build_file_hierarchy(file_experience)
            
            # ツリーマップを作成
            fig, ax = plt.subplots(figsize=(14, 10))
            
            self._draw_treemap(ax, hierarchy_data, 0, 0, 1, 1)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'ファイル・ディレクトリ経験度マップ - {developer_email}', 
                        fontsize=16, pad=20)
            
            # ファイル保存
            filename = f"file_experience_map_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ファイル経験度マップを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"ファイル経験度マップ生成エラー: {e}")
            return ""
    
    def generate_skill_progression_chart(self, 
                                       developer_data: Dict[str, Any], 
                                       developer_email: str) -> str:
        """スキル成長チャートを生成
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたスキル成長チャートファイルのパス
        """
        try:
            # スキル成長データを取得
            skill_progression = self._extract_skill_progression_data(developer_data, developer_email)
            
            if not skill_progression:
                logger.warning(f"開発者 {developer_email} のスキル成長データが見つかりません")
                return ""
            
            # 時系列チャートを作成
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'スキル成長チャート - {developer_email}', fontsize=16)
            
            # 各スキル領域のチャートを描画
            skill_areas = ['技術的専門性', 'レビュー品質', '協力関係', '学習速度']
            
            for i, skill_area in enumerate(skill_areas):
                ax = axes[i // 2, i % 2]
                
                if skill_area in skill_progression:
                    dates = skill_progression[skill_area]['dates']
                    values = skill_progression[skill_area]['values']
                    
                    ax.plot(dates, values, marker='o', linewidth=2)
                    ax.set_title(skill_area)
                    ax.set_xlabel('日付')
                    ax.set_ylabel('スキルレベル')
                    ax.grid(True, alpha=0.3)
                    
                    # トレンドラインを追加
                    if len(values) > 1:
                        z = np.polyfit(range(len(values)), values, 1)
                        p = np.poly1d(z)
                        ax.plot(dates, p(range(len(values))), "--", alpha=0.8, color='red')
                
            plt.tight_layout()
            
            # ファイル保存
            filename = f"skill_progression_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"スキル成長チャートを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"スキル成長チャート生成エラー: {e}")
            return ""
    
    def generate_collaboration_network_chart(self, 
                                           collaboration_data: List[Dict[str, Any]], 
                                           developer_email: str) -> str:
        """協力関係ネットワークチャートを生成
        
        Args:
            collaboration_data: 協力関係データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたネットワークチャートファイルのパス
        """
        try:
            # 協力関係ネットワークを構築
            network_data = self._build_collaboration_network(collaboration_data, developer_email)
            
            if not network_data:
                logger.warning(f"開発者 {developer_email} の協力関係データが見つかりません")
                return ""
            
            # ネットワークチャートを作成
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # ノードとエッジを描画
            self._draw_network_graph(ax, network_data, developer_email)
            
            ax.set_title(f'協力関係ネットワーク - {developer_email}', 
                        fontsize=16, pad=20)
            ax.axis('off')
            
            # ファイル保存
            filename = f"collaboration_network_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"協力関係ネットワークチャートを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"協力関係ネットワークチャート生成エラー: {e}")
            return ""
    
    def _extract_expertise_data(self, 
                              developer_data: Dict[str, Any], 
                              developer_email: str) -> Dict[str, float]:
        """開発者の専門性データを抽出
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            技術領域別専門性データ
        """
        expertise_data = {}
        
        try:
            dev_profile = developer_data.get(developer_email, {})
            
            # 技術領域別の専門性を抽出
            expertise_areas = [
                'Python', 'Java', 'JavaScript', 'C++', 'Go', 
                'Frontend', 'Backend', 'Database', 'DevOps', 'Testing'
            ]
            
            for area in expertise_areas:
                # 専門性スコアを計算（0-1の範囲）
                score = dev_profile.get(f'{area.lower()}_expertise', 0.0)
                expertise_data[area] = min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"専門性データ抽出エラー: {e}")
        
        return expertise_data
    
    def _extract_file_experience_data(self, 
                                    developer_data: Dict[str, Any], 
                                    developer_email: str) -> Dict[str, float]:
        """ファイル経験度データを抽出
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            ファイル・ディレクトリ別経験度データ
        """
        file_experience = {}
        
        try:
            dev_profile = developer_data.get(developer_email, {})
            file_history = dev_profile.get('file_history', {})
            
            for file_path, experience_score in file_history.items():
                file_experience[file_path] = experience_score
                
        except Exception as e:
            logger.warning(f"ファイル経験度データ抽出エラー: {e}")
        
        return file_experience
    
    def _extract_skill_progression_data(self, 
                                      developer_data: Dict[str, Any], 
                                      developer_email: str) -> Dict[str, Dict[str, List]]:
        """スキル成長データを抽出
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            スキル成長の時系列データ
        """
        skill_progression = {}
        
        try:
            dev_profile = developer_data.get(developer_email, {})
            skill_history = dev_profile.get('skill_history', {})
            
            for skill_area, history in skill_history.items():
                if isinstance(history, list) and history:
                    dates = [entry.get('date') for entry in history]
                    values = [entry.get('value', 0.0) for entry in history]
                    
                    skill_progression[skill_area] = {
                        'dates': pd.to_datetime(dates),
                        'values': values
                    }
                    
        except Exception as e:
            logger.warning(f"スキル成長データ抽出エラー: {e}")
        
        return skill_progression
    
    def _build_file_hierarchy(self, file_experience: Dict[str, float]) -> Dict[str, Any]:
        """ファイル階層構造を構築
        
        Args:
            file_experience: ファイル経験度データ
            
        Returns:
            階層構造データ
        """
        hierarchy = {}
        
        for file_path, experience in file_experience.items():
            parts = file_path.split('/')
            current = hierarchy
            
            for part in parts[:-1]:  # ディレクトリ部分
                if part not in current:
                    current[part] = {'children': {}, 'experience': 0.0}
                current = current[part]['children']
            
            # ファイル部分
            filename = parts[-1]
            current[filename] = {'experience': experience, 'children': {}}
        
        return hierarchy
    
    def _draw_treemap(self, ax, data: Dict[str, Any], x: float, y: float, 
                     width: float, height: float, level: int = 0):
        """ツリーマップを描画
        
        Args:
            ax: matplotlib軸
            data: 階層データ
            x, y: 描画位置
            width, height: 描画サイズ
            level: 階層レベル
        """
        if not data:
            return
        
        # 経験度に基づいてサイズを計算
        total_experience = sum(
            item.get('experience', 0) + sum(
                child.get('experience', 0) 
                for child in item.get('children', {}).values()
            )
            for item in data.values()
        )
        
        if total_experience == 0:
            return
        
        current_x = x
        current_y = y
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        for i, (name, item) in enumerate(data.items()):
            item_experience = item.get('experience', 0) + sum(
                child.get('experience', 0) 
                for child in item.get('children', {}).values()
            )
            
            if item_experience == 0:
                continue
            
            # サイズを計算
            item_width = width * (item_experience / total_experience)
            
            # 矩形を描画
            rect = plt.Rectangle(
                (current_x, current_y), item_width, height,
                facecolor=colors[i], edgecolor='white', linewidth=1,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # ラベルを追加
            if item_width > 0.05 and height > 0.05:  # 十分な大きさの場合のみ
                ax.text(
                    current_x + item_width/2, current_y + height/2,
                    name, ha='center', va='center',
                    fontsize=max(8 - level, 6), weight='bold'
                )
            
            # 子要素を再帰的に描画
            if item.get('children') and level < 3:  # 最大3階層まで
                self._draw_treemap(
                    ax, item['children'],
                    current_x, current_y, item_width, height * 0.7,
                    level + 1
                )
            
            current_x += item_width
    
    def _build_collaboration_network(self, 
                                   collaboration_data: List[Dict[str, Any]], 
                                   developer_email: str) -> Dict[str, Any]:
        """協力関係ネットワークを構築
        
        Args:
            collaboration_data: 協力関係データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            ネットワークデータ
        """
        network = {
            'nodes': {},
            'edges': []
        }
        
        # 中心ノード（対象開発者）を追加
        network['nodes'][developer_email] = {
            'x': 0.5, 'y': 0.5, 'size': 100, 'color': 'red'
        }
        
        # 協力相手を特定
        collaborators = {}
        for collab in collaboration_data:
            if collab.get('developer1') == developer_email:
                partner = collab.get('developer2')
            elif collab.get('developer2') == developer_email:
                partner = collab.get('developer1')
            else:
                continue
            
            if partner not in collaborators:
                collaborators[partner] = 0
            collaborators[partner] += collab.get('interaction_count', 1)
        
        # 協力相手ノードを円形に配置
        num_collaborators = len(collaborators)
        if num_collaborators > 0:
            for i, (partner, interaction_count) in enumerate(collaborators.items()):
                angle = 2 * math.pi * i / num_collaborators
                x = 0.5 + 0.3 * math.cos(angle)
                y = 0.5 + 0.3 * math.sin(angle)
                
                network['nodes'][partner] = {
                    'x': x, 'y': y, 
                    'size': min(50 + interaction_count * 5, 150),
                    'color': 'blue'
                }
                
                # エッジを追加
                network['edges'].append({
                    'from': developer_email,
                    'to': partner,
                    'weight': interaction_count
                })
        
        return network
    
    def _draw_network_graph(self, ax, network_data: Dict[str, Any], developer_email: str):
        """ネットワークグラフを描画
        
        Args:
            ax: matplotlib軸
            network_data: ネットワークデータ
            developer_email: 対象開発者のメールアドレス
        """
        # エッジを描画
        for edge in network_data['edges']:
            from_node = network_data['nodes'][edge['from']]
            to_node = network_data['nodes'][edge['to']]
            
            ax.plot(
                [from_node['x'], to_node['x']],
                [from_node['y'], to_node['y']],
                'k-', alpha=0.5, linewidth=edge['weight'] / 5
            )
        
        # ノードを描画
        for node_id, node in network_data['nodes'].items():
            circle = Circle(
                (node['x'], node['y']), 
                node['size'] / 1000,
                color=node['color'], alpha=0.7
            )
            ax.add_patch(circle)
            
            # ラベルを追加
            label = node_id.split('@')[0] if '@' in node_id else node_id
            ax.text(
                node['x'], node['y'] - node['size'] / 800,
                label, ha='center', va='top',
                fontsize=10, weight='bold' if node_id == developer_email else 'normal'
            )
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    
    def generate_comprehensive_chart_report(self, 
                                          developer_data: Dict[str, Any], 
                                          collaboration_data: List[Dict[str, Any]], 
                                          developer_email: str) -> Dict[str, str]:
        """包括的なチャートレポートを生成
        
        Args:
            developer_data: 開発者データ
            collaboration_data: 協力関係データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたチャートファイルパスの辞書
        """
        report_paths = {}
        
        try:
            # 専門性レーダーチャート
            expertise_path = self.generate_expertise_radar_chart(
                developer_data, developer_email
            )
            if expertise_path:
                report_paths['expertise_radar'] = expertise_path
            
            # ファイル経験度マップ
            file_experience_path = self.generate_file_experience_map(
                developer_data, developer_email
            )
            if file_experience_path:
                report_paths['file_experience'] = file_experience_path
            
            # スキル成長チャート
            skill_progression_path = self.generate_skill_progression_chart(
                developer_data, developer_email
            )
            if skill_progression_path:
                report_paths['skill_progression'] = skill_progression_path
            
            # 協力関係ネットワークチャート
            collaboration_network_path = self.generate_collaboration_network_chart(
                collaboration_data, developer_email
            )
            if collaboration_network_path:
                report_paths['collaboration_network'] = collaboration_network_path
            
            logger.info(f"開発者 {developer_email} の包括的チャートレポートを生成しました")
            
        except Exception as e:
            logger.error(f"包括的チャートレポート生成エラー: {e}")
        
        return report_paths