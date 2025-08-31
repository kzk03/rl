"""
ヒートマップ生成システム

開発者の行動パターンを可視化するためのヒートマップ生成機能を提供する。
レスポンス時間、受諾率、時間帯・曜日別パターンの可視化を行う。
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """ヒートマップ生成器
    
    開発者の行動パターンを時間軸で可視化するヒートマップを生成する。
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
        plt.rcParams['figure.figsize'] = config.get('figure_size', (12, 8))
        
        # カラーマップ設定
        self.colormap = config.get('colormap', 'YlOrRd')
        
    def generate_response_time_heatmap(self, 
                                     developer_data: List[Dict[str, Any]], 
                                     developer_email: str) -> str:
        """レスポンス時間ヒートマップを生成
        
        Args:
            developer_data: 開発者のレビューデータ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたヒートマップファイルのパス
        """
        try:
            # 開発者のデータをフィルタリング
            dev_reviews = [
                review for review in developer_data 
                if review.get('reviewer_email') == developer_email
            ]
            
            if not dev_reviews:
                logger.warning(f"開発者 {developer_email} のレビューデータが見つかりません")
                return ""
            
            # 時間帯・曜日別のレスポンス時間を集計
            heatmap_data = self._aggregate_response_times(dev_reviews)
            
            # ヒートマップを作成
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.1f',
                cmap=self.colormap,
                ax=ax,
                cbar_kws={'label': 'レスポンス時間 (時間)'}
            )
            
            ax.set_title(f'レスポンス時間ヒートマップ - {developer_email}', 
                        fontsize=16, pad=20)
            ax.set_xlabel('時間帯', fontsize=12)
            ax.set_ylabel('曜日', fontsize=12)
            
            # ファイル保存
            filename = f"response_time_heatmap_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"レスポンス時間ヒートマップを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"レスポンス時間ヒートマップ生成エラー: {e}")
            return ""
    
    def generate_acceptance_rate_heatmap(self, 
                                       developer_data: List[Dict[str, Any]], 
                                       developer_email: str) -> str:
        """受諾率ヒートマップを生成
        
        Args:
            developer_data: 開発者のレビューデータ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたヒートマップファイルのパス
        """
        try:
            # 開発者のレビュー依頼データをフィルタリング
            dev_requests = [
                request for request in developer_data 
                if request.get('potential_reviewer') == developer_email
            ]
            
            if not dev_requests:
                logger.warning(f"開発者 {developer_email} のレビュー依頼データが見つかりません")
                return ""
            
            # 時間帯・曜日別の受諾率を集計
            heatmap_data = self._aggregate_acceptance_rates(dev_requests)
            
            # ヒートマップを作成
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={'label': '受諾率'}
            )
            
            ax.set_title(f'レビュー受諾率ヒートマップ - {developer_email}', 
                        fontsize=16, pad=20)
            ax.set_xlabel('時間帯', fontsize=12)
            ax.set_ylabel('曜日', fontsize=12)
            
            # ファイル保存
            filename = f"acceptance_rate_heatmap_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"受諾率ヒートマップを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"受諾率ヒートマップ生成エラー: {e}")
            return ""
    
    def generate_collaboration_heatmap(self, 
                                     collaboration_data: List[Dict[str, Any]], 
                                     developer_email: str) -> str:
        """協力相手別コミュニケーション頻度ヒートマップを生成
        
        Args:
            collaboration_data: 協力関係データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたヒートマップファイルのパス
        """
        try:
            # 協力関係データを集計
            collab_matrix = self._aggregate_collaboration_frequency(
                collaboration_data, developer_email
            )
            
            if collab_matrix.empty:
                logger.warning(f"開発者 {developer_email} の協力関係データが見つかりません")
                return ""
            
            # ヒートマップを作成
            fig, ax = plt.subplots(figsize=(14, 10))
            
            sns.heatmap(
                collab_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                cbar_kws={'label': 'コミュニケーション回数'}
            )
            
            ax.set_title(f'協力相手別コミュニケーション頻度 - {developer_email}', 
                        fontsize=16, pad=20)
            ax.set_xlabel('時間帯', fontsize=12)
            ax.set_ylabel('協力相手', fontsize=12)
            
            # ファイル保存
            filename = f"collaboration_heatmap_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"協力関係ヒートマップを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"協力関係ヒートマップ生成エラー: {e}")
            return ""
    
    def generate_activity_pattern_heatmap(self, 
                                        activity_data: List[Dict[str, Any]], 
                                        developer_email: str,
                                        time_window_days: int = 30) -> str:
        """活動パターンヒートマップを生成
        
        Args:
            activity_data: 活動データ
            developer_email: 対象開発者のメールアドレス
            time_window_days: 分析対象期間（日数）
            
        Returns:
            生成されたヒートマップファイルのパス
        """
        try:
            # 指定期間の活動データを集計
            activity_matrix = self._aggregate_activity_patterns(
                activity_data, developer_email, time_window_days
            )
            
            if activity_matrix.empty:
                logger.warning(f"開発者 {developer_email} の活動データが見つかりません")
                return ""
            
            # ヒートマップを作成
            fig, ax = plt.subplots(figsize=(16, 8))
            
            sns.heatmap(
                activity_matrix,
                cmap='viridis',
                ax=ax,
                cbar_kws={'label': '活動強度'}
            )
            
            ax.set_title(f'活動パターンヒートマップ ({time_window_days}日間) - {developer_email}', 
                        fontsize=16, pad=20)
            ax.set_xlabel('時間帯', fontsize=12)
            ax.set_ylabel('日付', fontsize=12)
            
            # ファイル保存
            filename = f"activity_pattern_heatmap_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"活動パターンヒートマップを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"活動パターンヒートマップ生成エラー: {e}")
            return ""
    
    def _aggregate_response_times(self, reviews: List[Dict[str, Any]]) -> pd.DataFrame:
        """レスポンス時間を時間帯・曜日別に集計
        
        Args:
            reviews: レビューデータ
            
        Returns:
            集計されたレスポンス時間データ
        """
        # 曜日と時間帯のマトリックスを初期化
        weekdays = ['月', '火', '水', '木', '金', '土', '日']
        hours = list(range(24))
        
        response_matrix = pd.DataFrame(
            index=weekdays,
            columns=hours,
            dtype=float
        ).fillna(0)
        
        count_matrix = pd.DataFrame(
            index=weekdays,
            columns=hours,
            dtype=int
        ).fillna(0)
        
        for review in reviews:
            try:
                # タイムスタンプから曜日と時間を抽出
                timestamp = pd.to_datetime(review.get('timestamp'))
                weekday = weekdays[timestamp.weekday()]
                hour = timestamp.hour
                
                response_time = review.get('response_time_hours', 0)
                
                # 累積と回数を記録
                response_matrix.loc[weekday, hour] += response_time
                count_matrix.loc[weekday, hour] += 1
                
            except Exception as e:
                logger.warning(f"レスポンス時間集計エラー: {e}")
                continue
        
        # 平均レスポンス時間を計算
        avg_response_matrix = response_matrix / count_matrix.replace(0, 1)
        
        return avg_response_matrix
    
    def _aggregate_acceptance_rates(self, requests: List[Dict[str, Any]]) -> pd.DataFrame:
        """受諾率を時間帯・曜日別に集計
        
        Args:
            requests: レビュー依頼データ
            
        Returns:
            集計された受諾率データ
        """
        weekdays = ['月', '火', '水', '木', '金', '土', '日']
        hours = list(range(24))
        
        accepted_matrix = pd.DataFrame(
            index=weekdays,
            columns=hours,
            dtype=int
        ).fillna(0)
        
        total_matrix = pd.DataFrame(
            index=weekdays,
            columns=hours,
            dtype=int
        ).fillna(0)
        
        for request in requests:
            try:
                timestamp = pd.to_datetime(request.get('timestamp'))
                weekday = weekdays[timestamp.weekday()]
                hour = timestamp.hour
                
                is_accepted = request.get('action') == 'accept'
                
                if is_accepted:
                    accepted_matrix.loc[weekday, hour] += 1
                total_matrix.loc[weekday, hour] += 1
                
            except Exception as e:
                logger.warning(f"受諾率集計エラー: {e}")
                continue
        
        # 受諾率を計算
        acceptance_rate_matrix = accepted_matrix / total_matrix.replace(0, 1)
        
        return acceptance_rate_matrix
    
    def _aggregate_collaboration_frequency(self, 
                                         collaboration_data: List[Dict[str, Any]], 
                                         developer_email: str) -> pd.DataFrame:
        """協力関係頻度を集計
        
        Args:
            collaboration_data: 協力関係データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            集計された協力関係データ
        """
        # 協力相手を特定
        collaborators = set()
        for collab in collaboration_data:
            if collab.get('developer1') == developer_email:
                collaborators.add(collab.get('developer2'))
            elif collab.get('developer2') == developer_email:
                collaborators.add(collab.get('developer1'))
        
        if not collaborators:
            return pd.DataFrame()
        
        # 時間帯別の協力頻度を集計
        hours = list(range(24))
        collab_matrix = pd.DataFrame(
            index=list(collaborators),
            columns=hours,
            dtype=int
        ).fillna(0)
        
        for collab in collaboration_data:
            try:
                timestamp = pd.to_datetime(collab.get('timestamp'))
                hour = timestamp.hour
                
                if collab.get('developer1') == developer_email:
                    partner = collab.get('developer2')
                elif collab.get('developer2') == developer_email:
                    partner = collab.get('developer1')
                else:
                    continue
                
                if partner in collaborators:
                    collab_matrix.loc[partner, hour] += 1
                    
            except Exception as e:
                logger.warning(f"協力関係集計エラー: {e}")
                continue
        
        return collab_matrix
    
    def _aggregate_activity_patterns(self, 
                                   activity_data: List[Dict[str, Any]], 
                                   developer_email: str,
                                   time_window_days: int) -> pd.DataFrame:
        """活動パターンを集計
        
        Args:
            activity_data: 活動データ
            developer_email: 対象開発者のメールアドレス
            time_window_days: 分析対象期間
            
        Returns:
            集計された活動パターンデータ
        """
        # 対象期間のデータをフィルタリング
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)
        
        dev_activities = [
            activity for activity in activity_data
            if (activity.get('developer_email') == developer_email and
                start_date <= pd.to_datetime(activity.get('timestamp')) <= end_date)
        ]
        
        if not dev_activities:
            return pd.DataFrame()
        
        # 日付と時間帯のマトリックスを作成
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        hours = list(range(24))
        
        activity_matrix = pd.DataFrame(
            index=[date.strftime('%m-%d') for date in date_range],
            columns=hours,
            dtype=float
        ).fillna(0)
        
        for activity in dev_activities:
            try:
                timestamp = pd.to_datetime(activity.get('timestamp'))
                date_str = timestamp.strftime('%m-%d')
                hour = timestamp.hour
                
                activity_intensity = activity.get('intensity', 1.0)
                
                if date_str in activity_matrix.index:
                    activity_matrix.loc[date_str, hour] += activity_intensity
                    
            except Exception as e:
                logger.warning(f"活動パターン集計エラー: {e}")
                continue
        
        return activity_matrix
    
    def generate_comprehensive_heatmap_report(self, 
                                            developer_data: List[Dict[str, Any]], 
                                            developer_email: str) -> Dict[str, str]:
        """包括的なヒートマップレポートを生成
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたヒートマップファイルパスの辞書
        """
        report_paths = {}
        
        try:
            # レスポンス時間ヒートマップ
            response_path = self.generate_response_time_heatmap(
                developer_data, developer_email
            )
            if response_path:
                report_paths['response_time'] = response_path
            
            # 受諾率ヒートマップ
            acceptance_path = self.generate_acceptance_rate_heatmap(
                developer_data, developer_email
            )
            if acceptance_path:
                report_paths['acceptance_rate'] = acceptance_path
            
            # 協力関係ヒートマップ
            collaboration_path = self.generate_collaboration_heatmap(
                developer_data, developer_email
            )
            if collaboration_path:
                report_paths['collaboration'] = collaboration_path
            
            # 活動パターンヒートマップ
            activity_path = self.generate_activity_pattern_heatmap(
                developer_data, developer_email
            )
            if activity_path:
                report_paths['activity_pattern'] = activity_path
            
            logger.info(f"開発者 {developer_email} の包括的ヒートマップレポートを生成しました")
            
        except Exception as e:
            logger.error(f"包括的ヒートマップレポート生成エラー: {e}")
        
        return report_paths