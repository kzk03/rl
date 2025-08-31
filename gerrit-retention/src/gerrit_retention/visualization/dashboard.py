"""
ダッシュボードシステム

開発者のリアルタイムストレスレベル表示とストレス要因分解、
沸点リスク警告システムを提供するダッシュボード機能。
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

logger = logging.getLogger(__name__)


class DeveloperDashboard:
    """開発者ダッシュボード
    
    リアルタイムストレスレベル表示、ストレス要因分解、
    沸点リスク警告システムを提供する。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初期化
        
        Args:
            config: ダッシュボード設定
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'outputs/dashboards'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日本語フォント設定
        plt.rcParams['font.family'] = config.get('font_family', 'DejaVu Sans')
        
        # 警告閾値設定
        self.stress_thresholds = config.get('stress_thresholds', {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        })
        
        # 色設定
        self.colors = config.get('colors', {
            'low': '#2ecc71',      # 緑
            'medium': '#f39c12',   # オレンジ
            'high': '#e74c3c',     # 赤
            'critical': '#8e44ad'  # 紫
        })
        
    def generate_realtime_stress_dashboard(self, 
                                         developer_data: Dict[str, Any], 
                                         developer_email: str) -> str:
        """リアルタイムストレスダッシュボードを生成
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたダッシュボードファイルのパス
        """
        try:
            # 開発者の現在のストレス状態を取得
            stress_data = self._extract_current_stress_data(developer_data, developer_email)
            
            if not stress_data:
                logger.warning(f"開発者 {developer_email} のストレスデータが見つかりません")
                return ""
            
            # ダッシュボードを作成
            fig = plt.figure(figsize=(16, 12))
            
            # レイアウトを設定
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. 総合ストレスレベル（大きなゲージ）
            ax_main = fig.add_subplot(gs[0, :2])
            self._draw_stress_gauge(ax_main, stress_data['total_stress'], 
                                  f"{developer_email} - 総合ストレスレベル")
            
            # 2. 沸点リスク警告
            ax_risk = fig.add_subplot(gs[0, 2:])
            self._draw_boiling_point_warning(ax_risk, stress_data)
            
            # 3. ストレス要因分解（円グラフ）
            ax_factors = fig.add_subplot(gs[1, :2])
            self._draw_stress_factors_pie(ax_factors, stress_data['stress_factors'])
            
            # 4. 時系列ストレス変化
            ax_timeline = fig.add_subplot(gs[1, 2:])
            self._draw_stress_timeline(ax_timeline, stress_data['stress_history'])
            
            # 5. 推奨アクション
            ax_actions = fig.add_subplot(gs[2, :2])
            self._draw_recommended_actions(ax_actions, stress_data)
            
            # 6. ストレス詳細メトリクス
            ax_metrics = fig.add_subplot(gs[2, 2:])
            self._draw_stress_metrics_table(ax_metrics, stress_data['detailed_metrics'])
            
            # タイトルと更新時刻
            fig.suptitle(f'開発者ストレスダッシュボード - {developer_email}', 
                        fontsize=18, fontweight='bold')
            
            update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fig.text(0.99, 0.01, f'最終更新: {update_time}', 
                    ha='right', va='bottom', fontsize=10)
            
            # ファイル保存
            filename = f"stress_dashboard_{developer_email.replace('@', '_at_')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # インタラクティブ版も生成
            interactive_path = self._generate_interactive_dashboard(
                developer_data, developer_email, stress_data
            )
            
            logger.info(f"リアルタイムストレスダッシュボードを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"リアルタイムストレスダッシュボード生成エラー: {e}")
            return ""
    
    def generate_team_overview_dashboard(self, 
                                       team_data: List[Dict[str, Any]]) -> str:
        """チーム全体のストレス概要ダッシュボードを生成
        
        Args:
            team_data: チームメンバーのデータ
            
        Returns:
            生成されたダッシュボードファイルのパス
        """
        try:
            # チーム全体のストレス状況を集計
            team_stress_summary = self._aggregate_team_stress(team_data)
            
            # ダッシュボードを作成
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. チーム全体のストレス分布
            ax_distribution = fig.add_subplot(gs[0, :])
            self._draw_team_stress_distribution(ax_distribution, team_stress_summary)
            
            # 2. 高リスク開発者一覧
            ax_high_risk = fig.add_subplot(gs[1, 0])
            self._draw_high_risk_developers(ax_high_risk, team_stress_summary['high_risk'])
            
            # 3. ストレス要因ランキング
            ax_factors_ranking = fig.add_subplot(gs[1, 1])
            self._draw_stress_factors_ranking(ax_factors_ranking, team_stress_summary)
            
            # 4. チーム健全性指標
            ax_health = fig.add_subplot(gs[1, 2])
            self._draw_team_health_metrics(ax_health, team_stress_summary)
            
            # 5. 推奨チーム施策
            ax_team_actions = fig.add_subplot(gs[2, :])
            self._draw_team_recommended_actions(ax_team_actions, team_stress_summary)
            
            fig.suptitle('チームストレス概要ダッシュボード', 
                        fontsize=18, fontweight='bold')
            
            # ファイル保存
            filename = f"team_stress_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"チーム概要ダッシュボードを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"チーム概要ダッシュボード生成エラー: {e}")
            return ""
    
    def _extract_current_stress_data(self, 
                                   developer_data: Dict[str, Any], 
                                   developer_email: str) -> Dict[str, Any]:
        """現在のストレスデータを抽出
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            ストレスデータ
        """
        try:
            dev_profile = developer_data.get(developer_email, {})
            
            # 総合ストレスレベル
            total_stress = dev_profile.get('stress_level', 0.0)
            
            # ストレス要因分解
            stress_factors = {
                'タスク適合度': dev_profile.get('task_compatibility_stress', 0.0),
                'ワークロード': dev_profile.get('workload_stress', 0.0),
                '社会的要因': dev_profile.get('social_stress', 0.0),
                '時間的プレッシャー': dev_profile.get('temporal_stress', 0.0)
            }
            
            # 沸点関連データ
            boiling_point = dev_profile.get('boiling_point_estimate', 1.0)
            stress_margin = boiling_point - total_stress
            risk_level = self._calculate_risk_level(total_stress, boiling_point)
            
            # ストレス履歴
            stress_history = dev_profile.get('stress_history', [])
            
            # 詳細メトリクス
            detailed_metrics = {
                '現在のストレス': total_stress,
                '沸点推定値': boiling_point,
                'ストレス余裕度': stress_margin,
                'リスクレベル': risk_level,
                '最近の受諾率': dev_profile.get('recent_acceptance_rate', 0.0),
                '平均レスポンス時間': dev_profile.get('avg_response_time', 0.0),
                '協力関係品質': dev_profile.get('collaboration_quality', 0.0)
            }
            
            return {
                'total_stress': total_stress,
                'stress_factors': stress_factors,
                'boiling_point': boiling_point,
                'stress_margin': stress_margin,
                'risk_level': risk_level,
                'stress_history': stress_history,
                'detailed_metrics': detailed_metrics
            }
            
        except Exception as e:
            logger.warning(f"ストレスデータ抽出エラー: {e}")
            return {}
    
    def _draw_stress_gauge(self, ax, stress_level: float, title: str):
        """ストレスゲージを描画
        
        Args:
            ax: matplotlib軸
            stress_level: ストレスレベル（0-1）
            title: タイトル
        """
        # 半円ゲージを描画
        theta = np.linspace(0, np.pi, 100)
        
        # 背景円弧
        ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=20)
        
        # ストレスレベルに応じた色を決定
        color = self._get_stress_color(stress_level)
        
        # ストレス円弧
        stress_theta = np.linspace(0, np.pi * stress_level, int(100 * stress_level))
        if len(stress_theta) > 0:
            ax.plot(np.cos(stress_theta), np.sin(stress_theta), color, linewidth=20)
        
        # 針を描画
        needle_angle = np.pi * (1 - stress_level)
        needle_x = 0.8 * np.cos(needle_angle)
        needle_y = 0.8 * np.sin(needle_angle)
        ax.arrow(0, 0, needle_x, needle_y, head_width=0.05, head_length=0.05, 
                fc='black', ec='black')
        
        # 数値表示
        ax.text(0, -0.3, f'{stress_level:.2f}', ha='center', va='center', 
               fontsize=24, fontweight='bold')
        
        # 閾値マーカー
        for threshold_name, threshold_value in self.stress_thresholds.items():
            angle = np.pi * (1 - threshold_value)
            x = 1.1 * np.cos(angle)
            y = 1.1 * np.sin(angle)
            ax.plot([0.9 * np.cos(angle), 1.1 * np.cos(angle)], 
                   [0.9 * np.sin(angle), 1.1 * np.sin(angle)], 
                   'k-', linewidth=2)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.5, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    def _draw_boiling_point_warning(self, ax, stress_data: Dict[str, Any]):
        """沸点リスク警告を描画
        
        Args:
            ax: matplotlib軸
            stress_data: ストレスデータ
        """
        risk_level = stress_data['risk_level']
        stress_margin = stress_data['stress_margin']
        
        # リスクレベルに応じた背景色
        if risk_level == 'critical':
            bg_color = '#ffebee'
            text_color = '#c62828'
            warning_text = '⚠️ 緊急警告'
        elif risk_level == 'high':
            bg_color = '#fff3e0'
            text_color = '#ef6c00'
            warning_text = '⚠️ 高リスク'
        elif risk_level == 'medium':
            bg_color = '#f3e5f5'
            text_color = '#7b1fa2'
            warning_text = '⚠️ 注意'
        else:
            bg_color = '#e8f5e8'
            text_color = '#2e7d32'
            warning_text = '✅ 安全'
        
        # 背景を描画
        rect = Rectangle((0, 0), 1, 1, facecolor=bg_color, edgecolor='none')
        ax.add_patch(rect)
        
        # 警告テキスト
        ax.text(0.5, 0.7, warning_text, ha='center', va='center', 
               fontsize=16, fontweight='bold', color=text_color)
        
        # ストレス余裕度
        ax.text(0.5, 0.5, f'ストレス余裕度: {stress_margin:.2f}', 
               ha='center', va='center', fontsize=12)
        
        # 推定到達時間（仮想的な計算）
        if stress_margin > 0:
            estimated_days = max(1, int(stress_margin * 30))  # 簡易計算
            ax.text(0.5, 0.3, f'推定到達時間: {estimated_days}日', 
                   ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.3, '即座の対応が必要', 
                   ha='center', va='center', fontsize=12, color='red')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('沸点リスク警告', fontsize=14, fontweight='bold')
    
    def _draw_stress_factors_pie(self, ax, stress_factors: Dict[str, float]):
        """ストレス要因円グラフを描画
        
        Args:
            ax: matplotlib軸
            stress_factors: ストレス要因データ
        """
        labels = list(stress_factors.keys())
        sizes = list(stress_factors.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        # 円グラフを描画
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10}
        )
        
        ax.set_title('ストレス要因分解', fontsize=14, fontweight='bold')
    
    def _draw_stress_timeline(self, ax, stress_history: List[Dict[str, Any]]):
        """ストレス時系列変化を描画
        
        Args:
            ax: matplotlib軸
            stress_history: ストレス履歴データ
        """
        if not stress_history:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('ストレス時系列変化', fontsize=14, fontweight='bold')
            return
        
        # 時系列データを準備
        dates = [pd.to_datetime(entry['date']) for entry in stress_history]
        stress_values = [entry['stress_level'] for entry in stress_history]
        
        # 線グラフを描画
        ax.plot(dates, stress_values, marker='o', linewidth=2, markersize=4)
        
        # 閾値線を描画
        for threshold_name, threshold_value in self.stress_thresholds.items():
            if threshold_name != 'low':
                ax.axhline(y=threshold_value, color=self.colors[threshold_name], 
                          linestyle='--', alpha=0.7, label=threshold_name)
        
        ax.set_xlabel('日付')
        ax.set_ylabel('ストレスレベル')
        ax.set_title('ストレス時系列変化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _draw_recommended_actions(self, ax, stress_data: Dict[str, Any]):
        """推奨アクションを描画
        
        Args:
            ax: matplotlib軸
            stress_data: ストレスデータ
        """
        # ストレスレベルに基づいて推奨アクションを決定
        actions = self._generate_recommended_actions(stress_data)
        
        ax.text(0.05, 0.9, '推奨アクション:', fontsize=12, fontweight='bold', 
               transform=ax.transAxes)
        
        for i, action in enumerate(actions[:5]):  # 最大5つまで表示
            y_pos = 0.8 - i * 0.15
            ax.text(0.05, y_pos, f'• {action}', fontsize=10, 
                   transform=ax.transAxes, wrap=True)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('推奨アクション', fontsize=14, fontweight='bold')
    
    def _draw_stress_metrics_table(self, ax, metrics: Dict[str, Any]):
        """ストレス詳細メトリクステーブルを描画
        
        Args:
            ax: matplotlib軸
            metrics: メトリクスデータ
        """
        # テーブルデータを準備
        table_data = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f'{value:.3f}'
            else:
                formatted_value = str(value)
            table_data.append([key, formatted_value])
        
        # テーブルを描画
        table = ax.table(
            cellText=table_data,
            colLabels=['メトリクス', '値'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax.axis('off')
        ax.set_title('詳細メトリクス', fontsize=14, fontweight='bold')
    
    def _get_stress_color(self, stress_level: float) -> str:
        """ストレスレベルに応じた色を取得
        
        Args:
            stress_level: ストレスレベル
            
        Returns:
            色コード
        """
        if stress_level >= self.stress_thresholds['critical']:
            return self.colors['critical']
        elif stress_level >= self.stress_thresholds['high']:
            return self.colors['high']
        elif stress_level >= self.stress_thresholds['medium']:
            return self.colors['medium']
        else:
            return self.colors['low']
    
    def _calculate_risk_level(self, stress_level: float, boiling_point: float) -> str:
        """リスクレベルを計算
        
        Args:
            stress_level: 現在のストレスレベル
            boiling_point: 沸点推定値
            
        Returns:
            リスクレベル
        """
        if stress_level >= boiling_point * 0.9:
            return 'critical'
        elif stress_level >= boiling_point * 0.8:
            return 'high'
        elif stress_level >= boiling_point * 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommended_actions(self, stress_data: Dict[str, Any]) -> List[str]:
        """推奨アクションを生成
        
        Args:
            stress_data: ストレスデータ
            
        Returns:
            推奨アクションのリスト
        """
        actions = []
        stress_factors = stress_data['stress_factors']
        risk_level = stress_data['risk_level']
        
        # リスクレベル別の基本アクション
        if risk_level == 'critical':
            actions.append('緊急: 即座にワークロードを削減してください')
            actions.append('マネージャーとの緊急面談を実施')
        elif risk_level == 'high':
            actions.append('ワークロードの見直しを検討')
            actions.append('ストレス軽減策の実施')
        
        # ストレス要因別のアクション
        max_factor = max(stress_factors, key=stress_factors.get)
        
        if max_factor == 'タスク適合度' and stress_factors[max_factor] > 0.6:
            actions.append('専門性に合ったタスクへの再配分を検討')
            actions.append('スキル向上のための学習機会を提供')
        
        if max_factor == 'ワークロード' and stress_factors[max_factor] > 0.6:
            actions.append('同時進行タスク数の削減')
            actions.append('締切の再調整を検討')
        
        if max_factor == '社会的要因' and stress_factors[max_factor] > 0.6:
            actions.append('チーム内コミュニケーションの改善')
            actions.append('協力関係の見直し')
        
        if max_factor == '時間的プレッシャー' and stress_factors[max_factor] > 0.6:
            actions.append('レビュー期限の調整')
            actions.append('優先度の再評価')
        
        return actions
    
    def _generate_interactive_dashboard(self, 
                                      developer_data: Dict[str, Any], 
                                      developer_email: str,
                                      stress_data: Dict[str, Any]) -> str:
        """インタラクティブダッシュボードを生成
        
        Args:
            developer_data: 開発者データ
            developer_email: 対象開発者のメールアドレス
            stress_data: ストレスデータ
            
        Returns:
            インタラクティブダッシュボードファイルのパス
        """
        try:
            # HTMLダッシュボードを生成
            html_content = self._generate_html_dashboard(
                developer_email, stress_data
            )
            
            filename = f"interactive_dashboard_{developer_email.replace('@', '_at_')}.html"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"インタラクティブダッシュボードを生成しました: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"インタラクティブダッシュボード生成エラー: {e}")
            return ""
    
    def _generate_html_dashboard(self, developer_email: str, 
                               stress_data: Dict[str, Any]) -> str:
        """HTMLダッシュボードを生成
        
        Args:
            developer_email: 開発者のメールアドレス
            stress_data: ストレスデータ
            
        Returns:
            HTML文字列
        """
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>開発者ストレスダッシュボード - {developer_email}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .widget {{ border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
                .stress-gauge {{ text-align: center; }}
                .stress-level {{ font-size: 2em; font-weight: bold; }}
                .risk-warning {{ padding: 10px; border-radius: 5px; text-align: center; }}
                .risk-critical {{ background-color: #ffebee; color: #c62828; }}
                .risk-high {{ background-color: #fff3e0; color: #ef6c00; }}
                .risk-medium {{ background-color: #f3e5f5; color: #7b1fa2; }}
                .risk-low {{ background-color: #e8f5e8; color: #2e7d32; }}
            </style>
        </head>
        <body>
            <h1>開発者ストレスダッシュボード - {developer_email}</h1>
            
            <div class="dashboard">
                <div class="widget stress-gauge">
                    <h3>総合ストレスレベル</h3>
                    <div class="stress-level">{stress_data['total_stress']:.2f}</div>
                </div>
                
                <div class="widget">
                    <h3>沸点リスク警告</h3>
                    <div class="risk-warning risk-{stress_data['risk_level']}">
                        リスクレベル: {stress_data['risk_level'].upper()}
                    </div>
                    <p>ストレス余裕度: {stress_data['stress_margin']:.2f}</p>
                </div>
                
                <div class="widget">
                    <h3>ストレス要因</h3>
                    <ul>
                        {''.join([f'<li>{factor}: {value:.2f}</li>' for factor, value in stress_data['stress_factors'].items()])}
                    </ul>
                </div>
                
                <div class="widget">
                    <h3>詳細メトリクス</h3>
                    <table>
                        {''.join([f'<tr><td>{key}</td><td>{value}</td></tr>' for key, value in stress_data['detailed_metrics'].items()])}
                    </table>
                </div>
            </div>
            
            <script>
                // 自動更新機能（実装時に追加）
                setInterval(function() {{
                    // ダッシュボードデータを更新
                    console.log('ダッシュボード更新中...');
                }}, 60000); // 1分ごと
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def _aggregate_team_stress(self, team_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """チーム全体のストレス状況を集計
        
        Args:
            team_data: チームメンバーのデータ
            
        Returns:
            チームストレス集計データ
        """
        # 実装は簡略化（実際の実装では詳細な集計を行う）
        return {
            'average_stress': 0.5,
            'high_risk': ['developer1@example.com', 'developer2@example.com'],
            'stress_distribution': {'low': 3, 'medium': 5, 'high': 2, 'critical': 1},
            'common_factors': ['ワークロード', 'タスク適合度']
        }
    
    def _draw_team_stress_distribution(self, ax, team_summary: Dict[str, Any]):
        """チームストレス分布を描画"""
        # 簡略化された実装
        ax.text(0.5, 0.5, 'チームストレス分布', ha='center', va='center', 
               transform=ax.transAxes)
        ax.set_title('チームストレス分布')
    
    def _draw_high_risk_developers(self, ax, high_risk_devs: List[str]):
        """高リスク開発者一覧を描画"""
        ax.text(0.5, 0.5, f'高リスク開発者: {len(high_risk_devs)}名', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('高リスク開発者')
    
    def _draw_stress_factors_ranking(self, ax, team_summary: Dict[str, Any]):
        """ストレス要因ランキングを描画"""
        ax.text(0.5, 0.5, 'ストレス要因ランキング', ha='center', va='center', 
               transform=ax.transAxes)
        ax.set_title('ストレス要因ランキング')
    
    def _draw_team_health_metrics(self, ax, team_summary: Dict[str, Any]):
        """チーム健全性指標を描画"""
        ax.text(0.5, 0.5, 'チーム健全性指標', ha='center', va='center', 
               transform=ax.transAxes)
        ax.set_title('チーム健全性指標')
    
    def _draw_team_recommended_actions(self, ax, team_summary: Dict[str, Any]):
        """チーム推奨施策を描画"""
        ax.text(0.5, 0.5, 'チーム推奨施策', ha='center', va='center', 
               transform=ax.transAxes)
        ax.set_title('チーム推奨施策')
    
    def generate_comprehensive_dashboard_report(self, 
                                              developer_data: Dict[str, Any], 
                                              team_data: List[Dict[str, Any]], 
                                              developer_email: str) -> Dict[str, str]:
        """包括的なダッシュボードレポートを生成
        
        Args:
            developer_data: 開発者データ
            team_data: チームデータ
            developer_email: 対象開発者のメールアドレス
            
        Returns:
            生成されたダッシュボードファイルパスの辞書
        """
        report_paths = {}
        
        try:
            # 個人ストレスダッシュボード
            individual_path = self.generate_realtime_stress_dashboard(
                developer_data, developer_email
            )
            if individual_path:
                report_paths['individual_dashboard'] = individual_path
            
            # チーム概要ダッシュボード
            team_path = self.generate_team_overview_dashboard(team_data)
            if team_path:
                report_paths['team_dashboard'] = team_path
            
            logger.info(f"包括的ダッシュボードレポートを生成しました")
            
        except Exception as e:
            logger.error(f"包括的ダッシュボードレポート生成エラー: {e}")
        
        return report_paths