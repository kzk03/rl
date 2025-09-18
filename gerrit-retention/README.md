# Gerrit 開発者定着予測システム

開発者の長期的な貢献を促進する「開発者定着予測システム」です。このシステムは、開発者の「沸点」（ストレス限界点）を予測し、レビュー受諾行動を最適化することで、持続可能な開発者コミュニティを構築することを目的とします。

## 概要

- **定着予測**: 開発者の継続的な貢献確率を予測
- **ストレス分析**: 開発者の「沸点」を定量化・予測
- **レビュー最適化**: レビュー受諾行動の分析・改善
- **強化学習**: 長期定着を重視した推薦システム

## アーキテクチャ

```
GAT (埋め込み) → IRL (報酬) → RL (ポリシー)
```

## 主要機能

1. **Gerrit データ統合**: Gerrit API からのデータ抽出・変換
2. **定着予測モデル**: 開発者の定着確率予測
3. **ストレス・沸点分析**: 多次元ストレス指標と限界点予測
4. **レビュー行動分析**: 受諾確率・類似度・好み分析
5. **強化学習環境**: レビュー受諾最適化
6. **可視化システム**: ヒートマップ・ダッシュボード
7. **適応的戦略**: 動的推薦戦略調整

## セットアップ

```bash
# 依存関係のインストール
uv sync

# 環境変数の設定
cp .env.example .env
# .env ファイルを編集してGerrit接続情報を設定

# Gerrit接続のテスト
uv run python scripts/setup_gerrit_connection.py
```

## 使用方法

```bash
# フルパイプラインの実行
uv run python scripts/run_full_pipeline.py

# 個別コンポーネントの実行
uv run python training/retention_training/train_retention_model.py
uv run python training/stress_training/train_stress_model.py
uv run python training/rl_training/train_ppo_agent.py
```

## プロジェクト構造

詳細なプロジェクト構造については、`docs/development/project_structure.md` を参照してください。

## ライセンス

MIT License

## 最新出力参照先（Last Artifacts）

- 最終更新: 2025-09-18
- 現行の評価出力:
  - 招待の選定（IRL, Plackett–Luce）: `outputs/reviewer_invitation_irl_pl_full/`
  - 招待後の受諾/実参加: `outputs/reviewer_acceptance_after_invite_full/`
- 旧版/整理済み出力: `outputs/_legacy/`（旧特徴を含む成果物を移設）
- 評価と実行コマンドの詳細: `docs/poster_style_evaluation_guide.md`
