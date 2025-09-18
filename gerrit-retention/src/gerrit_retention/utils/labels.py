from __future__ import annotations

JP_LABELS = {
    'reviewer_total_reviews': '累積レビュー参加数',
    'reviewer_recent_reviews_30d': '過去30日レビュー参加数',
    'reviewer_gap_days': '最終参加からの経過日数',
    'match_off_specialty_flag': '専門外フラグ(過去30日同一プロジェクト経験無)',
    'off_specialty_recent_ratio': '専門外比率(=1-同一プロジェクト比率)',
    'reviewer_recent_reviews_7d': '過去7日レビュー参加数',
    'reviewer_proj_share_30d': '過去30日同一プロジェクト比率',
    'reviewer_active_flag_30d': '過去30日活動フラグ',
    'reviewer_proj_prev_reviews_30d': '過去30日同一プロジェクト参加数',
    'reviewer_file_tfidf_cosine_30d': '過去30日ファイルトークンTF-IDFコサイン',
    'reviewer_pending_reviews': '未クローズレビュー数',
    'reviewer_workload_deviation_z': '活動量Zスコア',
    'change_current_invited_cnt': '変更での招待人数',
    'reviewer_night_activity_share_30d': '過去30日夜間活動比率',
    'reviewer_overload_flag': '過負荷フラグ(平均+σ以上)',
    '_intercept': 'ベースライン切片',
}

def jp_label(name: str) -> str:
    return JP_LABELS.get(name, name)
