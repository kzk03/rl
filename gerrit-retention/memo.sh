#!/bin/bash
# 同期コマンド
# Local → Server
rsync -avz --progress /Users/kazuki-h/rl/gerrit-retention/ socsel:/mnt/data1/kazuki-h/gerrit-retention/


# Server → Local
rsync socsel:/mnt/data1/kazuki-h/rl/gerrit-retention/ /Users/kazuki-h/rl/


scripts/analysis/run_review_acceptance_cross_eval.py


