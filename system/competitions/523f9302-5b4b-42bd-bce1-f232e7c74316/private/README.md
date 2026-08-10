# 端到端模型赛道私榜评测

本目录为比赛 `523f9302-5b4b-42bd-bce1-f232e7c74316` 提供一次性、可审查的私榜批次流程。
它沿用 `public/` 的用户接口、资源限制和评分规则，但评测阶段不会更新后台分数。

## 当前口径

- 用户 Notebook 必须恰好一个，并定义 `main(datasources, start_date, end_date)`；
- 使用 `bar1m/bar5m/bar15m/bar30m` 四张 `_private` 数据表；
- 私榜起点暂设为公榜结束日次日：`2025-12-01 00:00:00`；
- 结束日为新批次首次启动当天的 `23:59:59`，续跑时从 manifest 恢复；
- 四项指标 `ic_mean/ic_ir/sharpe_ratio/stress_ic_ir` 分别做百分位排名，各占 25%；
- 没有因子池回归，失败提交记 `-2`；
- 默认串行执行，避免单张 GPU 上多个端到端模型争抢显存。

上线前应确认私榜数据表名和时间区间；如有变化，只需修改 `private.py` 的 `DATASETS`、
`DATE_START` 和结束日设置。

## 使用

```bash
python prepare_submissions.py
python private.py
```

评测产物位于 `system/files/<competition_id>/private/runs/<batch_id>/`。完成后 manifest 状态为
`review_pending`，重点检查 `artifacts/leaderboard_score.csv`、`leaderboard_final.csv`、
`submissions_summary.csv`、两个 score pool 以及各提交的 stdout/result.json。

预览发布时先在 `publish.py` 中设置 `DRY_RUN = True`；确认无误后恢复为 `False`，运行：

```bash
python publish.py
```

并输入 `PUBLISH`。只有此步骤会调用 API 写入 `private_score`。

断点续跑或指定重跑时，在 `private.py` 中固定原 `BATCH_ID`，设置 `RESUME = True`，并按需填写
`RERUN_SUBMISSION_IDS`。已发布批次不能续跑。
