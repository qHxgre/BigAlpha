  import os
  from collections import defaultdict

  # token：写到 RUNNER_BASE_DIR/cptjudge.jwt 或 ALPHATHON_API_TOKEN 任一即可
  os.environ.setdefault("ALPHATHON_API_BASE_URL", "http://localhost:8000/bigapis/alphathon/v1")
  os.environ.setdefault("ALPHATHON_API_TOKEN", "<your-jwt>")

  from alphathonapiserver.judgebase import AlphathonAPI

  COMPETITION_ID = "5c3f7783-4158-4196-97ab-171b27218c7c"

  api = AlphathonAPI()

  # 1) 比赛元信息（可选）
  competition = api.get_competition_by_id(COMPETITION_ID)
  print(f"比赛：{competition['name']}（{competition['id']}）")

  # 2) 拉该比赛的全部提交
  submissions = api.query_submissions(competition_id=COMPETITION_ID)
  print(f"提交总数：{len(submissions)}")

  # 3) 按 user_id 聚合 → 参赛者信息
  participants: dict[str, dict] = defaultdict(
      lambda: {"submission_count": 0, "best_public_score": None, "last_submitted_at": None}
  )
  for s in submissions:
      uid = s["user_id"]
      p = participants[uid]
      p["submission_count"] += 1

      score = s.get("public_score")
      if score is not None and (p["best_public_score"] is None or score > p["best_public_score"]):
          p["best_public_score"] = score

      created_at = s.get("created_at")
      if created_at and (p["last_submitted_at"] is None or created_at > p["last_submitted_at"]):
          p["last_submitted_at"] = created_at

  # 4) 打印
  print(f"参赛者数：{len(participants)}")
  for uid, info in participants.items():
      print(
          f"  user_id={uid}  submissions={info['submission_count']}  "
          f"best_public={info['best_public_score']}  last={info['last_submitted_at']}"
      )