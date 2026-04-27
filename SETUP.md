# 一次性設定 (One-time setup)

排程任務啟動前,你需要在 GitHub 上完成兩件事 — 都只做一次。

## 1. 在 GitHub 上建立 repo

1. 登入 https://github.com
2. 右上角 `+` → New repository
3. 名稱:`agent-policy-gateway`
4. **Private** (建議,可隨時改 public)
5. **不要**勾選 "Add a README" / .gitignore / license — 我們已經在本地有了
6. 建立後,GitHub 會給你一個 URL,例如:
   `https://github.com/<你的帳號>/agent-policy-gateway.git`

把這個 URL 記下來。

## 2. 產生 fine-grained Personal Access Token (PAT)

排程任務要在無人值守時 push,所以需要一個 token。

1. 到 https://github.com/settings/personal-access-tokens/new
2. **Token name:** `agent-policy-gateway-bot`
3. **Expiration:** 90 days (到期前我會提醒你續)
4. **Repository access:** Only select repositories → 選 `agent-policy-gateway`
5. **Repository permissions:**
   - Contents: **Read and write**
   - Metadata: Read (預設)
   - Pull requests: Read and write (之後可能需要)
6. 點 Generate token,**立刻複製整段 token**(只會顯示一次)

## 3. 把 token 存到本地

在這個資料夾建立檔案 `.gh-token`,內容是:

```
GITHUB_USERNAME=<你的帳號>
GITHUB_REPO_URL=https://github.com/<你的帳號>/agent-policy-gateway.git
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxx
```

`.gh-token` 已經被 `.gitignore` 排除,不會被推上去。

## 4. 確認

排程任務的第一次跑會自動驗證:`.gh-token` 能否解析、能否 `git clone`、能否 push。
如果有任何一步失敗,任務會停止並把錯誤寫到 `docs/work-log/YYYY-MM-DD.md`,**不會** push 任何東西。

---

設定完成後,排程會在每天指定時間自動執行,把專案往前推一個 roadmap 項目。
