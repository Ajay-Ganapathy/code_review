---
title: Code Review Environment Server
emoji: 🎳
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Code Review Environment

A reinforcement learning benchmark environment where an agent acts as a senior software engineer reviewing pull requests. The agent must identify bugs, suggest fixes, and make approval decisions across progressively harder code review tasks — spanning missing imports, logic errors, and security vulnerabilities.

---

## Motivation

Code review is a high-stakes, multi-step reasoning task that requires an agent to:

- **Detect bugs and security vulnerabilities** from raw code diffs
- **Generate corrective code** that resolves identified issues
- **Make a final judgment** (approve or reject) backed by technical reasoning

Existing benchmarks test code generation or comprehension in isolation. This environment tests the full review loop — detection, remediation, and decision-making — in a structured, scorable way. It is designed to evaluate whether LLMs can act as reliable automated reviewers in real software development pipelines.

---

## Setup and Usage

### Install dependencies

```bash
pip install openenv-core
```

```bash
git clone https://github.com/Ajay-Ganapathy/code_review && cd code_review
uv pip install -e .
```

### Run the server locally (optional)

```bash
uv run server --host 0.0.0.0 --port 8000
```

### Run the agent

```bash
uv run python inference.py
```

### Environment variables

Set the following before running:

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | The API endpoint for the LLM (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / API key |

### Key constants in `inference.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_STEPS` | `3` | Steps per episode |
| `NUM_EPISODES` | `16` | Number of PRs to review |
| `TEMPERATURE` | `0.2` | Sampling temperature (lower = more deterministic) |
| `MAX_TOKENS` | `512` | Max tokens per LLM response |
| `SUCCESS_SCORE_THRESHOLD` | `0.1` | Minimum score to count as success |

---

## Environment Description

The agent receives a pull request observation at each step and must respond with a structured JSON action. Each episode runs for up to `MAX_STEPS = 3` steps following a fixed workflow:

| Step | Expected Action | Purpose |
|------|----------------|---------|
| 1 | `comment` | Identify all issues in the diff |
| 2 | `suggest_fix` | Provide corrected code |
| 3 | `final_decision` | Approve or reject the PR |

Each step is independently scored. The final episode score is the maximum score achieved across all steps.

The environment automatically selects a grader tier (`easy`, `medium`, or `hard`) based on the `task_type` field of each dataset sample. No manual configuration is needed — the grader switches per episode as `reset()` is called.

---

## Action Space

Actions must be returned as JSON with the following fields:

```json
{
  "action_type": "comment | suggest_fix | final_decision",
  "comment": "Detailed description of identified issues (>30 characters)",
  "suggested_code": "Corrected code snippet, or null if not applicable",
  "decision": "approve | reject | null"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action_type` | `str` | Always | One of `comment`, `suggest_fix`, `final_decision` |
| `comment` | `str` | Recommended | Technical description of issues found |
| `suggested_code` | `str \| null` | Step 2 | Corrected code replacing the buggy diff |
| `decision` | `str \| null` | Step 3 | `approve` or `reject`; `null` otherwise |

---

## Observation Space

Each step returns a `CodeReviewObservation` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `pr` | `CodeReviewPullRequest` | The pull request under review |
| `pr.id` | `str` | Unique PR identifier |
| `pr.title` | `str` | Short title of the PR |
| `pr.description` | `str` | Brief description of intent |
| `pr.language` | `str` | Programming language (e.g. `python`) |
| `pr.diffs` | `List[CodeDiff]` | List of file diffs |
| `pr.diffs[].file_name` | `str` | Name of the changed file |
| `pr.diffs[].diff` | `str` | The actual code change |
| `previous_comments` | `List[str]` | Comments made in prior steps |
| `step_count` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps per episode (default: 3) |

---

## Scoring

### Grader tiers

The dataset contains three difficulty levels, each backed by a dedicated grader class in `graders.py`. The grader is selected automatically from `task_type` in the dataset sample.

| Tier | Class | Issue matching | Wrong decision | Done scoring |
|------|-------|---------------|---------------|--------------|
| `easy` | `EasyGrader` | Substring match | 0.2 partial credit | Max over full history |
| `medium` | `MediumGrader` | Token overlap + substring fallback | 0.1 partial credit | Recency-weighted max |
| `hard` | `HardGrader` | Token overlap + seq sim (threshold 0.3) | No credit | Final step only |

### Score components per tier

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| Issue detection | 40% | 42% | 45% |
| Fix quality | 30% | 30% | 28% |
| Decision accuracy | 30% | 28% | 27% |

**Fix quality** is computed as a weighted combination of token overlap, sequence similarity, and (for medium/hard) line-level exact matching. **Issue detection** checks how many ground-truth issues appear in the agent's comment. All scores are clamped to `[0.01, 0.99]`.

### Bonuses and penalties

| Condition | Easy | Medium | Hard |
|-----------|------|--------|------|
| Comment length > 30 chars | +0.15 | +0.10 | — |
| Correct decision at step 1 | +0.10 | +0.10 | +0.05 |
| Correct decision at step 2 | +0.10 | +0.05 | — |
| No comment on non-decision step | −0.05 | −0.08 | −0.12 |
| Step count > 3 | — | −0.04/step | −0.05 × (steps − 2) |

---

## Task Descriptions

### Easy

Straightforward single-file issues with an obvious fix. The `EasyGrader` uses simple substring matching — the agent gets full issue credit if the issue phrase appears anywhere in the comment.

| PR | Issue | Expected Decision |
|----|-------|------------------|
| Missing import | `datetime` used without import | reject |

**What the agent must do:** Detect the missing `from datetime import datetime` statement and supply the corrected import line.

---

### Medium

Logical or performance issues that require understanding of Python semantics. The `MediumGrader` uses token overlap so paraphrased descriptions still score well.

| PR | Issue | Expected Decision |
|----|-------|------------------|
| Division function | No guard against division by zero | reject |
| Inefficient loop | `range(len(arr))` pattern; can use `in` directly | approve |

**What the agent must do:** For the division task, add a `if b == 0: return None` guard. For the loop task, recognise it as a style issue but not a correctness bug — the correct decision is **approve**.

---

### Hard

Security vulnerabilities, injection attacks, and cross-file null-handling bugs. The `HardGrader` applies a minimum similarity threshold: vague or generic comments receive zero issue credit.

| PR | Issue | Expected Decision |
|----|-------|------------------|
| Authentication logic | Hardcoded plaintext password `admin123` | reject |
| SQL query | String concatenation exposes SQL injection | reject |
| Cross-file null bug | `get_user(None)` called without input validation | reject |

**What the agent must do:**
- **Auth:** Detect the hardcoded secret and propose `bcrypt`-based password comparison.
- **SQL:** Detect string concatenation and replace with a parameterised query using `%s` placeholder + `cursor.execute`.
- **Null bug:** Validate `id is not None` before the `db[id]` lookup and fix the call site in `controller.py`.

---

## Baseline Scores

Expected performance ranges by model capability:

| Score Range | Interpretation |
|-------------|---------------|
| 0.00 – 0.20 | Failing — agent cannot follow the JSON schema or identify basic issues |
| 0.20 – 0.50 | Partial — agent detects some issues but misses security vulnerabilities or gives wrong decisions |
| 0.50 – 0.75 | Competent — agent handles easy and medium tasks; struggles with hard security/null cases |
| 0.75 – 1.00 | Strong — agent reliably detects all issue types, generates correct fixes, and makes sound decisions |

### Step-level log format

```
[START] task=code_review env=code_review_benchmark model=meta-llama/Llama-3.1-8B-Instruct
[STEP]  step=1 action=comment        reward=0.55 done=false error=null
[STEP]  step=2 action=suggest_fix    reward=0.72 done=false error=null
[STEP]  step=3 action=final_decision reward=0.85 done=true  error=null
[END]   success=true steps=3 score=0.850 rewards=0.55,0.72,0.85
```

---

## Conclusion

The Code Review Environment provides a structured, reproducible benchmark for evaluating LLM-based agents on one of the most practically valuable tasks in software engineering. By decomposing the review process into three distinct steps — issue detection, fix generation, and final judgment — and by scaling difficulty through dedicated grader tiers, it rewards agents that reason carefully rather than those that simply pattern-match on surface-level symptoms.