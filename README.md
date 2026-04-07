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

A reinforcement learning benchmark environment where an agent acts as a senior software engineer reviewing pull requests. The agent must identify bugs, suggest fixes, and make approval decisions across progressively harder code review tasks.

## Quick Start
 
Install the OpenEnv core package:
```bash
pip install openenv-core
```

Clone the repo
```bash
git clone https://github.com/Ajay-Ganapathy/code_review && cd code_review
```

Install packages
```bash
uv pip install -e .
```

`[OPTIONAL]` To run server locally
```bash
uv run server --host 0.0.0.0 --port 8000
```

Run the agent in another terminal
```bash
uv run python inference.py
```

## Motivation
 
Code review is a high-stakes, multi-step reasoning task that requires an agent to:
 
- **Detect bugs and security vulnerabilities** from raw code diffs
- **Generate corrective code** that resolves identified issues
- **Make a final judgment** (approve/reject) backed by technical reasoning
 
Existing benchmarks test code generation or comprehension in isolation. This environment tests the full review loop — detection, remediation, and decision-making — in a structured, scorable way. It is designed to evaluate whether LLMs can act as reliable automated reviewers in software development pipelines.

## Environment Description
 
The agent receives a pull request observation at each step and must respond with a structured JSON action. The episode runs for up to `MAX_STEPS = 3` steps, following a prescribed workflow:
 
| Step | Expected Action | Purpose |
|------|----------------|---------|
| 1 | `comment` | Identify all issues in the diff |
| 2 | `suggest_fix` | Provide corrected code |
| 3 | `final_decision` | Approve or reject the PR |
 
Each step is independently scored, and the final episode score is the maximum score achieved across all steps.

## Action Space
 
Actions are instances of `CodeReviewAction` and must be returned as JSON with the following fields:
 
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

## Scoring
 
Each action is scored across three components:
 
| Component | Weight | Method |
|-----------|--------|--------|
| Issue Detection | 40% | Fraction of ground-truth issues mentioned in `comment` |
| Fix Quality | 30% | Token overlap + sequence similarity between `suggested_code` and ground-truth fix |
| Decision Accuracy | 30% | Exact match with ground-truth `approve`/`reject`; partial credit (0.2) for wrong decision |
 
**Bonuses and penalties applied per step:**
 
- `+0.1` — comment length > 30 characters (encourages detail)
- `+0.1` — correct final decision reached in step ≤ 2 (encourages efficiency)
- `-0.1` — no comment provided on a non-decision step (penalizes lazy steps)
- `-0.05` — step count exceeds 3 (penalizes long trajectories)
 
The final episode score is the **maximum** `grade_action` score across all steps in the episode. Scores are clamped to `[0.0, 1.0]`.

## Task Descriptions
 
The dataset contains tasks at three difficulty levels:
 
### Easy
 
Straightforward single-file issues with an obvious fix.
 
| PR | Issue | Expected Decision |
|----|-------|------------------|
| Missing import | `datetime` used without import | reject |
 
**What the agent must do:** Detect the missing `from datetime import datetime` statement and supply the corrected import.
 
---
 
### Medium
 
Logical or performance issues requiring understanding of Python semantics.
 
| PR | Issue | Expected Decision |
|----|-------|------------------|
| Division function | No guard against division by zero | reject |
| Inefficient loop | `range(len(arr))` pattern; can use `in` operator | approve |
 
**What the agent must do:** For the division task, add a `if b == 0: return None` guard. For the loop task, recognize it as a style/efficiency issue but not a correctness bug — the correct decision is **approve**.
 
---
 
### Hard
 
Security vulnerabilities, injection attacks, and cross-file null-handling bugs.
 
| PR | Issue | Expected Decision |
|----|-------|------------------|
| Authentication logic | Hardcoded plaintext password `admin123` | reject |
| SQL query | String concatenation exposes SQL injection | reject |
| Cross-file null bug | `get_user(None)` called without input validation | reject |
 
**What the agent must do:**
- For auth: detect the hardcoded secret and propose `bcrypt`-based password comparison.
- For SQL: detect string concatenation and replace with a parameterized query (`%s` placeholder + `cursor.execute`).
- For null bug: validate `id is not None` before the `db[id]` lookup, and fix the call site in `controller.py`.
 
The agent runs `NUM_EPISODES = 16` episodes (configurable) with each `MAX_STEPS = 3` and logs each step:
 
```
[START] task=code_review env=code_review_benchmark model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=... reward=0.55 done=false error=null
[STEP] step=2 action=... reward=0.72 done=false error=null
[STEP] step=3 action=... reward=0.85 done=true error=null
[END] success=true steps=12 score=0.850 rewards=0.55,0.72,0.85,0.60
```

## Configuration
 
Key constants in `inference.py`:
 
| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_STEPS` | `3` | Steps per episode |
| `NUM_EPISODES` | `16` | Number of PRs to review |
| `TEMPERATURE` | `0.2` | Sampling temperature (lower = more deterministic) |
| `MAX_TOKENS` | `256` | Max tokens per LLM response |
| `SUCCESS_SCORE_THRESHOLD` | `0.1` | Minimum normalized score to count as success |

## Score Interpretation
 
| Score Range | Interpretation |
|-------------|---------------|
| 0.00 – 0.20 | Failing — agent cannot follow the JSON schema or identify basic issues |
| 0.20 – 0.50 | Partial — agent detects some issues but misses security vulnerabilities or gives wrong decisions |
| 0.50 – 0.75 | Competent — agent handles easy and medium tasks; struggles with hard security/null cases |
| 0.75 – 1.00 | Strong — agent reliably detects all issue types, generates correct fixes, and makes sound decisions |

## Conclusion
 
The Code Review Environment provides a structured, reproducible benchmark for evaluating LLM-based agents on one of the most practically valuable tasks in software engineering. By decomposing the review process into three distinct steps — issue detection, fix generation, and final judgment — it rewards agents that reason carefully rather than those that simply pattern-match on surface-level symptoms.

