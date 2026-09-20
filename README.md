# HMM 解码服务

隐马尔可夫模型的**登记**与**解码**服务：Viterbi 最可能整段状态路径 + 前向后向逐时刻后验。
Python 3.12 / Flask，模型档写入进程内 SQLite，仅通过 HTTP 对外提供。

## 运行

```bash
pip install -r requirements.txt
python main.py                 # 监听 0.0.0.0:8000（PORT 可改）
```

Docker：

```bash
docker build -t hmm-decoder .
docker run -p 8000:8000 hmm-decoder
```

模型档 SQLite 路径由 `HMM_DB_PATH` 指定（默认 `./hmm_models.sqlite3`，容器内 `/data/hmm_models.sqlite3`）。

## API

### `POST /models` — 登记模型档

```json
{
  "name": "my-model",
  "states": ["s0", "s1"],            // 可选，缺省 S0..Sn-1
  "alphabet": ["x", "y"],
  "initial": [0.6, 0.4],
  "transition": [[0.9, 0.1], [0.1, 0.9]],
  "emission": [[0.9, 0.1], [0.1, 0.9]]
}
```

登记时校验（失败即拒绝，不会留到解码时）：状态数 ≥ 2；转移矩阵每行和为 1；
发射矩阵每个状态对字母表归一；初值和为 1（容差钉死 `1e-6`）；维数与状态数、
字母表长度一致；概率落在 `[0,1]`；字母表符号非空且唯一。

### `GET /models` / `GET /models/<name>` — 列出 / 查看

### `POST /decode` — 按名解码

```json
{"model": "demo-regime-switch", "observations": "aaaaaaaaaaaabbbbbbbbbbbb"}
```

`observations` 可为符号数组，或为字符串（按字符拆分）。返回：

```json
{
  "model": "demo-regime-switch",
  "length": 24,
  "path": ["regime_a", "...", "transit", "regime_b", "..."],
  "log_probability": -11.546...,
  "log_likelihood": -10.247...,
  "posteriors": [{"regime_a": 0.997, "regime_b": 0.001, "transit": 0.002}, ...],
  "tie_break": "lowest_state_index"
}
```

- `path`：最大化整段联合概率 P(路径, 观测) 的 Viterbi 路径（回溯得到）。
- `log_probability`：该路径的对数联合概率。
- `posteriors`：前向（t=0 为初值×发射）与后向（末尾取对数单位值 0 往回走）
  在同一时刻相加（对数域）再归一；每时刻各状态后验之和为 1。
- `tie_break`：并列打破规则，钉死为 `lowest_state_index`（扫描按状态下标升序，
  仅在严格更大时替换），前驱选择与最终状态选择共用此规则。

### `GET /demo` — 内置示范档的观测串

## 内置示范档 `demo-regime-switch`

三状态分叉例：两个黏性区段状态（`regime_a`/`regime_b`，自转 0.95，直接互转仅
0.002）加一个瞬态 `transit`（自转 0.1，是区段间最可信的通道）。内置观测串人为
嵌入一次区段切换（12 个 `a` 接 12 个 `b`）：

- Viterbi 跟上切换，并在边界**恰好经过 transit 一步**；
- 同一观测上，后验把切换点的不确定性摊到边界两个时刻（transit 后验各约 0.42），
  **逐时刻取最大永远选不到 transit** —— 因此「逐点后验最大串 ≠ Viterbi 路径」，
  测试 `test_posterior_argmax_is_not_the_viterbi_path` 用这份分叉卡住
  「用后验冒充 Viterbi」的实现。

## 计算约定

- 全部概率在**对数域**累加（logsumexp），长序列不会下溢成全零后再填一条随意路径；
  观测序列在模型下概率为 0 时返回 `422 zero_probability`，而不是伪造路径。
- 长度 1、2 的极短串走同一套 Viterbi / 前向后向代码，无启发式分支。

## 错误

错误响应统一为 `{"error": {"type": ..., "message": ...}}`：

| type | 状态码 | 含义 |
|---|---|---|
| `missing_field` | 400 | 缺必需字段 |
| `invalid_model` | 400 | 模型档校验失败（行和、维数、概率域等） |
| `invalid_body` / `invalid_observations` | 400 | 请求体形态错误 |
| `empty_observations` | 400 | 空观测串 |
| `unknown_symbol` | 400 | 观测符号不在字母表内 |
| `unknown_model` | 404 | 未登记的档名 |
| `zero_probability` | 422 | 观测序列在模型下概率为 0 |

## 模块划分

```
app/
  validation.py   模型档与观测校验（归一容差钉死在此）
  logdomain.py    对数域原语（to_log / logsumexp）
  viterbi.py      Viterbi 递推与回溯（并列打破规则钉死在此）
  posterior.py    前向后向与逐时刻后验
  store.py        模型档存取（进程内 SQLite，带锁）
  demo.py         内置三状态分叉示范档
  service.py      Flask 路由与错误映射
main.py           入口
tests/            端到端测试（经 HTTP 层）
```

## 测试

```bash
python -m pytest tests/ -q
```

覆盖：区段切换可跟上、后验每时刻求和为 1、分叉例上 Viterbi ≠ 逐点后验最大、
行和偏离 1 登记失败、空观测 / 未知符号 / 未知档名被拒、极短串正常解码、
长序列（5000）不下溢、单点观测改动使路径该时刻切换、发射模糊化后路径跟转移走、
转入概率压近 0 后路径不停留、两档并行解码互不写串。
