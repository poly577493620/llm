# 大模型截图识别实验（任务三）

## 项目用途

通过 Playwright 截图采集佛山土地交易详情页信息，结合多模态大模型（Kimi K2.6）识别截图中的结构化字段，与任务二参考值进行比对核验。

## 工作流程

```
读取任务二 JSON
       ↓
从前 10 页各选择 1 条已成交记录
       ↓
Playwright 打开详情页 → 穿过两层 iframe → 截图
       ↓
调用 Kimi K2.6 多模态模型识别截图中的字段
       ↓
保存原始 JSON 响应
       ↓
与任务二参考值规范化比较
       ↓
导出 task3_llm_results.xlsx
```

## 环境要求

- Python 3.14
- uv（版本管理工具）
- Windows 系统
- 系统已安装 Google Chrome 或 Microsoft Edge（截图用）
- OpenCode Go 多模态模型 API（OpenAI 兼容接口）

## 安装

```powershell
cd 03_llm_screenshot
uv sync
```

> **注意**：默认使用系统 Chrome，无需额外安装 Playwright Chromium。如需备用方案，可选运行：
> ```powershell
> uv run playwright install chromium
> ```

## 配置

复制环境变量模板并填入真实 API 密钥：

```powershell
Copy-Item .env.example .env
# 编辑 .env，填入 VISION_API_KEY 等值
```

## 运行

```powershell
# 完整流程（截图 + 模型 + 导出），使用系统 Chrome
uv run python -m foshan_scraper --browser-channel chrome

# 演示模式（只处理 2 条，方便录屏）
uv run python -m foshan_scraper --limit 2 --browser-channel chrome

# 离线模式（跳过截图和模型调用，从 raw_outputs 读取已有结果）
uv run python -m foshan_scraper --offline

# 仅截图（用于分阶段测试和录屏）
uv run python -m foshan_scraper --capture-only --browser-channel chrome

# 关闭断点续传（强制重新调用模型）
uv run python -m foshan_scraper --no-resume --browser-channel chrome
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--limit` | 限制处理记录数（默认全部，用于演示可设为 2） |
| `--source-json` | 任务二 JSON 文件路径 |
| `--output-dir` | 输出目录（默认 ./output） |
| `--headless/--no-headless` | Playwright 无头/可视模式 |
| `--browser/--no-browser` | 是否启动浏览器采集截图 |
| `--browser-channel` | 浏览器渠道：`chrome`（默认）或 `msedge` |
| `--capture-only` | 仅完成截图采集，不调用模型也不导出 Excel |
| `--offline` | 离线模式，从已有 raw_outputs 读取 |
| `--resume/--no-resume` | 启用断点续传（默认启用，跳过已存在的 raw JSON） |

## 模型与服务商

- **模型**：Kimi K2.6
- **服务商**：OpenCode Go
- **接口格式**：OpenAI 兼容 Chat Completions
- **图片传输**：base64 编码 PNG
- **特殊配置**：关闭 thinking、启用 JSON 模式（`response_format: json_object`）

## 模型返回字段

模型从截图中识别以下 8 个字段：

| 字段名 | 含义 |
|--------|------|
| `transaction_date` | 成交时间 |
| `winner` | 竞得人 |
| `transaction_area` | 交易土地面积 |
| `transaction_price` | 成交地价 |
| `land_location` | 地块位置 |
| `land_usage` | 土地实际用途（详情页实际用途，非 API 分类） |
| `uncertain_fields` | 模型无法确认的字段列表 |
| `notes` | 备注 |

## 土地用途口径差异说明

**API 中的 `land_usage` 是土地用途分类**，由后端系统按固定类别编码；
**截图中的 `land_usage` 是土地实际用途**，由详情页展示的具体文字描述。

两者可能同时正确但文本不同（例如：API 返回 "商服用地"，详情页显示 "商业用地（零售商业用地、餐饮用地、旅馆用地），兼容娱乐用地"）。

遇到这种情况：
- **不直接判定为 OCR 识别错误**；
- 单独标记为 **"字段口径不同，需人工核验"**；
- 总体状态为 `needs_review`，但属于可理解的差异。

## 核验结果说明

Excel 中核验状态：

- `match` — 所有字段匹配，无口径差异
- `needs_review` — 有差异需人工核验（含字段口径差异和普通识别差异）

差异分类：

- **普通识别差异**：模型识别值与参考值规范化后不一致，属于 OCR 识别错误或待核验字段
- **字段口径差异**：API 土地用途分类与详情页土地实际用途含义不同，两边可能同时正确

## 输出文件

- `data/reference_data.json` — 任务二参考数据（已成交记录，供比对使用）
- `output/screenshots/screenshot_01.png` ~ `screenshot_10.png` — 详情页截图（10 张）
- `output/raw_outputs/raw_01.json` ~ `raw_10.json` — 模型原始响应（10 份）
- `output/task3_llm_results.xlsx` — 比对核验结果

## 最终提交材料

1. **任务二参考数据** — `data/reference_data.json`
2. **10 张截图** — `output/screenshots/` 目录
3. **10 份原始模型 JSON** — `output/raw_outputs/` 目录
4. **Excel 核验结果** — `output/task3_llm_results.xlsx`
5. **字段识别提示词** — `prompts/字段识别提示词.md`
6. **技术说明** — `docs/任务三_技术说明.md`
7. **数据核验说明** — `docs/数据核验说明.md`
8. **个人心得** — `docs/任务三_个人心得.md`
9. **执行录屏** — `evidence/任务三_执行录屏.mp4`

