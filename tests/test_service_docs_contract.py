"""Source-backed service docs checks without loading models or starting servers."""

import ast
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/openai_api"
READMES = [EXAMPLE / name for name in ("README.md", "README_zh.md", "README_ja.md", "README_ko.md")]
DOCS = [EXAMPLE / "CLIENTS.md", *READMES]
PACKAGED = ROOT / "funasr/bin/_server_app.py"
WORKFLOW_DOCS = [EXAMPLE / name for name in ("WORKFLOWS.md", "WORKFLOWS_zh.md")]

# Frozen from the two pre-edit workflow guides, not translated from one language.
WORKFLOW_HEADINGS = {
    "WORKFLOWS.md": "Low-code Workflow Recipes for the FunASR OpenAI-Compatible API|Server preflight|Postman smoke test|Multipart HTTP request|Dify custom tool or HTTP node|Direct file upload path|Audio URL path|n8n HTTP Request node|n8n OpenAI Audio node|Webhook worker pattern|Production guardrails|Troubleshooting",
    "WORKFLOWS_zh.md": "FunASR OpenAI 兼容 API 低代码工作流配方|服务预检|Postman smoke test|Multipart HTTP 请求|Dify 自定义工具或 HTTP 节点|直接上传文件|音频 URL 转写|n8n HTTP Request 节点|n8n OpenAI Audio 节点|Webhook worker 模式|生产环境护栏|故障排查",
}

# Frozen from the pre-edit README inventory, excluding fenced code comments.
LEGACY_HEADINGS = {
    "README.md": "FunASR OpenAI-Compatible API Server|API Contract|Quick Start|End-to-end smoke test|Browser demo with Gradio|Usage with OpenAI SDK (Python)|Usage with curl|Available Models|API Endpoints|Agent Framework Integration|LangChain Example|Docker Deployment|Kubernetes Deployment|Configuration|Troubleshooting",
    "README_zh.md": "FunASR OpenAI 兼容 API 服务|API Contract|快速开始|端到端 smoke test|Gradio 浏览器 Demo|使用 OpenAI SDK|使用 curl|可用模型|API 端点|Agent 与低代码工作流|Docker 部署|Kubernetes 部署|配置|故障排查",
    "README_ja.md": "FunASR OpenAI 互換 API サーバー|クイックスタート|エンドツーエンド smoke test|Gradio ブラウザデモ|OpenAI SDK で使う|curl で使う|利用できるモデル|API エンドポイント|エージェントとローコードワークフロー|Docker デプロイ|Kubernetes デプロイ|設定|トラブルシューティング",
    "README_ko.md": "FunASR OpenAI 호환 API 서버|빠른 시작|엔드투엔드 smoke test|Gradio 브라우저 데모|OpenAI SDK로 사용하기|curl로 사용하기|사용 가능한 모델|API 엔드포인트|에이전트 및 로우코드 워크플로|Docker 배포|Kubernetes 배포|설정|문제 해결",
}


def supported_module_command(args):
    """Allow only the documented environment setup commands, not arbitrary modules."""
    return args in (["python3.11", "-m", "venv", ".venv"],
                    ["python", "-m", "pip", "install", "-e", "."],
                    ["python", "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart"],
                    ["python", "-m", "pip", "install", "openai"],
                    ["python", "-m", "pip", "install", "gradio"],
                    ["python", "-m", "pip", "check"])


def blocks(text, language):
    return re.findall(r"^```" + language + r"\n(.*?)^```", text, re.M | re.S)


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def function(path, name):
    return next(node for node in ast.walk(tree(path))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name)


def form_defaults(node):
    defaults = {}
    for arg, value in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Form":
            defaults[arg.arg] = ast.literal_eval(next(
                keyword.value for keyword in value.keywords if keyword.arg == "default"))
    return defaults


def headings(text):
    text = re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", text, flags=re.M | re.S)
    result = set()
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(f"{slug}-{count}" if count else slug)
    return result


class ServiceDocsContract(unittest.TestCase):
    def test_preserves_existing_readme_headings(self):
        for path in READMES:
            text = path.read_text(encoding="utf-8")
            source = re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", text, flags=re.M | re.S)
            actual = re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", source, re.M)
            expected = LEGACY_HEADINGS[path.name].split("|")
            self.assertEqual([value for value in actual if value in expected], expected, path.name)

    def test_long_contract_and_model_explanations_are_readable_lists(self):
        model_headings = ("Available Models", "可用模型", "利用できるモデル", "사용 가능한 모델")
        configs = next(node for node in ast.walk(tree(EXAMPLE / "server.py"))
                       if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                       and target.id == "MODEL_CONFIGS" for target in node.targets))
        aliases = [ast.literal_eval(key) for key in configs.value.keys]
        for path, model_heading in zip(READMES, model_headings):
            text = path.read_text(encoding="utf-8")
            for heading in ("API Contract", model_heading):
                section = text.split("## " + heading + "\n", 1)[1].split("\n## ", 1)[0]
                self.assertNotRegex(section, r"(?m)^\|", path.name)
                if heading == model_heading:
                    self.assertEqual(re.findall(r"(?m)^- `([^`]+)`", section), aliases, path.name)

    def test_module_command_allowlist_rejects_unchecked_variants(self):
        self.assertTrue(supported_module_command(["python", "-m", "pip", "check"]))
        for args in (["python", "-m", "missing"], ["python", "-m", "pip", "uninstall", "funasr"],
                     ["python3.11", "-m", "venv"], ["python", "-m", "pip", "check", "--bogus"]):
            self.assertFalse(supported_module_command(args))

    def test_readme_checkout_and_loopback_recipes(self):
        for path in READMES:
            text = path.read_text(encoding="utf-8")
            quick_start = blocks(text, "bash")[0]
            ordered = ["git clone https://github.com/modelscope/FunASR.git FunASR-api", "cd FunASR-api",
                       "git checkout --detach e19029adca384a06a2f60bd8c18cb98f1a0499aa",
                       "python3.11 -m venv .venv", "source .venv/bin/activate",
                       "python -m pip install -e .", "python -m pip install fastapi uvicorn python-multipart",
                       "python -m pip check", "cd examples/openai_api",
                       "python server.py --host 127.0.0.1 --model sensevoice --device cpu --port 8000"]
            positions = [quick_start.index(command) for command in ordered]
            self.assertEqual(positions, sorted(positions), path.name)
            self.assertIn("SECURITY", text[:text.index("```bash")])
            for marker in ("/v1/models", "/openapi.json", "Fun-ASR-MLT-Nano", '--model-path', '--hub',
                           'model="custom"', 'whisper-1', 'sentence_info', 'ctc_timestamps', '0.0.0.0'):
                self.assertIn(marker, text, path.name)
            self.assertIn("FUNASR_HOST_PORT=127.0.0.1:8000 docker compose up --build", text)
            for code in blocks(text, "bash"):
                for line in code.replace("\\\n", " ").splitlines():
                    args = shlex.split(line, comments=True)
                    if args[:2] == ["docker", "run"]:
                        self.assertEqual(args[args.index("-p") + 1], "127.0.0.1:8000:8000")
            self.assertNotRegex(text, r"(?:170|120|3\.6)[x×倍]")

    def test_readme_clients_close_files_and_use_supported_form_fields(self):
        allowed = set(form_defaults(function(EXAMPLE / "server.py", "transcribe"))) | {"file"}
        for path in READMES:
            formats = []
            for code in blocks(path.read_text(encoding="utf-8"), "python"):
                parsed = ast.parse(code)
                calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "create"]
                for call in calls:
                    kwargs = {kw.arg: kw.value for kw in call.keywords}
                    formats.append(ast.literal_eval(kwargs["response_format"]) if "response_format" in kwargs else None)
                    self.assertEqual(ast.literal_eval(kwargs["model"]), "sensevoice")
                    self.assertLessEqual(set(kwargs), allowed)
                    self.assertIsInstance(kwargs["file"], ast.Name)
                    contexts = [node for node in ast.walk(parsed) if isinstance(node, ast.With)
                                and call in list(ast.walk(node))]
                    self.assertTrue(any(isinstance(item.optional_vars, ast.Name)
                                        and item.optional_vars.id == kwargs["file"].id
                                        and isinstance(item.context_expr, ast.Call)
                                        and isinstance(item.context_expr.func, ast.Name)
                                        and item.context_expr.func.id == "open"
                                        and len(item.context_expr.args) == 2
                                        and ast.literal_eval(item.context_expr.args[1]) == "rb"
                                        for node in contexts for item in node.items))
            self.assertGreaterEqual(len(formats), 2, path.name)
            self.assertEqual(set(formats), {None, "verbose_json"}, path.name)

    def test_relative_links_and_owned_page_anchors(self):
        owned = {path.resolve() for path in DOCS}
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            self.assertIn("## API Contract", text)
            self.assertIn("](#api-contract)", text)
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                parts = urlsplit(link)
                if parts.scheme or parts.netloc:
                    continue
                target = (path.parent / unquote(parts.path)).resolve() if parts.path else path.resolve()
                self.assertIn(ROOT.resolve(), target.parents)
                self.assertNotIn(".mcp-tasks", target.parts)
                self.assertTrue(target.exists(), f"{path.name}: {link}")
                if parts.fragment and target in owned:
                    self.assertIn(unquote(parts.fragment), headings(target.read_text(encoding="utf-8")), link)

    def test_sdk_links_and_unsupported_claims(self):
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            for link in ("../../docs/python_api.md", "../../docs/python_api_zh.md"):
                self.assertIn(f"]({link})", text)
            for marker in ("spk=true", "segments=[]", "--model auto", "OpenAI API"):
                self.assertIn(marker, text)
            for stale in ("170x realtime", "120x realtime", "3.6x realtime",
                          "Server starts in ~20s", "Works with **any agent framework**",
                          "Fast default with language, emotion, and event tags",
                          "Returns: text, segments (with start/end/speaker), duration"):
                self.assertNotIn(stale, text)

    def test_python_and_shell_examples_are_syntactically_valid(self):
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            for index, code in enumerate(blocks(text, "python")):
                compile(code, f"{path}:{index}", "exec")
            for index, code in enumerate(blocks(text, "bash")):
                result = subprocess.run(["bash", "-n"], input=code, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, f"{path}:{index}: {result.stderr}")

    def test_local_script_commands_and_options_match_source(self):
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            if path.name.startswith("README"):
                quick_start = blocks(text, "bash")[0]
                self.assertIn("cd examples/openai_api", quick_start)
                self.assertLess(quick_start.index("cd examples/openai_api"), quick_start.index("python server.py"))
            for code in blocks(text, "bash"):
                for line in code.replace("\\\n", " ").splitlines():
                    args = shlex.split(line, comments=True)
                    if len(args) < 2 or args[0] not in ("python", "python3.11", "bash"):
                        continue
                    if args[1] == "-m":
                        self.assertTrue(supported_module_command(args), args)
                        continue
                    script = EXAMPLE / args[1]
                    self.assertTrue(script.is_file(), str(script))
                    if args[0] != "python":
                        continue
                    options = {ast.literal_eval(arg) for node in ast.walk(tree(script))
                               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                               and node.func.attr == "add_argument"
                               for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
                    for arg in args[2:]:
                        if arg.startswith("--"):
                            self.assertIn(arg.split("=", 1)[0], options, str(script))

    def test_form_defaults_and_alias_boundaries_match_source(self):
        example = form_defaults(function(EXAMPLE / "server.py", "transcribe"))
        packaged = form_defaults(function(PACKAGED, "transcribe"))
        self.assertEqual(example["model"], "sensevoice")
        self.assertNotIn("spk", example)
        self.assertEqual(packaged["model"], "fun-asr-nano")
        self.assertIs(packaged["spk"], False)
        for path, name, expected in ((EXAMPLE / "server.py", "MODEL_CONFIGS", True),
                                     (PACKAGED, "FALLBACK_CONFIGS", False)):
            assignment = next(node for node in ast.walk(tree(path))
                              if isinstance(node, ast.Assign) and any(
                                  isinstance(target, ast.Name) and target.id == name for target in node.targets))
            aliases = [ast.literal_eval(key) for key in assignment.value.keys]
            self.assertEqual("paraformer-en" in aliases, expected)

    def test_example_serializes_sentence_info_not_raw_sdk_timestamps(self):
        handler = function(EXAMPLE / "server.py", "transcribe")
        branch = next(node for node in ast.walk(handler) if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Compare)
                      and isinstance(node.test.left, ast.Name)
                      and node.test.left.id == "response_format")

        class CaptureReturn(ast.NodeTransformer):
            def visit_Return(self, node):
                return ast.copy_location(ast.Assign(
                    targets=[ast.Name(id="response", ctx=ast.Store())], value=node.value), node)

        executable = ast.fix_missing_locations(CaptureReturn().visit(ast.Module(body=[branch], type_ignores=[])))
        clean = function(EXAMPLE / "server.py", "clean_text")
        namespace = {"re": re, "JSONResponse": lambda value: value, "response_format": "verbose_json",
                     "elapsed": 0.42, "language": None, "model": "sensevoice", "text": "hello"}
        exec(compile(ast.Module(body=[clean], type_ignores=[]), "clean-text", "exec"), namespace)
        raw_token = {"token": "hello", "start_time": 100, "end_time": 400}
        cases = [({"timestamp": [[100, 400]], "timestamps": [raw_token], "ctc_timestamps": [raw_token]}, []),
                 ({"sentence_info": [{"start": 100, "end": 400, "text": "<|HAPPY|>hello"}]},
                  [{"start": 0.1, "end": 0.4, "text": "hello", "speaker": None}])]
        for result, expected in cases:
            namespace["result"] = [result]
            exec(compile(executable, "example-verbose-branch", "exec"), namespace)
            self.assertEqual(namespace["response"]["segments"], expected)
            self.assertEqual(namespace["response"]["language"], "auto")
            self.assertEqual(namespace["response"]["duration"], 0.42)

    def test_workflow_alias_and_compose_host_binding_match_source(self):
        resolver = function(EXAMPLE / "server.py", "resolve_openai_transcription_model")
        namespace = {"N8N_OPENAI_MODEL_ALIAS": "whisper-1", "DEFAULT_MODEL": "sensevoice"}
        exec(compile(ast.Module(body=[resolver], type_ignores=[]), "workflow-alias", "exec"), namespace)
        self.assertEqual(namespace[resolver.name]("whisper-1"), "sensevoice")
        self.assertEqual(namespace[resolver.name]("paraformer"), "paraformer")
        compose = (EXAMPLE / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn('"${FUNASR_HOST_PORT:-8000}:8000"', compose)
        dockerfile = (EXAMPLE / "Dockerfile").read_text(encoding="utf-8")
        command = next(json.loads(line[4:]) for line in dockerfile.splitlines() if line.startswith("CMD "))
        args = shlex.split(command[-1])
        self.assertEqual(args[args.index("--host") + 1], "0.0.0.0")

    def test_startup_defaults_match_source(self):
        for path, expected in ((EXAMPLE / "server.py", "sensevoice"),
                               (ROOT / "funasr/bin/server.py", "auto")):
            option = next(node for node in ast.walk(tree(path)) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                          and node.args and isinstance(node.args[0], ast.Constant)
                          and node.args[0].value == "--model")
            self.assertEqual(next(ast.literal_eval(kw.value) for kw in option.keywords
                                  if kw.arg == "default"), expected)
        startup = function(PACKAGED, "create_app")
        selection = next(node for node in ast.walk(startup) if isinstance(node, ast.IfExp)
                         and isinstance(node.body, ast.Constant) and node.body.value == "fun-asr-nano")
        for device, expected in (("cuda:0", "fun-asr-nano"), ("cpu", "sensevoice"), ("mps", "sensevoice")):
            self.assertEqual(eval(compile(ast.Expression(selection), str(PACKAGED), "eval"), {"device": device}), expected)

    def test_packaged_verbose_example_matches_real_builder(self):
        samples = [json.loads(code) for code in blocks(DOCS[0].read_text(encoding="utf-8"), "json")]
        self.assertEqual(len(samples), 3)
        namespace = {"resolve_transcription_language": lambda requested, result: requested or result["language"]}
        builder = function(PACKAGED, "build_openai_verbose_json")
        exec(compile(ast.Module(body=[builder], type_ignores=[]), str(PACKAGED), "exec"), namespace)
        result = {"text": "recognized speech", "language": "en", "duration": 3.2,
                  "segments": [{"start": 0.0, "end": 3.2, "text": "recognized speech"}]}
        self.assertEqual(namespace[builder.name](result), samples[1])
        self.assertNotIn("speaker", samples[1]["segments"][0])
        self.assertLessEqual(samples[1]["segments"][0]["end"], samples[1]["duration"])

    def test_example_verbose_sample_and_duration_meaning(self):
        samples = [json.loads(code) for code in blocks(DOCS[0].read_text(encoding="utf-8"), "json")]
        handler = function(EXAMPLE / "server.py", "transcribe")
        response = next(node for node in ast.walk(handler) if isinstance(node, ast.Dict)
                        and any(isinstance(key, ast.Constant) and key.value == "duration" for key in node.keys))
        expected = eval(compile(ast.Expression(response), "example-response", "eval"),
                        {"text": "recognized speech", "segments": [], "language": None,
                         "elapsed": 0.42, "model": "sensevoice"})
        self.assertEqual(expected, samples[2])
        source = PACKAGED.read_text(encoding="utf-8")
        self.assertIn("float(sf.info(audio_path).duration)", source)
        self.assertIn('"duration": len(audio_data) / sr', source)
        clients = DOCS[0].read_text(encoding="utf-8")
        self.assertIn("not audio duration", clients)
        self.assertIn("Audio duration in seconds", clients)
        self.assertIn("does not enable diarization", clients)


class WorkflowDocsContract(unittest.TestCase):
    def test_workflow_explanatory_sections_use_complete_readable_lists(self):
        sections = [
            (("Multipart HTTP request", "Multipart HTTP 请求"),
             ["Method", "URL", "Body type", "File field", "Text field", "Text field", "Timeout"],
             [("POST",), ("/v1/audio/transcriptions",), ("multipart/form-data",), ("`file`",),
              ("model=sensevoice",), ("response_format=verbose_json",), ("300",)]),
            (("n8n HTTP Request node", "n8n HTTP Request 节点"),
             ["Method", "URL", "Send Body", "Body Content Type", "Binary file field", "Additional form fields", "Response Format", "Timeout"],
             [("POST",), ("/v1/audio/transcriptions",), ("enabled",), ("Form-Data", "multipart"),
              ("`file`",), ("model=sensevoice", "response_format=verbose_json"), ("JSON",), ()]),
        ]
        symptoms = [
            ["Workflow can call `/health` but transcription fails", "`localhost` connection fails from Dify or n8n",
             "Response has no usable `segments`", "Requests time out", "First request is slow", "Unknown model alias"],
            ["工作流能访问 `/health`，但转写失败", "Dify 或 n8n 访问 `localhost` 失败", "响应中没有可用 `segments`",
             "请求超时", "第一次请求很慢", "模型别名未知"],
        ]
        fixes = [("multipart/form-data", "`file`"), ("Compose", "Kubernetes"),
                 ("sentence_info", "verbose_json"), ("HTTP timeout",), ("--model sensevoice", "/health"), ("/v1/models",)]
        for language, path in enumerate(WORKFLOW_DOCS):
            text = path.read_text(encoding="utf-8")
            cases = [*sections, (("Troubleshooting", "故障排查"), symptoms[language], fixes)]
            for titles, labels, tokens in cases:
                with self.subTest(path=path.name, section=titles[language]):
                    section = text.split("## " + titles[language] + "\n", 1)[1].split("\n## ", 1)[0]
                    self.assertNotRegex(section, r"(?m)^\s*\|")
                    self.assertNotRegex(section, r"(?i)<table\b")
                    lines = section.splitlines()
                    start = next((i for i, line in enumerate(lines) if line.startswith("- **")), None)
                    self.assertIsNotNone(start, "Missing explanation list")
                    items = []
                    for line in lines[start:]:
                        if not line.startswith("- "):
                            break
                        match = re.fullmatch(r"- \*\*(.+?):\*\* (.+)", line)
                        self.assertIsNotNone(match, line)
                        items.append(match.groups())
                    self.assertEqual([label for label, _ in items], labels)
                    for (_, value), required in zip(items, tokens):
                        for token in required:
                            self.assertIn(token, value)

    def test_preserves_workflow_headings_and_links(self):
        for path in WORKFLOW_DOCS:
            text = path.read_text(encoding="utf-8")
            source = re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", text, flags=re.M | re.S)
            actual = re.findall(r"^(#{1,6}) (.+)$", source, re.M)
            titles = WORKFLOW_HEADINGS[path.name].split("|")
            levels = [1, 2, 2, 2, 2, 3, 3, 2, 3, 2, 2, 2]
            self.assertEqual([(len(level), title) for level, title in actual], list(zip(levels, titles)))
            suffix = "_zh" if path.stem.endswith("_zh") else ""
            preserved = {"WORKFLOWS.md" if suffix else "WORKFLOWS_zh.md",
                         f"POSTMAN{suffix}.md", f"OPENAPI{suffix}.md", f"SECURITY{suffix}.md",
                         f"../../docs/moss_transcribe_diarize{suffix}.md"}
            links = set(re.findall(r"\[[^\]]+\]\(([^)]+)\)", text))
            self.assertLessEqual(preserved, links)
            for link in links:
                parts = urlsplit(link)
                if parts.scheme or parts.netloc:
                    continue
                target = (path.parent / unquote(parts.path)).resolve()
                self.assertIn(ROOT.resolve(), target.parents)
                self.assertNotIn(".mcp-tasks", target.parts)
                self.assertTrue(target.is_file(), link)
                if parts.fragment:
                    self.assertIn(unquote(parts.fragment), headings(target.read_text(encoding="utf-8")), link)

    def test_workflow_preflight_uses_prepared_loopback_server(self):
        options = {ast.literal_eval(arg) for node in ast.walk(tree(EXAMPLE / "server.py"))
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "add_argument" for arg in node.args
                   if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        for path in WORKFLOW_DOCS:
            text = path.read_text(encoding="utf-8")
            prefix = text.split("```bash", 1)[0]
            suffix = "_zh" if path.stem.endswith("_zh") else ""
            self.assertIn(f"](README{suffix}.md", prefix)
            self.assertIn(f"](SECURITY{suffix}.md)", prefix)
            self.assertIn(f"](../../docs/agent_integration{suffix}.md)", text)
            startup = blocks(text, "bash")[0]
            commands = [shlex.split(line, comments=True) for line in startup.splitlines() if line.strip()]
            self.assertIn(["source", "../../.venv/bin/activate"], commands)
            launches = [args for args in commands if args[:2] == ["python", "server.py"]]
            self.assertEqual(len(launches), 1)
            args = launches[0]
            for option, expected in (("--host", "127.0.0.1"), ("--model", "sensevoice"), ("--device", "cpu")):
                self.assertIn(option, args)
                self.assertEqual(args[args.index(option) + 1], expected)
            self.assertLessEqual({arg for arg in args if arg.startswith("--")}, options)

    def test_workflow_url_assignments_are_copyable_and_checks_are_explicit(self):
        for path in WORKFLOW_DOCS:
            codes = blocks(path.read_text(encoding="utf-8"), "bash")
            assignments = []
            for code in codes:
                self.assertNotRegex(code, r"<[^\n>]+>")
                result = subprocess.run(["bash", "-n"], input=code, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for line in code.replace("\\\n", " ").splitlines():
                    args = shlex.split(line, comments=True)
                    if args[:1] == ["export"]:
                        assignments.extend(arg for arg in args[1:] if arg.startswith("FUNASR_BASE_URL="))
            self.assertEqual(assignments, ["FUNASR_BASE_URL=http://127.0.0.1:8000"])
            for endpoint in ("health", "v1/models", "openapi.json"):
                self.assertIn(f'"$FUNASR_BASE_URL/{endpoint}"', "\n".join(codes))

    def test_workflow_json_examples_match_actual_service_response_builders(self):
        handler = function(EXAMPLE / "server.py", "transcribe")
        response = next(node for node in ast.walk(handler) if isinstance(node, ast.Dict)
                        and any(isinstance(key, ast.Constant) and key.value == "duration" for key in node.keys))
        example = eval(compile(ast.Expression(response), "example-response", "eval"),
                       {"text": "recognized speech", "segments": [], "language": None,
                        "elapsed": 0.42, "model": "sensevoice"})
        namespace = {}
        functions = [function(PACKAGED, name) for name in
                     ("_split_text_for_openai_segments", "build_openai_fallback_segments",
                      "resolve_transcription_language", "build_openai_verbose_json")]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(PACKAGED), "exec"), namespace)
        result = {"text": "recognized speech", "duration": 3.2, "language": "en",
                  "segments": namespace["build_openai_fallback_segments"]("recognized speech", 3.2)}
        packaged = namespace["build_openai_verbose_json"](result)
        for path in WORKFLOW_DOCS:
            text = path.read_text(encoding="utf-8")
            samples = [json.loads(code) for code in blocks(text, "json")]
            self.assertEqual(samples, [example, packaged], path.name)
            self.assertIn("](CLIENTS.md#response-formats)", text)
            for marker in ("sentence_info", "segments=[]", "spk=true", "ctc_timestamps", "use_itn", "hotwords"):
                self.assertIn(marker, text)
        for speaker in (None, 0, "SPK0", "S01"):
            result["segments"][0]["speaker"] = speaker
            segment = namespace["build_openai_verbose_json"](result)["segments"][0]
            if speaker is None:
                self.assertNotIn("speaker", segment)
            else:
                self.assertEqual(segment["speaker"], speaker)

    def test_workflow_url_worker_warns_before_code_and_declares_dependency(self):
        for path in WORKFLOW_DOCS:
            text = path.read_text(encoding="utf-8")
            section = text.split("### Audio URL path" if path.name == "WORKFLOWS.md" else "### 音频 URL 转写", 1)[1]
            warning = section.split("```python", 1)[0]
            markers = ("not implemented", "redirect", "private-network", "byte", "trusted") if path.name == "WORKFLOWS.md" else (
                "未实现", "重定向", "私网", "字节", "可信")
            for marker in markers:
                self.assertIn(marker, warning)
            self.assertIn("python -m pip install requests", text[:text.index("```python")])

    def test_workflow_python_examples_are_equivalent_and_send_real_multipart_fields(self):
        snippets = [blocks(path.read_text(encoding="utf-8"), "python") for path in WORKFLOW_DOCS]
        self.assertEqual([len(items) for items in snippets], [2, 2])
        self.assertEqual([ast.dump(ast.parse(code)) for code in snippets[0]],
                         [ast.dump(ast.parse(code)) for code in snippets[1]])
        allowed = set(form_defaults(function(EXAMPLE / "server.py", "transcribe"))) | {"file"}
        for code, name in zip(snippets[0], ("transcribe_from_url", "transcribe_bytes")):
            calls = []
            handles = []
            payload = b"test audio bytes, no acoustic inference"

            def post(url, *, files, data, timeout):
                self.assertEqual(urlsplit(url).path, "/v1/audio/transcriptions")
                self.assertEqual(set(files), {"file"})
                self.assertLessEqual(set(data) | set(files), allowed)
                self.assertEqual(data, {"model": "sensevoice", "response_format": "verbose_json"})
                self.assertGreater(timeout, 0)
                body = files["file"][1]
                if hasattr(body, "read"):
                    handles.append(body)
                    body = body.read()
                self.assertEqual(body, payload)
                calls.append(url)
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"text": "hello", "segments": []})

            downloads = []

            def get(url, *, timeout):
                downloads.append(url)
                self.assertGreater(timeout, 0)
                return SimpleNamespace(content=payload, raise_for_status=lambda: None)

            parsed = ast.parse(code)
            # Substitute only the HTTP boundary; execute the documented worker body unchanged.
            parsed.body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
            namespace = {"Path": Path, "tempfile": tempfile, "requests": SimpleNamespace(get=get, post=post)}
            exec(compile(parsed, name, "exec"), namespace)
            args = ("https://storage.example.invalid/approved.wav",) if name == "transcribe_from_url" else ("audio.wav", payload)
            self.assertEqual(namespace[name](*args), {"text": "hello", "segments": []})
            self.assertEqual(len(calls), 1)
            self.assertEqual(downloads, list(args) if name == "transcribe_from_url" else [])
            self.assertTrue(all(handle.closed for handle in handles))


if __name__ == "__main__":
    unittest.main()
