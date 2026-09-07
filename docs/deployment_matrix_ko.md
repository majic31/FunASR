# FunASR 배포 선택 매트릭스

제품, 데모, 벤치마크, 내부 워크플로에 맞는 가장 짧은 배포 경로를 고르기 위한 가이드입니다. 먼저 요구를 만족하는 최소 구성에서 시작하고, throughput, latency, integration 요구가 명확해질 때 더 무거운 runtime으로 이동하세요.

## 빠른 결정 표

| Path | 적합한 용도 | 시작 문서 | 운영 메모 |
|---|---|---|---|
| Colab notebook | 브라우저 smoke test, 첫 평가, 공유 가능한 demo | [Colab 빠른 시작](../examples/colab/README_ko.md) | 로컬 환경이 필요 없습니다. 첫 실행은 모델을 다운로드하며 GPU runtime이 더 빠릅니다. |
| Python API | Notebook, offline job, 첫 model evaluation | [README quick start](../README_ko.md#빠른-시작) | 가장 단순한 경로입니다. batching, retry, file 관리는 호출 측에서 담당합니다. |
| OpenAI 호환 API | Private speech API, Agent, Dify/LangChain/AutoGen style clients | [OpenAI API example](../examples/openai_api/README_ko.md) | OpenAI audio API를 이미 지원하는 앱에 가장 쉽게 연결됩니다. |
| Docker Compose API | 재현 가능한 local smoke test, 작은 internal service | [OpenAI API Docker docs](../examples/openai_api/README_ko.md#docker-배포) | 기본은 CPU입니다. CUDA를 쓰기 전에 CUDA-capable image로 조정하세요. |
| Kubernetes API | Cluster service용 internal speech API | [Kubernetes template](../examples/openai_api/kubernetes/) | private `ClusterIP`부터 시작합니다. 범위를 넓히기 전에 auth, TLS, network policy, GPU scheduling을 추가하세요. |
| Runtime WebSocket service | Live captions, meeting, call-center stream | [Runtime service docs](../runtime/readme.md) | partial result, endpointing, long-lived audio stream이 중요할 때 사용합니다. |
| vLLM acceleration | Fun-ASR-Nano의 native 파일 전사 또는 split-engine 디코딩 | [공식 native 검증(영문)](./vllm_official_native_validation.md), [split-engine guide](./vllm_guide.md) | 두 경로는 checkpoint와 API가 다릅니다. 비자기회귀 Paraformer에는 적용되지 않습니다. |
| MOSS-Transcribe-Diarize | 긴 다중 화자 transcription, timestamp, speaker label | [Third-party MOSS guide](./moss_transcribe_diarize.md) | OpenMOSS의 Apache-2.0 model이며 FunASR `AutoModel`에 통합되어 있습니다. local HF(`backend="hf"`), vLLM(`backend="vllm"`) 또는 SGLang Omni(`backend="sglang"`)를 선택할 수 있습니다. model의 공개 및 유지 관리는 계속 OpenMOSS가 담당합니다. |
| MCP server | Claude/Cursor/desktop agent speech tool | [MCP example](../examples/mcp_server/) | ASR 결과를 local tool로 Agent에 전달할 때 유용합니다. |
| Subtitle generator | 긴 audio/video에서 SRT/VTT 생성 | [Subtitle example](../examples/subtitle/) | readability가 중요하면 verbose segment와 speaker label을 사용합니다. |
| Batch ASR script | Archive, meeting, dataset, 반복 offline run | [Batch example](../examples/batch_asr_improved.py) | production에서는 queue, manifest, retry log를 추가하세요. |

## 자주 쓰는 선택

### Fun-ASR-Nano를 vLLM으로 실행하고 싶다

- **Split-engine**: 기본 모델 `FunAudioLLM/Fun-ASR-Nano-2512`를 사용하며, FunASR가 오디오를 처리하고 LLM decoder를 vLLM에서 실행합니다. [split-engine guide](./vllm_guide.md)를 참고하세요.
- **Native**: vLLM 자체가 오디오 모델을 실행합니다. 공식 checkpoint는 `FunAudioLLM/Fun-ASR-Nano-2512-vllm`입니다. 기본 모델용 설정이나 checkpoint와 서로 바꾸어 사용하지 마세요.

[공식 native 검증 기록(영문)](./vllm_official_native_validation.md)은 모델 revision
`a4362c943d48951f98ca2a62181cc028970270c5`, vLLM 0.27.1, 기존 H100 환경에서의 기능 확인입니다.
모델 revision은 FunASR 패키지 버전이 아닙니다. 의존성을 포함한 준비 및 실행 절차는 검증 기록을 참고하세요.

확인한 API는 파일 입력의 `/v1/audio/transcriptions`입니다. `/v1/realtime`, 클린 설치, 긴 오디오, 화자 분리, 프로덕션 처리 용량은 검증 대상이 아닙니다. 이를 FunASR SDK나 다른 checkpoint의 검증으로 간주하지 말고, 실제 운영 오디오와 부하로 별도 평가하세요.

### 5분 안에 FunASR을 시험하고 싶다

브라우저만으로 확인하려면 [Colab 빠른 시작](../examples/colab/README_ko.md)을 사용하세요. 로컬에서 작업하려면 README의 Python API부터 시작합니다. 어떤 모델을 고를지 고민된다면 [모델 선택 가이드](./model_selection_ko.md)를 참고하세요.

### Cloud transcription의 local replacement가 필요하다

OpenAI 호환 API를 사용하세요. 주요 진입점은 다음과 같습니다.

- `/v1/audio/transcriptions`: 파일 전사
- `/v1/models`: 모델 목록
- `/health`: 상태 확인
- Swagger docs: API 확인

먼저 `sensevoice`로 smoke test를 실행하고 기존 SDK나 HTTP client를 [OpenAI API example](../examples/openai_api/README_ko.md)에 맞춰 연결하세요.

### 재현 가능한 container demo가 필요하다

`examples/openai_api/docker-compose.yml`을 CPU mode smoke test로 사용합니다.

```bash
cd examples/openai_api
cp .env.example .env
docker compose up --build
```

CUDA를 사용하려면 CUDA-capable PyTorch/FunASR image를 만든 뒤 `FUNASR_DEVICE=cuda`로 바꾸고 같은 smoke test로 확인하세요.

### Streaming 또는 live captioning이 필요하다

Runtime WebSocket service를 사용하세요. production 전에 chunk size, VAD, endpointing, punctuation, speaker diarization, reconnect, client backpressure를 실제 오디오로 검증하세요.

## Readiness checklist

- model alias를 정하고 deployment note에 고정합니다.
- FunASR version, model version, device, CUDA/PyTorch version, Docker image tag, command line을 기록합니다.
- public smoke sample과 realistic private sample을 최소 1개씩 실행합니다.
- request마다 audio duration, model, device, latency, response format, error type을 로깅합니다.
- trusted network 밖으로 API를 노출하기 전에 upload-size limit, authentication, TLS, rate limit을 넣습니다. [Security guide](../examples/openai_api/SECURITY.md)도 확인하세요.
- 막히면 deployment path, command/config, logs, model, device, audio characteristics를 포함해 [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md)를 열어 주세요.
