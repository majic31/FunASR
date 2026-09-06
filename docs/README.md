# FunASR Documentation

[中文文档](https://www.funasr.com/docs/) |
[English documentation](https://www.funasr.com/en/docs/) |
[Model Zoo](../model_zoo/modelscope_models.md) |
[Releases](https://github.com/modelscope/FunASR/releases)

## Start Here

| Your task | Guide |
| --- | --- |
| Choose a model, language, or timestamp capability | [Model selection](model_selection.md) / [中文](model_selection_zh.md) |
| Run the first transcription | [Quickstart](https://www.funasr.com/en/quickstart.html) / [CLI](cli.md) |
| Move from Whisper or a cloud API | [Migration](migration_from_whisper.md) / [中文](migration_from_whisper_zh.md) |

## Models and Deployment

| Your task | Guide |
| --- | --- |
| Compare CPU, GPU, realtime and API runtimes | [Deployment matrix](deployment_matrix.md) / [中文](deployment_matrix_zh.md) |
| Transcribe and diarize with third-party MOSS | [MOSS](moss_transcribe_diarize.md) / [中文](moss_transcribe_diarize_zh.md) |
| Accelerate with the FunASR vLLM split engine | [vLLM](vllm_guide.md) / [中文](vllm_guide_zh.md) |
| Evaluate native vLLM serving | [Validation record](vllm_native_funasr_validation.md) |
| Deploy llama.cpp, TensorRT, Docker or Kubernetes | [Deployment manuals](https://www.funasr.com/en/deploy/) |

## Evaluate and Operate

- [Performance methodology](benchmark/rtf_reproducibility.md) and [realtime WebSocket benchmarks](benchmark/realtime_ws_benchmark.md)
- [Troubleshooting](troubleshooting.md) / [中文](troubleshooting_zh.md)
- [Use cases](use_case_showcase.md), [community integrations](community_projects.md), and [repository responsibilities](repository_roles.md)

The product-site documentation is rendered from these Markdown sources. Edit the
source once; the site build updates article content, navigation and local search.
The catalogue lives in [documentation.json](../web-pages/product-site/data/documentation.json).
Model weights retain their individual licenses; the FunASR toolkit is MIT licensed.

## Build the Product Documentation

From the repository root:

```sh
python -m pip install -r web-pages/product-site/requirements-site.txt
python web-pages/product-site/build.py --output /tmp/funasr-docs
python web-pages/product-site/validate.py /tmp/funasr-docs
python -m http.server --directory /tmp/funasr-docs 8000
```

## Build the Sphinx Reference

For convenience, we provide users with the ability to generate local HTML manually.

First, you should install the following packages, which is required for building HTML:

```sh
pip3 install -U "funasr[docs]"
```

Then you can generate HTML manually.

```sh
cd docs
make html
```

The generated files are all contained in the "FunASR/docs/_build" directory. You can access the FunASR documentation by simply opening the "html/index.html" file in your browser from this directory.
