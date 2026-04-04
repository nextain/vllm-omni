# MiniCPM-o 4.5

**Language / 언어**: [English](README.md) | 한국어

## 설정

하드웨어에 맞는 메모리 할당 설정은 [stage configuration 문서](https://vllm-omni.readthedocs.io/en/latest/configuration/stage_configs/)를 참조하세요.

### 필수 패키지

```bash
pip install soundfile librosa
sudo apt-get install ffmpeg  # librosa MP3 백엔드 필요
```

## 예제 실행 (MiniCPM-o 4.5)

### 서버 시작

```bash
# 2× RTX 3090 — async_chunk 스트리밍 (권장, TTFP ~0.07s)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000

# 2× RTX 3090 — 동기 모드
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000

# 단일 24 GB GPU
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000
```

> RTX 3090 (NVLink 없음) 환경에서는 `NCCL_P2P_DISABLE=1`이 필수입니다.

Stage config 옵션: [`minicpmo.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo.yaml) · [`minicpmo_48gb_2gpu.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml) · [`minicpmo_async_chunk.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml)

### 요청 전송

예제 폴더로 이동:
```bash
cd examples/online_serving/minicpm_o/
```

#### 텍스트 대화

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "안녕하세요, 잘 지내세요?"}]
  }'
```

#### 오디오 출력 (텍스트 → 음성)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "vLLM을 한 문장으로 설명해줘."}],
    "modalities": ["audio"]
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
audio = data['choices'][0]['message']['audio']['data']
open('output.wav', 'wb').write(base64.b64decode(audio))
print('output.wav 저장 완료')
"
```

#### 이미지 + 텍스트 → 오디오

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"}},
        {"type": "text", "text": "이 이미지에 무엇이 있나요? 한 문장으로 답해주세요."}
      ]
    }],
    "modalities": ["audio"]
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
audio = data['choices'][0]['message']['audio']['data']
open('output_image.wav', 'wb').write(base64.b64decode(audio))
print('output_image.wav 저장 완료')
"
```

#### 오디오 입력 → 오디오 출력

```bash
# 로컬 오디오 파일을 base64로 인코딩
AUDIO_B64=$(base64 -w 0 /path/to/audio.wav)

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"openbmb/MiniCPM-o-4_5\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"input_audio\", \"input_audio\": {\"data\": \"${AUDIO_B64}\", \"format\": \"wav\"}},
        {\"type\": \"text\", \"text\": \"이 오디오에서 무슨 말을 하고 있나요?\"}
      ]
    }],
    \"modalities\": [\"audio\"]
  }" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
audio = data['choices'][0]['message']['audio']['data']
open('output_audio.wav', 'wb').write(base64.b64decode(audio))
print('output_audio.wav 저장 완료')
"
```

## Modality 제어

생성할 출력 modality를 지정할 수 있습니다.

### 지원 modality

| Modalities | 출력 |
|------------|------|
| `["text"]` | 텍스트만 (오디오 생성 건너뜀 — 더 빠름) |
| `["audio"]` | 오디오만 |
| `["text", "audio"]` | 텍스트 + 오디오 |
| 미지정 | 텍스트 + 오디오 (기본값) |

### 텍스트 전용 (Talker + Code2Wav 스테이지 건너뜀)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "vLLM을 간단히 설명해줘."}],
    "modalities": ["text"]
  }'
```

### OpenAI Python SDK 사용

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# 텍스트 전용
response = client.chat.completions.create(
    model="openbmb/MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "안녕하세요!"}],
    modalities=["text"],
)
print(response.choices[0].message.content)

# 오디오 출력
response = client.chat.completions.create(
    model="openbmb/MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "한 문장으로 인사해줘."}],
    modalities=["audio"],
)
audio_data = base64.b64decode(response.choices[0].message.audio.data)
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

## 스트리밍 출력

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "짧은 이야기를 들려줘."}],
    "modalities": ["text"],
    "stream": true
  }'
```

오디오 스트리밍은 `async_chunk` stage config 사용 (TTFP ~0.07s):
- [`minicpmo_async_chunk.yaml`](../../../vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml)

## 평가 스크립트

| 스크립트 | 목적 |
|--------|------|
| [`conversation_benchmark.py`](conversation_benchmark.py) | 멀티턴 영어 대화 벤치마크 (6 시나리오) |
| [`language_test.py`](language_test.py) | EN / ZH / KO 비교 (CER + 의미 유사도) |
| [`voicebench_runner.py`](voicebench_runner.py) | VoiceBench (지식 / 지시 / 견고성 / 안전) |
| [`e2e_conversation_test.py`](e2e_conversation_test.py) | Speaker / Monitor 핵심 프레임워크 |
| [`metrics/cjk_metrics.py`](metrics/cjk_metrics.py) | CER, 의미 유사도 (sentence-transformers) |
| [`metrics/conversation_quality.py`](metrics/conversation_quality.py) | 관련성, 일관성, 지식 유지 |

```bash
# 대화 벤치마크 실행
python conversation_benchmark.py --omni

# VoiceBench 실행
python voicebench_runner.py

# 언어 테스트 (EN / ZH / KO)
python language_test.py
```

전체 벤치마크 결과: [BENCHMARK.ko.md](BENCHMARK.ko.md)

## FAQ

**librosa 백엔드 오류:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**RTX 3090에서 NCCL P2P 오류:**
```bash
export NCCL_P2P_DISABLE=1
```

**한국어 TTS가 왜곡된 오디오를 출력:**  
Code2Wav 백본인 CosyVoice2가 한국어로 학습되지 않았습니다. 텍스트 생성은 정상이나 음성 합성은 파인튜닝이 필요합니다. [BENCHMARK.ko.md § 한국어](BENCHMARK.ko.md) 참조.

**오디오가 항상 ~20초:**  
Talker 스테이지가 현재 학습 분포에서 stop token 6561을 거의 생성하지 않습니다. `_trim_silence()` 후처리로 무음 구간을 제거합니다. upstream PR과 별도 이슈로 추적 중입니다.
