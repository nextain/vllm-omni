# MiniCPM-o 4.5 수정 내용 설명

## 배경: 원래 뭐가 문제였나?

MiniCPM-o 4.5는 텍스트를 읽으면 음성으로 바꿔주는 AI 모델입니다. 3단계로 동작합니다:

1. **Thinker** (생각하기): 사용자 입력을 이해하고 텍스트 응답 생성
2. **Talker** (말하기): 텍스트를 음성 토큰으로 변환
3. **Code2Wav** (소리 만들기): 음성 토큰을 실제 WAV 오디오로 변환

문제는 **2단계 Talker**에서 원본 모델과 완전히 다른 방식으로 구현되어 있었다는 것입니다.

---

## 수정 1: emb_text 가중치 누락 (A1)

### 뭐가 잘못됐나?
원본 모델에는 `emb_text`라는 15만 단어 × 768차원 테이블이 있습니다. "안녕하세요"라는 단어를 768차원 벡터로 변환하는 사전입니다. Talker가 음성을 만들 때 이 사전을 참고합니다.

우리 코드는 이 사전을 **아예 로드하지 않았습니다** (`skip_prefixes=["emb_text."]`로 무시).

### 어떻게 고쳤나?
`skip_prefixes`에서 `emb_text`를 제거하고, WeightsMapper에 `"tts.emb_text." → "emb_text."` 매핑을 추가했습니다. 이제 HF 체크포인트에서 `tts.emb_text.weight` (152064×768)가 정상 로드됩니다.

---

## 수정 2: Projector 입력이 바뀌어 들어감 (A2)

### 뭐가 잘못됐나?
원본:
- `projector_semantic` = LLM이 **생각한 결과** (hidden states)를 768차원으로 변환
- `emb_text` = **단어 자체**를 768차원으로 변환

우리 코드:
- `projector_semantic`에 **단어 임베딩** (input embeddings)을 넣음 ← 잘못!
- `projector_spk`에 **hidden states**를 넣음 ← 이것도 잘못!

비유하면: 번역가한테 프랑스어 텍스트를 줘야 하는데 스페인어를 줬고, 통역사한테는 녹음 파일을 줘야 하는데 자막을 준 격입니다.

### 어떻게 고쳤나?
- `semantic_projection`에 hidden states를 넣도록 수정
- `emb_text`에 token IDs를 넣도록 수정
- `projector_spk`는 원본에서 추론 시 사용하지 않으므로, 가중치만 로드하고 호출하지 않음

---

## 수정 3: 활성함수 ReLU vs SiLU (A7, R4)

### 뭐가 잘못됐나?
원본 `MultiModalProjector`:
```python
self.relu = nn.ReLU()  # 음수는 0, 양수는 그대로
```

우리 코드:
```python
self.act_fn = nn.SiLU()  # 부드러운 곡선, 음수도 일부 통과
```

가중치는 ReLU로 학습됐는데 SiLU로 추론하면 다른 결과가 나옵니다. Talker projector뿐 아니라 **Audio projector** (Whisper 출력 변환)도 같은 문제가 있었습니다.

### 어떻게 고쳤나?
`MiniCPMOTalkerResizeMLP`와 `MiniCPMOAudioProjectionMLP` 모두 `nn.SiLU()` → `nn.ReLU()`로 변경.

---

## 수정 4: L2 정규화 누락 (A3)

### 뭐가 잘못됐나?
원본:
```python
hidden_embeds = projector_semantic(hidden_states)
if config.normalize_projected_hidden:  # True
    hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)  # 단위 벡터로 정규화
```

우리 코드: 정규화 코드 자체가 없었음.

HF config에서 `normalize_projected_hidden=True`인데 코드에서 무시했습니다. 벡터 크기가 제각각이 되어 음성 품질이 떨어집니다.

### 어떻게 고쳤나?
`build_conditioning()` 메서드에 `if self.config.normalize_projected_hidden: F.normalize(...)` 추가.

---

## 수정 5: 경계 토큰 누락 (A4)

### 뭐가 잘못됐나?
원본 Talker 입력:
```
[conditioning, text_eos_embed, audio_bos_embed] → AR 디코딩 시작
```

우리 코드: conditioning만 넣고 `text_eos`와 `audio_bos` 경계 토큰을 안 넣었음.

이 경계 토큰은 Talker에게 "텍스트 끝났어, 이제 오디오 시작해"라고 알려주는 신호입니다.

### 어떻게 고쳤나?
`talker_preprocess`에서 conditioning 뒤에 `emb_text([TEXT_EOS_TOKEN_ID, AUDIO_BOS_TOKEN_ID])`를 concatenate.

---

## 수정 6: EOS 토큰 잘못됨 (A9)

### 뭐가 잘못됐나?
원본: Talker가 오디오 토큰 `6561`을 생성하면 멈춤 (= `num_audio_tokens - 1 = 6562 - 1`)
우리 코드:
- `pipeline.yaml`: stop_token_ids 없음 → 무한 생성
- `minicpm_o_ci.yaml`: `6563` → vocab 범위 밖이라 절대 생성 안 됨 → 역시 무한 생성

### 어떻게 고쳤나?
`pipeline.yaml`과 CI config 모두 `stop_token_ids: [6561]`로 수정.

---

## 수정 7: tts_bound 필터링 (A10)

### 뭐가 잘못됐나?
원본: LLM이 생성한 전체 시퀀스에서 `<|tts_bos|>` ~ `<|tts_eos|>` 사이만 잘라서 Talker에 전달.
우리 코드: 시스템 프롬프트, 사용자 메시지 포함 **전체 시퀀스**를 Talker에 전달.

### 어떻게 고쳤나?
`_find_tts_bound()` 함수 추가. `<|tts_bos|>` (151703)과 `<|tts_eos|>` (151704) 위치를 찾아 해당 구간만 슬라이싱.

---

## 수정 8: BOS skip 제거 (A11)

### 뭐가 잘못됐나?
`talker2code2wav`에서 `output.token_ids[1:]`로 첫 토큰을 건너뛰었음. 원본에서 `audio_bos`는 conditioning의 마지막에 들어가고, 생성된 토큰에는 BOS가 없음. 유효한 첫 번째 codec 토큰을 잘라먹고 있었음.

### 어떻게 고쳤나?
`list(output.token_ids)` — 전체 사용, skip 없음.

---

## 수정 9: Config 기본값 8개 (A5)

HF `config.json`과 비교해서 틀린 기본값 전부 수정:

**MiniCPMTTSConfig (Talker):**

| 파라미터 | 잘못된 값 | 올바른 값 |
|----------|----------|----------|
| `normalize_projected_hidden` | `False` | `True` |
| `audio_tokenizer_type` | `"wavtokenizer"` | `"s3tokenizer"` |
| `audio_tokenizer_sample_rate` | `24000` | `16000` |
| `attention_type` | `"sliding_recompute"` | `"full_attention"` |

**MiniCPMOConfig (전체 모델):**

| 파라미터 | 잘못된 값 | 올바른 값 |
|----------|----------|----------|
| `drop_vision_last_layer` | `True` | `False` |
| `max_position_embeddings` | `32768` | `40960` |
| `stream_input` | `False` | `True` |

**MiniCPMVSliceConfig (비전):**

| 파라미터 | 잘못된 값 | 올바른 값 |
|----------|----------|----------|
| `max_slice_nums` | `9` | `1` |

---

## 수정 10: 토큰 ID (A6)

placeholder로 넣었던 잘못된 토큰 ID를 `tokenizer_config.json`에서 확인한 정확한 값으로 교체:

| 토큰 | 잘못된 ID | 올바른 ID |
|------|----------|----------|
| `<\|tts_bos\|>` | 151859 | 151703 |
| `<\|tts_eos\|>` | 151860 | 151704 |
| `audio_bos` | — | 151687 |
| `text_eos` | — | 151692 |

---

## 수정 11: 텐서 크기 불일치 (28 vs 27)

### 뭐가 잘못됐나?
RunPod E2E에서 `RuntimeError: The size of tensor a (28) must match the size of tensor b (27)` 크래시.

`token_ids` (Python list)와 `hidden_states` (PyTorch tensor)의 길이가 1 차이. EOS 토큰이 `token_ids`에는 포함되지만 `hidden_states`에는 포함 안 되는 경우.

### 어떻게 고쳤나?
`hidden_states.shape[0]`을 기준 길이로 사용하고, `token_ids`를 거기에 맞춰 trim. bounds clamping 추가.

---

## 수정 12: 디버그 로그 + 코드 정리

- `[DBG load]`, `[DBG extract]`, `[DBG fwd]`, `[DBG mm]`, `[DBG gpu_ar]`, `[DBG thinker2talker]` — 7곳 제거
- CosyVoice3 → CosyVoice2 참조 4곳 수정
- "Phase 5", "Phase 6" 프로젝트 내부 용어 제거
- 깨진 테스트 4개 수정 (linear_fc1→linear1, cosyvoice_config 제거, flow_model→_flow)
