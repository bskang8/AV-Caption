# caption_refine 파이프라인 상세 설명

## 목적

1차 AI가 생성한 주행 영상 캡션에는 두 가지 구조적 문제가 있습니다.

1. **Hallucination** — 영상에 없는 객체·이벤트·표지판 등이 캡션에 서술됨
2. **정보 누락** — 날씨·차선 수·신호 상태·보행자 행동 등 ODD 분석에 필요한 세부 정보가 빠져 있음

`caption_refine`은 Cosmos-Reason2 비전-언어 모델을 통해 영상을 직접 보면서 캡션을 **검증 → 구조화 추출 → 재검증 → 정제**하는 **4-Stage 파이프라인**입니다.

---

## 파일 구조

```
caption_refine/
├── config.py             전역 설정 (경로, 모델, 토큰 예산, 프레임 수 등)
├── prompts.py            4개 Stage 프롬프트 템플릿
├── cosmos_client.py      vLLM API 클라이언트 (영상 → 멀티모달 content 변환)
├── pipeline.py           단일 클립 4-Stage 오케스트레이션
├── batch_runner.py       배치 처리 + 진행 추적 (progress.json)
└── stages/
    ├── stage1_ground.py  Stage 1: 기존 캡션 검증 (hallucination 탐지)
    ├── stage2_extract.py Stage 2: ODD 정보 구조화 추출
    ├── stage3_verify.py  Stage 3: 저확신 필드 재검증
    └── stage4_refine.py  Stage 4: 정제 캡션 생성
```

출력 디렉터리 (기본값: `$CR_OUTPUT_ROOT`):
```
caption_v2/
├── captions/{clip_id}.camera_front_wide_120fov.txt   정제된 캡션
├── odd/{clip_id}.json                                구조화 ODD 정보
├── diff/{clip_id}_diff.json                          변경 내역 (hallucination 목록)
└── progress.json                                     배치 처리 진행 상태
```

---

## 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│ batch_runner.py                                                  │
│                                                                  │
│  clip_id 목록 로드 (gap / longtail / all / ids-file)             │
│  progress.json 읽기 → 완료/에러 클립 제외                         │
│  asyncio.Semaphore(N)로 동시 처리 수 제한                         │
│                                                                  │
│  ┌─── process_clip(clip_id) ─────────────────────────────────┐  │
│  │                                                            │  │
│  │  MP4 + 원본 캡션(.txt) 읽기                                │  │
│  │         │                                                  │  │
│  │         ▼                                                  │  │
│  │  ┌── Stage 1: 캡션 검증 ──────────────────────────────┐   │  │
│  │  │  16 frames (force_frames) × 3회 병렬 (temp=0.3)    │   │  │
│  │  │  → hallucinated[], missed[], grounded[]             │   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  │         │                                                  │  │
│  │         ▼                                                  │  │
│  │  ┌── Stage 2: ODD 추출 ──────────────────────────────┐    │  │
│  │  │  4 frames (video 모드에서 token 절약)              │    │  │
│  │  │  → 15개 ODD 필드 + confidence + evidence           │    │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  │         │                                                  │  │
│  │         ▼                                                  │  │
│  │  ┌── Stage 3: 저확신 재검증 ─────────────────────────┐    │  │
│  │  │  confidence < 0.7 필드만 선별해 재확인             │    │  │
│  │  │  → verified_odd (확정된 ODD dict)                  │    │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  │         │                                                  │  │
│  │         ▼                                                  │  │
│  │  ┌── Stage 4: 캡션 정제 ─────────────────────────────┐    │  │
│  │  │  6 frames + 원본캡션 + hal/miss + verified_odd     │    │  │
│  │  │  → 150~300 단어 정제 캡션 (텍스트)                 │    │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  │         │                                                  │  │
│  │         ▼                                                  │  │
│  │  captions/ odd/ diff/ 저장                                 │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  10개 처리마다 progress.json 저장 → 중단 후 재시작 가능           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 영상 입력 처리 (cosmos_client.py)

Cosmos-Reason2는 비전-언어 모델입니다. HTTP API로는 바이너리를 직접 전송할 수 없으므로 두 가지 입력 방식을 지원합니다.

### frames 모드 (Stage 1·2·3 기본)

```
MP4 파일 (예: 20초, 30fps = 600 프레임)
│
▼  cv2.VideoCapture
총 프레임 수 파악
│
▼  np.linspace(0, 599, N) → N개 인덱스
   영상 전체에서 균등 간격으로 N개 프레임 선택
   (Stage별 N이 다름 — 아래 "프레임 수 결정" 참고)
│
▼  MAX_FRAME_W=854, MAX_FRAME_H=480 이하로 리사이즈
│
▼  JPEG 품질 85로 압축 후 base64 인코딩
│
▼  OpenAI API content 배열로 조립
[
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}},
  ...  (N개)
  {"type": "text", "text": "<프롬프트>"}
]
│
▼  vLLM POST → Cosmos-Reason2가 이미지 시퀀스를 시간축 영상으로 해석
```

### video 모드 (Stage 4, CR_VIDEO_MODE=video 시)

```bash
export CR_VIDEO_MODE=video
export CR_VIDEO_FPS=1      # 20초 × 1fps = 20프레임 ≈ 28,540 토큰
```

```
MP4 파일 전체를 file:// URL로 전달
→ vLLM 서버가 --allowed-local-media-path 경로에서 직접 읽음
→ client payload 최소화, 시간 흐름 정보 보존

※ Stage 1은 force_frames=True로 항상 frames 모드 사용
   (video 모드에서 hallucination 탐지 보수적으로 동작하는 문제 회피)
```

### 프레임 수 결정 (Stage별)

| Stage | 프레임 수 | 이유 |
|-------|----------|------|
| Stage 1 | **16** (`NUM_FRAMES_STAGE1`) | hallucination 탐지는 넓은 커버리지 필요 |
| Stage 2 | **4** (`NUM_FRAMES // 3`) | ODD JSON이 길어서 토큰 초과 방지 |
| Stage 3 | **12** (`NUM_FRAMES`) | 기본값, 재검증 대상 필드 소수 |
| Stage 4 | **6** (`NUM_FRAMES // 2`) | 프롬프트에 ODD+캡션 텍스트가 많아 토큰 절약 |

**토큰 계산 근거 (8B 모델, max_model_len=32768):**  
854×480 → ~430 tokens/frame  
16 frames × 430 = ~6,880 visual tokens → 출력 2,048 토큰 확보 가능  
4 frames × 430 = ~1,720 visual tokens → 출력 4,096 토큰 + Qwen3 thinking 여유

---

## Stage 1 — 기존 캡션 검증 (stage1_ground.py)

### 목적

원본 캡션에서 영상에 없는 내용(hallucination)과 누락된 내용(missed)을 탐지합니다.

### 비결정성 문제와 3-pass 해결책

bitsandbytes INT8 양자화 모델은 `temperature=0`이어도 GPU 커널 비결정성으로 인해 실행마다 다른 결과를 낼 수 있습니다. 단일 실행은 일부 hallucination을 탐지하지 못하는 경우가 있습니다.

**해결: temperature=0.3으로 3회 병렬 실행 → hallucinated 합집합(union) 취합**

```
Pass 1 (temp=0.3) → {sentences 4, 5, 8}  탐지
Pass 2 (temp=0.3) → {sentences 4, 5, 6, 8}  탐지
Pass 3 (temp=0.3) → {sentences 4, 5, 6, 8, 9, 10}  탐지
                       ↓
         Union = {4, 5, 6, 8, 9, 10}  → 6개 모두 탐지
```

3회 실행은 `asyncio.gather`로 동시 실행되므로 **벽시계 시간은 단일 실행과 동일**합니다.

### 프롬프트 전략

**기존 방식 (실패):** "캡션에서 직접 모순되는 내용을 찾아라" → 불확실하면 건너뜀  
**현재 방식 (채택):** "각 문장의 주장을 영상에서 확인하라 — 근거를 찾을 수 없으면 flag"

```
각 번호 문장을 영상 프레임과 비교해:
- 해당 문장이 주장하는 객체/행동/표지판이 어떤 프레임에서도 보이지 않으면 hallucinated로 분류
- 영상에는 있지만 캡션에 없는 것은 missed로 분류

반환 형식:
{
  "hallucinated": [
    {"sentence_num": 5, "reason": "white minibus는 어떤 프레임에도 없음"}
  ],
  "missed": [
    "16번 프레임에서 빨간 트럭이 반대편 차선에서 진행 중"
  ]
}
```

### 출력 — GroundingResult

```python
@dataclass
class GroundingResult:
    grounded:     list[str]   # 영상과 일치하는 문장
    hallucinated: list[str]   # 영상에 없는 내용 + [이유]
    missed:       list[str]   # 누락된 관찰 (최대 5개, 길이 내림차순)
```

### 필터링 로직

**hallucinated 필터:**
- `sentence_num`이 범위 밖이면 무시
- `reason`이 해당 문장 자체와 Jaccard 유사도 > 0.5이면 무시 (이유가 문장을 그대로 반복한 경우)

**missed 필터:**
- 기존 문장들과 Jaccard 유사도 > 0.35 또는 content-word 커버리지 > 0.55이면 제외 (이미 언급된 내용)
- 일반적 표현 필터 (`"road is clean"`, `"driver's actions"` 등 17개 stopphrase 제외)
- 중복 제거 후 최대 5개

---

## Stage 2 — ODD 정보 추출 (stage2_extract.py)

### 목적

영상에서 **15개 ODD 필드**를 구조화된 JSON으로 추출합니다. 각 필드에 `confidence(0~1)`와 `evidence(관찰 근거)`를 함께 요청합니다.

### 설계 원칙

- **캡션 미포함**: 원본 캡션을 프롬프트에 넣지 않음 → 캡션에 의한 편향 없이 영상 자체에서 추출
- **4 frames**: ODD JSON은 15+ 필드로 응답이 길어 토큰 초과 방지를 위해 이미지 토큰 최소화
- **max_tokens=4096**: Stage 2 응답 전용 토큰 예산 (Stage 1·3·4의 2,048보다 2배)

### 추출 필드 목록

| 필드 | 설명 | 예시 값 |
|------|------|---------|
| `time_of_day` | 시간대 | day / night / dawn / dusk |
| `weather` | 날씨 | clear / cloudy / rainy / foggy / snowy |
| `road_type` | 도로 유형 | highway / urban / intersection / rural / parking_lot |
| `num_lanes` | 차선 수 (정수) | 4 |
| `ego_lane_position` | 자차 차선 위치 | leftmost / second_from_right / rightmost |
| `road_surface` | 노면 상태 | dry / wet / icy / unpaved |
| `road_markings` | 노면 표시 목록 | ["lane_lines", "crosswalk", "bicycle_lane"] |
| `traffic_density` | 교통 밀도 | free / light / moderate / congested |
| `surrounding_vehicles` | 주변 차량 종류·수·행동 | types, count_estimate, notable_behaviors |
| `ego_actions` | 자차 행동 목록 | ["straight", "braking", "right_turn"] |
| `pedestrians` | 보행자 유무·수·행동 | present, count_estimate, behavior |
| `traffic_signals` | 신호등 상태 | present, state (red/green/yellow) |
| `road_signs` | 표지판 종류·내용 | types, details |
| `hazard_level` | 위험도 | low / medium / high + rationale |
| `lighting_condition` | 조명 조건 | daylight / artificial / mixed / dark |

### confidence 기반 분기

```python
CONFIDENCE_THRESHOLD = 0.7   # CR_CONF_THRESHOLD 환경변수로 조정

low_confidence = {
    필드명: 필드값
    for 필드명, 필드값 in 응답
    if 필드값["confidence"] < 0.7
}
# → Stage 3으로 전달
```

---

## Stage 3 — 저확신 재검증 (stage3_verify.py)

### 목적

Stage 2에서 모델이 확신하지 못한 필드만 **영상을 다시 보며 재확인**합니다.  
전체 필드 재검증 대신 선별함으로써 API 호출 비용을 절감합니다.

### 프롬프트 구조

```
영상을 다시 보고 아래 필드들만 집중적으로 확인하세요:
- weather: 현재 'clear' (confidence 0.55) — 근거: 렌즈에 빗방울 없음, 하늘 일부 흐림

각 필드에 대해:
1. 영상에서 관찰한 내용
2. CONFIRM (원래 답이 맞음) 또는 CORRECT (새 값 제시)

반환 JSON:
{
  "weather": {
    "observation": "노면 반사와 앞유리 물방울 확인",
    "verdict": "CORRECT",
    "corrected_value": "rainy"
  }
}
```

### 처리 로직

```
Stage 3 응답
│
├─ verdict = "CONFIRM" → Stage 2 값 유지, verified=True 표시
│
└─ verdict = "CORRECT"
      ├─ corrected_value로 기존 값 교체
      └─ confidence를 0.85로 상향 (재검증 완료 표시)
```

Stage 3 결과가 `verified_odd` — Stage 4로 전달되는 최종 ODD 딕셔너리입니다.

---

## Stage 4 — 캡션 정제 (stage4_refine.py)

### 목적

Stage 1·2·3의 결과를 통합해 **정확하고 정보가 풍부한 최종 캡션**을 생성합니다.

### 입력

| 정보 | 출처 |
|------|------|
| 원본 캡션 | `.txt` 파일 |
| 제거할 내용 (`hallucinated`) | Stage 1 |
| 추가할 내용 (`missed`) | Stage 1 |
| 검증된 장면 정보 (`verified_odd`) | Stage 3 |
| 영상 6 frames | cosmos_client (force_frames=False, 기본 모드) |

### ODD 압축 전달 방식

`verified_odd` 전체 JSON을 그대로 넣으면 토큰이 과다합니다.  
`_compact_odd()`로 핵심 값만 추출해 텍스트 한 줄씩 전달합니다.

```python
# _compact_odd() 결과 예시
time_of_day: day
weather: clear
road_type: urban
num_lanes: 4
ego_lane_position: rightmost
ego_actions: ['straight', 'right_turn']
traffic_signals: True (green)
```

### 프롬프트 구조

```
영상을 보며 아래 정보를 바탕으로 정제된 캡션을 작성하세요.

=== 원본 캡션 (오류 포함 가능) ===
{original_caption}

=== 검증된 장면 정보 ===
{compact_odd}

=== 제거할 내용 (영상에 없음) ===
- {hallucinated[0]}
- {hallucinated[1]}
...

=== 추가할 내용 (영상에 있음, 누락됨) ===
- {missed[0]}
...

작성 규칙:
1. 위 "제거할 내용" 삭제
2. 검증된 장면 정보를 자연스럽게 통합
3. 그라운딩된 원본 내용 보존
4. "추가할 내용" 포함
5. 시간순 서술 (영상 시작 → 끝)
6. 150~300 단어, 3인칭 과거시제 ("The ego-vehicle...")
7. 텍스트만 출력 (JSON·헤더 금지)
```

---

## API 재시도 구조 (cosmos_client.py)

두 층의 재시도로 네트워크 오류와 JSON 파싱 실패를 모두 처리합니다.

```
chat_json() / chat_text() 호출
│
└─ 외부 루프 (최대 MAX_RETRIES=3회): JSON 파싱 실패 시 재시도
      │
      └─ _chat() 내부 루프 (최대 3회): API 오류 시 지수 백오프
            ├─ 시도 1 실패 → 2초 대기
            ├─ 시도 2 실패 → 4초 대기
            └─ 시도 3 실패 → 예외 전파

JSON 파싱 (_extract_json):
  1. 마크다운 코드블록 제거 (```json ... ```)
  2. 첫 번째 { 또는 [ 위치 탐색
  3. 표준 json.loads() 시도
  4. 실패 시 json_repair 라이브러리로 자동 복구
     (누락 쉼표, 잘린 배열, 닫히지 않은 괄호 등)
  5. 모두 실패 시 ValueError 전파 → 외부 루프에서 재시도
```

---

## 배치 처리 (batch_runner.py)

### 동시성 모델

```python
sem = asyncio.Semaphore(N)    # N = --concurrent 인자 (기본 2)

tasks = [create_task(process_clip(clip_id)) for clip_id in pending]
# 수백 개 Task를 한꺼번에 생성하지만
# Semaphore로 실제 실행은 N개씩 제한

# 완료된 순서대로 결과 수집 → 다음 대기 Task 자동 시작
for result in as_completed(tasks):
    ...
```

**concurrent=2 기준 vLLM 서버 부하:**  
클립 1개당 Stage 1(3 pass) + Stage 2 + Stage 3 + Stage 4 = **최소 6회 API 호출**  
concurrent=2이면 동시에 최대 12개 요청 → vLLM 내부에서 배치 처리

### 진행 상태 추적

```json
// progress.json 구조
{
  "done":    ["clip_id_1", "clip_id_2", ...],
  "error":   ["clip_id_x"],
  "skipped": []
}
```

- **10개 처리마다** 저장 → 프로세스 강제 종료 시 손실 최소화
- **재시작 시** `done` + `error` 클립은 자동 스킵
- `--reset`으로 초기화 가능

### 분산 처리 (샤딩)

```bash
# 2개 GPU, 각각 다른 vLLM 서버 실행 시
CR_VLLM_URL=http://localhost:8001/v1 \
uv run python -m caption_refine.batch_runner \
    --source all --shard-index 0 --total-shards 2

CR_VLLM_URL=http://localhost:8002/v1 \
uv run python -m caption_refine.batch_runner \
    --source all --shard-index 1 --total-shards 2
```

스트라이드 방식으로 클립을 분배합니다 (`ids[shard::total]`).  
자세한 내용은 [DISTRIBUTED.md](DISTRIBUTED.md) 참조.

---

## 실행 방법

### 전제 조건

```bash
# vLLM, bitsandbytes 설치 여부 확인
cd /Data1/home/bskang/cosmos-reason2
uv pip list | grep -E "vllm|bitsandbytes|json-repair"

# 비디오·캡션 파일 존재 확인
ls /Data1/home/bskang/cds-data/front_camera_videos/*.mp4 | head -3
ls /Data1/home/bskang/cds-data/caption_v2/captions/*.txt | head -3
```

---

### Step 1 — vLLM 서버 기동

---

#### 양자화(Quantization) 켜기 / 끄기

8B 모델(BF16 원본)을 RTX 4090 24GB에 올리려면 양자화가 필수입니다.  
아래 표를 참고해 상황에 맞는 옵션을 선택하세요.

| 모드 | 추가 플래그 | 가중치 VRAM | max-model-len 가능 | 품질 |
|------|------------|------------|-------------------|------|
| **BF16 (양자화 없음)** | `--dtype bfloat16` | ~15 GB | ≤ 8192 (여유 없음) | 최고 |
| **INT8 (현재 사용)** | `--load-format bitsandbytes --quantization bitsandbytes` | ~7.5 GB | 32768 | 우수 |
| **사전 양자화 모델** | `--quantization fp8` (모델에 따라 다름) | 3~5 GB | 32768+ | 모델마다 상이 |

**Cosmos-Reason2-8B VRAM 소비 실측치 (RTX 4090 24GB, `--gpu-memory-utilization 0.92`)**:
```
가중치(INT8) ≈ 7.5 GB
프레임워크 오버헤드 ≈ 6 GB
KV cache (남은 용량) ≈ 8.4 GB  → 61,296 토큰
```

##### ① 양자화 켬 — INT8 (권장, 현재 설정)

```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve nvidia/Cosmos-Reason2-8B \
    --port 8001 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --load-format bitsandbytes \
    --quantization bitsandbytes \
    --allowed-local-media-path /Data1/home/bskang/cds-data \
    --reasoning-parser qwen3 \
    > vllm_8001.log 2>&1 &
echo "vLLM PID: $!"
```

> `--load-format bitsandbytes --quantization bitsandbytes` 두 플래그를 **함께** 지정해야 합니다.  
> `--quantization`만 쓰면 weight 로딩이 BF16으로 유지되어 OOM이 발생합니다.

##### ② 양자화 끔 — BF16 원본 (24GB에 빠듯하게 올라감)

두 양자화 플래그를 제거하고 `--dtype`과 `--max-model-len`을 조정합니다.

```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve nvidia/Cosmos-Reason2-8B \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.70 \
    --allowed-local-media-path /Data1/home/bskang/cds-data \
    --reasoning-parser qwen3 \
    > vllm_8001.log 2>&1 &
```

> BF16로는 가중치만 ~15 GB이므로, `--gpu-memory-utilization 0.70`(≈16.8 GB)으로 시작해  
> CUDA OOM이 없으면 0.75~0.80까지 올릴 수 있습니다.  
> `--max-model-len 8192`로 KV cache를 줄여야 나머지 VRAM을 가중치에 할당할 수 있습니다.  
> **video 모드(~28,540 토큰)는 BF16 + 24GB에서 불가** → frames 모드(`CR_VIDEO_MODE=frames`)로 사용하세요.

##### ③ 사전 양자화 모델 사용 (INT4 수준, 메모리 여유 확보)

INT4 수준으로 더 압축하려면 이미 양자화된 모델 변형을 사용합니다.  
(vLLM bitsandbytes는 INT8이 기본이며, 기본 모델에서 런타임 INT4는 지원이 제한적입니다.)

```bash
# NVFP4 양자화 모델 예시 (가중치 ~3.5 GB)
CUDA_VISIBLE_DEVICES=0 nohup vllm serve vrfai/Cosmos-Reason2-8B-NVFP4 \
    --port 8001 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --allowed-local-media-path /Data1/home/bskang/cds-data \
    --reasoning-parser qwen3 \
    > vllm_8001.log 2>&1 &
```

> 사전 양자화 모델은 HuggingFace에 별도 레포로 올라와 있습니다.  
> 다운로드: `huggingface-cli download vrfai/Cosmos-Reason2-8B-NVFP4`

---

#### `--allowed-local-media-path` — 경로 지정 방법

`video` 모드에서 클라이언트는 `file:///path/to/video.mp4` 형태의 URL을 서버에 전달합니다.  
서버는 `--allowed-local-media-path`에 지정한 경로가 해당 파일의 **상위 디렉터리인지**를 검사합니다.

**내부 검사 로직 (vLLM 소스 코드 기준):**
```python
# vllm/multimodal/utils.py
if allowed_local_media_path not in filepath.resolve().parents:
    raise ValueError(f"파일 경로가 허용된 경로의 하위가 아닙니다")
```

`Path.parents`는 파일 경로의 **모든 상위 디렉터리 목록**이므로,  
`--allowed-local-media-path`에 충분히 상위의 경로를 지정하면 그 아래 **모든 하위 경로가 허용**됩니다.

**⚠️ 주의: 단일 경로만 지정 가능**  
`--allowed-local-media-path`는 문자열 하나만 받습니다. 여러 개를 나열할 수 없습니다.

---

##### 방법 1 — 공통 상위 경로 사용 (가장 간단)

접근이 필요한 모든 경로들의 **공통 상위 디렉터리**를 지정합니다.

```
# 접근 필요 경로들
/Data1/home/bskang/cds-data/front_camera_videos/clip.mp4
/Data1/home/bskang/cds-data/side_camera_videos/clip.mp4
/Data1/home/bskang/extra-data/clips/clip.mp4

# 공통 상위: /Data1/home/bskang (또는 /Data1)
```

```bash
--allowed-local-media-path /Data1/home/bskang
```

> 보안 고려: 경로를 넓게 열수록 서버가 해당 디렉터리 아래의 임의 파일을 읽을 수 있게 됩니다.  
> 최소한으로 필요한 공통 조상 경로를 사용하세요.

##### 방법 2 — 심볼릭 링크 (경로를 하나의 디렉터리로 통합)

실제 파일은 분산되어 있지만, 하나의 허용 디렉터리 아래에 심볼릭 링크를 모읍니다.

```bash
# 허용 디렉터리 생성
mkdir -p /Data1/home/bskang/vllm-media

# 다른 위치의 경로들을 링크
ln -s /Data1/home/bskang/cds-data/front_camera_videos \
      /Data1/home/bskang/vllm-media/front_camera_videos

ln -s /Data2/other-server/clips \
      /Data1/home/bskang/vllm-media/other_clips

# 서버 기동
--allowed-local-media-path /Data1/home/bskang/vllm-media
```

> symlink를 따라가면 실제 파일은 다른 디스크에 있어도 됩니다.  
> `filepath.resolve()`가 심볼릭 링크를 해석하므로 실제 경로 기준으로 검사됩니다.  
> **주의**: resolve() 후 실제 경로가 허용 경로의 하위여야 합니다 — symlink 목적지가 허용 경로 밖이면 거부됩니다.

##### 방법 3 — bind mount (Linux, 가장 견고)

별개의 마운트 포인트를 하나의 디렉터리 아래에 묶습니다.

```bash
# 허용 디렉터리 생성
mkdir -p /mnt/vllm-media/front_camera_videos
mkdir -p /mnt/vllm-media/other_clips

# bind mount (root 권한 또는 sudo 필요)
sudo mount --bind /Data1/home/bskang/cds-data/front_camera_videos \
                  /mnt/vllm-media/front_camera_videos

sudo mount --bind /Data2/other-server/clips \
                  /mnt/vllm-media/other_clips

# 서버 기동
--allowed-local-media-path /mnt/vllm-media
```

> bind mount는 resolve() 후에도 `/mnt/vllm-media/` 하위로 나타나므로 심볼릭 링크 문제를 회피합니다.  
> 재부팅 후 유지하려면 `/etc/fstab`에 등록하거나 시작 스크립트에 추가하세요.

---

#### 현재 서버 기동 명령어 (요약)

```bash
cd /Data1/home/bskang/cosmos-reason2

# 8B 모델 — INT8 양자화 (현재 권장)
CUDA_VISIBLE_DEVICES=0 nohup vllm serve nvidia/Cosmos-Reason2-8B \
    --port 8001 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --load-format bitsandbytes \
    --quantization bitsandbytes \
    --allowed-local-media-path /Data1/home/bskang/cds-data \
    --reasoning-parser qwen3 \
    > vllm_8001.log 2>&1 &
echo "vLLM PID: $!"
```

> `--allowed-local-media-path /Data1/home/bskang/cds-data`  
> → `front_camera_videos/`, `caption_v2/` 등 cds-data 하위 모든 경로 허용

서버 준비 확인 (로딩 3~5분 소요):

```bash
watch -n 10 'curl -sf http://localhost:8001/health && echo "READY" || echo "Loading..."'
tail -f vllm_8001.log | grep -E "Application startup|ERROR"
```

#### Cosmos-Reason2-2B (양자화 없이 가능, 빠른 실행)

```bash
CUDA_VISIBLE_DEVICES=1 nohup vllm serve nvidia/Cosmos-Reason2-2B \
    --port 8000 \
    --max-model-len 32768 \
    --allowed-local-media-path /Data1/home/bskang/cds-data \
    --reasoning-parser qwen3 \
    > vllm_8000.log 2>&1 &
```

> 2B 모델은 BF16 기준 ~4GB이므로 양자화 없이도 여유롭게 올라갑니다.

---

### Step 2 — 환경변수 설정

```bash
# 8B 모델 (port 8001) + video 모드 (현재 기본 설정)
export CR_VLLM_URL="http://localhost:8001/v1"
export CR_VLLM_MODEL="nvidia/Cosmos-Reason2-8B"
export CR_VIDEOS_DIR="/Data1/home/bskang/cds-data/front_camera_videos"
export CR_CAPTIONS_DIR="/Data1/home/bskang/cds-data/caption_v2/captions"
export CR_OUTPUT_ROOT="/Data1/home/bskang/cds-data/caption_v2_8b_video"
export CR_VIDEO_MODE="video"   # Stage 4에서 file:// URL로 전송 (Stage 1은 항상 frames)
export CR_VIDEO_FPS="1"        # 20초 × 1fps = 20 frames ≈ 28,540 토큰
```

또는 `test_8b.py`처럼 스크립트 내에서 `os.environ`으로 설정:

```python
os.environ["CR_VLLM_URL"]     = "http://localhost:8001/v1"
os.environ["CR_VLLM_MODEL"]   = "nvidia/Cosmos-Reason2-8B"
os.environ["CR_VIDEO_MODE"]   = "video"
os.environ["CR_VIDEO_FPS"]    = "1"
```

---

### Step 3 — 단일 클립 테스트

```bash
cd /Data1/home/bskang/cosmos-reason2

# 특정 5개 클립으로 빠른 검증
.venv/bin/python test_8b.py
```

또는 batch_runner로 1개만:

```bash
uv run python -m caption_refine.batch_runner \
    --source gap --limit 1 --concurrent 1
```

정상 실행 로그:
```
19:35:24 INFO pipeline — [e8770620] Stage 1: grounding check
19:35:33 INFO stage1_ground — Stage 1 done — hal=6 missed=1 (3-pass union, temp=0.3)
19:35:33 INFO pipeline — [e8770620] Stage 2: ODD extraction
19:35:47 INFO pipeline — [e8770620] Stage 3: self-verify (0 low-conf fields)
19:35:47 INFO pipeline — [e8770620] Stage 4: caption refine
19:35:52 INFO pipeline — [e8770620] Done — hal=6 missed=1 low_conf=0
```

결과 확인:

```bash
CLIP="e8770620-5491-4b43-ba35-8f4efcc8d660"
BASE="/Data1/home/bskang/cds-data/caption_v2_8b_video"

echo "=== 정제 캡션 ==="
cat "${BASE}/captions/${CLIP}.camera_front_wide_120fov.txt"

echo "=== 변경 내역 (hallucination 목록) ==="
python3 -m json.tool "${BASE}/diff/${CLIP}_diff.json"

echo "=== ODD 구조화 정보 ==="
python3 -m json.tool "${BASE}/odd/${CLIP}.json"
```

---

### Step 4 — 소규모 배치 검증 (5~10개)

```bash
uv run python -m caption_refine.batch_runner \
    --source gap \
    --limit 5 \
    --concurrent 2
```

진행 상황 모니터링:

```bash
# 별도 터미널에서
watch -n 30 'python3 -c "
import json
d = json.load(open(\"/Data1/home/bskang/cds-data/caption_v2_8b_video/progress.json\"))
print(f\"완료: {len(d['\''done'\''])}, 에러: {len(d['\''error'\''])}\")"'
```

---

### Step 5 — 전체 배치 실행

```bash
# gap 클립 전체
uv run python -m caption_refine.batch_runner --source gap --concurrent 2

# 중단 후 재시작 (동일 명령어 — 완료 클립 자동 스킵)
uv run python -m caption_refine.batch_runner --source gap --concurrent 2

# 실패 클립만 재처리
python3 -c "
import json
d = json.load(open('/Data1/home/bskang/cds-data/caption_v2_8b_video/progress.json'))
json.dump(d['error'], open('/tmp/retry.json', 'w'))
print(len(d['error']), '개 재처리')
"
uv run python -m caption_refine.batch_runner --ids-file /tmp/retry.json --concurrent 1

# 처음부터 다시 시작
uv run python -m caption_refine.batch_runner --source gap --reset
```

---

## 주요 설정값 (config.py)

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `CR_VLLM_URL` | `http://localhost:8000/v1` | vLLM 서버 주소 |
| `CR_VLLM_MODEL` | `nvidia/Cosmos-Reason2-2B` | 서빙 모델 이름 |
| `CR_VIDEO_MODE` | `frames` | `frames`: base64 이미지 / `video`: file:// URL |
| `CR_VIDEO_FPS` | `4` | video 모드에서 서버에 전달하는 샘플링 FPS |
| `CR_NUM_FRAMES` | `12` | Stage 2·3 기본 프레임 수 |
| `CR_NUM_FRAMES_STAGE1` | `16` | Stage 1 전용 프레임 수 (hallucination 탐지) |
| `CR_CONF_THRESHOLD` | `0.7` | Stage 3 재검증 기준 confidence |
| `CR_CONCURRENT` | `2` | 동시 처리 클립 수 |
| `CR_DATA_ROOT` | `/Data1/home/bskang/cds-data` | 데이터 루트 경로 |
| `CR_VIDEOS_DIR` | `$DATA_ROOT/front_camera_videos` | MP4 파일 디렉터리 |
| `CR_CAPTIONS_DIR` | `$DATA_ROOT/captions` | 원본 캡션 디렉터리 |
| `CR_OUTPUT_ROOT` | `$DATA_ROOT/caption_v2` | 출력 루트 디렉터리 |

**토큰 예산 (변경 불가, config.py 직접 수정):**

| 상수 | 값 | 사용 Stage |
|------|----|-----------|
| `MAX_TOKENS_STAGE1` | 2048 | Stage 1 (3-pass 각각) |
| `MAX_TOKENS_STAGE2` | 4096 | Stage 2 (ODD JSON 긴 응답) |
| `MAX_TOKENS_STAGE3` | 1024 | Stage 3 (부분 재검증) |
| `MAX_TOKENS_STAGE4` | 1024 | Stage 4 (정제 캡션) |

---

## 출력 파일 구조

### 1. 정제 캡션 (`captions/{clip_id}.camera_front_wide_120fov.txt`)

원본과 동일한 파일명 규칙 → 기존 파이프라인에 디렉터리만 바꿔 교체 가능.

### 2. 구조화 ODD (`odd/{clip_id}.json`)

```json
{
  "clip_id": "e8770620-5491-4b43-ba35-8f4efcc8d660",
  "odd_compat": {
    "time_of_day": "day",
    "weather": "clear",
    "road_type": "urban",
    "traffic_density": "light",
    "hazard_level": "medium",
    "agent_type": ["car"],
    "ego_action": ["straight", "right_turn"]
  },
  "odd_extended": {
    "time_of_day": {"value": "day", "confidence": 0.95, "evidence": "밝은 햇빛"},
    "num_lanes":   {"value": 4,     "confidence": 0.88, "evidence": "차선 4개 확인"},
    "traffic_signals": {"present": true, "state": "green", "confidence": 0.91, "evidence": "..."}
  }
}
```

- `odd_compat`: 기존 `odd_tags.json` 스키마 호환 → **현재 시스템에 바로 대체 가능**
- `odd_extended`: confidence·evidence 포함 완전 데이터 → **사후 품질 분석용**

### 3. 변경 내역 (`diff/{clip_id}_diff.json`)

```json
{
  "clip_id": "e8770620-5491-4b43-ba35-8f4efcc8d660",
  "grounded": ["A black sedan merges into the rightmost lane..."],
  "hallucinated": [
    "During this maneuver, a white minibus stalls at the curb... [white minibus는 어떤 프레임에도 없음]",
    "Ahead, a school zone warning is visible... [school zone 표지판 미확인]"
  ],
  "missed": [
    "A red truck is visible on the left side of the road, moving forward."
  ],
  "low_conf_fields": []
}
```

`hallucinated` 수가 많은 클립 분석 → 원본 캡션 생성 모델의 문제 유형 파악에 활용.

---

## 처리 시간 추정 (Cosmos-Reason2-8B, RTX 4090)

| Stage | 내용 | 소요 시간 |
|-------|------|----------|
| Stage 1 | 16 frames × 3 pass 병렬 (temp=0.3) | ~7~15초 |
| Stage 2 | 4 frames, ODD 15 필드 추출 | ~10~20초 |
| Stage 3 | 저확신 필드만 재검증 (평균 0~3개) | 0~8초 |
| Stage 4 | 6 frames, 정제 캡션 생성 | ~5~10초 |
| 합계 | | **~25~55초/클립** |

`concurrent=2` 기준 처리량:

| 클립 수 | 예상 시간 |
|---------|----------|
| 5개 | ~2~5분 |
| 50개 | ~20~50분 |
| 200개 (gap 전체) | ~80~200분 |

> Stage 1이 3-pass 병렬이므로 벽시계 시간은 단일 pass와 동일.  
> concurrent를 올리면 vLLM 내부 배치 효율이 높아져 처리 속도가 개선됩니다.  
> `concurrent=4` 시 약 1.5~2배 속도 향상 예상 (GPU 메모리·KV cache 여유 확인 후 조정).

---

## 외부 서버 실행 — 경로 설정 가이드

로컬 개발 환경과 외부 서버(astrago 등)의 데이터 경로가 다를 때, 모든 경로는 환경변수로 오버라이드합니다.
코드를 수정할 필요가 없습니다.

### 경로 설정 원칙

```
config.py 기본값 (로컬)
  CR_DATA_ROOT = /Data1/home/bskang/cds-data
    ├── VIDEOS_DIR   = $CR_DATA_ROOT/front_camera_videos
    ├── CAPTIONS_DIR = $CR_DATA_ROOT/captions
    └── OUTPUT_ROOT  = $CR_DATA_ROOT/caption_v2

환경변수 우선순위:
  CR_DATA_ROOT 하나만 설정 → 하위 3개 경로 자동 결정  (구조가 같을 때)
  CR_VIDEOS_DIR / CR_CAPTIONS_DIR / CR_OUTPUT_ROOT 개별 설정 → 구조가 다를 때
  CR_INDEX_DIR / CR_SANFLOW_GAP_PATH → clip_id 목록 위치
```

### 방법 1 — CR_DATA_ROOT만 교체 (구조 동일, 루트만 다를 때)

외부 서버의 데이터 구조가 로컬과 동일하고 마운트 위치만 다른 경우입니다.

```bash
# 외부 서버에서 실행 전 설정
export CR_DATA_ROOT="/root/kadap/MyDisk/cds-data"
# → VIDEOS_DIR   자동으로 /root/kadap/MyDisk/cds-data/front_camera_videos
# → CAPTIONS_DIR 자동으로 /root/kadap/MyDisk/cds-data/captions
# → OUTPUT_ROOT  자동으로 /root/kadap/MyDisk/cds-data/caption_v2

export CR_VLLM_URL="http://localhost:8000/v1"

uv run python -m caption_refine.batch_runner --source gap --concurrent 2
```

### 방법 2 — 경로 개별 지정 (입력/출력 구조가 다를 때)

입력 데이터와 출력 경로가 서로 다른 디스크나 디렉터리에 있는 경우입니다.

```bash
# 입력 경로 (원본 영상, 기존 캡션)
export CR_VIDEOS_DIR="/data/input/videos"
export CR_CAPTIONS_DIR="/data/input/captions"

# 출력 경로 (정제 결과물)
export CR_OUTPUT_ROOT="/data/output/caption_v2"

# clip_id 목록 경로 (--source gap / longtail / all 사용 시)
export CR_INDEX_DIR="/data/index"
export CR_SANFLOW_GAP_PATH="/data/index/sanflow_gaps.json"

# vLLM
export CR_VLLM_URL="http://localhost:8000/v1"

uv run python -m caption_refine.batch_runner --source gap --concurrent 2
```

### 방법 3 — --ids-file 사용 (clip_id 목록 별도 파일)

`--source gap/longtail/all`의 JSON 경로를 맞추기 번거로울 때, clip_id 목록을 직접 파일로 전달합니다.

```bash
# clip_id 목록 파일 준비 (JSON 배열)
cat > /tmp/my_clips.json << 'EOF'
[
  "e8770620-5491-4b43-ba35-8f4efcc8d660",
  "47302f29-4895-4ed0-8053-11601eec80e1"
]
EOF

export CR_VIDEOS_DIR="/data/input/videos"
export CR_CAPTIONS_DIR="/data/input/captions"
export CR_OUTPUT_ROOT="/data/output/caption_v2"

uv run python -m caption_refine.batch_runner \
    --ids-file /tmp/my_clips.json \
    --concurrent 2
```

### 외부 서버 실행 전 체크리스트

```bash
# 1. 입력 파일 존재 확인
ls ${CR_VIDEOS_DIR}/*.mp4 | head -3
ls ${CR_CAPTIONS_DIR}/*.txt | head -3

# 2. vLLM 서버 응답 확인
curl http://localhost:8000/health

# 3. 출력 디렉터리 쓰기 권한 확인 (config.py import 시 자동 생성됨)
python3 -c "import os; os.environ['CR_OUTPUT_ROOT']='/data/output/caption_v2'; \
    from caption_refine import config; print('출력 경로 OK:', config.OUTPUT_ROOT)"

# 4. 단일 클립 테스트 (전체 실행 전 검증)
uv run python -m caption_refine.batch_runner --ids-file /tmp/my_clips.json --limit 1 --concurrent 1
```

### 외부 서버 전용 실행 스크립트 예시

매번 환경변수를 설정하는 번거로움을 줄이려면 실행 스크립트로 묶어 사용합니다.

```bash
#!/bin/bash
# run_astrago.sh — 외부 서버(astrago) 전용 실행 스크립트

set -e

# ── 경로 설정 ──────────────────────────────────────────────────
export CR_DATA_ROOT="/root/kadap/MyDisk/cds-data"
export CR_OUTPUT_ROOT="/root/kadap/MyDisk/cds-data/caption_v2"

# ── vLLM 설정 ──────────────────────────────────────────────────
export CR_VLLM_URL="http://localhost:8000/v1"
export CR_VLLM_MODEL="nvidia/Cosmos-Reason2-2B"

# ── 실행 ───────────────────────────────────────────────────────
cd /root/kadap/MyDisk/cosmos-reason2
source .venv/bin/activate

python -m caption_refine.batch_runner \
    --source gap \
    --concurrent 2 \
    "$@"     # 추가 인자 전달 가능 (예: --limit 10 --reset)
```

```bash
chmod +x run_astrago.sh

# 기본 실행
./run_astrago.sh

# 추가 인자 전달
./run_astrago.sh --limit 10
./run_astrago.sh --reset
```

---

## 건너뛰기(Skip) / 재시작(Resume) 동작 분석

### 스킵 판단 기준 — progress.json이 유일한 근거

`batch_runner.py:106-107`의 핵심 코드:

```python
already_done = set(state["done"]) | set(state["error"])
pending = [cid for cid in clip_ids if cid not in already_done]
```

**출력 파일 존재 여부는 확인하지 않습니다.** 스킵 여부는 `progress.json`의 `done` + `error` 목록에만 의존합니다.

### 동작 시나리오별 분석

| 상황 | progress.json | 출력 파일 | 처리 결과 |
|------|--------------|----------|----------|
| 정상 완료 후 재시작 | `done`에 있음 | 존재 | **스킵** (재처리 없음) |
| 에러 후 재시작 | `error`에 있음 | 없음/불완전 | **스킵** (자동 재시도 안 됨) |
| progress.json 삭제 후 재시작 | 없음 | 존재 | **전체 재처리 + 덮어쓰기** |
| 새 클립 추가 | 없음 | 없음 | 정상 처리 |
| 입력 파일 없는 클립 | 없음 | 없음 | `no_video` / `no_caption` → `error`로 기록 → 이후 스킵 |

### 에러 클립 재처리 방법

에러 클립은 `error` 목록에 기록되어 다음 실행에서도 자동으로 스킵됩니다.
재처리하려면 아래 중 하나를 선택하세요.

**방법 A — 에러 클립만 ids-file로 추출해 재처리 (권장)**

```bash
# 에러 목록을 파일로 추출
python3 -c "
import json
d = json.load(open('$CR_OUTPUT_ROOT/progress.json'))
json.dump(d['error'], open('/tmp/retry.json', 'w'))
print(len(d['error']), '개 에러 클립 추출')
"

# 재처리 (--concurrent 1로 낮춰서 원인 파악 용이)
uv run python -m caption_refine.batch_runner \
    --ids-file /tmp/retry.json --concurrent 1
```

> 이 방법은 완료된 클립은 그대로 유지하면서 에러 클립만 재시도합니다.  
> 단, 위 명령은 새 progress.json이 없는 상태로 시작하므로 에러 클립들이 `done`에 추가됩니다.  
> 기존 progress.json의 `done` 목록은 유지하고 싶다면 방법 B를 사용하세요.

**방법 B — progress.json에서 에러 목록만 수동 제거**

```bash
python3 -c "
import json
path = '$CR_OUTPUT_ROOT/progress.json'
d = json.load(open(path))
print('에러:', len(d['error']), '개 → 제거 후 재처리 대상으로 전환')
d['error'] = []
json.dump(d, open(path, 'w'), indent=2)
"

# 재실행 시 에러였던 클립들이 pending으로 처리됨
uv run python -m caption_refine.batch_runner --source gap --concurrent 2
```

**방법 C — 전체 초기화 후 처음부터**

```bash
# progress.json 삭제 + 재시작 (완료 클립도 모두 재처리됨, 출력 파일 덮어쓰기)
uv run python -m caption_refine.batch_runner --source gap --reset
```

### 출력 파일 기반 스킵이 없는 이유

현재 구현에서 출력 파일 존재 여부로 스킵하지 않는 이유:

1. **원자성 보장**: 파일만 있고 progress.json에 없다면 Stage 4까지 완료됐는지 불확실 (중간 충돌 가능)
2. **progress.json이 단일 진실 소스**: 3개 출력 파일(caption / odd / diff) 중 일부만 생성된 불완전 상태를 방지
3. **덮어쓰기 의도적 허용**: `--reset` 시 기존 출력을 갱신하는 것이 설계 의도

> **실용 팁**: 외부 서버에서 처음 실행할 때 progress.json이 없으면 전체 클립을 새로 처리합니다.  
> 이전 서버에서 생성한 progress.json을 복사해 오면 이미 처리된 클립을 스킵할 수 있습니다.

---

## 트러블슈팅

### `No valid JSON found in response`

모델이 JSON을 마크다운 코드블록으로 감싸거나 쉼표를 빠뜨린 경우.  
`_extract_json()`이 json_repair 라이브러리로 자동 복구합니다.  
반복 실패 시 → `MAX_RETRIES` 소진 후 해당 클립 `error` 상태로 기록.

### Stage 1 `hal=0` (hallucination 미탐지)

- `NUM_FRAMES_STAGE1`이 너무 적으면 커버리지 부족 → 기본값 16 유지 권장
- `_STAGE1_PASSES`를 3→4로 올리거나 `_STAGE1_TEMPERATURE`를 0.3→0.5로 조정 가능  
  (`stage1_ground.py` 상단 상수 직접 수정)

### video 모드에서 `finish_reason: length` (응답 잘림)

- `CR_VIDEO_FPS`를 낮춰 토큰 수 감소 (예: `1` → `0.5`)
- 또는 `CR_VIDEO_MODE=frames`로 전환 후 `CR_NUM_FRAMES`로 조정

### vLLM `CUDA out of memory`

- `--gpu-memory-utilization`을 낮춤 (예: 0.92 → 0.85)
- `--max-model-len`을 줄임 (32768 → 16384, video 모드 불가해짐)
- `--quantization bitsandbytes` 확인 (8B 모델 필수)



# GPU 0
CR_VLLM_URL=http://localhost:8000/v1 \
uv run python -m caption_refine.batch_runner \
    --ids-file /Data1/home/bskang/cosmos-reason2/caption_refine/clip_ids_shard00.json \
    --shard-index 0 --total-shards 12 --concurrent 2

# GPU 1
CR_VLLM_URL=http://localhost:8001/v1 \
uv run python -m caption_refine.batch_runner \
    --ids-file /Data1/home/bskang/cosmos-reason2/caption_refine/clip_ids_shard01.json \
    --shard-index 1 --total-shards 12 --concurrent 2