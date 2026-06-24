# caption_refine_v2 — 파이프라인 상세 분석

> AV(자율주행) 클립에서 **ODD 태깅** + **자연어 캡션**을 동시에 생성하는 5-Stage 멀티모달 파이프라인.  
> 모델: `nvidia/Cosmos-Reason2-2B` (vLLM 서빙)

---

## 1. 전체 구조

```
클립 입력
  ├── 영상 파일   : {clip_id}.camera_front_wide_120fov.mp4
  └── egomotion  : {clip_id}.egomotion.offline.parquet

         ┌─────────────────────────────────┐
         │   Stage 0  Motion Pre-processing│  ← egomotion 파싱, 속도·곡률·이벤트 계산
         └────────────┬────────────────────┘
                      │ MotionSummary
         ┌────────────▼────────────────────┐
         │   Phase 1  센서 확정값 생성      │  ← curvature, gradient, speed, time_of_day
         └────────────┬────────────────────┘
                      │ sensor_confirmed dict
            ┌─────────┴──────────┐
            │  병렬 실행          │
   ┌────────▼──────┐   ┌────────▼──────┐
   │   Stage 1     │   │   Stage 2     │
   │  NL 시나리오   │   │  ODD JSON     │
   │  (8프레임)     │   │  (4프레임)     │
   └────────┬──────┘   └────────┬──────┘
            │                   │
            │          ┌────────▼──────┐
            │          │   Stage 3     │
            │          │  unknown 재확인│  ← unknown 필드만 8프레임 재판단
            │          └────────┬──────┘
            │                   │ odd_grouped
            └──────────┬────────┘
                       │ nl_scenario + odd_grouped
         ┌─────────────▼────────────────────┐
         │   Stage 4  교차검증              │  ← 텍스트 전용, 영상 없음
         │   NL → implied ODD 추출          │
         │   + 규칙 기반 충돌 감지·해소      │
         └─────────────┬────────────────────┘
                       │ odd_final + consistency_score
         ┌─────────────▼────────────────────┐
         │   Stage 5  최종 캡션 합성        │  ← 12프레임, 모든 결과 통합
         └─────────────┬────────────────────┘
                       │
             ┌─────────┴──────────┐
             │                    │
   captions/{clip_id}.txt    odd/{clip_id}.json
                            crossval/{clip_id}_crossval.json
```

---

## 2. Stage별 상세 설명

### Stage 0 — Motion Pre-processing

**목적**: egomotion 센서 데이터를 분석해 ODD 확정값과 VLM 프롬프트용 동적 서사를 동시에 생성.

**입력**: `{clip_id}.egomotion.offline.parquet`

| 컬럼 | 내용 |
|------|------|
| `timestamp` | UNIX 타임스탬프 (초) |
| `x, y, z` | 차량 위치 (미터, ENU 좌표계) |
| `qx, qy, qz, qw` | 회전 쿼터니언 |

**처리 과정**:

1. **속도 계산**: 위치 차분의 arc-length로 스칼라 속도 산출  
   `speed = √(Δx²+Δy²+Δz²) / Δt` → 이동평균 스무딩  
   (vx·vy 방향성분 합이 아닌 크기 기반 — 급회전 전후 속도 스파이크 방지)

2. **곡률 계산**: 2D 수평면(x,y) 궤적에서 곡률반경 추출  
   `κ = (x'y'' - y'x'') / (x'²+y'²)^1.5` → 대표 반경 R = 1/|κ|

3. **IMU pitch 추출**: 쿼터니언 → pitch(앞뒤 경사) 변환, 중앙값으로 대표값 산출

4. **이벤트 감지**:
   - **Turn**: 곡률 ≥ 1/200m 구간이 ≥2s 지속
   - **Lane Change**: 횡방향 변위 ≥2.5m, 2~8초 이내 완료
   - **Accel/Decel**: 5초 윈도우에서 속도 변화 ≥10 km/h (직선 구간만)

5. **20초 구간 분할 서사 생성**:
   ```
   [Vehicle kinematics — 20 s]
   0–5 s: 42 km/h, straight
   5–10 s: 35 km/h, straight
   10–15 s: 28 km/h, sharp right turn, downhill
   15–20 s: 45 km/h, straight, downhill
   Key: right turn at 11 s (r≈28 m) · deceleration 10 km/h at 5 s
   ```

**출력 (`MotionSummary`)**:

| 메서드/속성 | 용도 |
|------------|------|
| `to_phase1_dict()` | Phase 1 → `{curvature_radius_m, imu_pitch_mean_deg, speed_mean_kph, timestamp_hour}` |
| `narrative` | Stage 1·5 프롬프트에 주입되는 동적 서사 텍스트 (~80 토큰) |
| `valid` | parquet 파일 존재·파싱 성공 여부 (False면 meta.json fallback) |

---

### Phase 1 — 센서 확정값 생성

Stage 0 결과를 ODD 카테고리로 변환. 수치 임계값 기반으로 결정하므로 VLM 추측 없이 **확정(ground truth)** 처리됨.

| 입력 수치 | 임계값 | 출력 |
|----------|--------|------|
| `curvature_radius_m` | >500m → straight, >200m → gentle, 이하 → sharp | `road_curvature` |
| `imu_pitch_mean_deg` | 절댓값 <1.5° → flat, + → uphill, - → downhill | `road_gradient` |
| `speed_mean_kph` | <50 → low, <100 → mid, ≥100 → high | `speed_range` |
| `timestamp_hour` | 일출(6)·일몰(19) ±30분 기준 | `time_of_day` |

egomotion 파일이 없는 클립은 `meta.json` fallback → 없으면 `sensor_confirmed = {}` (Phase 1 스킵).

---

### Stage 1 — NL 시나리오 생성

**목적**: 영상을 보고 **4섹션 구조화 자연어 시나리오**를 생성. Stage 4 교차검증의 원천 텍스트.

**프레임 수**: 8 (균등 샘플링)  
**온도**: 0.2 (다양성 허용)  
**실행 전략**: 동일 프롬프트로 2회 병렬 실행 → **내용어(content words) 수가 더 많은 쪽 채택**

**4섹션 구조**:
```
Section 1 — Road & Environment   : 도로 유형, 날씨, 조도
Section 2 — Surrounding Situation: 주변 도로 사용자, 이상 요소
Section 3 — Key Challenge        : AV에게 핵심 난이도 요소
Section 4 — Events               : 클립 중 발생한 변화/이벤트
```

**Stage 0 연동**: `motion_narrative`가 있으면 프롬프트 맨 앞에 삽입.  
Section 4 힌트에 "Use the kinematics data above to accurately describe turns, lane changes, and speed changes." 추가.

---

### Stage 2 — 구조화 ODD 추출

**목적**: 영상에서 **4차원 16개 필드 ODD JSON**을 직접 추출. Stage 1과 완전 독립 실행.

**프레임 수**: 4 (토큰 절약)  
**온도**: 0.0 (결정론적)

**ODD 스키마 (4차원 × 4필드 = 16필드)**:

```
road_structure   : road_type, lane_count, lane_marking_quality, junction_proximity
environment      : lighting_condition, precipitation, fog, road_surface, backlight
dynamic_elements : traffic_density, road_user_types, construction_zone, special_event
scene_complexity : occlusion_level, visibility_range, scene_ambiguity, unexpected_element
```

**설계 원칙**: confidence 없음 — 확신하지 못하면 반드시 `"unknown"` 사용 (추측 금지).

**판단 기준 (프롬프트 내 명시)**:
- `road_type`: 중앙분리대+고속 → highway, 2차로+속도표지 → national_road, 신호+횡단보도 밀집 → urban
- `precipitation`: 와이퍼 자국, 빗방울, 젖은 노면 반사 중 하나라도 보이면 → rain
- `occlusion_level`: 전방 시야 차단 비율 — <30% → low, 30~60% → medium, >60% → high
- `road_user_types`: 승용차만 → cars_only, 다양한 유형 → mixed

**JSON 파싱**: `json_repair` 라이브러리로 VLM이 생성한 불완전 JSON도 복구.

**unknown_ratio 경고**:
- >0.3 → 프롬프트 검토 권장
- >0.5 → 영상 품질 점검 필요

---

### Stage 3 — unknown 필드 재확인

**목적**: Stage 2에서 `"unknown"`으로 남은 필드만 골라 **더 구체적인 시각 힌트**와 함께 재판단.

**프레임 수**: 8 (더 많은 프레임으로 정밀 재검토)  
**실행 조건**: unknown 필드가 0개면 즉시 스킵

**프롬프트 구성**: unknown 필드별로 **허용값 + 시각 단서 힌트** 쌍을 생성.

예시:
```
- environment.road_surface: allowed = [dry | wet | unknown]
    hint: Look for reflective sheen or puddles on road = wet; otherwise dry
```

**응답 형식**: `"group.field": "resolved_value"` 형태의 JSON. 해소 가능한 것만 반환, 나머지는 unknown 유지.

---

### Stage 4 — 교차검증

**목적**: Stage 1 NL 텍스트와 Stage 2·3 ODD JSON의 **일관성 검증 + 충돌 해소**.  
**특징**: 영상 없음 — 텍스트 전용 LLM 호출 (`text_chat_json`).

**2단계 처리**:

**Step 1 — NL → implied ODD 추출 (VLM)**:  
NL 시나리오 텍스트만 입력해 암묵적으로 내포된 ODD 속성 추출.
```json
{
  "implied_precipitation":   "none",
  "implied_fog":             null,
  "implied_road_surface":    "dry",
  "implied_lighting":        "poorly_lit",
  "implied_road_type":       "urban",
  "implied_traffic_density": "moderate",
  "implied_special_event":   null,
  "implied_challenges":      ["sharp left turn", "vehicle braking"]
}
```

**Step 2 — 규칙 기반 충돌 감지·해소**:  
7개 속성을 대상으로 ODD값 vs NL implied값 비교.

| 우선순위 | 속성 | 이유 |
|---------|------|------|
| **NL 우선** | precipitation, fog, road_surface | 장면 맥락 추론에 강함 |
| **ODD 우선** | road_type, traffic_density, lighting | 시각적 분류에 강함 |
| **flag** | 나머지 | 수동 검토 필요 (`review_needed`) |

**consistency_score**: `1.0 - (충돌 수 / 비교 가능 속성 수)`

| 점수 | 등급 | 의미 |
|------|------|------|
| ≥0.9 | good | 자동 확정 가능 |
| 0.7~0.9 | warn | 충돌 항목 검토 권장 |
| <0.7 | poor | 전면 수동 검토 필요 |

출력 `odd_final`: 충돌 해소 규칙이 반영된 최종 ODD dict.

---

### Stage 5 — 최종 캡션 합성

**목적**: 지금까지 수집된 모든 정보를 통합해 **150~300단어 서사 캡션** 생성.

**프레임 수**: 12 (가장 많은 시각 컨텍스트)  
**온도**: 0.0

**프롬프트 입력 5개 블록**:

```
=== SCENE ANALYSIS (Stage 1 NL 시나리오) ===
=== SENSOR-CONFIRMED FACTS (Phase 1 확정값) ===
=== VEHICLE KINEMATICS (Stage 0 동적 서사) ===
=== VERIFIED ODD FACTS (odd_final) ===
=== CROSS-VALIDATION RESULT (충돌 해소 내역) ===
```

**캡션 작성 규칙**:
1. sensor_confirmed + motion_narrative → 절대적 ground truth
2. 충돌 발생 시 Stage 4에서 결정된 winner 값 사용
3. ODD 정보를 자연스럽게 문장에 녹임
4. 시간 순서 서술 (시작 → 중간 → 끝)
5. 3인칭 과거 시제 ("The ego-vehicle...")

**실패 시 fallback**: Stage 5 예외 발생 시 Stage 1 NL 시나리오를 그대로 반환.

---

## 3. 데이터 흐름 요약

| 단계 | 입력 | 출력 | VLM 호출 | 프레임 수 |
|------|------|------|---------|---------|
| Stage 0 | parquet | MotionSummary | 없음 | — |
| Phase 1 | MotionSummary | sensor_confirmed dict | 없음 | — |
| Stage 1 | video + narrative | NL 시나리오 (텍스트) | 2회 병렬 | 8 |
| Stage 2 | video | ODD JSON (16필드) | 1회 | 4 |
| Stage 3 | video + unknown 목록 | 해소된 ODD dict | 0~1회 | 8 |
| Stage 4 | NL 시나리오 텍스트 | implied ODD + conflicts | 1회 (텍스트 전용) | 0 |
| Stage 5 | video + 모든 결과 | 최종 캡션 (텍스트) | 1회 | 12 |

**클립당 총 VLM 호출**: 최소 5회 (Stage 3 스킵 시) ~ 6회

---

## 4. 병렬 실행 구조

```python
# Stage 1 + Stage 2 는 동시 실행 (서로 참조 없음)
nl_scenario, extraction = await asyncio.gather(
    stage1_caption.run(client, vid, motion_narrative),
    stage2_odd.run(client, vid),
)
```

- Stage 3은 Stage 2 완료 후 실행 (unknown 필드 목록 필요)
- Stage 4는 Stage 1·3 완료 후 실행 (NL + ODD 둘 다 필요)
- Stage 5는 Stage 4 완료 후 실행 (최종 종합)

---

## 5. 파일 구조

### 입력 데이터

```
/Data1/home/bskang/cds-data/
├── front_camera_videos/
│   └── {clip_id}.camera_front_wide_120fov.mp4
└── egomotion_offline/
    └── {clip_id}.egomotion.offline.parquet
```

### 출력 데이터

```
/Data1/home/bskang/cds-data/caption_v3/
├── captions/
│   └── {clip_id}.camera_front_wide_120fov.txt   ← Stage 5 최종 캡션
├── odd/
│   └── {clip_id}.json                            ← ODD 전체 결과
└── crossval/
    └── {clip_id}_crossval.json                   ← 교차검증 결과
```

### ODD JSON 구조

```json
{
  "clip_id": "...",
  "sensor_confirmed": {
    "road_curvature": "sharp",
    "road_gradient":  "flat",
    "speed_range":    "low"
  },
  "odd_raw":   { /* Stage 2 원본 */ },
  "odd_final": { /* Stage 4 충돌 해소 후 */ },
  "odd_compat": {
    "road_type": "...", "weather": "...", "traffic_density": "...",
    "agent_type": "...", "lane_count": "...", "lighting": "...", "scene_ambiguity": "..."
  },
  "_meta": {
    "unknown_ratio":     0.007,
    "consistency_score": 0.80,
    "model":             "nvidia/Cosmos-Reason2-2B",
    "tagging_version":   "2.0"
  }
}
```

---

## 6. 설정 및 실행

### 환경변수 (모두 선택 사항)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CR_VLLM_URL` | `http://localhost:8000/v1` | vLLM 서버 주소 |
| `CR_VLLM_MODEL` | `nvidia/Cosmos-Reason2-2B` | 모델 ID |
| `CR_DATA_ROOT` | `/Data1/home/bskang/cds-data` | 데이터 루트 |
| `CR_EGOMOTION_DIR` | `{DATA_ROOT}/egomotion_offline` | egomotion parquet 디렉토리 |
| `CR_CONCURRENT` | `2` | 동시 처리 클립 수 |

### 실행 명령

```bash
# gap 클립 200개 (검증용)
uv run python -m caption_refine_v2.batch_runner --source gap --concurrent 2

# 전체 클립 (~299,180개)
uv run python -m caption_refine_v2.batch_runner --source all --concurrent 4

# 특정 ID 목록
uv run python -m caption_refine_v2.batch_runner --ids-file my_clips.json

# 분산 실행 (서버 3대 × shard)
uv run python -m caption_refine_v2.batch_runner \
    --source all --shard-index 0 --total-shards 3 --concurrent 4
```

---

## 7. 설계 핵심 원칙

### 독립성 (Stage 1 ↔ Stage 2)
Stage 1(NL)과 Stage 2(ODD)는 같은 영상을 보지만 **서로의 결과를 모른다**.  
이후 Stage 4가 두 독립 판단의 일치도를 검증하는 구조.

### unknown 우선 정책
VLM이 확신하지 못하면 `"unknown"` → Stage 3에서 재시도.  
v1의 confidence 점수 방식 대비 훨씬 명확한 불확실성 표현.

### 센서 우선 계층
```
Stage 0 egomotion  >  Phase 1 확정값  >  Stage 4 교차검증  >  Stage 2·1 VLM 추론
```
물리 센서 데이터가 VLM 추론보다 항상 우선.

### Fallback 체계
- egomotion 없음 → meta.json → sensor_confirmed = {}
- Stage 3 unknown 없음 → 즉시 스킵
- Stage 5 실패 → Stage 1 NL 시나리오 반환

---

## 8. 200 clips 실측 결과 (Gap 파이프라인)

| 지표 | 수치 |
|------|------|
| 처리 시간 | 13분 (190 clips, concurrent=2) |
| 오류율 | 0% |
| avg consistency_score | 0.80 |
| good (≥0.9) | 35% |
| warn (0.7~0.9) | 46% |
| poor (<0.7) | 20% |
| avg unknown_ratio | 0.007 |
| egomotion 미보유 클립 | 20% (meta.json fallback) |

**ODD 분포**:
- Road type: rural 48%, national_road 28%, urban 15%, highway 5%
- Lighting: well_lit 82%, poorly_lit 14%
- Traffic: sparse 84%, moderate 11%, dense 1%

**poor 클립 충돌 패턴**: `lighting`, `road_type`, `traffic_density` 3개 필드에서 NL ↔ ODD 불일치 집중.  
→ 같은 영상을 보고 Stage 1(자유 서술)과 Stage 2(JSON 분류)가 다른 판단을 내린 케이스.  
→ 프롬프트 개선 또는 VLM 추론 한계 사례로 활용 가능.
