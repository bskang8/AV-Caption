## — 이중 파이프라인 (구조화 ODD + 자연어 시나리오) 교차 검증 시스템

**작성 목적**: 주행 영상 + 웨이포인트(no GPS) + IMU + 속도 + 곡률 데이터를 기반으로  
ODD를 체계적으로 추출하고, 자연어 시나리오와 교차 검증으로 정합성을 높이는 실용 절차  
**참조**: `05_overviews/ODD_추출_가이드라인.md` (Safety Case용), `05_overviews/VLM_기반_ODD_추출_비판적분석.md`  
**작성일**: 2026-06

---

## 0. 전체 아키텍처

```
입력 데이터
  영상 + 웨이포인트(no GPS) + IMU(rot) + 속도 + 곡률
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
[Phase 1]           [Phase 1]
센서 확정 속성       센서 확정 속성
(동일 결과)          (동일 결과)
    │                   │
    ▼                   ▼
[Phase 2-A]         [Phase 2-B]
VLM: 구조화          VLM: 자연어
ODD 추출             시나리오 추출
(JSON)               (자유 기술)
    │                   │
    └─────────┬─────────┘
              ▼
        [Phase 3]
        교차 검증 레이어
        일치·불일치 탐지 + 보완
              ▼
    [최종 통합 레코드]
    ODD 태그 + 시나리오 + 정합성 점수
              ▼
        [Phase 4]
        분포 확인 → 슬라이스 정의
```

**두 파이프라인 독립성 원칙**: Phase 2-A와 Phase 2-B는 서로의 결과를 참조하지 않고 독립 실행한다. 교차 검증은 Phase 3에서만 수행한다.

---

## 핵심 개념 정리

### ODD 명세 vs 조건 어노테이션

| | ODD 명세 | 조건 어노테이션 |
|---|---|---|
| 정의 | 시스템이 동작하도록 설계된 조건의 집합 | 이 클립에서 실제 발생한 조건 |
| 시간성 | **정적** — 설계 시점에 한 번 작성 | **동적** — 시간에 따라 변함 |
| 형식 | ISO 34503 Table B.1 | 어노테이션 DB |

영상에서 시간에 따라 변하는 것은 ODD가 아니라 **운행 조건**이다.  
어노테이션은 "이 조건이 ODD 안에 있는가"를 판정하는 것과 "실제 조건이 무엇인가"를 기록하는 것을 동시에 한다.

### 어노테이션 시간 단위

| 속성 유형 | 단위 | 이유 |
|---|---|---|
| 도로 구조, 지형 (경사·곡률) | 클립 | 클립 안에서 변화 없음 |
| 시간대, 조명 | 클립 (황혼 전환은 세그먼트) | 느린 변화 |
| **강수, 역광** | **세그먼트** | 급변하고 성능 영향 큼 |
| 교통 밀도, 참여자 | 클립 중앙값 | 순간 변동보다 대표값이 의미 있음 |
| 장면 복잡도 | 클립 | VLM 전체 장면 판단 |

경계 조건 클립 처리: 전환 구간은 `transitional` 플래그 부여 후 세그먼트 분할.

---

## 입력 데이터 소스와 한계

| 소스 | 제공 정보 | GPS 없는 영향 |
|---|---|---|
| 영상 | 시각 장면 전체 | — |
| 웨이포인트 (no GPS) | 상대 경로·곡률·분기점 토폴로지 | 지도 매칭 불가 → 도로 유형·지역 유형은 VLM으로 |
| IMU (rot: pitch/roll/yaw) | 경사, 뱅킹, 회전율 | — |
| 속도 | 자차 속도 | — |
| 곡률 | 도로 기하 | — |
| 타임스탬프 | 시간대 계산 | 위치 없어도 한국 기준 고정값으로 주야간 구분 가능 |

**GPS 부재의 의미**: Phase 1에서 지도 매칭 기반 속성(도로 유형, 지역 유형, 교차로 유형, 속도 제한)을 확정할 수 없다. 이 속성들은 VLM(Phase 2-A)이 영상에서 직접 추출한다. GPS 부재가 VLM의 역할을 더 크게 만든다.

---

## ODD 4차원 속성 체계

4개 차원으로 체계화한다. 차원④(장면 복잡도)는 NL 시나리오와 교차 검증의 핵심 연결고리다.

### 차원 ① 도로 구조 (Road Structure)

| 속성 | 값 | 소스 |
|---|---|---|
| `road_type` | `highway` `national_road` `urban` `rural` | VLM |
| `road_curvature` | `straight` `gentle` `sharp` | 곡률 데이터 |
| `road_gradient` | `flat` `uphill` `downhill` | IMU pitch |
| `junction_proximity` | `none` `approaching` `in_junction` `post_junction` | 웨이포인트 토폴로지 |
| `lane_count` | `single` `double` `multi` | VLM |
| `lane_marking_quality` | `clear` `faint` `absent` | VLM |

### 차원 ② 환경 조건 (Environmental Conditions)

| 속성 | 값 | 소스 |
|---|---|---|
| `time_of_day` | `day` `dusk_dawn` `night` | 타임스탬프 |
| `lighting_condition` | `well_lit` `moderate` `poorly_lit` | VLM |
| `precipitation` | `none` `rain` `snow` | VLM |
| `fog` | `none` `present` | VLM |
| `road_surface` | `dry` `wet` `unknown` | VLM |
| `backlight` | `none` `present` | VLM |

### 차원 ③ 동적 요소 (Dynamic Elements)

| 속성 | 값 | 소스 |
|---|---|---|
| `traffic_density` | `sparse` `moderate` `dense` | VLM |
| `road_user_types` | `cars_only` `mixed` `pedestrians` `cyclists` `emergency` | VLM |
| `construction_zone` | `none` `present` | VLM |
| `special_event` | `none` `accident` `obstacle` `emergency` | VLM |

### 차원 ④ 장면 복잡도 (Scene Complexity)

NL 시나리오에는 자연스럽게 등장하지만 기존 ODD 목록에는 없던 차원이다.  
이 차원이 두 파이프라인 교차 검증의 주요 연결고리 역할을 한다.

| 속성 | 값 | 소스 |
|---|---|---|
| `occlusion_level` | `low` `medium` `high` | VLM |
| `visibility_range` | `good` `moderate` `poor` | VLM |
| `scene_ambiguity` | `low` `medium` `high` | VLM |
| `unexpected_element` | `none` `present` | VLM |

---

## Phase 1 — 센서 확정 속성 (4개)

GPS 없이 센서 데이터만으로 확정적으로 추출 가능한 속성. 양쪽 파이프라인에 동일하게 적용된다.

| 속성 | 소스 | 처리 |
|---|---|---|
| `road_curvature` | 곡률 데이터 | R>500m → straight, 200~500 → gentle, <200 → sharp |
| `road_gradient` | IMU pitch | ±1.5° → flat, 양수 → uphill, 음수 → downhill |
| `speed_range` | 속도 데이터 | <50 kph → low, 50~100 → mid, 100+ → high |
| `time_of_day` | 타임스탬프 | 한국 기준 일출·일몰 ±30분 → dusk_dawn, 사이 → day, 외 → night |

```python
def extract_phase1(clip_meta: dict) -> dict:
    r = clip_meta["curvature_radius_m"]
    pitch = clip_meta["imu_pitch_mean_deg"]
    v = clip_meta["speed_mean_kph"]
    hour = clip_meta["timestamp"].hour  # 타임스탬프만 사용

    # 한국 평균 일출 6시, 일몰 19시 기준 (정밀도 필요 시 ephem 라이브러리 사용)
    SUNRISE, SUNSET = 6, 19
    if SUNRISE + 0.5 < hour < SUNSET - 0.5:
        tod = "day"
    elif abs(hour - SUNRISE) <= 0.5 or abs(hour - SUNSET) <= 0.5:
        tod = "dusk_dawn"
    else:
        tod = "night"

    return {
        "road_curvature": "straight" if r > 500 else ("gentle" if r > 200 else "sharp"),
        "road_gradient":  "flat" if abs(pitch) < 1.5 else ("uphill" if pitch > 0 else "downhill"),
        "speed_range":    "low" if v < 50 else ("mid" if v < 100 else "high"),
        "time_of_day":    tod,
    }
```

---

## Phase 2-A — VLM 구조화 ODD 추출

목표: 4차원 ODD 속성을 JSON 형식으로 추출.  
원칙: 확신 없으면 반드시 `unknown`. 추측으로 채우지 않는다.

### 프롬프트

```
다음 주행 영상 클립을 보고 아래 JSON 형식으로만 답해라.
확신할 수 없는 항목은 반드시 "unknown"을 사용하고 절대 추측하지 마라.
설명이나 부연 없이 JSON만 출력하라.

{
  "road_structure": {
    "road_type": "highway" | "national_road" | "urban" | "rural" | "unknown",
    "lane_count": "single" | "double" | "multi" | "unknown",
    "lane_marking_quality": "clear" | "faint" | "absent" | "unknown",
    "junction_proximity": "none" | "approaching" | "in_junction" | "post_junction" | "unknown"
  },
  "environment": {
    "lighting_condition": "well_lit" | "moderate" | "poorly_lit" | "unknown",
    "precipitation": "none" | "rain" | "snow" | "unknown",
    "fog": "none" | "present" | "unknown",
    "road_surface": "dry" | "wet" | "unknown",
    "backlight": "none" | "present" | "unknown"
  },
  "dynamic_elements": {
    "traffic_density": "sparse" | "moderate" | "dense" | "unknown",
    "road_user_types": "cars_only" | "mixed" | "pedestrians" | "cyclists" | "emergency" | "unknown",
    "construction_zone": "none" | "present" | "unknown",
    "special_event": "none" | "accident" | "obstacle" | "emergency" | "unknown"
  },
  "scene_complexity": {
    "occlusion_level": "low" | "medium" | "high" | "unknown",
    "visibility_range": "good" | "moderate" | "poor" | "unknown",
    "scene_ambiguity": "low" | "medium" | "high" | "unknown",
    "unexpected_element": "none" | "present" | "unknown"
  }
}

판단 기준:
- road_type: 중앙분리대+고속 → highway, 왕복 2차로+제한속도 표지 → national_road, 신호·횡단보도 밀집 → urban
- precipitation: 와이퍼 흔적, 빗방울, 노면 물 반사 중 하나라도 보이면 rain
- fog: 100m 이상 물체 윤곽이 흐릿하면 present
- occlusion_level: 전방 시야의 30% 미만 가림 → low, 30~60% → medium, 60% 이상 → high
- scene_ambiguity: 차선·신호·교통 참여자 의도가 명확하면 low, 판단이 모호하면 high
```

### 구현

```python
def run_pipeline_a(clip_frames: list, phase1_tags: dict) -> dict:
    response = vlm_call(
        frames=sample_frames(clip_frames, n=6),
        prompt=ODD_EXTRACTION_PROMPT,
        temperature=0,
        response_format="json"
    )
    odd = json.loads(response)

    # Phase 1 확정값 덮어쓰기 (센서 값이 VLM보다 신뢰도 높음)
    odd["road_structure"]["road_curvature"] = phase1_tags["road_curvature"]
    odd["road_structure"]["road_gradient"]  = phase1_tags["road_gradient"]
    odd["environment"]["time_of_day"]       = phase1_tags["time_of_day"]

    # unknown 비율 계산
    all_vals = [v for dim in odd.values() for v in dim.values()]
    odd["_meta"] = {
        "unknown_ratio": all_vals.count("unknown") / len(all_vals),
        "model": VLM_MODEL_ID,
        "pipeline": "A_structured_odd"
    }
    return odd
```

---

## Phase 2-B — VLM 자연어 시나리오 추출

목표: 같은 클립에서 무슨 일이 일어나고 있는지 자유롭게 기술.  
원칙: ODD JSON을 참조하지 않는다. 독립성이 교차 검증의 전제다.

### 프롬프트

```
다음 주행 영상 클립에서 일어나고 있는 상황을 설명해라.

아래 순서로 작성하되 각 항목은 1~2문장으로 제한한다:

1. 도로 및 환경: 어떤 도로에서 어떤 날씨·조명 조건인가?
2. 주변 상황: 주변 교통 참여자와 특이 요소가 있는가?
3. 핵심 도전: 이 장면에서 자율주행 시스템이 직면하는 주된 어려움은 무엇인가?
4. 이벤트: 클립 내에서 조건이 변하거나 주목할 사건이 있었는가?

JSON이나 정해진 레이블 없이 자연어로만 답하라.
```

### 구현

```python
def run_pipeline_b(clip_frames: list) -> dict:
    # Phase 2-A 결과를 참조하지 않음 — 독립 실행
    response = vlm_call(
        frames=sample_frames(clip_frames, n=8),  # A보다 더 많은 프레임
        prompt=NL_SCENARIO_PROMPT,
        temperature=0.2,   # 자연어 다양성을 위해 소폭 허용
        response_format="text"
    )
    return {
        "nl_description": response,
        "_meta": {
            "model": VLM_MODEL_ID,
            "pipeline": "B_nl_scenario"
        }
    }
```

---

## Phase 3 — 교차 검증 레이어

두 파이프라인 결과를 대조해 불일치를 탐지하고 최종 ODD를 확정한다.  
세 번째 VLM 호출 또는 규칙 기반으로 처리한다.

### 3-1. NL → ODD 키워드 추출

NL 시나리오에서 ODD 속성 관련 키워드를 추출해 구조화된 비교 형태로 변환한다.

```python
NL_TO_ODD_PROMPT = """
다음 자연어 시나리오에서 아래 항목에 해당하는 단서가 있으면 추출하라.
없으면 null로 표시하라. JSON으로만 답하라.

시나리오: {nl_description}

{{
  "implied_precipitation": "none" | "rain" | "snow" | null,
  "implied_fog": "none" | "present" | null,
  "implied_road_surface": "dry" | "wet" | null,
  "implied_lighting": "well_lit" | "moderate" | "poorly_lit" | null,
  "implied_road_type": "highway" | "national_road" | "urban" | "rural" | null,
  "implied_traffic_density": "sparse" | "moderate" | "dense" | null,
  "implied_special_event": "none" | "accident" | "obstacle" | "emergency" | null,
  "implied_scene_challenge": "<핵심 어려움 키워드 최대 3개, 없으면 []>"
}}
"""
```

### 3-2. 불일치 탐지 규칙

```python
def detect_conflicts(odd: dict, nl_implied: dict) -> list:
    conflicts = []

    checks = [
        ("precipitation",    odd["environment"]["precipitation"],
                             nl_implied["implied_precipitation"]),
        ("fog",              odd["environment"]["fog"],
                             nl_implied["implied_fog"]),
        ("road_surface",     odd["environment"]["road_surface"],
                             nl_implied["implied_road_surface"]),
        ("lighting",         odd["environment"]["lighting_condition"],
                             nl_implied["implied_lighting"]),
        ("road_type",        odd["road_structure"]["road_type"],
                             nl_implied["implied_road_type"]),
        ("traffic_density",  odd["dynamic_elements"]["traffic_density"],
                             nl_implied["implied_traffic_density"]),
    ]

    for attr, odd_val, nl_val in checks:
        if nl_val is None or odd_val == "unknown":
            continue
        if odd_val != nl_val:
            conflicts.append({
                "attribute":  attr,
                "odd_value":  odd_val,
                "nl_implied": nl_val,
                "resolution": resolve(attr, odd_val, nl_val)
            })
    return conflicts


def resolve(attr: str, odd_val: str, nl_val: str) -> dict:
    """불일치 해소 규칙 — 어느 파이프라인을 신뢰할지 결정"""

    # NL이 맥락 추론에 강한 속성: NL 우선
    nl_preferred = {"precipitation", "fog", "road_surface"}
    # ODD가 시각적 분류에 강한 속성: ODD 우선
    odd_preferred = {"road_type", "traffic_density", "lighting"}

    if attr in nl_preferred:
        return {"winner": "nl", "final_value": nl_val,
                "reason": "NL이 장면 맥락에서 추론하는 속성"}
    elif attr in odd_preferred:
        return {"winner": "odd", "final_value": odd_val,
                "reason": "ODD가 시각적 분류로 직접 판단하는 속성"}
    else:
        return {"winner": "flag", "final_value": "review_needed",
                "reason": "규칙 미정의, 수동 검토 필요"}
```

### 3-3. 정합성 점수 계산

```python
def consistency_score(odd: dict, conflicts: list) -> float:
    """비교 가능한 속성 중 일치 비율"""
    comparable = sum(1 for _, o, n in ... if n is not None and o != "unknown")
    if comparable == 0:
        return 1.0
    return 1.0 - len(conflicts) / comparable
```

**점수 해석 기준**:

| 점수 | 의미 | 조치 |
|---|---|---|
| 0.9 이상 | 양호 | 자동 확정 |
| 0.7 ~ 0.9 | 주의 | 갈등 항목만 검토 |
| 0.7 미만 | 불량 | 전체 수동 검토 또는 재추출 |

---

## Phase 4 — 분포 확인 후 슬라이스 정의

Phase 1~3 완료 후 실제 데이터 분포를 확인하고 나서 슬라이스를 정의한다.  
사전에 µODD 슬라이스를 정의하지 않는다 — 분포 모르는 상태에서 정의하면 대부분 공슬라이스가 된다.

```python
import pandas as pd

df = pd.read_json("final_records.jsonl", lines=True)

# 단변량 분포
for col in ["road_type", "time_of_day", "precipitation", "scene_ambiguity"]:
    print(df[col].value_counts())

# 교차 분포 (성능 분산이 큰 2차원 조합)
pd.crosstab(df["time_of_day"], df["precipitation"])
pd.crosstab(df["road_type"],   df["scene_ambiguity"])

# 정합성 점수 분포
df["consistency_score"].hist(bins=20)
```

**슬라이스 정의 기준**:

| 기준 | 조치 |
|---|---|
| 클립 수 < 50 | 데이터 추가 수집 또는 합성 데이터 |
| 클립 수 ≥ 500 | 학습 시 서브샘플링 |
| `unknown` 비율 > 30% | Phase 2 프롬프트 재검토 |
| 정합성 < 0.7 비율 > 20% | 양쪽 프롬프트 품질 재검토 |

---

## 통합 레코드 스키마

```yaml
# E2E ODD + Scenario Record Schema v2.0

record:
  # 메타데이터
  clip_id: str
  duration_sec: float
  timestamp: datetime
  tagging_version: "2.0"

  # Phase 1: 센서 확정 (수정 불가)
  sensor_confirmed:
    road_curvature: [straight, gentle, sharp]
    road_gradient:  [flat, uphill, downhill]
    speed_range:    [low, mid, high]
    time_of_day:    [day, dusk_dawn, night]

  # Phase 2-A: 구조화 ODD (VLM)
  odd_raw:
    road_structure:
      road_type:            [highway, national_road, urban, rural, unknown]
      lane_count:           [single, double, multi, unknown]
      lane_marking_quality: [clear, faint, absent, unknown]
      junction_proximity:   [none, approaching, in_junction, post_junction, unknown]
    environment:
      lighting_condition:   [well_lit, moderate, poorly_lit, unknown]
      precipitation:        [none, rain, snow, unknown]
      fog:                  [none, present, unknown]
      road_surface:         [dry, wet, unknown]
      backlight:            [none, present, unknown]
    dynamic_elements:
      traffic_density:      [sparse, moderate, dense, unknown]
      road_user_types:      [cars_only, mixed, pedestrians, cyclists, emergency, unknown]
      construction_zone:    [none, present, unknown]
      special_event:        [none, accident, obstacle, emergency, unknown]
    scene_complexity:
      occlusion_level:      [low, medium, high, unknown]
      visibility_range:     [good, moderate, poor, unknown]
      scene_ambiguity:      [low, medium, high, unknown]
      unexpected_element:   [none, present, unknown]
    _meta:
      model: str
      unknown_ratio: float

  # Phase 2-B: 자연어 시나리오 (VLM)
  scenario_raw:
    nl_description: str      # 자유 기술 원문
    _meta:
      model: str

  # Phase 3: 교차 검증
  validation:
    nl_implied: {...}        # NL에서 추출한 ODD 키워드
    conflicts:               # 불일치 목록
      - attribute: str
        odd_value: str
        nl_implied: str
        resolution:
          winner: str        # "odd" | "nl" | "flag"
          final_value: str
          reason: str
    consistency_score: float # 0.0 ~ 1.0

  # 최종 확정 ODD (Phase 3 반영)
  odd_final:                 # odd_raw에서 conflicts 반영 후 확정값
    road_structure: {...}
    environment: {...}
    dynamic_elements: {...}
    scene_complexity: {...}

  # ODD 멤버십
  in_odd: bool
  odd_exit_reason: str | null

  # 세그먼트 (강수·역광 변화 시)
  segments:
    - seg_id: str
      time_range: [float, float]
      precipitation: str
      backlight: str
      in_odd: bool
```

---

## VLM 운용 지침

### 모델 선택

| 모델 | 용도 | 비고 |
|---|---|---|
| GPT-4o | Phase 2-A·B 모두 | 가장 정확, 유료 |
| InternVL2-8B | 로컬 실행 Phase 2-A | A100 기준 ~3 fps, 무료 |
| Qwen2-VL-7B | 로컬 실행 Phase 2-B | 비디오 입력 지원 |
| 소형 분류 모델 | 특정 속성 대체 가능 | 강수·교통밀도는 YOLO+CNN이 더 빠름 |

### 프레임 샘플링 전략

```python
def sample_frames(clip_frames: list, n: int) -> list:
    """균등 샘플링 — 클립 시작·중간·끝을 반드시 포함"""
    if len(clip_frames) <= n:
        return clip_frames
    indices = [int(i * (len(clip_frames) - 1) / (n - 1)) for i in range(n)]
    return [clip_frames[i] for i in indices]
```

### unknown 비율 관리

```
unknown_ratio > 0.5  → 해당 클립 영상 품질 점검 (너무 어둡거나 흐린 경우)
unknown_ratio > 0.3  → 프롬프트 판단 기준 재검토
unknown_ratio < 0.1  → 과신 가능성 (VLM이 추측으로 채우고 있을 수 있음)
```

---

## Safety Case 연결 경로

이 가이드의 최종 출력은 E2E 학습·검증용이다. Safety Case 증거로 승격이 필요한 시점에 추가 작업이 필요하다.

```
이 가이드 출력 (E2E용)
  odd_final + consistency_score
        ↓  Safety Case 필요 시
  VLM 태그 → 전용 모델로 재검증 (확률 캘리브레이션)
  ISO 34503 Table B.1 명세 문서 추가 작성
  Syntactic 수치 범위 매핑 추가
  µODD 슬라이스 공식 정의 + 커버리지 집계
        ↓
  Safety Case 증거 → ODD_추출_가이드라인.md 참조
```
