# ODD Schema — caption_refine_v2

소스 기준: `stage2_odd.py`, `prompts.py`, `pipeline.py`, `config.py`  
데이터 출력 경로: `caption_v3/odd/{clip_id}.json`  
검증: 전체 **101,741개** 파일 전수 조사 기준 (2026-07-07)

---

## JSON 최상위 구조

```
{clip_id}.json
├── clip_id            — 클립 UUID
├── sensor_confirmed   — Phase 1: 센서 확정 속성
├── odd_raw            — Phase 2-A: VLM 원본 응답 (4차원)
├── odd_final          — Phase 2-A + Stage 3 보정 적용 결과 (4차원)
├── odd_compat         — odd_final에서 파생한 평탄화 호환 레코드
└── _meta              — 품질 메타 정보
```

---

## 1. `sensor_confirmed` — 센서 확정 속성 (Phase 1)

센서 데이터(곡률·IMU·속도·타임스탬프)에서 결정적으로 계산. VLM 불필요.

| 필드 | 값 종류 | 판단 기준 (config.py 임계값) |
|------|---------|------------------------------|
| `road_curvature` | `straight` \| `gentle` \| `sharp` | 곡률 반경 > 500 m → straight, 200–500 m → gentle, < 200 m → sharp |
| `road_gradient` | `flat (mean N°)` \| `uphill (mean +N°)` \| `downhill (mean -N°)` | IMU pitch: \|p\| < 1.5° → flat, p > 0 → uphill, p < 0 → downhill (수치 포함 문자열로 저장) |
| `speed_range` | `low (…km/h)` \| `mid (…km/h)` \| `high (…km/h)` | 평균 속도: < 50 kph → low, 50–100 → mid, 100+ → high (범위 수치 포함 문자열로 저장) |
| `time_of_day` | `day` \| `dusk_dawn` \| `night` | 타임스탬프(KST): 일출 6시±30분·일몰 19시±30분 → dusk_dawn |

> `time_of_day`는 메타 파일에 타임스탬프가 있을 때만 생성됨. **전체 101,741개 파일에서 미관측 — 현재 데이터셋에 없음.**

---

## 2. `odd_raw` / `odd_final` — 4차원 ODD (Phase 2-A + Stage 3)

VLM(Cosmos-Reason2)이 영상을 보고 추출. 불확실하면 `"unknown"` 사용.

### 2-1. `road_structure` — 도로 구조

| 필드 | 허용 값 | 판단 기준 |
|------|---------|-----------|
| `road_type` | `highway` \| `national_road` \| `urban` \| `rural` \| `tunnel` \| `unknown` | 중앙분리대+고속→highway, 왕복 2차로+제한속도→national_road, 신호+횡단보도→urban, 시골 개방도로→rural, 터널 구간→tunnel |
| `lanes_ego_direction` | `1` \| `2` \| `2+` \| `3` \| `3+` \| `4` \| `5` \| `unknown` | 자차와 같은 방향 차선 수. 좌측 차선 마킹 개수로 판단 |
| `lanes_opposite` | `0` \| `1` \| `2+` \| `unknown` | `0` = 일방통행 (반대 방향 차로 자체 없음). 대향 차량이 보이거나 분리대로 나뉘면 `1` 이상 |
| `road_divider` | `none` \| `dashed` \| `solid` \| `barrier` \| `unknown` | 물리 방호벽→barrier, 실선→solid, 점선→dashed, 분리 없음→none |
| `lane_marking_quality` | `clear` \| `faint` \| `absent` \| `unknown` | 차선 도색 상태 |
| `junction_proximity` | `none` \| `approaching` \| `in_junction` \| `post_junction` \| `unknown` | 교차로/합류 전·내·후 여부 |

### 2-2. `environment` — 환경 조건

| 필드 | 허용 값 | 판단 기준 |
|------|---------|-----------|
| `lighting_condition` | `well_lit` \| `moderate` \| `poorly_lit` \| `unknown` | 전체 밝기 수준 |
| `precipitation` | `none` \| `rain` \| `snow` \| `unknown` | 와이퍼 흔적·빗방울·노면 반사 중 하나라도 보이면 rain |
| `fog` | `none` \| `present` \| `unknown` | ~100 m 이상 물체 윤곽이 흐릿하면 present |
| `road_surface` | `dry` \| `wet` \| `snow` \| `uneven` \| `unpaved` \| `dirt` \| `gravel` \| `sandy` \| `dusty` \| `snow-dusted` \| `unknown` | 노면 광택·물 반사로 판단. 프롬프트 정의값은 `dry\|wet\|unknown`이나 VLM이 추가 값 생성 |
| `backlight` | `none` \| `present` \| `unknown` | 태양·헤드라이트 렌즈 플레어 |

### 2-3. `dynamic_elements` — 동적 요소

| 필드 | 허용 값 | 판단 기준 |
|------|---------|-----------|
| `traffic_density` | `sparse` \| `moderate` \| `dense` \| `unknown` | 가시 차량 수: 0–2→sparse, 3–6→moderate, 7+→dense |
| `road_user_types` | `cars_only` \| `mixed` \| `pedestrians` \| `cyclists` \| `obstacles` \| `emergency` \| `unknown` | 장면 내 모든 도로 사용자 유형. `obstacles`는 프롬프트 미정의 값이나 실제 데이터에 존재 |
| `construction_zone` | `none` \| `present` \| `unknown` | 공사 콘·표지·작업자 존재 여부 |
| `special_event` | `none` \| `obstacle` \| `unknown` | 경로 상 특수 상황. 프롬프트에 `accident`·`emergency` 정의되어 있으나 실제 데이터에 미관측 |

### 2-4. `scene_complexity` — 장면 복잡도

| 필드 | 허용 값 | 판단 기준 |
|------|---------|-----------|
| `occlusion_level` | `low` \| `medium` \| `high` \| `unknown` | 전방 시야 가림 비율: < 30%→low, 30–60%→medium, > 60%→high |
| `visibility_range` | `good` \| `moderate` \| `poor` \| `unknown` | 유효 시야 거리: > 200 m→good, 50–200 m→moderate, < 50 m→poor |
| `scene_ambiguity` | `low` \| `medium` \| `high` \| `unknown` | 차선·신호·참여자 의도 해석 난이도 |
| `unexpected_element` | `none` \| `present` \| `unknown` | 운전자를 놀라게 할 비정상 요소 존재 여부 |

---

## 3. `odd_compat` — 평탄화 호환 레코드

`odd_final`에서 핵심 10개 필드를 추출·파생. 파이프라인 외부 필터링·통계용.

| 필드 | 값 종류 | 출처 |
|------|---------|------|
| `road_type` | (road_structure 동일) | odd_final.road_structure |
| `weather` | `clear` \| `rain` \| `snow` \| `unknown` | precipitation: none→clear, 나머지 동일 |
| `traffic_density` | (dynamic_elements 동일) | odd_final.dynamic_elements |
| `agent_type` | (road_user_types 동일) | odd_final.dynamic_elements |
| `lanes_ego_direction` | (road_structure 동일) | odd_final.road_structure |
| `lanes_opposite` | (road_structure 동일) | odd_final.road_structure |
| `lane_summary` | `"ego:N opp:M"` 형식 문자열 | lanes_ego_direction + lanes_opposite 조합 |
| `road_divider` | (road_structure 동일) | odd_final.road_structure |
| `lighting` | (lighting_condition 동일) | odd_final.environment |
| `scene_ambiguity` | (scene_complexity 동일) | odd_final.scene_complexity |

---

## 4. `_meta` — 품질 메타

| 필드 | 타입 | 설명 |
|------|------|------|
| `unknown_ratio` | float | odd_final 전체 16 필드 중 `"unknown"` 비율 (> 0.3 경고, > 0.5 위험) |
| `consistency_score` | float | odd_raw ↔ odd_final 일치율 (Stage 3 보정 후 변경된 필드 반영) |
| `model` | str | 사용 VLM 모델명 (예: `nvidia/Cosmos-Reason2-8B`) |
| `tagging_version` | str | 파이프라인 버전 (`"2.0"`) |

---

## 필드 수 요약

| 섹션 | 필드 수 |
|------|---------|
| sensor_confirmed | 3–4 (time_of_day는 타임스탬프 유무에 따라 선택) |
| odd_raw / odd_final | **16** (4그룹 × 4필드) |
| odd_compat | **10** |
| _meta | 4 |
