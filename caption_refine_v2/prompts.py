"""
5개 Stage별 프롬프트 + Phase 1 판단 기준.

Stage 1: video → 4섹션 구조화 NL 시나리오 (Phase 2-B)
Stage 2: video → 4차원 ODD JSON, confidence 없음, unknown 사용 (Phase 2-A)
Stage 3: video → unknown 필드 재확인
Stage 4: NL 시나리오 텍스트 → implied ODD 키워드 추출 (Phase 3 Step 1, 영상 없음)
Stage 5: video + NL 시나리오 + odd_final + 교차검증 결과 → 최종 캡션

설계 원칙 (odd_tagging_practical.md):
  - Phase 2-A (구조화 ODD)와 Phase 2-B (NL 시나리오)는 독립 실행, 서로 참조 안 함
  - 확신 없으면 반드시 "unknown" — 추측으로 채우지 않음
  - NL 우선 속성: precipitation, fog, road_surface (맥락 추론)
  - ODD 우선 속성: road_type, traffic_density, lighting (시각 분류)
"""

# ── Stage 1: NL 시나리오 (Phase 2-B) ─────────────────────────────────────────

def stage1_nl_scenario(motion_narrative: str = "") -> str:
    motion_block = ""
    if motion_narrative:
        motion_block = (
            f"\n=== VEHICLE KINEMATICS (sensor data, ground truth) ===\n"
            f"{motion_narrative}\n\n"
        )
    return motion_block + """\
Watch this autonomous driving video carefully.

Describe the driving scene in 4 structured sections.

Section 1 — Road & Environment:
What type of road is it (highway, urban street, rural road)?
Count lanes carefully: how many lanes travel in the SAME direction as the ego-vehicle, \
and how many in the OPPOSITE direction? (e.g., "2 lanes same direction, 1 opposite" or \
"single bidirectional lane") Note weather and lighting conditions.

Section 2 — Surrounding Agents:
First, list every visible road user using this format (one per line):
  - [type] [position relative to ego] [action] [safety relevance]
  Types: car/truck/bus/motorcycle/bicycle/pedestrian/other
  Position: front/rear/left/right + estimated distance (e.g., front-left ~20m)
  Action: constant_speed / accelerating / decelerating / turning / stopped / \
crossing / cut-in / parked
  Safety relevance: low / medium / HIGH
Then in 1–2 sentences summarize the overall traffic situation.

Section 3 — Key Challenge:
What is the main difficulty this scene presents for an autonomous driving system? \
(e.g., limited visibility, complex intersection, pedestrian unpredictability, occlusion)

Section 4 — Events:
Lane change detection: observe how the dashed lane markings move relative to the camera \
center over time. If the RIGHT dashed line drifts toward center → ego moved LEFT. \
If the LEFT dashed line drifts toward center → ego moved RIGHT. \
If markings remain roughly symmetric → no lane change.
Then describe all noteworthy events (traffic signal change, vehicle cut-in, sudden braking, \
lane change, construction zone entry, ego turns, on-ramp merge, junction entry). \
For junction entry / on-ramp merge: look for early frames where the road approaches at an angle \
(the main road visible to the side), then the camera perspective rotates to align with that road. \
Use the kinematics data above (if provided) as ground truth for ego speed and maneuvers.
For each STOP event in the kinematics: watch the video frames around that timestamp and describe \
what CAUSED the stop — e.g. traffic light, STOP sign, vehicle blocking ahead, \
person exiting a vehicle in front, pedestrian crossing. \
Do NOT infer the cause from the kinematic hint label; observe it directly from the video. \
A stop cause that is visible in the video always overrides any hint.

Write in third-person past tense ("The ego-vehicle...").
Output the 4 sections as plain text — no JSON, no numbered headers."""


# ── Stage 2: 구조화 ODD 추출 (Phase 2-A) ─────────────────────────────────────

def stage2_odd_extract() -> str:
    return """\
Watch this autonomous driving video carefully.
Think step by step about each field before answering, then reply ONLY with valid JSON.
If you are not confident, use "unknown" — never guess.

{
  "road_structure": {
    "road_type":              "highway | national_road | urban | rural | unknown",
    "lanes_ego_direction":    "1 | 2 | 3+ | unknown",
    "lanes_opposite":         "0 | 1 | 2+ | unknown",
    "road_divider":           "none | dashed | solid | barrier | unknown",
    "lane_marking_quality":   "clear | faint | absent | unknown",
    "junction_proximity":     "none | approaching | in_junction | post_junction | unknown"
  },
  "environment": {
    "lighting_condition": "well_lit | moderate | poorly_lit | unknown",
    "precipitation":      "none | rain | snow | unknown",
    "fog":                "none | present | unknown",
    "road_surface":       "dry | wet | unknown",
    "backlight":          "none | present | unknown"
  },
  "dynamic_elements": {
    "traffic_density":  "sparse | moderate | dense | unknown",
    "road_user_types":  "cars_only | mixed | pedestrians | cyclists | emergency | unknown",
    "construction_zone": "none | present | unknown",
    "special_event":    "none | accident | obstacle | emergency | unknown"
  },
  "scene_complexity": {
    "occlusion_level":    "low | medium | high | unknown",
    "visibility_range":   "good | moderate | poor | unknown",
    "scene_ambiguity":    "low | medium | high | unknown",
    "unexpected_element": "none | present | unknown"
  }
}

Judgment criteria:
- road_type: median divider + high speed → highway; 2-lane + speed limit signs → national_road; \
signals + crosswalks dense → urban; open country road → rural
- lanes_ego_direction: count ONLY lanes going the SAME direction as the ego-vehicle. \
Look at lane markings to the LEFT of ego: each dashed/solid line = one additional lane.
- lanes_opposite: "0" means this is a ONE-WAY road — NO opposing traffic direction exists at all \
(e.g., expressway exit ramp, one-way city street). \
If ANY vehicles travel toward you, or if a divider/barrier separates your side from opposing flow, \
use "1" or "2+" instead. \
If opposing lanes exist but you cannot count them precisely, use "unknown" — do NOT use "0".
- road_divider: physical barrier or raised median → barrier; \
solid painted line only → solid; dashed painted line → dashed; \
no separation at all (bidirectional sharing same lane space) → none. \
NOTE: a high-speed road with opposing traffic almost always has at least solid or barrier.
- precipitation: wiper marks OR raindrops OR wet road reflections visible in ANY frame → rain
- fog: objects beyond ~100 m appear hazy → present
- backlight: direct sun or headlight glare into camera lens → present
- occlusion_level: <30% of forward view blocked → low; 30–60% → medium; >60% → high
- scene_ambiguity: lane markings, signals, and road user intent all clear → low; \
any element ambiguous or hard to interpret → high
- road_user_types: only cars visible → cars_only; any mix of types → mixed; \
pedestrians visible → pedestrians; cyclists dominant → cyclists; emergency vehicle → emergency

Do NOT add keys, comments, or text outside the JSON."""


# ── Stage 3: unknown 필드 재확인 ─────────────────────────────────────────────

# 필드별 허용 값 (Stage 3 프롬프트용)
_ALLOWED_VALUES: dict[str, str] = {
    "road_structure.road_type":            "highway | national_road | urban | rural | unknown",
    "road_structure.lanes_ego_direction":  "1 | 2 | 3+ | unknown",
    "road_structure.lanes_opposite":       "0 | 1 | 2+ | unknown",
    "road_structure.road_divider":         "none | dashed | solid | barrier | unknown",
    "road_structure.lane_marking_quality": "clear | faint | absent | unknown",
    "road_structure.junction_proximity":   "none | approaching | in_junction | post_junction | unknown",
    "environment.lighting_condition":      "well_lit | moderate | poorly_lit | unknown",
    "environment.precipitation":           "none | rain | snow | unknown",
    "environment.fog":                     "none | present | unknown",
    "environment.road_surface":            "dry | wet | unknown",
    "environment.backlight":               "none | present | unknown",
    "dynamic_elements.traffic_density":    "sparse | moderate | dense | unknown",
    "dynamic_elements.road_user_types":    "cars_only | mixed | pedestrians | cyclists | emergency | unknown",
    "dynamic_elements.construction_zone":  "none | present | unknown",
    "dynamic_elements.special_event":      "none | accident | obstacle | emergency | unknown",
    "scene_complexity.occlusion_level":    "low | medium | high | unknown",
    "scene_complexity.visibility_range":   "good | moderate | poor | unknown",
    "scene_complexity.scene_ambiguity":    "low | medium | high | unknown",
    "scene_complexity.unexpected_element": "none | present | unknown",
}

# 필드별 시각적 단서 힌트
_FIELD_HINTS: dict[str, str] = {
    "road_structure.road_type":            "Look for: median divider, number of lanes, road signs, signals, crosswalks",
    "road_structure.lanes_ego_direction":  "Count lane markings to the LEFT of the ego-vehicle; each line = 1 additional lane in same direction",
    "road_structure.lanes_opposite":       "Count lanes in the OPPOSITE direction. 0 = truly one-way road (no opposing direction whatsoever). If you see any oncoming vehicles, or a divider/barrier separating opposing traffic, use 1 or 2+. If unsure of count but opposing traffic exists, use unknown — not 0.",
    "road_structure.road_divider":         "Is traffic separated by a barrier/raised median (barrier), solid painted line (solid), dashed painted line (dashed), or nothing at all — same lane space shared bidirectionally (none)? High-speed roads with opposing traffic almost always have at least solid or barrier.",
    "road_structure.lane_marking_quality": "Check if white/yellow lines are clearly painted, faded, or absent",
    "road_structure.junction_proximity":   "approaching: intersection/merge point visible ahead; in_junction: ego is currently inside an intersection or merge zone (road visible at an angle in early frames, ego crossing or joining); post_junction: just cleared an intersection; none: open road with no junction nearby",
    "environment.lighting_condition":      "Assess overall brightness: streetlights on → well_lit; dim but visible → moderate",
    "environment.precipitation":           "Check for wiper smears, raindrops on lens, wet reflections on road surface",
    "environment.fog":                     "Check if distant objects (>100 m) have hazy/blurred outlines",
    "environment.road_surface":            "Look for reflective sheen or puddles on road = wet; otherwise dry",
    "environment.backlight":               "Check if sun or headlights create glare directly into the camera",
    "dynamic_elements.traffic_density":    "Count visible vehicles: 0–2 → sparse; 3–6 → moderate; 7+ → dense",
    "dynamic_elements.road_user_types":    "Identify all visible road user categories in the scene",
    "dynamic_elements.construction_zone":  "Look for: orange cones, barriers, construction signs, workers",
    "dynamic_elements.special_event":      "Look for: collision debris, obstacles in path, emergency vehicle",
    "scene_complexity.occlusion_level":    "Estimate what percentage of the forward view is blocked by vehicles/objects",
    "scene_complexity.visibility_range":   "Estimate usable viewing distance: >200 m → good; 50–200 m → moderate; <50 m → poor",
    "scene_complexity.scene_ambiguity":    "Would a human driver need extra caution to interpret this scene?",
    "scene_complexity.unexpected_element": "Is there anything atypical that would surprise a driver?",
}


def stage3_resolve_unknown(
    unknown_fields: list[str],
    sensor_confirmed: dict | None = None,
) -> str:
    lines = []
    for key in unknown_fields:
        allowed = _ALLOWED_VALUES.get(key, "unknown")
        hint    = _FIELD_HINTS.get(key, "")
        lines.append(f"  - {key}: allowed = [{allowed}]")
        if hint:
            lines.append(f"      hint: {hint}")
    fields_text = "\n".join(lines)

    # JSON 응답 예시 구성
    example_keys = {k: "your_value" for k in unknown_fields[:3]}
    import json as _json
    example_json = _json.dumps(example_keys, indent=2)

    sensor_block = ""
    if sensor_confirmed:
        sensor_lines = "\n".join(f"  {k}: {v}" for k, v in sensor_confirmed.items())
        sensor_block = f"""\
=== SENSOR-CONFIRMED CONTEXT (ground truth — use to inform your judgment) ===
{sensor_lines}

"""

    # Special-case bias note for lanes_opposite
    lanes_opp_note = ""
    if "road_structure.lanes_opposite" in unknown_fields:
        lanes_opp_note = """\
IMPORTANT — lanes_opposite: use "0" ONLY when certain this is a one-way road.
If you see any oncoming vehicle or a divider separating two traffic directions,
return "1" or "2+" (or "unknown" if count is unclear). Never use "0" as a default.

"""

    return f"""\
Watch this video again carefully. The fields below were undetermined (value: "unknown").
Look for the specific visual evidence described in each hint and provide your best judgment.

{sensor_block}{lanes_opp_note}Unknown fields to resolve:
{fields_text}

Reply ONLY with valid JSON where each key is "group.field" and value is your resolved category.
If you genuinely cannot determine a value even after careful inspection, return "unknown" — \
but for most fields a definite answer should be reachable.

Example format:
{example_json}

Do NOT include fields not listed above. Do NOT add commentary outside the JSON."""


# ── Stage 4: NL → implied ODD 추출 (Phase 3 Step 1, 텍스트 전용) ─────────────

def stage4_nl_to_odd(nl_scenario: str) -> str:
    return f"""\
A natural language description of a driving scene is provided below.
Extract any values for the following ODD attributes that are clearly implied by the text.
If the text gives no evidence for an attribute, return null.

=== SCENARIO ===
{nl_scenario}

Reply ONLY with valid JSON:

{{
  "implied_precipitation":   "none" | "rain" | "snow" | null,
  "implied_fog":             "none" | "present" | null,
  "implied_road_surface":    "dry" | "wet" | null,
  "implied_lighting":        "well_lit" | "moderate" | "poorly_lit" | null,
  "implied_road_type":       "highway" | "national_road" | "urban" | "rural" | null,
  "implied_traffic_density": "sparse" | "moderate" | "dense" | null,
  "implied_special_event":   "none" | "accident" | "obstacle" | "emergency" | null,
  "implied_challenges":      []
}}

Rules:
- implied_challenges: list of up to 3 key challenge keywords extracted from the scene description \
(e.g. ["limited visibility", "pedestrian crossing", "occlusion"]).
- Only return a non-null value when the text CLEARLY implies that specific category.
- Do NOT infer or guess — only extract what is explicitly described.
- Do NOT add commentary outside the JSON.

Disambiguation rules (common false positives to avoid):
- implied_precipitation="snow": ONLY when the text says it IS snowing / snowfall visible in the air. \
Snow visible on the GROUND, roadside verges, or mountains = historical accumulation = return null.
- implied_precipitation="rain": ONLY when the text says it is raining. Wet road surface alone \
does NOT imply active rain — return null for precipitation if rain is not explicitly mentioned.
- implied_lighting="moderate": ONLY when the text explicitly mentions poor visibility, dim light, \
or reduced illumination. Daytime driving (including overcast or partly cloudy skies) is typically \
well_lit — if adequate visibility is described, return "well_lit". Use "moderate" only for \
genuinely challenging low-light conditions.
- implied_road_type: Use the ODD taxonomy strictly. "highway" = motorway/freeway with no grade-level \
intersections. "national_road" = designated national/state roads (can look urban with traffic lights). \
"urban" = city street. "rural" = countryside. If the text describes an intersection or traffic \
light but does NOT clearly specify the road designation, return null rather than guessing.
- implied_traffic_density="dense": ONLY when the text describes heavy, congested traffic with \
many vehicles. A short queue of 2–4 vehicles at a red light = "moderate". If uncertain, return null."""


# ── Stage 5: 최종 캡션 합성 ───────────────────────────────────────────────────

def _format_odd_final(odd_final: dict) -> str:
    """odd_final 구조화 딕셔너리를 읽기 쉬운 텍스트로 변환."""
    lines = []
    group_labels = {
        "road_structure":   "Road Structure",
        "environment":      "Environment",
        "dynamic_elements": "Dynamic Elements",
        "scene_complexity": "Scene Complexity",
    }
    for group_key, label in group_labels.items():
        group = odd_final.get(group_key, {})
        if not group:
            continue
        lines.append(f"[{label}]")
        for field, value in group.items():
            if value and value != "unknown":
                lines.append(f"  {field}: {value}")
    return "\n".join(lines) if lines else "  (no confirmed data)"


def _format_conflicts_for_stage5(conflicts: list[dict]) -> str:
    if not conflicts:
        return "  None — both sources agreed."
    lines = []
    for c in conflicts:
        w = c.get("resolution", {})
        lines.append(
            f"  [{w.get('winner','?').upper()} wins] {c['attribute']}: "
            f"ODD={c['odd_value']}, NL={c['nl_implied']} → final={w.get('final_value','?')}"
        )
    return "\n".join(lines)


def stage5_final_caption(
    nl_scenario: str,
    odd_final: dict,
    sensor_confirmed: dict,
    conflicts: list[dict],
    consistency_score: float,
    nl_challenges: list[str],
    motion_narrative: str = "",
) -> str:
    odd_text       = _format_odd_final(odd_final)
    conflict_text  = _format_conflicts_for_stage5(conflicts)
    challenge_text = ", ".join(nl_challenges) if nl_challenges else "none noted"

    sensor_lines = "\n".join(
        f"  {k}: {v}" for k, v in sensor_confirmed.items()
    ) if sensor_confirmed else "  (no sensor data available)"

    kinematics_block = motion_narrative if motion_narrative else "(not available)"

    return f"""\
Watch this driving video and write the final, accurate narrative caption.

=== SCENE ANALYSIS (from video — 4 sections) ===
{nl_scenario}

=== SENSOR-CONFIRMED FACTS (from vehicle data, highest reliability) ===
{sensor_lines}

=== VEHICLE KINEMATICS (time-series, highest reliability) ===
{kinematics_block}

=== VERIFIED ODD FACTS (structured, after cross-validation) ===
{odd_text}

=== CROSS-VALIDATION RESULT (consistency score: {consistency_score:.2f}) ===
{conflict_text}

=== KEY CHALLENGES NOTED ===
{challenge_text}

Before writing the caption, mentally verify the following safety checklist.
Do NOT output the checklist — use it only to ensure nothing is missed:
  [1] Ego trajectory: speed, any lane changes (sensor + visual evidence), turns
  [2] All agents: for each vehicle/pedestrian/cyclist noted above — position, action, threat
  [3] Lane count: how many lanes each direction (from scene analysis)
  [4] Safety-critical events: cut-in, sudden braking, pedestrian crossing path, signal violation
  [5] Road hazard: wet surface, ice, construction, poor visibility
  [6] Any conflict between sensor data and visual observation — use sensor as ground truth

Now write a final narrative caption that:
1. Uses sensor-confirmed facts and motion narrative as absolute ground truth.
2. For any conflict, uses the winning value shown above.
3. Accurately states the number of lanes in each direction (from verified ODD and scene analysis). \
   If verified ODD says lanes_opposite=0 but the scene analysis or video shows vehicles traveling \
   in the opposite direction, report the actual opposing lanes observed — do NOT write "no opposing traffic".
4. Describes each visible road user with their position, action, and relevance to the ego-vehicle.
5. Preserves specific events and challenges from the scene analysis.
6. Describes events chronologically (beginning → middle → end of clip).
7. Is 150–300 words, third-person past tense ("The ego-vehicle...").

Output the caption text ONLY — no JSON, no headers, no checklist, no explanation."""
