# SPDX-FileCopyrightText: Copyright (c) 2026 Byungsu Kang. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
4개 Stage별 프롬프트 템플릿.
각 함수는 최종 user 메시지 문자열을 반환한다.
"""


SYSTEM_PROMPT = (
    "You are an expert autonomous driving video analyst. "
    "You reason carefully about what you observe in the video before answering. "
    "Always ground your answers in visual evidence from the video."
)


def stage1_grounding(sentences: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    n = len(sentences)
    return f"""\
Watch these driving video frames carefully.

Here is an existing caption split into {n} numbered sentences:

{numbered}

Your tasks:
A) For each sentence that makes a specific, verifiable claim about what is visible \
(a particular object, sign, action, or event), check the video frames for supporting \
evidence. If you can find NO visual evidence supporting a specific claim \
(e.g., a sign described is not visible in any frame, an action did not occur, \
a described object is not present), mark that sentence as ungrounded.

B) List important observations clearly visible in the frames that the caption does NOT \
mention at all. Be specific with location and timing.

Reply ONLY with valid JSON matching this schema:

{{
  "hallucinated": [
    {{"sentence_num": <integer>, "reason": "<what the sentence claims vs what you actually see>"}}
  ],
  "missed": [
    "<specific observation visible in frames, not covered by any of the {n} sentences>"
  ]
}}

Rules:
- hallucinated: flag a sentence if its specific claim (object, sign, action) has NO \
supporting evidence in the frames. You do NOT need to be 100% certain it is wrong — \
if you cannot find it anywhere in the frames, flag it with your observation.
- missed: only include things clearly visible in the frames and completely absent from \
the caption. Max 5 items, most significant first.
- Return empty arrays if all claims are visually confirmed and nothing significant is missing.
- Do NOT add commentary outside the JSON."""


def stage2_extract() -> str:
    return """\
Watch this autonomous driving video carefully.

Think step by step about each field before committing to an answer, then reply ONLY with valid JSON:

{
  "time_of_day": {
    "value": "<day|night|dawn|dusk|unknown>",
    "confidence": 0.0,
    "evidence": "brief visual cue (e.g. sun position, streetlight state, sky color)"
  },
  "weather": {
    "value": "<clear|cloudy|rainy|foggy|snowy|unknown>",
    "confidence": 0.0,
    "evidence": "brief visual cue"
  },
  "road_type": {
    "value": "<highway|urban|intersection|rural|parking_lot|tunnel|bridge|unknown>",
    "confidence": 0.0,
    "evidence": "brief visual cue"
  },
  "num_lanes": {
    "value": null,
    "confidence": 0.0,
    "evidence": "how you counted"
  },
  "ego_lane_position": {
    "value": "<leftmost|second_from_left|center|second_from_right|rightmost|unknown>",
    "confidence": 0.0,
    "evidence": "brief visual cue"
  },
  "road_surface": {
    "value": "<dry|wet|icy|unpaved|unknown>",
    "confidence": 0.0,
    "evidence": "brief visual cue"
  },
  "road_markings": {
    "value": ["<lane_lines|crosswalk|stop_line|turn_arrow|bicycle_lane|none>"],
    "confidence": 0.0
  },
  "traffic_density": {
    "value": "<free|light|moderate|congested|unknown>",
    "confidence": 0.0,
    "evidence": "brief visual cue"
  },
  "surrounding_vehicles": {
    "types": ["<car|truck|bus|motorcycle|cyclist|emergency_vehicle|none>"],
    "count_estimate": null,
    "notable_behaviors": ["e.g. stalled vehicle blocking lane, vehicle running red light"],
    "confidence": 0.0
  },
  "ego_actions": {
    "value": ["<straight|braking|lane_change|left_turn|right_turn|stopping|u_turn|reversing>"],
    "confidence": 0.0,
    "evidence": "brief visual cue"
  },
  "pedestrians": {
    "present": false,
    "count_estimate": null,
    "behavior": "e.g. crossing, waiting at curb, jaywalking",
    "confidence": 0.0
  },
  "traffic_signals": {
    "present": false,
    "state": "<red|yellow|green|not_visible|none>",
    "confidence": 0.0,
    "evidence": "brief visual cue"
  },
  "road_signs": {
    "types": ["<speed_limit|warning|direction|prohibition|information|none>"],
    "details": "e.g. speed limit 60, railway crossing warning",
    "confidence": 0.0
  },
  "hazard_level": {
    "value": "<low|medium|high|unknown>",
    "rationale": "why this hazard level",
    "confidence": 0.0
  },
  "lighting_condition": {
    "value": "<daylight|artificial|mixed|dark|unknown>",
    "confidence": 0.0,
    "evidence": "brief visual cue"
  }
}

Rules:
- num_lanes must be an integer or null (not a string).
- count_estimate must be an integer or null.
- All confidence values must be between 0.0 and 1.0.
- Do NOT add commentary outside the JSON."""


def stage3_verify(low_confidence_fields: dict) -> str:
    fields_text = "\n".join(
        f"- {field}: currently '{info.get('value', info.get('present', '?'))}' "
        f"(confidence {info.get('confidence', '?'):.2f}) — evidence: {info.get('evidence', info.get('rationale', ''))}"
        for field, info in low_confidence_fields.items()
    )
    return f"""\
Watch this video again and focus ONLY on the fields listed below, which had low confidence:

{fields_text}

For each field:
1. Describe exactly what you observe in the video relevant to this field.
2. State your verdict: CONFIRM (original answer is correct) or CORRECT (provide new value).

Reply ONLY with valid JSON:

{{
  "field_name": {{
    "observation": "what you see in the video",
    "verdict": "<CONFIRM|CORRECT>",
    "corrected_value": null
  }}
}}

- corrected_value must match the allowed values for that field (see Stage 2 schema).
- Set corrected_value to null when verdict is CONFIRM.
- Do NOT include fields not listed above.
- Do NOT add commentary outside the JSON."""


def _compact_odd(verified_odd: dict) -> str:
    """confidence/evidence 없이 핵심 값만 추출해 짧은 요약 생성."""
    lines = []
    for key, val in verified_odd.items():
        if not isinstance(val, dict):
            continue
        v = val.get("value", val.get("present", val.get("types", "?")))
        lines.append(f"  {key}: {v}")
    return "\n".join(lines) if lines else "  (no data)"


def stage4_refine(
    original_caption: str,
    verified_odd: dict,
    hallucinated: list[str],
    missed: list[str],
) -> str:
    odd_summary = _compact_odd(verified_odd)
    hal_text = "\n".join(f"- {h}" for h in hallucinated) if hallucinated else "None"
    miss_text = "\n".join(f"- {m}" for m in missed) if missed else "None"

    return f"""\
Watch this driving video. You will write a corrected, factual caption.

=== Original caption (may contain errors) ===
{original_caption}

=== Verified scene information ===
{odd_summary}

=== Claims from original caption NOT supported by the video ===
{hal_text}

=== Important events visible in video but missing from original caption ===
{miss_text}

Write a refined caption that:
1. Removes all hallucinated content listed above.
2. Naturally incorporates the verified scene information (weather, road type, time of day, etc.).
3. Preserves grounded content from the original caption.
4. Adds any missed observations.
5. Describes events chronologically (beginning → middle → end of clip).
6. Is 150–300 words.
7. Uses third-person past tense ("The ego-vehicle...").

Output the caption text ONLY — no JSON, no headers, no explanation."""