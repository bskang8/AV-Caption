# NVIDIA PhysicalAI-AV Egomotion (Waypoint / Rotation) 다운로드 가이드

## Waypoint / Rotation 데이터 위치

| 항목 | 경로 | 형식 |
|------|------|------|
| **Egomotion (waypoint+rot)** | `labels/egomotion/` | Parquet chunk 파일 |
| **Egomotion 정밀판** | `labels/egomotion.offline/` | Parquet chunk 파일 (LiDAR 기반 정밀 추정, 298k 클립) |

**데이터 스키마** (`EgomotionState` 클래스 기준):
```
pose         → RigidTransform (x, y, z 위치 + qx, qy, qz, qw 쿼터니언)
velocity     → [vx, vy, vz]
acceleration → [ax, ay, az]
curvature    → 스칼라 (차선 변경·회전 감지에 핵심)
```

- **Waypoint = pose의 x, y, z** (timestamp 0 기준 상대 좌표)
- **Rotation = pose의 쿼터니언 [qw, qx, qy, qz]** → yaw(heading) 변화로 변환 가능
- 좌표계: 클립 시작 시점(timestamp 0) 기준 로컬 좌표, yaw=0

---

## 다운로드 방법

### 1단계: 인증 설정 (최초 1회)

```bash
pip install huggingface_hub physical_ai_av

# HuggingFace 로그인 (토큰 필요)
huggingface-cli login
# → https://huggingface.co/settings/tokens 에서 토큰 생성
# → 데이터셋 페이지에서 NVIDIA 라이선스 동의 필수
```

### 2단계: Egomotion 청크 파일만 다운로드

```bash
# egomotion.offline 전체 (더 정확, 298k 클립)
huggingface-cli download nvidia/PhysicalAI-Autonomous-Vehicles \
    --repo-type dataset \
    --local-dir ./cds-data \
    --include "labels/egomotion.offline/*"

# 용량이 부담되면 일부 청크만 먼저 확인
huggingface-cli download nvidia/PhysicalAI-Autonomous-Vehicles \
    --repo-type dataset \
    --local-dir ./cds-data \
    --include "labels/egomotion.offline/egomotion.offline.chunk_0000.parquet"
```

### 3단계: clip_id로 waypoint/rotation 추출

```python
import pandas as pd
import numpy as np
from pathlib import Path

def quat_to_yaw(qw, qx, qy, qz):
    """쿼터니언 → yaw 각도(도) 변환."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.degrees(np.arctan2(siny, cosy))

def load_egomotion_for_clips(clip_ids: list[str], egomotion_dir: str) -> dict:
    """clip_id 목록에 해당하는 waypoint/rotation 데이터를 반환."""
    egomotion_dir = Path(egomotion_dir)
    clip_set = set(clip_ids)
    result = {}

    for parquet_file in sorted(egomotion_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet_file)

        # 해당 청크에 우리 clip_id가 있는지 확인
        matched = df[df["clip_uuid"].isin(clip_set)]
        if matched.empty:
            continue

        for clip_id, group in matched.groupby("clip_uuid"):
            group = group.sort_values("timestamp_ns")

            # waypoint: x, y (z는 보통 무시)
            waypoints = group[["timestamp_ns", "x", "y", "z"]].to_dict("records")

            # rotation → yaw 변환
            group["yaw_deg"] = quat_to_yaw(
                group["qw"], group["qx"], group["qy"], group["qz"]
            )
            group["delta_yaw"] = group["yaw_deg"].diff().fillna(0)

            result[clip_id] = {
                "waypoints": waypoints,
                "yaw": group[["timestamp_ns", "yaw_deg", "delta_yaw"]].to_dict("records"),
            }

        # 모두 찾으면 조기 종료
        if len(result) == len(clip_ids):
            break

    return result


# 사용 예시
if __name__ == "__main__":
    import json

    clip_ids = json.loads(open("caption_refine/my_clips.json").read())
    data = load_egomotion_for_clips(
        clip_ids,
        egomotion_dir="./cds-data/labels/egomotion.offline"
    )

    for clip_id, motion in data.items():
        print(f"\n[{clip_id[:8]}]")
        print(f"  waypoints: {len(motion['waypoints'])} points")
        print(f"  yaw range: {min(r['yaw_deg'] for r in motion['yaw']):.1f}° ~ "
              f"{max(r['yaw_deg'] for r in motion['yaw']):.1f}°")
        print(f"  max Δyaw: {max(abs(r['delta_yaw']) for r in motion['yaw']):.1f}°/step")
```

---

## Parquet 컬럼 실제 확인 방법 (다운로드 후)

```bash
python -c "
import pandas as pd
df = pd.read_parquet('./cds-data/labels/egomotion.offline/egomotion.offline.chunk_0000.parquet')
print(df.columns.tolist())
print(df.dtypes)
print(df.head(2))
"
```

컬럼명이 확인되면 `qw/qx/qy/qz` 또는 `rotation_w/x/y/z` 등 실제 이름에 맞게 코드를 조정하면 됩니다.

---

## 주의사항

| 항목 | 내용 |
|------|------|
| **라이선스** | 데이터셋 페이지에서 NVIDIA 라이선스 직접 동의 필수 (없으면 403) |
| **용량** | egomotion.offline 전체 = 수십 GB 예상 (청크 단위 선택 가능) |
| **정밀도** | `egomotion.offline` > `egomotion` (LiDAR 기반 후처리, 회전 정확도 더 높음) |
| **좌표** | 절대 GPS 아닌 클립 시작점 기준 상대 좌표 → 모델에 전달하기에 적합 |
