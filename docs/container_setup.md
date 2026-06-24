# 컨테이너 교체 후 vllm 환경 복구 가이드

컨테이너가 새로 생성될 때마다 `.venv`의 Python 경로가 깨지는 문제를 해결하는 절차입니다.

---

## 문제 원인

`.venv`는 마운트된 디스크(`MyDisk`)에 있어 컨테이너가 교체되어도 파일은 남지만,
내부적으로 이전 컨테이너의 Python 경로를 하드코딩하고 있어 실행이 불가능해집니다.

| 깨지는 항목 | 위치 | 증상 |
|-------------|------|------|
| Python 심볼릭 링크 | `.venv/bin/python*` | `bad interpreter: No such file or directory` |
| pyvenv.cfg home 경로 | `.venv/pyvenv.cfg` | `No module named '_ctypes'` |

---

## 1회성 사전 준비 (계정/권한)

### HuggingFace 모델 접근 승인

`nvidia/Cosmos-Reason2-2B`는 Gated Repository입니다.
아래 페이지에서 **Accept** 버튼을 눌러 접근 승인을 받아야 합니다.

```
https://huggingface.co/nvidia/Cosmos-Reason2-2B
```

### HuggingFace 토큰 발급

```
https://huggingface.co/settings/tokens
```

`Read` 권한 이상의 토큰을 발급받아 보관해둡니다.

---

## 컨테이너 교체 시 복구 절차

### Step 1: Python 3.12 설치

Ubuntu 22.04 기본 저장소에는 Python 3.12가 없으므로 deadsnakes PPA를 추가합니다.

```bash
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y python3.12 python3.12-venv python3.12-dev

# 설치 확인
/usr/bin/python3.12 --version
```

### Step 2: venv Python 심볼릭 링크 수정

`$(which python3.12)` 대신 반드시 절대경로를 사용합니다.
(uv가 설치된 환경에서 `which` 명령이 uv Python을 반환할 수 있습니다.)

```bash
ln -sf /usr/bin/python3.12 /root/kadap/MyDisk/cosmos-reason2/.venv/bin/python
ln -sf /usr/bin/python3.12 /root/kadap/MyDisk/cosmos-reason2/.venv/bin/python3
ln -sf /usr/bin/python3.12 /root/kadap/MyDisk/cosmos-reason2/.venv/bin/python3.12

# 확인
ls -la /root/kadap/MyDisk/cosmos-reason2/.venv/bin/python*
```

### Step 3: pyvenv.cfg home 경로 수정

venv는 `pyvenv.cfg`의 `home` 값으로 표준 라이브러리 경로를 결정합니다.
이 값이 이전 컨테이너(또는 uv)의 Python 경로를 가리키면 `_ctypes` 등 표준 모듈 임포트가 실패합니다.

```bash
sed -i 's|^home = .*|home = /usr/bin|' /root/kadap/MyDisk/cosmos-reason2/.venv/pyvenv.cfg

# 확인
cat /root/kadap/MyDisk/cosmos-reason2/.venv/pyvenv.cfg
# home = /usr/bin 이어야 함
```

### Step 4: pip 및 setuptools 설치

Python 심볼릭 링크 교체 후 venv 내부의 pip이 끊어집니다.
`ensurepip`으로 pip을 먼저 복구한 뒤 setuptools를 설치해야 합니다.

> **주의**: `pip install setuptools`가 "already satisfied"를 출력해도 실제 패키지 파일이 없는 경우가 있습니다.
> dist-info 메타데이터만 남고 실제 `setuptools/` 디렉토리가 유실된 상태입니다. 반드시 `--force-reinstall`을 사용하세요.

```bash
source /root/kadap/MyDisk/cosmos-reason2/.venv/bin/activate

# pip 부트스트랩 (인터넷 불필요, Python 표준 라이브러리 내장)
python -m ensurepip --upgrade

# setuptools 설치 (--force-reinstall: 메타데이터만 있고 실제 파일 없는 경우 대비)
# vllm 0.12.0은 setuptools<81.0.0 을 요구하므로 버전 범위를 명시
python -m pip install "setuptools>=77.0.3,<81.0.0" --force-reinstall

# 확인
python -c "import setuptools; print(setuptools.__version__)"
```

### Step 5: HuggingFace 로그인

```bash
source /root/kadap/MyDisk/cosmos-reason2/.venv/bin/activate
huggingface-cli login
# 프롬프트에 토큰 붙여넣기
```

모델이 이미 캐시에 있는 경우 로그인 없이도 실행 가능합니다.

```bash
# 캐시 확인
ls ~/.cache/huggingface/hub/ | grep -i cosmos
```

### Step 6: vllm 실행

```bash
source /root/kadap/MyDisk/cosmos-reason2/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0 nohup vllm serve nvidia/Cosmos-Reason2-2B \
    --port 8000 --max-model-len 16384 \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --reasoning-parser qwen3 \
    > vllm_8000.log 2>&1 &
echo "PID: $!"
```

---

## 자동화 스크립트

위 절차를 한 번에 실행하는 스크립트입니다.
컨테이너 교체 후 이 스크립트 하나만 실행하면 됩니다.

```bash
#!/bin/bash
set -e

VENV=/root/kadap/MyDisk/cosmos-reason2/.venv

echo "=== Step 1: Python 3.12 설치 ==="
if ! /usr/bin/python3.12 --version &>/dev/null; then
    apt-get install -y software-properties-common
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get update
    apt-get install -y python3.12 python3.12-venv python3.12-dev
else
    echo "Python 3.12 이미 설치됨: $(/usr/bin/python3.12 --version)"
fi

echo "=== Step 2: 심볼릭 링크 수정 ==="
ln -sf /usr/bin/python3.12 "$VENV/bin/python"
ln -sf /usr/bin/python3.12 "$VENV/bin/python3"
ln -sf /usr/bin/python3.12 "$VENV/bin/python3.12"

echo "=== Step 3: pyvenv.cfg 수정 ==="
sed -i 's|^home = .*|home = /usr/bin|' "$VENV/pyvenv.cfg"
echo "현재 pyvenv.cfg:"
cat "$VENV/pyvenv.cfg"

echo "=== Step 4: pip 및 setuptools 설치 ==="
source "$VENV/bin/activate"
python -m ensurepip --upgrade
python -m pip install "setuptools>=77.0.3,<81.0.0" --force-reinstall

echo "=== 완료 ==="
echo "다음 명령으로 vllm을 실행하세요:"
echo "  source $VENV/bin/activate"
echo "  huggingface-cli login  # 캐시 없을 경우에만"
echo "  CUDA_VISIBLE_DEVICES=0 vllm serve nvidia/Cosmos-Reason2-2B --port 8000 ..."
```

스크립트 저장 및 실행:

```bash
# 저장
vi /root/kadap/MyDisk/cosmos-reason2/fix_venv.sh

# 실행 권한 부여 후 실행
chmod +x /root/kadap/MyDisk/cosmos-reason2/fix_venv.sh
bash /root/kadap/MyDisk/cosmos-reason2/fix_venv.sh
```

---

## 실행 확인

```bash
# 프로세스 확인
ps aux | grep vllm

# 로그 실시간 확인 (서버 준비 완료까지 수 분 소요)
tail -f /root/kadap/MyDisk/cosmos-reason2/vllm_8000.log

# API 서버 응답 확인 ("Application startup complete." 이후)
curl http://localhost:8000/health
```

---

## 에러별 원인 요약

| 에러 메시지 | 원인 | 해결 |
|------------|------|------|
| `bad interpreter: No such file or directory` | `.venv/bin/python` 심볼릭 링크 broken | Step 2 |
| `No module named '_ctypes'` | `pyvenv.cfg home`이 uv Python 경로를 가리킴 | Step 3 |
| `No module named 'vllm'` | 시스템 Python으로 실행 (venv 미활성화) | `source .venv/bin/activate` 후 실행 |
| `No module named pip` | Python 교체 후 venv의 pip 끊어짐 | Step 4 (`ensurepip`) |
| `No module named 'setuptools'` | deadsnakes Python에 setuptools 미포함, 또는 dist-info만 있고 실제 파일 유실 | Step 4 (`--force-reinstall`) |
| `pip` says "already satisfied" but `import setuptools` fails | dist-info 메타데이터만 존재, 실제 `setuptools/` 디렉토리 없음 | `python -m pip install setuptools --force-reinstall` |
| `setuptools 82.x` 설치 후 vllm dependency conflict 경고 | vllm 0.12.0이 `setuptools<81.0.0` 요구 | `python -m pip install "setuptools>=77.0.3,<81.0.0"` |
| `GatedRepoError: 401` | HuggingFace 인증 없음 | Step 5 |
