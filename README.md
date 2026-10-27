# FaultVLA-Finetuning_learning

SO-101 팔 5축 + AmazingHand 손 8축의 정상 컵 운반을 위한 서버 이전 패키지.
기본 시작 모델은 **VLA-Adapter/LIBERO-Object-Pro**이며, 검증된 정상 시연 200개를 사용합니다.

**학습 범위:** 백본과 대부분의 행동 헤드를 동결하고 마지막 `fc2` 출력층만 학습합니다.
전체 모델/전체 행동 헤드/LoRA 학습이나 저수준 모터 제어 학습은 아닙니다.
컵을 다른 테이블로 옮기는 데이터이며 바구니 투입, 고장 주입, Phase A/B는 포함하지 않습니다.

## 서버에서 시작하기

Ubuntu Linux x86_64, Conda, Git, GitHub CLI(`gh`), NVIDIA 드라이버와 CUDA BF16 지원 GPU가 필요합니다.
현재 코드의 PyTorch는 2.2.0/CUDA 12.1 계열입니다. 더 새로운 GPU는 별도 PyTorch 호환성 검사가 필요합니다.
전체 데이터 특징 캐시를 위해 RAM 16GB 이상(32GB 권장), 여유 디스크 20GB 이상을 확보하세요.
8GB GPU에서는 기존 소량 출력층 학습을 검증했으며, 서버 사양에서의 전체 실행 시간/성공률은 미측정입니다.

```bash
# 비공개 저장소 읽기 권한이 있는 계정으로 로그인
# GitHub CLI가 없다면 https://cli.github.com/ 의 설치 안내를 따릅니다.
gh auth login
gh repo clone dla921/FaultVLA-Finetuning_learning
cd FaultVLA-Finetuning_learning

conda create -n FaultVLA-Finetuning python=3.10 pip -y
conda activate FaultVLA-Finetuning
bash server/install.sh

# 비공개 Release의 4개 압축 파일 다운로드 → SHA-256 검사 → 데이터 복원
python server/restore_data.py

# 공식 Hugging Face의 고정 리비전 다운로드 (약 2.8GB)
python server/download_checkpoint.py --model object
python server/preflight.py --model object --inference

# 실제 역전파/체크포인트 저장 연결 검사: 시연 1개씩, 2개 표본씩
python server/train.py --model object --smoke --epochs 1 --output results/object_smoke

# train 160개·validation 20개 전체로 출력층 학습
python server/train.py --model object --epochs 20 --output results/object_wide200
```

출력 디렉터리는 매번 새 경로여야 합니다. 기존 결과를 덮어쓰지 않습니다.
특징 추출이 오래 걸릴 수 있으므로 서버에서는 `tmux` 등의 세션 유지 도구를 사용하세요.
현재 학습기는 중간 특징 추출/epoch 재개를 지원하지 않습니다.

인터넷 연결은 설치·Release 다운로드·체크포인트 다운로드에 필요합니다. 학습 중에는 로컬 파일을 사용합니다.
접근 토큰을 저장소 파일에 쓰지 마세요. `gh auth login` 인증은 사용 중인 서버의 GitHub CLI가 관리합니다.

## 학습 후 실제 VLA 평가

헤드리스 서버는 EGL을 사용합니다. 필요하면 관리자가 `libegl1 libgl1 libglfw3` 시스템 패키지를 설치해야 합니다.

```bash
MUJOCO_GL=egl python scripts/check_renderer.py
python scripts/evaluate_limited.py \
  --data data/cartesian_demos_v2_wide200 \
  --training-run results/object_wide200 \
  --episode validation_0000 --render-backend egl \
  --output results/object_wide200_validation0000
```

470스텝 전체 평가 후 `summary.json`의 `task_success`를 확인합니다. 성공 종료코드 0, 작업 실패 2, 오류 1입니다.
`--steps 5`는 연결 검사이며 작업 성공을 판단하지 않습니다.
평가기는 초기 상태만 시연에서 읽고 이후 실제 VLA 출력으로 움직입니다. 전문가 고정 궤적을 실행하지 않습니다.
시연 이미지는 WSL GLFW/D3D12에서 생성됐으므로 서버 EGL 이미지와 픽셀 단위 동일성을 보장하지 않습니다.
카메라·해상도·그림자/반사 설정은 유지하며 평가 backend를 결과에 별도 기록합니다.

## 데이터와 모델

- 데이터: `data/cartesian_demos_v2_wide200/`, train 160 / validation 20 / test 20.
- 200 × 470스텝 = 94,000 관측·행동 쌍; front/wrist RGB 188,000장.
- 테이블 가장자리 간격 11.03~13.97cm. 실패 시도와 중단된 부분 폴더는 학습 패키지에 넣지 않았습니다.
- `configs/data_release.json`: Release 자산 이름·크기·SHA-256. 대용량 데이터는 Git 이력이 아닌 **같은 비공개 저장소의 Release**에 있습니다.
- `configs/checkpoints.json`: Object-Pro/Spatial-Pro의 공식 저장소와 고정 리비전. 모델 가중치는 GitHub에 중복 업로드하지 않습니다.
- Object-Pro: `afd5c551e4851fb434deaba272c7688e87586cc4`, 통계 키 `libero_object_no_noops`.
- 새 행동·상태 정규화는 항상 train 160개에서만 계산합니다. validation/test를 통계나 학습에 섞지 않습니다.
- 손 열기/닫기는 이진 행동이며 8개 손 모터를 독립 예측하지 않습니다.
- 명령 관절 제한은 준수했지만 실제 손목은 MuJoCo soft constraint 때문에 최대 약 0.0597도 초과했습니다.

전체 이미지 무결성·정렬 재검사:
```bash
python scripts/validate_wide_dataset.py
python -m unittest discover -s tests -v
```

Spatial-Pro로 비교하려면 `server/download_checkpoint.py --model spatial`과 `server/train.py --model spatial ...`을 사용합니다.
출력 체크포인트는 원본 가중치를 심볼릭 링크로 참조합니다. 원본 `checkpoints/`를 유지하세요.
학습 완료 결과를 다른 장비에 옮길 때는 `cp -aL 결과/checkpoint 목적지`처럼 링크를 실제 파일로 복사해야 합니다.
평가기는 `--training-run` 아래 checkpoint를 사용하므로 학습 결과 설정에 남은 이전 절대 경로에 의존하지 않습니다.

## 포함 파일과 검증 범위

- `server/`: 설치, 데이터 복원, 체크포인트 다운로드, 사전 검사, 학습 실행기.
- `src/faultvla/`, `scripts/`, `configs/`: Action Adapter와 학습·평가 코드.
- `environments/so101_wrist_side/`: 원본 MJCF·메시·제어 코드.
- `VLA-Adapter/`: 사용한 upstream 커밋의 실행 소스 스냅샷. MIT LICENSE 유지.
- `reports/`: 수집·행동 공간 감사, 소스 버전, 패키지 시험 결과.
- `docs/`: 이전 실험 기록. 과거 Spatial-Pro/v1 데이터 명령보다 이 README의 서버 명령을 우선하세요.

구체적인 시험은 `reports/server_package_validation.json`을 참고하세요.
새 서버에서의 환경 신규 설치, 장시간 전체 학습, 학습 후 운반 성공률, 실물 로봇은 별도 검증 대상입니다.
원본 Object-Pro가 SO-101에 학습된 모델이라는 의미는 아닙니다.

## 출처

[공식 LIBERO Object-Pro 모델](https://huggingface.co/VLA-Adapter/LIBERO-Object-Pro),
[VLA-Adapter 소스](https://github.com/OpenHelix-Team/VLA-Adapter).
이 패키지의 로봇 환경과 데이터는 사용자가 제공한 실험 자산입니다. 비공개 저장소로 보관합니다.
