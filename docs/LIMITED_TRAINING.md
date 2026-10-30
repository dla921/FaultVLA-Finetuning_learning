> LEGACY: This document describes the previous Cartesian/wide200 experiment. For current training use README.md and docs/STABLEGRASP200.md.

﻿# 8GB GPU용 최소 출력층 학습

2026-10-27. 현재 구현은 **행동 헤드의 마지막 선형층(fc2)만 학습하는 제한 실험**이다.
백본, 행동 헤드의 나머지 부분, proprio projector는 동결한다. 전체 헤드 파인튜닝이나
LoRA가 아니며, 이 결과만으로 컵 운반 정책이 학습됐다고 판단하지 않는다.

## 완료된 실제 시험

`results/limited_smoke_01/summary.json`과 `results/limited_rollout_smoke_01/summary.json`:

| 항목 | 실측 결과 |
| --- | --- |
| GPU | RTX 4060 8GB |
| 학습 대상 | fc2의 6,279개 파라미터 |
| 배치 / 행동 청크 | 배치 1 / 8개 연속 행동 |
| 시험 데이터 | train_00의 16개 시작 시점, val_00의 16개 시작 시점 |
| 학습 | 10 epoch, 실제 backward와 AdamW 갱신 |
| 학습 normalized L1 | 0.420624 → 0.239910 |
| 검증 normalized L1 | 0.397256 → 최저 0.228807 (epoch 8) |
| PyTorch 최대 할당 VRAM | 3,640.59MiB (약 3.56GiB) |
| PyTorch 최대 예약 VRAM | 4,074MiB (약 3.98GiB) |
| 동결 파라미터의 gradient | 없음 |
| 저장 후 온라인 추론 일치 검사 | 통과 |
| 새 프로세스 체크포인트 로드 + MuJoCo | 실제 VLA 출력으로 5/5스텝 완료, 오류/IK 거부/클리핑 0 |

메모리 수치는 해당 프로세스의 PyTorch 통계이며 디스플레이·드라이버 등 전체 GPU 사용량과는 다르다.
5스텝은 0.25초 연결 검사다. 운반 성공은 평가하지 않아 `task_success: null`이다.
`final_metrics.success: false`는 5스텝 종료 시점의 조건만 나타낸다.
정규화·에피소드 경계·분할 격리 단위 검사 3개도 통과했다(`results/limited_training_tests.txt`).

사용자 재개 요청 후 확인 결과, 학습과 연결 검사는 이미 정상 종료되어 재학습하지 않았다.
남은 문서화만 이어서 완료했다. 실패하거나 중단된 학습을 성공으로 간주한 것이 아니다.

## 학습 방법

WSL Ubuntu 터미널에서 아래를 실행한다. 모든 출력 디렉터리는 **새 경로**여야 한다.
이미 존재하면 스크립트가 종료하며 기존 결과를 덮어쓰지 않는다.

```bash
conda activate FaultVLA
cd /home/ubuntu/FaultVLA

# 첫 단계: 시연 한 개의 전체 유효 관측으로 제한 학습
# train_00/val_00 각각 463개 시작 시점, 각 시점당 8개 행동
python scripts/train_limited.py \
  --train-episodes train_00 \
  --validation-episodes val_00 \
  --all-samples \
  --epochs 20 \
  --learning-rate 0.0001 \
  --output results/limited_train_01
```

이 명령은 아직 실행하지 않았다. 완료된 것은 위의 16+16개 관측 최소 시험이다.
전체 시연 학습 시간은 실제로 측정하지 않았다. 이미지 특징 추출이 먼저 진행되어
`Features train: ...`가 출력되고, 이후 epoch별 train/validation 손실이 출력된다.

검증 손실이 가장 낮은 가중치를 `checkpoint/`로 내보낸다. epoch 0보다 좋아지지 않으면
초기 출력층이 선택될 수 있으므로 `best_validation_epoch`를 확인한다. 손실이 내려가더라도
실제 운반 성공을 의미하지 않으므로 아래 정책 평가가 필요하다.

```bash
# 학습 배치에서 실제 정책이 작업을 재현하는지 검사
python scripts/evaluate_limited.py \
  --training-run results/limited_train_01 \
  --episode train_00 \
  --output results/limited_train_01_rollout_train

# 별도 검증 배치에서 평가
python scripts/evaluate_limited.py \
  --training-run results/limited_train_01 \
  --episode val_00 \
  --output results/limited_train_01_rollout_val
```

기본 470스텝(23.5초)이며 `--viewer`를 추가하면 MuJoCo 창을 표시한다.
`--steps 5`는 연결 검사만 수행하고 작업 성공을 판정하지 않는다.
정상 종료지만 470스텝 작업 실패일 때 종료코드는 2, 예외 발생은 1이다.
결과는 각 출력 경로의 `summary.json`, 실제 모델 행동은 `actions.jsonl`, 관측은 `rgb/`에 저장한다.

이 평가기는 시연의 **첫 물리 상태만** 초기화에 사용한다. 이후 전문가 궤적·행동을 읽거나
재생하지 않고 매 스텝 현재 RGB·상태·명령으로 VLA를 호출한다. 시연과 동일하게 그림자·반사·MSAA를
끈 224×224 front/wrist 관측을 사용한다. 원래 실행기 `run_vla.py`의 기본 렌더링/초기 자세와
차이가 있으므로 이번 제한 실험은 제공한 평가기를 사용한다.

첫 시연 학습의 손실 및 작업 결과를 확인한 뒤, 학습 범위를 늘리려면:

```bash
python scripts/train_limited.py \
  --train-episodes train_00 train_01 train_02 train_03 train_04 train_05 train_06 train_07 \
  --validation-episodes val_00 val_01 \
  --all-samples --epochs 20 \
  --output results/limited_train_all_01
```

테스트 배치(test_00/test_01)는 학습 또는 체크포인트 선택에 사용하지 않는다.
설정을 확정한 다음 평가기의 `--episode test_00`과 `--episode test_01`로 최종 검사한다.

## 데이터·학습 계약

- `data/cartesian_demos_v1`의 성공 시연만 사용한다. RLDS 변환 없이 이 최소 학습기가 직접 읽는다.
  기존 표준 VLA 학습 스크립트가 이 폴더를 그대로 읽을 수 있다는 의미는 아니다.
- 정규화 통계는 **train 분할 8개만** 사용하며 validation/test는 제외한다. 한 시연만 학습할 때도
  통계는 동일한 train 8개에서 계산한다. key는 `faultvla_normal_v1`이다.
- 행동 처음 6개는 q01/q99로 [-1,1] 정규화하고 범위 밖은 잘라낸다. 손은 0/1 그대로다.
  8D 상태도 train 분할 통계를 적용한다. 거의 고정된 차원은 작은 폭을 부여해 0으로 나누지 않는다.
- 현재 관측에서 시작하는 8개 연속 행동을 목표로 사용한다. 에피소드 끝을 넘어가는 시작 시점은
  제외하므로 470행당 463개 청크다. 다른 에피소드의 행동을 이어 붙이지 않는다.
- 실제 체크포인트의 이미지 처리·Qwen 명령·상태 입력 경로로 fc2 직전 특징을 추출한다.
  모든 앞단을 동결하므로 이 특징을 CPU에 보관한 뒤 출력층에만 역전파한다.
- 행동 출력층은 FP32로 학습하고 기존 헤드와 같은 BF16으로 내보낸다.
  기존 체크포인트의 파일과 원본 데모는 수정하지 않는다.
- 출력 체크포인트는 큰 백본 파일을 원본 체크포인트에 대한 심볼릭 링크로 참조한다.
  **원본 `checkpoints/LIBERO-Spatial-Pro`를 삭제하거나 옮기면 안 된다.** 다른 장비로 복사할 때는
  링크 대상도 함께 복사해야 한다.

## 파일과 제한

- `scripts/train_limited.py`: 특징 추출, 출력층 학습, 검증 선택, 체크포인트 내보내기.
- `src/faultvla/limited_training.py`: train 전용 통계, 행동 정규화, 에피소드 내부 청크.
- `scripts/evaluate_limited.py`: 시연과 일치하는 렌더링/초기화에서 실제 VLA 폐루프 평가.
- 각 학습 결과: `settings.json`, `dataset_statistics.json`, `features.pt`, `history.json`,
  `output_layer_training.pt`, `summary.json`, `inference_config.json`, `checkpoint/`.

현재 학습 스크립트는 중간 재개 옵션을 제공하지 않는다. 실행이 실제로 중단되어 summary와
체크포인트가 완성되지 않았다면 새 출력 경로로 다시 실행한다. 이번 최소 시험은 완료되었으므로
재실행할 필요가 없다. 오래 걸리는 대규모 학습용 복구 기능까지 구현한 것은 아니다.

이 방법은 선형 출력 매핑만 바꾸므로 표현력에 한계가 있다. 컵 운반이 실패하면 epoch 수만
늘려 해결된다고 가정하지 말고, 닫힌 루프의 손 타이밍·행동 오차를 확인해 학습 대상 확대나
시연 보강을 판단해야 한다. 전체 헤드 학습의 8GB 수용 여부와 정상 작업 성공률은 아직 미검증이다.
Phase A/B와 고장 주입은 추가하지 않았다.
