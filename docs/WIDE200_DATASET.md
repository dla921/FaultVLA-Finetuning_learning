# 넓어진 테이블 정상 시연 200개

검증을 완료한 데이터: `data/cartesian_demos_v2_wide200/`.
기존 12개 시연과 원본 MJCF는 보존했다. 이번 작업은 정상 전문가 시연 수집이며
VLA 학습, 고장 주입, Phase A/B 알고리즘 추가는 수행하지 않았다.

## 거리 및 수집 범위

- 기존 테이블 중심 y 간격 16cm, 가장자리 사이 간격 6cm.
- 새 중심 y 간격 약 21~24cm, 실제 포함 시연의 가장자리 간격
  **11.03~13.97cm**.
- 테이블 크기는 동일하다. 첫 테이블은 고정하고 목적 테이블 위치만 초기화 때 변경한다.
- 컵 시작 x=0.260~0.268m, y=-0.086~-0.074m; 목표 x=0.254~0.266m, y=0.130~0.160m.
- 목표 y=0.12/0.14/0.16m의 기준 시험은 성공했고 0.18m는 원본 전문가부터 실패했다.
  실제 수집 범위는 성공 시험이 확인된 범위 안에서 정했다.
- 위치 조합은 seed=20261027로 미리 생성하고 분할을 지정했다. 실패하면 같은 분할의 다음
  후보를 사용한다. 성공 여부를 보고 train/test 사이로 에피소드를 옮기지 않는다.

## 실제 결과

| 분할 | 시연 | 관측·행동 쌍 | RGB |
| --- | ---: | ---: | ---: |
| train | 160 | 75,200 | 150,400 |
| validation | 20 | 9,400 | 18,800 |
| test | 20 | 9,400 | 18,800 |
| 합계 | 200 | 94,000 | 188,000 |

각 시연은 20Hz·470스텝·23.5초다. 7차원 실제 Action Adapter 행동, 8차원 상태,
front/wrist RGB 224×224, 작업 명령을 정렬하여 저장했다. 처음 자세 설정 및 0.5초
안정화는 시연 시작 전이며 운반 중 컵을 순간이동시키지 않는다.

성공 조건은 컵 15mm 초과 상승, 목표 xy 오차 30mm/25mm 미만, upright cosine>0.95,
목적 테이블 접촉, 손 접촉 해제다. 추가로 마지막 0.5초 컵 위치 변화가 1mm 미만이어야 한다.
이번 데이터의 상승량은 23.05~32.03mm,
마지막 0.5초 최대 위치 변화는 0.0601mm였다.

성공한 200개 모두 오류·IK 거부·행동 클리핑 없이 완료했다. 실패/품질 미달 시도
**47개**는 최종 분할에서 제외하고 원인과 로그를 보존했다.
`episodes/`에는 성공 시연만 남기고 실패 시도는 `excluded/`로 분리했다. 학습에서는
`manifest.json`과 분할 JSON을 기준으로 읽어야 한다.

전체 이미지 188,000장의 RGB 모드·크기·파일 무결성, 관측·행동·시간 정렬, 유한값,
행동 범위, 시연 경계, 분할 간 동일 배치 중복 없음, 기존 제한 학습 로더 호환성을 검사했다.
검증 결과는 `validation_report.json`, 수집/제외 목록은 `manifest.json`, 진행 상태는
`progress.json`, 개별 판정은 `attempts/`와 `episodes/*/quality.json`에 있다.

넓어진 배치의 `train_0000`은 기록된 7차원 행동만 고정 재생해 운반 성공을 재현했다.
470개 시점의 전체 물리 상태·관절 위치·컵 위치가 원 시연과 정확히 일치했다.
증거: `results/wide_action_replay_validation.json`. 모든 200개를 각각 고정 재생한 것은 아니다.

## 렌더링과 재현

이번 데이터는 기존 WSL의 GLFW/D3D12 경로로 RTX 4060에서 렌더링했다.
그림자·반사·MSAA는 기존 시연과 같이 끄고 카메라 위치와 224×224 크기를 유지했다.
초기에 CPU로 생성한 이미지도 같은 NVIDIA 렌더러로 다시 만들었다.
각 에피소드 `rendering.json`에 실제 렌더러를 기록했다. 물리 상태와 행동은 재렌더링으로
변경하지 않았다. GPU 렌더링은 동시 4개로 제한하며 수집 파이프라인은 최대 8개 작업을 병렬로 실행한다.

```bash
conda activate FaultVLA
cd /home/ubuntu/FaultVLA

# 완료 상태에서는 기존 성공 결과를 재사용한다. 다른 수집 실행과 중복 실행하지 않는다.
python scripts/collect_wide200.py --workers 8

# 전체 데이터 감사
python scripts/validate_wide_dataset.py
```

미완료 에피소드는 단계별 메타데이터를 기준으로 재개한다. 메타데이터 없이 남은
부분 폴더는 삭제하지 않고 `_incomplete_...`로 보존하고 해당 단계만 새로 실행한다.
최종 시연은 `episodes/`, 중간 원본 13모터 전문가 기록은 `data/wide_raw_v2/`, 후보 계획은
`configs/demos_wide200/`에 있다. 기존 데이터와 섞지 않는다.

## 이후 제한 학습에 연결하기

아래는 실행 예시이며 이번 데이터 수집 작업에서 학습을 실행하지 않았다.

```bash
# 새 데이터 한 시연으로 학습 연결을 먼저 확인
python scripts/train_limited.py \
  --data data/cartesian_demos_v2_wide200 \
  --train-episodes train_0000 \
  --validation-episodes validation_0000 \
  --all-samples --epochs 20 \
  --output results/limited_wide_01

# 새 배치와 같은 초기화/렌더링 조건에서 실제 VLA 출력으로 평가
python scripts/evaluate_limited.py \
  --data data/cartesian_demos_v2_wide200 \
  --training-run results/limited_wide_01 \
  --episode validation_0000 \
  --output results/limited_wide_01_eval
```

전체 학습 시에는 `train.json`의 160개 ID만 학습에, `validation.json`의 20개를 검증에
사용한다. `test.json`의 20개는 최종 평가용으로 남긴다. 새로운 정규화 통계는 train 160개에서만
계산해야 한다. 현재 제한 학습기는 `--data` 경로를 기준으로 이 분할을 읽는다.

## 해석 범위

200개는 서로 다른 위치 배치의 **정상 전문가 성공 시연**이지 VLA의 200회 성공 결과가 아니다.
같은 컵·손 동작·명령·카메라를 사용하며 물체 종류/조명/임의 초기 자세의 일반화는 포함하지 않는다.
생산 데이터는 시연의 초기 옆잡기 자세부터 시작한다. 손 8모터의 독립 조작은 학습하지 않고
기존 이진 열기/닫기를 사용한다. 실제 하드웨어 작업 공간이나 고장 보상 성능은 미검증이다.
`training_ready: false`는 표준 RLDS 학습 통합을 아직 하지 않았다는 의미이며,
제공한 제한 학습 로더로는 직접 읽을 수 있음을 확인했다.

## 중단 후 최종 정리 및 행동 공간 검토

수집은 200개까지 완료되었다. 최종 정리 중 남아 있던 부분 폴더 12개는 `interrupted/`로 옮기고 `interrupted_artifacts.json`에 기록했다. 현재 `episodes/`에는 manifest에 등록된 성공 시연 200개만 있다. 원본 환경 82개 파일은 ZIP과 대조하여 변경 없음을 확인했다.

[SO-101 행동 공간 검토](SO101_ACTION_COMPATIBILITY.md)에서 7D→팔 5축·손 8축 연결을 확인했다. 목표 명령은 제한 안이지만 실제 wrist_flex는 soft constraint 때문에 최대 약 0.0597도 초과했다. 엄격한 실제 관절 한계 무위반 데이터로 해석하면 안 된다.
