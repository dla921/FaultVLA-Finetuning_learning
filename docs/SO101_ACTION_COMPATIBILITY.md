> LEGACY: This document describes the previous Cartesian/wide200 experiment. For current training use README.md and docs/STABLEGRASP200.md.

﻿# SO-101 + AmazingHand 행동 공간 호환성 검토

2026-10-27 별도 Codex 검토: 현재 MuJoCo 모델과 Action Adapter를 통해 정상 컵 운반용 7차원 행동을 팔 5축·손 8축 명령으로 연결할 수 있다. 원본 LIBERO 정책의 작업 성공이나 실물 호환성을 뜻하지 않는다.

## 인터페이스

| 항목 | 현재 구현 |
| --- | --- |
| 행동 | `[dx, dy, dz, drx, dry, drz, open]` |
| 앞 6축 | 역정규화 후 dimensionless 제어 입력, [-1, 1] 범위 |
| 위치·회전 스케일 | 위치 0.05m, 회전 0.5rad를 곱함 |
| 좌표계 | 현재 설정은 MuJoCo world 축, 회전은 `Exp(delta_rotvec) @ R` |
| 팔 5축 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll; bounded IK로 변환 |
| 손 8축 | open > 0.5이면 열기, 그 외 닫기; 고정 8모터 자세 사이 전환 |
| 로봇 명령 단위 | MuJoCo 회전 관절 rad |
| 상태 8축 | palm xyz, tool rotvec, aperture 및 -aperture; 손 관절로 만든 proxy |
| 시간·청크 | 20Hz, 에피소드 경계를 넘지 않는 행동 8개 묶음 |

근거: `src/faultvla/bridge.py:70-150`, `src/faultvla/limited_training.py:35-60`.
5DOF 팔이 임의의 6D 위치·자세를 모두 만족할 수는 없다. IK 위치 오차 10mm 또는 회전 오차 0.15rad를 넘으면 손 동작까지 포함한 목표 갱신을 거부한다. 손 8모터를 독립적으로 예측하는 행동 공간은 아니다.

## 200개 데이터 검토 결과

- 94,000개 행동의 앞 6축 최대 절댓값 0.3040075, 손 행동은 0 또는 1.
- 기록된 13모터 목표 명령은 관절 제한 안에 있다.
- 200개 모두 수집 성공, 오류·IK 거부·행동 클리핑 없음.
- 저장된 20Hz 표본의 실제 관절 속도 최대 7.70071rad/s. 전체 physics step 최대치로 해석하면 안 된다.
- 팔 1rad/s·손 2rad/s는 목표 명령 변화율 제한이다. 실제 속도 제한과 다르며, 구현은 각 physics step에서 실제 속도 12rad/s 초과 시 중단한다.
- train_0000의 기록된 7D 행동만 재생한 470스텝은 물리 상태·관절·컵 위치 오차 0으로 성공했다. 200개 전부를 별도 재생한 것은 아니다.

**실제 위치 제한은 완전히 지켜지지 않았다.** wrist_flex의 MJCF 하한은 -1.658063rad이나 실제 최솟값은 -1.659105543rad였다. 6,264개 저장 시점에서 하한을 넘었고 최대 초과량은 0.001042543rad, 약 0.0597도다. 명령 최솟값 -1.657063017rad는 제한 안에 있다. MuJoCo의 soft constraint에 따른 작은 초과이며, 현재 bridge에는 실제 위치 초과 정지 로직이 없다. 따라서 이 데이터는 정상 작업 성공 시연이지만 엄격한 실제 관절 한계 무위반 데이터는 아니다. 학습 라벨이나 물리 설정을 사후 변경하지 않았다.

근거: `environments/so101_wrist_side/robot.xml:135`, `results/wide_action_space_audit.json`, `results/wide_action_replay_validation.json`.

## 체크포인트와 정규화

현재 기반 체크포인트는 `checkpoints/LIBERO-Spatial-Pro`다. 입력은 front/wrist RGB, 작업 명령, 상태 8축이며 출력 행동 청크는 8×7이다. 형상이 같아도 Panda용 정책이 SO-101의 기구학과 손 동작을 학습한 것은 아니다.

제한 학습기는 새 데이터의 train 160개만 사용하여 q01/q99를 계산한다. 행동 앞 6축을 정규화하고 이진 손 값은 그대로 둔다. 상태 8축도 새 통계를 사용한다. 새 통계는 학습 모델·추론 설정·출력 체크포인트에 함께 저장된다. 기존 LIBERO 통계를 새 데이터 통계 대신 사용하면 안 된다.

근거: `scripts/train_limited.py:55-61`, `scripts/train_limited.py:130`, `src/faultvla/policy.py`, `VLA-Adapter/experiments/robot/openvla_utils.py:803`, `VLA-Adapter/prismatic/extern/hf/modeling_prismatic.py:786`.

## 미검증 범위

이번에는 데이터 수집·검증만 완료했다. 이 200개로 학습한 VLA의 폐루프 성공률은 아직 측정하지 않았다. 현재 작업은 컵을 다른 테이블로 운반하는 작업이며 바구니 투입은 검증하지 않았다. 임의 초기 자세, 물체 일반화, 손 독립 조작, Fault-Tolerant 알고리즘, 실물 서보 단위·오프셋·교정도 미검증이다. 실물 모터 인터페이스에는 별도 변환과 검증이 필요하다.
