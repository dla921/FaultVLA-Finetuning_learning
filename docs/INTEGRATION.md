# SO-101 + AmazingHand / VLA-Adapter-Pro 연동

## 현재 상태 (2026-10-26)

정상 작업용 연결 코드를 구현했다. 실제 Pro 가중치는 없으므로 모델 로딩, 실제 추론,
VLA에 의한 컵 운반 성공은 **미검증**이다. 카메라 렌더링, 실제 정규화 함수,
Action Adapter, 제한 및 물리 실행을 시험했다. 합성 행동 시험은 VLA 성공 시험이 아니다.
Phase A/B는 추가하지 않았다.

첨부 장면은 바구니가 아닌 두 테이블 사이의 컵 운반 장면이다. 이를 변경하지 않았다.
`environments/so101_wrist_side/`는 ZIP 원본이며 기존 결과도 유지한다.
`VLA-Adapter/` 원본 소스는 수정하지 않았다.

## 실행

WSL에서:

```bash
conda activate FaultVLA
cd /home/ubuntu/FaultVLA

# 기존 데모: 매번 새 복사본에서 실행하여 원본 결과 보존
python scripts/run_baseline.py
# GUI가 연결된 WSL 세션에서
python scripts/run_baseline.py --viewer

# 가중치 없이 전방/손목 RGB 및 로봇 상태 확인 (VLA 추론 아님)
MUJOCO_GL=egl python scripts/run_vla.py --check-only

# 테스트
python -m unittest discover -s tests -v
```

추가된 코드는 기존 환경의 mujoco, numpy, scipy, Pillow 및 VLA 의존성을 사용한다.
원본 requirements.txt를 무제약으로 다시 설치할 필요는 없다.

## 필요한 체크포인트

- 이름: `VLA-Adapter/LIBERO-Spatial-Pro`
- 리비전: `cb023c6771fc62ea729d218f09ac0583c14f59d4`
- 기본 경로: `/home/ubuntu/FaultVLA/checkpoints/LIBERO-Spatial-Pro`
- 전체 저장소 파일 필요: `model.safetensors`, `action_head--checkpoint.pt`,
  `proprio_projector--checkpoint.pt`, config, processor/tokenizer 파일, dataset_statistics.json.
- 현재 저장한 `docs/checkpoint/`는 작은 메타데이터뿐이며 실행 가능한 가중치가 아니다.

```bash
python scripts/download_checkpoint.py
MUJOCO_GL=egl python scripts/run_vla.py \
  --checkpoint checkpoints/LIBERO-Spatial-Pro \
  --allow-cross-embodiment --steps 400
# GUI 선택: --viewer (MUJOCO_GL=egl 지정 없이 실행)
```

`--allow-cross-embodiment`는 Panda와 다른 손/좌표/카메라를 사용하는 실험임을 명시하는
실행 옵션이다. 호환성이나 성공을 보장하는 옵션이 아니다. 가중치가 없으면 즉시 실패하며,
스크립트 궤적이나 가짜 정책으로 대체하지 않는다.
작업 변경은 `--task "..."`, 설정 변경은 `--config ...`를 사용한다.
정상 종료이지만 작업 실패는 exit 2, 예외는 비정상 종료, 인터페이스 검사는 exit 0이다.

## 실제 체크포인트 입력·출력 계약

조사한 로컬 VLA 코드 커밋: `23fa0c9c159e2aa04341cdd3e924f44061311060`.
공개 체크포인트의 config/statistics와 API 파일 목록은 `docs/checkpoint/`에 보관했다.

| 항목 | 규격 |
|---|---|
| 이미지 | 전방 RGB + 손목 RGB, uint8 H×W×3; 256×256 렌더링 |
| 전처리 | 원본 get_vla_action: 224×224 JPEG/Lanczos resize, 면적 0.9 center crop, processor |
| 언어 | Qwen 형식 프롬프트에 실제 task 문자열 전달 |
| 상태 | [eef xyz(미터), 회전벡터 xyz(라디안), Panda 손가락 2개 변위(미터)], 8차원 |
| 출력 | 회귀 head의 8×7 chunk: [dx,dy,dz,drx,dry,drz,gripper] |
| 통계 키 | libero_spatial_no_noops |
| 정규화 | q01/q99 → [-1,1], proprio는 clip 적용 |
| 역정규화 | predict_action 내부에서 수행; Action Adapter에서 중복 수행하지 않음 |
| 손 명령 | 역정규화 출력의 마지막 값 >0.5 열기, 그 외 닫기 (0 닫힘, 1 열림) |
| 실행 주기 | 20 Hz (50 ms/action), 기본 1개 실행 후 재추론; 설정으로 최대 8개 |

역정규화된 첫 6개 값은 **직접 미터/라디안이 아니라 LIBERO OSC 입력값**이다.
robosuite 1.4.1 기본 OSC_POSE 규격을 적용한다:
입력 [-1,1]로 clip, 위치 ×0.05 m, 회전벡터 ×0.5 rad.
위치는 현재 world 좌표에 더하고, 회전은 `Exp(delta) @ current_rotation`으로 좌측 곱한다.
이 규격은 공개 LIBERO 기본 controller 설정으로 확인했으며 체크포인트 자체는 단위 정보를
내장하지 않는다. 다른 데이터셋으로 학습한 체크포인트에는 이 계약을 그대로 사용하면 안 된다.

첫 6차원 action의 q01:
`[-0.745473, -0.661607, -0.9375, -0.107143, -0.206786, -0.184286]`
q99:
`[0.9375, 0.875893, 0.932143, 0.103929, 0.176786, 0.145714]`
action mask는 `[true,true,true,true,true,true,false]`.
원본 전체 수치는 `docs/checkpoint/dataset_statistics.json`에 있다.

## SO-101 연결에서 명시적으로 도입한 근사

- 제어점: `palm_target`, 방향: `r_wrist_interface` 회전에 tool 보정 회전을 곱한다.
- policy frame = R × world + t. 기본은 R=I, t=0, tool 회전=I.
  이것은 좌표 정합이 완료됐다는 의미가 아니다. 임의로 통계 평균에 맞춰 이동시키지 않았다.
- AmazingHand 8개 모터의 **실측 qpos**를 open/closed posture 사이 비율로 환산하고,
  가상의 Panda 손가락 변위 `[+0.04*(1-c), -0.04*(1-c)]`로 표현한다.
  실제 Panda 센서값/정확한 손가락 간격이 아니며 이종 로봇 상태 proxy다.
- 이 초기 상태의 x,z,회전벡터 x,y가 학습 q01/q99 밖에 있다. 로더는 정상적인 원본
  clipping을 수행하고, 실행 로그에 범위 밖 차원을 기록한다. 미세학습이 필요할 가능성이 높다.
- 새 `vla_wrist`는 손목 body에 부착된 **가상 오프셋 카메라**다. 초기 자세에서 컵/손을
  볼 수 있도록 생성했고, 손목을 따라 움직인다. 실제 하드웨어 장착 및 LIBERO 시점과
  동일하지 않다. 전방/손목 원본 RGB를 파일로 기록한다.
- 직접 MuJoCo Renderer RGB를 사용하며 LIBERO의 raw framebuffer 회전 보정을
  무조건 복사하지 않는다. 실제 저장 영상을 눈으로 확인했다.

## Action Adapter와 제한

원본 데모의 bounded least-squares IK 방법과 손가락 posture를 사용한다.
단, IK 목표 위치/전체 방향은 매번 VLA 출력에서 계산하며 컵 위치나 고정 경유점을 사용하지 않는다.
5축 팔은 임의의 6D 자세를 실현할 수 없으므로 위치/회전 잔차가 기준을 넘으면
손 열기까지 포함해 전체 행동을 거부한다. 실패 시 기존 명령 유지, 남은 chunk 폐기 후 재관측한다.

- 13개 관절/actuator 이름으로 매핑하며 passive joint는 명령하지 않는다.
- 위치 목표: joint range와 actuator ctrlrange의 교집합.
- 목표 변화율: 팔 1 rad/s, 손 2 rad/s (시뮬레이션 초기 설정값, 실측 사양 아님).
- 실제 속도: physics step마다 감시, 12 rad/s 초과 시 기록 가능한 예외로 중단.
  target 변화율 제한은 실제 속도 보장이 아니므로 별도로 감시한다.
- IK 허용 오차: 10 mm, 0.15 rad. 근사 자세 수용 여부와 오차를 매 action 기록.
- 손가락: 열림 4쌍 [-0.8,+0.8], 닫힘 처음 3쌍 [+0.5,-0.5], 마지막 [+1.5,-1.5].
- 접촉을 고려한 경로 계획·충돌 회피는 구현하지 않았다. MuJoCo 접촉 물리는 실행된다.
- 모델 추론 중 시뮬레이션은 정지한다. 따라서 실제 로봇 실시간 지연 검증이 아니다.

## 검증 결과와 미검증 범위

1. ZIP 원본 정상 데모 실행 성공(exit 0).
   `results/baseline/demo/results/side_transfer.json`:
   상승 0.028479 m, 최종 xyz [0.263281,0.080063,0.119891] m,
   upright 0.999999878, table2 접촉, 손 접촉 없음.
2. 실제 MuJoCo EGL 전방·손목 RGB 렌더링 및 8차원 상태 생성 통과.
   `results/vla_20261026_231700_419381/`에 영상과 상태 보고서.
3. 11개 테스트 통과: 실제 FK에서 유도한 행동으로 모터 변화, 좌표 변환,
   그리퍼 방향, 비정상 입력, 도달 불가 거부, 제한, 실제 물리 step, 과속 중단,
   가중치 누락 거부, 모델 호출 인자 전달(stub), 원본 정규화 함수.
   로그: `results/test_report.txt`.
4. **미검증:** 실제 가중치 로딩/추론, GPU 메모리 요구, 실제 VLA 행동으로 컵 운반,
   일반화 성공률, 다양한 초기 상태/고장, 실제 하드웨어. 정책 호출 시험의 stub은
   테스트에만 존재하며 production 실행에는 fallback이 없다.
5. 성공 판정은 기존 데모와 같은 컵 상승/목적 위치/직립/접촉 기준이다.
   별도의 정착 시간·최종 속도 기준이 없으므로 결과 해석에 유의한다.

실행별 `config.json`, `summary.json`, `actions.jsonl`, RGB, 생성 MJCF를 보관한다.
로그에는 상태 범위, 실제 정책 행동 chunk, 모터 목표/실제 위치/속도, IK 잔차를 남긴다.

## 근거

- https://huggingface.co/VLA-Adapter/LIBERO-Spatial-Pro/tree/cb023c6771fc62ea729d218f09ac0583c14f59d4
- VLA-Adapter/experiments/robot/openvla_utils.py: get_vla_action, normalize_proprio
- VLA-Adapter/experiments/robot/libero/run_libero_eval.py: prepare_observation, process_action
- VLA-Adapter/prismatic/extern/hf/modeling_prismatic.py: _unnormalize_actions
- https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py
- https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/config/osc_pose.json
- https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/utils/control_utils.py

추가 보존 검사: ZIP의 원본 82개 파일 모두 추출본과 바이트 단위 동일함을 확인했다.
`results/provenance.json`에 ZIP/메타데이터 해시와 원본 코드 커밋을 기록했다.
`pip check` 통과. 가중치 없이 실제 실행하면 명확한 FileNotFoundError로 중단됨을 확인했다.

## 2026-10-27 과속 중단 수정

실제 정책 실행의 step 180에서 `finger1_motor1`이 시뮬레이션 t=9.021 s에
12.300339 rad/s에 도달하는 현상을 저장 행동으로 재현했다. 당시 위치는 -0.254464 rad,
닫힘 목표는 +0.5 rad였다. 목표 slew 제한만으로는 접촉·추종 지연 때문에 누적된
위치 오차에 따른 급가속을 막을 수 없었다.

중단 기준 12 rad/s는 유지했다. 50 ms마다 계단식으로 목표를 넣던 실행기를 변경하여
1 ms physics step마다 목표를 갱신한다. 실제 관절 위치와 목표 간 오차를 팔 0.3 rad,
손 0.15 rad 이내로 제한하려고 하되, 목표 slew 제한도 동시에 적용한다.
기존 명령이 이미 이 범위를 벗어나면 순간 점프시키지 않고 slew 한도 안에서 복귀한다.
이는 모델 제약/접촉을 다루는 정상 제어 보완이며 Phase A 알고리즘은 아니다.

예외에는 관절명, 실제 속도, 임계값, 시각이 포함되며 summary.json의 speed_stop에
위치와 적용 목표까지 기록한다. requested_motor_targets_rad는 정책을 IK로 변환한 목표,
motor_targets_rad는 실제 마지막 physics step에 적용한 목표다.

수정 후 과거 실제 VLA 행동 181개를 open-loop 재생하면 과속 없이 완료했다.
기존 11개 시험 및 추가 physics-step 목표 slew 시험을 통과했다.
이 재생 결과는 새 정책 관측에 따른 작업 성공이나 임의 접촉에서의 과속 방지를 보장하지 않는다.

실제 체크포인트 재검증 결과:
`results/vla_20261027_003213_630135/summary.json`에서 실제 VLA 추론 200회,
200스텝(시뮬레이션 10초)을 과속 예외 없이 완료했다. 거부 행동 3개.
컵 운반은 실패(task_success=false): 최종 [0.270985,-0.107489,0.119891] m,
상승 약 0.42 mm, table1에 남았다. 이 결과로 모델 로딩과 실제 추론은 확인되었으며,
위 10월 26일의 '가중치 미설치/추론 미검증' 기록은 당시 상태다.
현재 남은 문제는 이종 로봇 정책의 정상 작업 성능이다. 과속 수정이 작업 성공을 뜻하지 않는다.

## 중단 지점에서 재개

기존 00:27 실행은 180번째 행동(0-based step 180) 도중 t=9.021 s에 중단됐다.
`recover_legacy_run.py`로 마지막 완결 스텝 경계인 t=9.000 s를 복원했고,
180개 제어 스텝에서 기록된 13개 관절 위치와 최대 오차 0 rad로 일치했다.
복원은 기록된 실제 모터 명령 재생이며 고정 운반 데모가 아니다.

```bash
# 이번에 생성한 복원 상태에서 재개 (남은 220스텝)
python scripts/run_vla.py --allow-cross-embodiment \
  --resume-state results/recovered_002735/last_good.npz \
  --reset-tracking-targets --steps 220
```

--reset-tracking-targets는 qpos/qvel/접촉 상태를 유지하고 위치 제어 목표만 현재 관절
위치로 재설정해 누적 오차를 해제한다. 원래 중단 직후 9.021 s의 과속 상태를 강제로
이어가는 것이 아니라 직전 완결 상태에서 제어기를 재초기화한다. 기존 chunk는 버리고
복원된 RGB와 상태에서 실제 VLA를 새로 호출한다.

이제 각 성공한 제어 스텝 후 `last_good.npz`를 원자적으로 갱신한다.
종료 시 `final_state.npz`도 저장한다. 과속 후에는 last_good 사용을 권장한다.
--steps는 추가 실행 스텝 수이며 summary에 previous_steps/start/end 시각을 기록한다.
동일 MJCF와 MuJoCo 버전에서 사용해야 하며, 물리 설정을 변경한 모델로 복원하지 않는다.
복원 테스트에서 위치·속도·제어와 다음 50 ms 물리 진행까지 동일함을 확인했다.

재개 실행 결과: `results/vla_20261027_020200_544841/summary.json`.
9.000초에서 실제 VLA 추론 220회/220스텝을 추가 실행하여 총 20.000초까지
과속 예외 없이 완료했다. 새 구간의 IK 거부는 0회다. 컵은 옆으로 넘어졌고
작업 성공 판정은 false다(upright_cos 약 -0.0000186).
따라서 중단 지점 복구·실행 재개는 확인했지만 정상 컵 운반 정책 성능은 확보되지 않았다.
재개 가능한 `last_good.npz`와 `final_state.npz`, 모든 관측 RGB 및 행동 로그를 보관했다.

2026-10-27 작업 성능 비교와 원시 성공 시연 수집 결과는 [성능 실험 보고서](PERFORMANCE.md)를 참고한다.
3개 후보 모두 작업 실패여서 기본 정책 설정은 변경하지 않았다. 실제 결과와 준비된 학습 자료를 구분했다.

7차원 행동 구조와 정상 시연 확장 결과는 [행동 계약·시연 보고서](ACTION_CONTRACT_AND_DEMOS.md)를 참고한다. 파인튜닝과 Phase A/B는 수행하지 않았다.
