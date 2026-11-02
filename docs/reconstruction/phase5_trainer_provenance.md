# Rain/Snow frozen-base trainer 복원 근거

구현: [scene_graph_pbr_weather.py](../../models/trainers/scene_graph_pbr_weather.py).
검증: [test_weather_trainer.py](../../tests/test_weather_trainer.py),
[실행 로그](../../logs/reconstruction/phase5_trainer_tests.log).

2026-10-17에 아래 실제 과거 소스를 대조하여 복원했다. 장시간 weather 학습의
완료를 뜻하지 않으며, 명령 실행 전 구조·loss·gradient·checkpoint 검증 결과다.

## 사용한 실제 소스

- `/home/user/.codex/sessions/2026/10/07/rollout-2026-10-07T21-38-23-01a1165f-3e32-7c10-ba19-9d99e1fdbba9.jsonl:36`:
  `_bounded_logit`, `_roughness_logit`, `_masked_charbonnier`, `_erode_binary_mask`,
  `_weather_material_mask`, `_pop_delta_regularizer`, `_material_losses`,
  `_rgb_losses`, `compute_weather_losses`, `compute_metrics`의 실제 소스 출력.
  이 helper와 loss 수식·reduction·threshold를 그대로 가져왔다.
- `/home/user/.codex/sessions/2026/10/13/rollout-2026-10-13T14-03-00-01a133a4-7c6d-7793-8b70-aac285121d21.jsonl:251`:
  당시 `models/trainers/scene_graph_pbr.py` 766–1021행 출력.
  생성자, stage 선택, optimizer, raw reference 렌더, frozen base step 유지의 근거다.
- [E09](evidence/E09_weather_overlay.txt): 실제 weather YAML.
- [E10](evidence/E10_weather_schedule.txt): 64×2 decoder와 material 30k →
  environment 3k → joint 5k 실행 설정. 이 명령 자체를 학습 완료 근거로 사용하지 않는다.

## 학습·렌더 계약

기존 Base v4 renderer를 상속하고, Gaussian 생성 시 `weathers=[raw]`를 사용한다.
Rain/Snow residual은 raw 재질 MLP와 별도 이름으로 등록한다. 현재 복원 범위에서
재질 residual은 Background에만 활성화하며, 조명 residual은 공유 HDR을 통해
surface·sky·동적 물체의 조명에 적용한다.

모든 base 파라미터를 먼저 freeze한 후 material 단계에는 weather decoder만,
environment 단계에는 해당 HDR residual만, joint에는 두 그룹만 Adam에 넣는다.
원래 Adam `eps=1e-15`, weight decay 0, 상수 stage별 LR을 유지한다. Raw geometry,
opacity, feature, raw material, tangent normal, HDR, camera response, CamPose,
affine은 optimizer에 포함되지 않는다.

`base_model_step`을 weather optimizer의 `step`과 분리한다. Raw 30k에서 시작하면
weather 단계가 step 1이어도 Gaussian node의 동작 기준 step은 30000이다.
Frozen node의 refinement/preprocess callback을 호출하지 않으므로 split, clone,
cull, opacity reset 및 SMPL KNN 갱신을 실행하지 않는다.

`render_raw_reference(raw_infos, cam_infos)`를 weather forward 전에 실행한다.
No-grad raw 렌더의 albedo·roughness·opacity만 보관하고, weather forward가 만든
residual cache는 loss 계산 시 pop한다. 호출 순서를 뒤집어 raw forward가 weather
cache를 지우는 실수를 피한다.

Material strength와 environment strength는 별도로 조절한다. 두 값이 0이면 같은
checkpoint의 raw와 RGB가 정확히 일치하는 것을 CUDA 검사에서 확인했다.

## Paired material loss와 confidence

감독은 Gaussian별 residual 자체와 pseudo target을 직접 비교하는 것이 아니라,
렌더한 raw/weather 재질맵의 logit 차이를 비교한다. Albedo logit은 epsilon 1e-4,
roughness는 `(r - 0.01) / 0.99`로 먼저 정규화한다. 기본 `gt_relative` target은
weather pseudo logit − raw pseudo logit이며 A/R 각각 ±4/±5로 제한한다.
`reconstruction`은 과거에 있던 별도 옵션으로 남겨 두며 기본 실행에서는 사용하지 않는다.

Confidence mask는 raw sky·ego·dynamic을 제외한 mask를 목표 해상도로 crop/resize하고
`>0.99`로 이진화한다. 이어 frozen raw opacity ≥0.5, **raw/weather pseudo normal 간**
cosine ≥0.5, frozen raw albedo의 pseudo raw 대비 채널평균 절대오차 ≤0.25,
roughness 절대오차 ≤0.35를 적용하고 1px erosion한다. 렌더 normal과 pseudo normal의
cosine을 비교하는 다른 mask로 바꾸지 않는다. 유효 픽셀이 64보다 작으면 오류다.

Charbonnier는 `sqrt(diff² + eps²) - eps`, eps=0.001이다. 유효 채널 수로 나누므로
albedo 3채널과 roughness 1채널의 reduction을 균형 있게 유지하고 residual 0일 때
loss도 정확히 0이다. Paired A/R=1/0.5, absolute A/R=0.1/0.05,
Gaussian residual L1 A/R=0.001/0.001이다.

Environment/joint에는 RGB surface=0.8, sky=0.1, SSIM=0.02와 HDR residual
magnitude/TV를 사용한다. RGB surface는 sky·ego·dynamic 제외, sky loss는 ego 제외다.
과거 코드대로 SSIM은 ego만 제외한 전역 영상에 적용하여 sky 경계에 새 이진 경계를
추가하지 않는다. Raw RGB/SSIM, boundary LPIPS, VGG, depth, metallic/normal loss는
weather objective로 상속되지 않는다.

## Checkpoint와 복원 시 명시적으로 바꾼 연결부

과거 trainer가 들어 있던 큰 `scene_graph_pbr.py` 대신 별도 subclass 파일을 만들었다.
`MultiTrainer_pbr_snow` 이름은 alias로 유지하지만 Rain/Snow 모두 동일한 구현을 쓴다.
Raw 모델용 기존 train 경로를 건드리지 않고 weather 전용 실행기가 이 API를 호출한다.

새 stage는 `init_from_base_state`/`init_from_base_checkpoint`로 이전 stage의 가중치를
가져와 optimizer와 step을 초기화한다. `load_state_dict`는 같은 stage의 모델 상태를
복구하며 `weather_training_state`의 weather/version/base step을 검증한다. Optimizer,
RNG 복구는 별도 실행기 utility의 책임이다.

Raw checkpoint에는 새 weather residual이 없으므로 **전체가 없는 해당 branch만**
초기화된 zero-output decoder/zero HDR residual로 채운다. 일부 tensor만 없는 branch,
다른 weather의 metadata, weather checkpoint에서 누락된 residual은 실패한다.
나머지 base 모델은 strict load한다. Checkpoint에 없는 빈 Gaussian class를 제거하는
기존 Base v3/v4 동작은 그대로 유지한다.

## 검증 결과

`MPLCONFIGDIR="$PWD/.cache/matplotlib" .conda/bin/python -m unittest tests.test_weather_trainer -v`

9개 테스트 통과:

- Charbonnier의 채널 균형, 정확한 zero, mask 밖 gradient 0.
- confidence 각 조건, inclusive opacity threshold, 1px erosion의 정확한 영역.
- `gt_relative`가 frozen raw 오차를 유지하는 수식과 `reconstruction`의 차이,
  roughness 정규화, target cap, 최소 유효 픽셀 실패.
- RGB surface/sky mask와 전역 SSIM 계약.
- 세 stage의 optimizer 선택, base freeze, node callback 미실행, base step 유지.
- 실제 collection 경로에서 Background에만 weather를 전달하고 Rigid/SMPL/Deformable에는
  raw를 전달하여, dynamic 재질은 고정한 채 공통 weather 조명만 적용하는 계약.
- 아직 point_ids/template을 복구하지 않은 실제 Rigid/SMPL 객체의 checkpoint 사전 검사.
  이때 custom state_dict를 호출하지 않으며, loader가 pop하는 metadata도 입력 checkpoint에
  남도록 모델별 mapping을 복사한다.
- 실제 CUDA gsplat + HDR64 + camera response에서 material → environment → joint
  각각 한 번의 forward/loss/backward/Adam update와 finite/nonzero selected gradient.
- 각 update 후 frozen parameter의 byte 동일성, strengths=(0,0)의 raw RGB 정확 일치,
  기존 raw 렌더와 최대 2e-6 허용 오차, checkpoint tensor byte 동일성 및 RGB round-trip
  2e-6 허용 오차, 누락 residual/다른 weather checkpoint 거부.

CUDA 테스트는 16 Gaussian, 32×32 합성 장면이다. 충분한 유효 target 영역을 만들기
위해 이 **검사에서만** opacity threshold를 0으로 설정했으며, 실제 실행 설정의
confidence threshold를 완화하지 않았다. 전체 788 데이터의 confidence 유효율과
실제 train 명령 검사는 별도 실행기 검증 기록을 따른다.
