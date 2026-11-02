# 실험 3: Base v4 + bounded teacher material calibration

이번 작업은 **구현·검증만** 진행한다. 사용자 요청에 따라 scene 학습·평가 실험을 시작하지 않는다. 기본 실행기 호출은 읽기 전용 preview이며, `--execute`를 명시해야 학습과 평가를 실행한다.

## 검증할 가설

기존 Base v4의 프레임별 G-buffer teacher가 같은 표면을 서로 다른 albedo·roughness·metallic 값으로 추정하면, 공유된 3D 재질 모델에 상충하는 supervision이 들어간다. 실험 3은 teacher의 프레임 단위 편차를 적은 수의 affine parameter로 흡수하면 렌더 결과의 temporal consistency도 개선되는지 확인한다.

근거는 [Intrinsic Image Fusion for Multi-View 3D Material Reconstruction](https://arxiv.org/html/2512.13157v1)의 §3.1이다. 원 논문은 객체·예측별 albedo 3×4, roughness·metallic 각각 1×2 affine 변환을 사용하고, 여러 후보 예측의 분포를 3D로 융합한 뒤 inverse path tracing으로 재질을 정련한다.

여기서는 기존 teacher 한 세트를 그대로 사용하고, **train 이미지별 대각 gain/bias만** Base v4와 공동 학습한다. Multi-candidate 생성, 객체별 분포 matching, 3D texture fusion, inverse path tracing은 도입하지 않는다. 따라서 IIF의 저차원 보정 아이디어를 응용한 실험이며 논문 전체 재현이 아니다.

## 모델과 제약

각 train image에 albedo RGB·roughness·metallic의 gain 5개와 bias 5개를 배정한다. 기본 scene의 train 88장에는 **880개 학습 parameter**가 추가된다. Held-out image에는 parameter를 만들지 않는다.

```text
p: N_train × 2 × 5의 영 초기화 parameter
u = tanh(p)
z = 0.5 × (u - 같은 카메라 train 이미지들의 u 평균)
gain = 1 + strength(step) × 0.15 × z_gain
bias =     strength(step) × 0.05 × z_bias
teacher_corrected = clamp(gain × teacher + bias, 0, 1)
```

Camera별 train parameter의 평균 편차를 0으로 고정해 모든 teacher의 색·재질 scale이 함께 이동하는 자유도를 제한한다. Gain은 0.85–1.15, bias는 ±0.05 범위이며, 양의 대각 gain은 채널별 값의 순서를 유지한다. 평균 gain/bias 제약은 clamp 이전 계수에 대한 것이며, 보정된 material 값의 평균까지 보존하지는 않는다. Clamp에 걸리는 영역은 기록한다.

보정은 sky·dynamic·ego-car를 제외한 정적 영역 중 opacity ≥ 0.95인 위치에만 적용한다. **원래 material loss의 sky/ego mask·분모·가중치는 유지**하고, 동적 영역에서는 원래 teacher 값을 그대로 사용한다. Normal teacher, RGB loss, geometry·lighting·density schedule도 유지한다. `image_infos`와 디스크의 teacher 파일을 수정하지 않는다.

원래 G-buffer crop/resize를 적용한 후, 기존 G-buffer loss가 사용하는 값의 공간에서 teacher를 보정한다. Teacher tensor와 static routing은 detach하지만 gain/bias에는 gradient를 남겨 3D scene과 공동 fitting한다. 현재 scene의 재질·rasterization gradient 경로는 유지한다.

| 설정 | C 기본값 |
|---|---:|
| 시작 / ramp | 3000 / 1000 step |
| Calibration LR | 0.001 |
| 최대 gain 편차 / bias | 0.15 / 0.05 |
| Identity prior weight | 0.01 |
| Parameter smoothness weight | 0.01 |
| Smoothness 최대 frame gap | 2 |
| Static support opacity | 0.95 |

Step 3000까지 strength는 0이며, 3001부터 증가해 4000에서 1이 된다. Identity prior는 `strength × 0.01 × mean(tanh(p)^2)`이다. Smoothness는 같은 카메라의 인접 train frame 중 gap ≤ 2인 쌍에서 `strength × 0.01 × mean(((z_next-z_previous)/gap)^2)`로 정의한다. Test frame parameter는 사용하지 않으며, test frame을 사이에 둔 두 train frame은 gap 2로 연결할 수 있다.

이 smoothness는 **보정 parameter에 대한 약한 prior**이며 같은 표면을 재투영해서 비교하는 실험 2의 loss와 다르다. 시간에 따라 teacher bias가 급격하게 변하면 smoothness가 적절한 보정을 방해할 수도 있다. 전체 작은 parameter table의 prior는 활성 training update마다 계산한다.

영 초기화와 warmup은 원래 teacher 값을 유지한다. 원래 G-buffer loss 세 항에 corrected teacher를 넣고 identity/smoothness 항만 추가한다. 실험 1 DiffReg와 실험 2 temporal loss는 켜지 않는다.

## 비용과 해석 범위

추가 연산은 작은 parameter table과 teacher map의 pointwise affine 보정이다. 이웃 프레임 로딩·렌더링, diffusion 추론, 새 G-buffer 생성은 없다. 추론에는 보정 head를 호출하지 않으므로 원래 Base v4의 renderer를 사용한다. Checkpoint에는 재현을 위해 head와 config·train ID metadata를 저장한다.

이는 프레임 전체에 공통적인 teacher 편차를 설명하는 가설이다. 국소적인 잘못된 재질 분할, 환각 패턴, 움직이는 물체의 대응 오류를 직접 고치지 않는다. Scene과 teacher correction을 함께 fitting하므로 scene 오차를 teacher가 따라가는 위험이 있다. 작은 범위, camera-mean 제약, identity prior는 이를 제한하지만 성능 향상을 보장하지 않는다. 실제 학습시간 증가율과 temporal 품질 개선량은 아직 측정하지 않았다.

## 동일한 Base v4 비교

| Arm | 초기화 | Supervision | 업데이트 |
|---|---|---|---|
| B0 | Dataset에서 처음부터 | 원래 Base v4 teacher | 0..30000 |
| C | B0와 같은 seed·초기화 | 제한된 teacher 보정 + parameter priors | 0..30000 |

실행기는 `output/pbr/rebuild/scene_788_100f/base_v4_raw/config.yaml`의 설정만 읽으며 기존 checkpoint로 초기화하지 않는다. 기본은 scene 788, camera 0, frame 000–099, seed 0, train 88/test 12다. 원래 loop가 inclusive이므로 각 arm은 30,001번 업데이트한다. 두 설정은 `trainer.teacher_calibration.enabled`만 다르다.

이미지 평가는 원래 `tools.eval`로 test 12장, native 1920×1280, 원래 affine 정책과 PSNR·SSIM·LPIPS 및 occupied/dynamic/human/vehicle 집계를 유지한다. 동일 flow·mask를 쓰는 기존 `tools.eval_gbuffer_attention`으로 paired temporal residual·raw warp·edge·coverage도 기록한다. 전체 경로에는 train frame이 포함되므로 연속 held-out 영상 성능으로 해석하지 않으며 train→test pair 진단도 따로 확인한다.

공통 실행 설정은 최종 checkpoint만 저장하고, 학습 완료 후 별도 image stage에서 평가하는 것이다. Optimizer·resolution·density·teacher supervision 가중치는 원래 Base v4 설정을 유지한다.

## 실행기

실험을 시작하지 않는 preview:

```bash
cd /home/user/Desktop/3ableGS
.conda/bin/python -m tools.run_teacher_calibration_experiments
```

향후 사용자가 실행을 요청한 뒤 사용할 명령:

```bash
cd /home/user/Desktop/3ableGS
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .conda/bin/python -u -m tools.run_teacher_calibration_experiments \
  --output-dir output/pbr/teacher_calibration/scene788_from_scratch_v1 \
  --execute
```

```bash
.conda/bin/python -m tools.run_teacher_calibration_experiments \
  --output-dir output/pbr/teacher_calibration/scene788_from_scratch_v1 --status
```

기본 순서는 `B0_s0_train → B0_s0_images → C_s0_train → C_s0_images → C_s0_temporal`이다. `--steps 3000` 이하에서는 보정이 활성화되지 않으므로 짧은 실행을 품질 실험으로 해석하지 않는다.

출력에는 공통 queue의 `manifest.json`, `state.json`, `summary.json/csv`, `logs/`, `stages/`, `reference_flow/`가 포함된다. Summary는 `image_delta_C_minus_B0`, `training_stage_wall_time_ratio_C_over_B0`, temporal residual 감소율 및 `calibration_stats/raw/*`의 마지막 기록 window를 담는다. Gain/bias 크기, 유효 비율, clamp 비율, 평균 teacher 변화량과 두 prior loss를 함께 확인한다. 시간에는 초기화·최종 저장이 포함되며, 마지막 로그 window는 전체 학습 평균이 아니다.

Checkpoint의 config·train IDs·카메라 수와 calibration parameter·train-ID buffer를 검증한다. Camera grouping과 smoothness 연결은 검증된 split/config로 다시 만든다. B0에 calibration state가 섞이거나 C에 누락된 경우를 거부한다. 기존 queue의 lock, code/config/input identity, artifact hash, 실패 attempt 보존과 완료 stage skip도 유지한다. 중단된 학습 stage의 재시도는 새 attempt에서 처음부터 시작하며 optimizer/RNG exact resume가 아니다. 앞선 중단 실험은 재개하거나 삭제하지 않는다.

## 구현 검증

2026-11-01 기준 전체 CPU 회귀 테스트 521개 중 **517개 통과, CUDA 필수 기존 항목 4개 skip**이다. 새 core 테스트 18개는 identity·warmup, gain/bias 경계와 camera 평균, split·camera·gap 연결, teacher gradient 차단과 합성 bias fitting을 확인한다. Integration 10개는 원래 RGB·normal loss와 동적 supervision 보존, crop/resize, scene/head gradient, optimizer 등록, checkpoint 재로드와 잘못된 split 거부, 추론 출력 불변을 검증한다. Queue 테스트 10개는 가짜 child를 사용해 실행 계획과 오류 복구를 검사했다.

로그는 `logs/teacher_calibration_validation/cpu_tests.log`에 저장했다. 기본 preview, Python 문법 검사와 `git diff --check`도 통과했다. 실제 scene 학습·평가 실험과 background service는 실행하지 않았고, 기본 실험 출력 폴더도 생성하지 않았다. 품질 개선량과 실제 전체 학습시간은 미측정이다.
