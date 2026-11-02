# 실험 4: Base v4 + frozen teacher confidence

이번 작업은 **구현과 검증만** 진행한다. 학습·평가 실험이나 background service를 시작하지 않는다. 실행기의 기본 호출은 읽기 전용 preview이며, `--execute`를 명시해야 실제 프로세스를 실행한다.

## 목적과 연구 근거

실험 4는 정적 표면을 여러 train 프레임에서 관찰했을 때 G-buffer teacher가 얼마나 일관적인지 계산하고, 그 정보를 **기존 material supervision의 상대 가중치**로 사용한다. Teacher를 시간축으로 평균해 target을 흐리게 만들거나 추론 결과를 필터링하지 않는다.

근거는 [SVGF: Spatiotemporal Variance-Guided Filtering](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/)의 temporal accumulation·variance 추정과 [NVIDIA NRD](https://github.com/NVIDIA-RTX/NRD)의 G-buffer 기반 history 신뢰도 판단이다. 원 연구와 라이브러리는 렌더링 노이즈 제거를 다룬다. 여기서는 재투영과 history rejection, moment 기반 불확실성이라는 원리를 **train teacher loss confidence**에 옮긴다. SVGF의 wavelet filtering이나 NRD 전체 denoiser 구현·성능을 재현하는 실험은 아니다.

## 처음부터 학습하며 신뢰도 만들기

V는 기존 Base v4와 동일하게 dataset에서 초기화한다. **0..2999 update를 마친 뒤, step 3000의 main update 직전** 현재 모델로 confidence cache를 한 번 생성한다. 기존 Base v4 checkpoint를 geometry 생성이나 학습 초기값으로 로드하지 않는다.

1. Train split의 이미지들을 카메라별 시간순으로 읽는다. 기본 scene에서는 88장이다. Test 12장의 teacher·RGB·geometry는 cache 생성에 사용하지 않는다.
2. 고정된 warmup 모델로 각 train view의 depth·normal·opacity를 `no_grad` 상태에서 한 번 렌더한다. 기본 크기는 native의 가로·세로 1/4인 480×320이다. RGB·teacher 입력은 renderer에 전달하지 않는다.
3. 원래 G-buffer crop/resize 규칙으로 teacher와 geometry를 정렬한다. Crop에 맞춰 principal point를 옮기고 resize 비율로 K를 조정하며, 현재 renderer가 사용한 camera pose를 저장한다.
4. 같은 카메라의 이전 train view로 재투영한다. 최대 frame gap은 2이며 실제 frame 번호를 사용한다. 예를 들어 frame 9는 held-out frame 8을 건너뛰고 train frame 7의 history를 사용할 수 있다. 더 큰 gap에서는 history를 초기화한다.
5. 현재·이전의 static mask, opacity, depth, normal, bilinear footprint, 왕복 재투영을 통과한 위치에서 teacher의 first/second moment를 누적한다. Geometry gate는 material 값의 일치 여부로 history를 미리 버리지 않는다. 불일치 자체가 confidence를 정하는 관측이기 때문이다.

Cache 생성은 RNG·camera·render cache·dataset downscale 상태를 보존한다. 현재 main forward 이전에 완료하고, geometry나 optimizer를 추가로 업데이트하지 않는다. 생성 중 실패하면 부분 cache를 활성화하지 않는다. 이후 optimization에서는 cache를 고정해 모델이 confidence를 바꿔 자신의 loss를 줄이는 경로를 차단한다.

## Moment와 원시 confidence

Albedo는 material moment 재투영 전에 linear 공간으로 바꾸고, roughness·metallic은 scalar 값을 사용한다. 유효 history의 effective count를 최대 4로 제한한다. 이는 엄밀한 최근 4프레임 sliding window가 아니라 history 비중에 상한이 있는 누적이다.

```text
x = [linear albedo RGB, roughness, metallic]
w = min(n_previous / (n_previous + 1), 3/4)
m1 = (1-w) * x    + w * warp(m1_previous)
m2 = (1-w) * x²   + w * warp(m2_previous)
variance = max(m2 - m1², 0)
score = (mean_RGB(variance_albedo) / sigma_albedo²
       + variance_roughness / sigma_roughness²
       + variance_metallic / sigma_metallic²) / 3
confidence = 0.25 + 0.75 * exp(-0.5 * score)
```

새 관측은 `m1=x`, `m2=x²`, count 1에서 시작한다. History가 없거나 가림·경계·정합 검사에서 탈락한 위치의 **원시 confidence는 1**이다. 검증하지 못한 teacher를 곧바로 잘못된 teacher로 간주하지 않는다. 아래 static 평균 정규화를 적용하면 최종 loss weight는 1보다 커질 수 있으므로, 원시 confidence 1이 모든 상황에서 최종 weight 1을 뜻하지는 않는다.

| 설정 | V 기본값 |
|---|---:|
| Cache 생성 step / weight ramp | 3000 / 1000 step |
| Cache render downscale | 4 |
| 최대 true frame gap / effective history | 2 / 4 |
| Confidence floor | 0.25 |
| Albedo / roughness / metallic sigma | 각 0.1 |
| Opacity 최소값 | 0.95 |
| Depth tolerance | 0.05 + 0.02 × projected depth |
| Normal 기준 | absolute cosine ≥ 0.9 |
| 왕복 허용오차 | cache 해상도에서 1 pixel |
| Static erosion / depth edge 상대 기준 | 1 pixel / 0.1 |
| Cache 메모리 상한 | 256 MiB |

Confidence는 RGB·roughness·metallic 분산을 묶은 단일 채널이다. 각 재질의 teacher 값은 원래 값 그대로 유지하고, 동일한 confidence로 세 material loss를 가중한다. Normal loss에는 confidence를 적용하지 않는다.

사용자 설정의 `confidence_floor`는 float16 cache에서 0으로 내려가지 않도록 `[2^-14, 1]`로 제한한다. Checkpoint에서도 저장 정밀도로 반올림한 하한, 양수 여부와 finite 값을 검사한다.

## Loss 연결과 비용

Step 3000에서는 cache를 만들지만 적용 strength는 0이다. 3001부터 선형으로 증가하고 4000부터 1이다. Cache를 현재 teacher loss 크기로 resize한 뒤 **현재 annotation의 static mask를 다시 적용**해 dynamic 경계로 가중치가 번지는 것을 막는다.

```text
q = 1 - strength + strength * confidence
weight_static = q * N_static / sum_static(q)
weight_elsewhere = 1
L_material = 기존 material_w * sum(기존 mask * weight * |prediction - teacher|)
             / (기존 mask 합 * material channel 수)
```

Static 영역의 가중치 평균을 1로 정규화하므로, 단순히 전체 G-buffer loss를 약하게 하는 비교가 아니다. 원래 denominator와 dynamic 영역의 weight 1을 유지한다. 모든 static residual이 같은 경우 원래 loss와 같은 값을 얻으며, confidence가 모두 동일한 프레임에서는 상대 가중치가 달라지지 않는다. 원시 confidence는 0.25–1이지만 정규화된 weight는 최대 4까지 커질 수 있다.

추가 학습 parameter는 **0개**다. 기본 scene의 cache 구축에는 quarter-resolution render 88회가 한 번 필요하며, 이후 매 step에는 cache resize·mask·가중치 계산만 추가된다. 모든 teacher·geometry packet을 보관하지 않고 history를 스트리밍한다. CPU float16 confidence 88×320×480×1은 약 **25.8 MiB**이며, 재현을 위해 최종 checkpoint에 포함한다. 이론적인 cache 크기이며 renderer 작업 메모리와 checkpoint의 나머지 모델 크기는 별도다.

추론 renderer에는 confidence 연산이나 temporal filtering이 추가되지 않는다. 평가 시 checkpoint에 저장된 cache를 검증·복원하되 teacher를 다시 읽거나 geometry를 재렌더해 cache를 만들지 않는다. 실제 학습시간 증가율은 아직 측정하지 않았으며, 같은 seed의 B0/V 전체 wall time과 cache 생성 시간을 함께 확인한다.

## 비교와 평가

| Arm | 초기화 | Material supervision | 업데이트 |
|---|---|---|---|
| B0 | Dataset에서 처음부터 | 원래 Base v4 | 0..30000 |
| V | 같은 seed로 처음부터 | 고정 teacher confidence로 static 상대 가중 | 0..30000 |

기본은 scene 788, camera 0, frame 000–099, seed 0, train 88/test 12다. 원래 inclusive loop에 따라 각 arm은 **30,001 optimizer updates**를 수행한다. 두 arm의 설정은 `trainer.teacher_confidence.enabled`만 다르다. 실험 1의 DiffReg, 실험 2의 temporal loss, 실험 3의 teacher calibration은 켜지 않는다.

실행기는 기존 `output/pbr/rebuild/scene_788_100f/base_v4_raw/config.yaml`의 설정만 복사한다. 기존 RGB·normal·geometry·lighting loss와 optimizer·density·resolution schedule을 유지한다. 저장 공간을 위해 두 arm 모두 최종 checkpoint만 저장하고, 학습 후 독립된 image stage에서 평가한다.

이미지 평가는 원래 `tools.eval`로 test 12장, native 1920×1280, 원래 affine 정책과 PSNR·SSIM·LPIPS 및 occupied/dynamic/human/vehicle 집계를 유지한다. 기존 `tools.eval_gbuffer_attention`의 같은 reference flow·mask로 temporal residual·raw warp·edge·coverage도 비교한다. 전체 경로에는 train frame이 포함되므로 연속 held-out 영상 성능으로 해석하지 않으며 train→test pair 진단도 따로 확인한다.

## 실행기와 결과

실험을 시작하지 않는 preview:

```bash
cd /home/user/Desktop/3ableGS
.conda/bin/python -m tools.run_teacher_confidence_experiments
```

향후 사용자가 실행을 요청한 뒤 사용할 명령:

```bash
cd /home/user/Desktop/3ableGS
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .conda/bin/python -u -m tools.run_teacher_confidence_experiments \
  --output-dir output/pbr/teacher_confidence/scene788_from_scratch_v1 \
  --execute
```

```bash
.conda/bin/python -m tools.run_teacher_confidence_experiments \
  --output-dir output/pbr/teacher_confidence/scene788_from_scratch_v1 --status
```

기본 순서는 `B0_s0_train → B0_s0_images → V_s0_train → V_s0_images → V_s0_temporal`이다. `--steps 3000` 미만은 cache를 만들지 않는 짧은 검증용 설정이다. `--steps 3000`에서는 cache만 생성하고 confidence strength가 0이므로 품질 실험으로 해석하지 않는다.

출력은 공통 queue의 `manifest.json`, `state.json`, `summary.json/csv`, `logs/`, `stages/`, `reference_flow/` 구조를 사용한다. Summary에는 `image_delta_V_minus_B0`, `training_stage_wall_time_ratio_V_over_B0`, temporal residual 감소율, 마지막 `confidence_stats/raw/*`와 material loss를 기록한다. Training wall time에는 초기화·cache 생성·최종 저장이 모두 포함된다. 마지막 logger window의 평균이며 전체 학습 평균을 새로 계산한 값은 아니다.

Checkpoint의 config·train IDs·카메라 수·crop·build step·map shape/dtype/range·cache byte 수를 검증한다. 완료된 V checkpoint에 cache가 없거나 B0에 cache가 섞인 경우를 거부한다. Code/config/input identity, artifact hash, queue lock, 실패 attempt 보존과 완료 stage skip은 기존 실행기를 재사용한다. 미완료 학습은 새 attempt에서 처음부터 시작하며 optimizer/RNG exact resume가 아니다. 앞선 중단 실험은 재개하거나 삭제하지 않는다.

## 해석의 한계

Warmup step 3000의 geometry가 부정확하면 correspondence와 confidence도 부정확할 수 있고, 고정된 cache는 이후 geometry 개선을 반영하지 않는다. Quarter-resolution 정합과 moment 보간은 고주파 텍스처 경계를 불확실성으로 판단할 수 있다. 서로 일치하지만 잘못된 teacher는 높은 confidence를 받을 수 있다. 원시 variance는 teacher 오류의 확률이 아니며, dynamic 표면의 temporal consistency를 직접 개선하는 제약도 아니다.

평가 결과가 나오기 전에는 temporal 개선량이나 학습시간 증가율을 주장하지 않는다. Temporal residual이 낮아져도 LPIPS·edge 보존과 정적/동적 진단을 함께 봐야 한다.

## 구현 검증

2026-11-01 기준 전체 CPU 회귀 테스트 569개 중 **565개 통과, CUDA 필수 기존 항목 4개 skip**이다. 신규 검증은 moments·geometry·정규화 core 20개, cache runtime 11개, 실제 PBR 메서드의 합성 integration 6개, 가짜 child를 사용한 queue 11개다. 알려진 이동·회전·subpixel 대응, 가림·cycle 검사, train-only cache 생성과 RNG·render 상태 복원, 원래 RGB·normal loss 및 dynamic gradient 보존, checkpoint 복원과 추론 시 teacher 없이 동작하는 경로를 확인했다.

로그는 `logs/teacher_confidence_validation/cpu_tests.log`에 저장했다. 기본 preview와 Python 문법 검사, `git diff --check`도 통과했다. 실제 scene 학습·cache 생성·평가 실험과 background service는 실행하지 않았으며, 기본 실험 출력 폴더도 생성하지 않았다. 실제 품질과 전체 학습시간은 미측정이다.
