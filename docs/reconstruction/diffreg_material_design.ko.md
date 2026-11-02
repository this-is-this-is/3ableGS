# Base v4 + G-buffer material relation regularization

사용자 요청에 따라 **기존 checkpoint를 미세조정하지 않고**, Base v4와 재질 관계 regularizer를 추가한 R을 각각 데이터 초기화부터 학습한다. 기본은 scene 788, camera 0, frame 000–099, seed 0, nominal 30,000 step이다. 원래 loop가 0..30000 inclusive라서 각 arm은 **30,001 optimizer updates**를 수행한다.

## 무엇을 검증하는가

기존 G-buffer loss는 각 프레임의 예측 재질을 teacher의 값에 맞춘다. 추가 항은 teacher의 albedo·roughness·metallic **관계**를 이용하여, 같은 재질로 보이는 인접 표면에서 학습 중인 material map이 불필요하게 달라지는 것을 억제한다. 이 공간 제약이 시점 변화에 따른 flicker도 줄이는지 검증한다. 직접 temporal loss나 attention은 넣지 않는다.

근거는 [Diffusion-Based Material Regularization for Physics-Based Inverse Rendering, ECCV 2026](https://arxiv.org/html/2606.31065v1)과 [공식 구현](https://github.com/gerwang/DiffReg-PBIR)이다. 이 실험은 전체 논문 재현이 아니라 **Base v4용 sampled local approximation**이다. 원 논문의 전역 permutohedral filtering, mesh reconstruction, inverse path tracing은 도입하지 않는다.

## 구현 및 설정

- [models/diffreg_material.py](../../models/diffreg_material.py): 현재 프레임에서 최대 4096개의 anchor를 샘플링하고, 각 anchor의 7×7 이웃에 대해 같은 teacher material kernel을 계산한다. 이미지 전체의 H×W×49 tensor를 만들지 않는다.
- Teacher의 `[albedo RGB, roughness, metallic]` 차이로 하나의 joint Gaussian kernel을 만든다. `sigma_g=0.1`, spatial `sigma_xy=2.0`이다. 원래 teacher 값의 표현을 그대로 쓰며, 이 대역폭은 이번 실험의 사전 고정값이다.
- Static annotation으로 sky·dynamic·ego-car를 제외하고, 유효 opacity/depth 및 depth·normal의 불연속을 검사한다. Teacher와 geometry routing은 detach하고 예측된 현재/이웃 material에는 gradient를 전달한다. RGB, opacity, geometry, material, lighting에 대한 기존 Base v4 학습 경로는 유지한다.
- Albedo는 log 공간에서 이웃을 평균하고, center의 detached albedo로 residual을 가중한다. 즉 `stopgrad(Ap) * abs(log(Ap) - weighted_mean(log(Aq)))` 형태이다. Roughness·metallic은 선형 공간에서 비교한다. 이것은 roughness까지 log 변환하는 공식 구현과 구별되는 Base v4 변형이다.
- Albedo는 RGB 평균, roughness/metallic은 각각 scalar 평균 후 세 재질 항을 동일 비중으로 평균한다. 기존 teacher loss 가중치는 그대로 유지한다.
- 시작 step 3000, ramp 1000, 최대 가중치 0.05. 비활성·무효 support에서 graph-safe 0을 반환한다. 별도 sampler RNG를 사용해 Base v4의 초기화·프레임 샘플링·boundary LPIPS 샘플링 stream을 바꾸지 않는다.
- 추론에는 추가 연산이 없고, 학습에도 다른 프레임의 렌더나 diffusion 실행이 없다. 실제 시간 증가율은 같은 seed의 전체 학습 시간과 별도 GPU smoke/profile로 확인한다.

| Arm | 초기화 | Loss | Step |
|---|---|---|---|
| B0 | 원래 dataset 초기화 | Base v4 | 0..30000 |
| R | B0와 같은 초기화/seed | Base v4 + local material relation | 0..30000 |

[B0 overlay](../../configs/diffreg_material/B0.yaml)와 [R overlay](../../configs/diffreg_material/R.yaml)는 loss 항만 정의한다. 실행기는 기록된 `output/pbr/rebuild/scene_788_100f/base_v4_raw/config.yaml`을 읽어 원래 scene/teacher/optimizer/density schedule을 복사한다. 이 경로의 checkpoint 파일은 읽거나 학습 초기값으로 사용하지 않는다.

두 arm에 공통으로 적용하는 실행상 차이는 다음 두 가지이다.

1. 중간 checkpoint 축적을 막기 위해 `saveckpt_freq=30000`으로 최종 checkpoint만 저장한다. 최종 모델 약 1GB씩과 평가/flow 파일을 제외한 중간 모델 저장을 줄인다.
2. 학습 config에서는 `render_test=false`로 자동 평가를 미루고, 학습이 성공한 뒤 독립된 image stage에서 `render_test=true`를 복원해 원래 `tools.eval`을 실행한다. 평가가 실패해도 완료된 30k 학습은 다시 하지 않는다.

이 두 항목은 optimization loss·update·data split·평가 metric을 바꾸지 않는다. 공식 이미지 평가는 같은 stride-8 test 12장, native 1920×1280, 원래 affine 정책, 원래 PSNR/SSIM/LPIPS 및 occupied/dynamic/human/vehicle 집계이다.

## 실행

기본 호출은 GPU 실행과 출력 디렉터리 생성이 없는 preview다.

```bash
cd /home/user/Desktop/3ableGS
.conda/bin/python -m tools.run_diffreg_experiments
```

Base v4 B0부터 R까지 학습·평가를 순차 실행한다.

```bash
cd /home/user/Desktop/3ableGS
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .conda/bin/python -u -m tools.run_diffreg_experiments \
  --output-dir output/pbr/diffreg_material/scene788_from_scratch_v1 \
  --execute
```

```bash
.conda/bin/python -m tools.run_diffreg_experiments \
  --output-dir output/pbr/diffreg_material/scene788_from_scratch_v1 --status
```

`--steps`는 검증용 짧은 run에도 사용할 수 있다. 3000 이하로 실행하면 regularizer가 본격 활성화되지 않으므로 품질 실험으로 해석하지 않는다. Full run은 설정을 바꾸지 않고 30k를 수행한다. 추가 seed가 필요하면 **새 output directory**에서 `--seeds 0 1 2`를 사용한다.

## 평가와 결과 파일

실행 순서는 `B0_s0_train → B0_s0_images → R_s0_train → R_s0_images → R_s0_temporal`이다. Temporal stage는 두 final checkpoint를 [기존 evaluator](../../tools/eval_gbuffer_attention.py)에 전달한다. 같은 reference optical flow/mask로 full-sequence residual·raw warp·edge error/strength·coverage를 계산한다. 이는 **train frame을 포함한 reconstruction-path 진단**이며 연속 held-out video 성능을 뜻하지 않는다. Train→test pair 집계도 따로 유지한다.

출력 루트에는 다음 파일을 남긴다.

- `manifest.json`: source config/overlay/code hash, scene·teacher의 size/mtime, train/test IDs, 초기화/평가 정책과 명령.
- `state.json`: 실행 중인 stage, PID, attempt별 성공/실패 및 시간.
- `summary.json`, `summary.csv`: 원래 이미지 지표, R−B0 차이, temporal residual 감소율, stage 시간, 마지막으로 기록된 DiffReg loss/support 통계.
- `logs/`: 각 child process의 독립 로그.
- `stages/`: 최종 모델과 기존 평가 결과. Image stage checkpoint는 가능하면 hard link로 공유해 디스크 중복을 줄인다.
- `reference_flow/`: B0/R에 공통인 평가 flow cache.

Training wall-time ratio는 초기화와 checkpoint 저장을 포함한다. 별도 microbenchmark 시간을 전체 학습시간으로 대체하지 않는다. Blur만으로 temporal residual이 낮아지는 경우를 걸러내기 위해 LPIPS와 edge 진단도 같이 해석한다. 이 실험의 spatial regularizer가 dynamic 영역의 temporal consistency를 개선한다고 미리 가정하지 않는다.

실행기는 중복 queue를 lock으로 막고 stage 시작 전 가용 디스크를 확인한다(기본 최소 20GiB). 같은 명령을 재실행하면 완료된 artifact hash를 검증한 뒤 건너뛰고, 미완료 stage는 새 `attempt_NNN`에서 다시 시작한다. 이는 **optimizer/RNG를 복원하는 정확한 학습 resume가 아니다**. Code·config·입력 identity가 바뀌면 새 output directory를 요구한다. 실패한 attempt와 기존 실험 파일은 삭제하지 않는다.

Checkpoint 선택은 final step으로 고정하며 test 지표를 보고 중간 checkpoint/loss weight를 선택하지 않는다. 이 씬에서 설정을 바꾸는 후속 비교는 exploratory로 기록하고, 일반화 주장은 별도 scene/seed 확인 뒤에 한다.

## 구현 검증

2026-11-01 기준 전체 CPU 회귀 검증 448개 중 444개 통과, CUDA 전용 4개 skip이다. 별도로 [CUDA 검사](../../tools/check_diffreg_material.py)를 실제 GPU에서 실행하여, 동일 seed의 초기 state·기존 loss 보존, 새 regularizer만의 material decoder/feature/geometry gradient, densification AbsGrad 및 strict checkpoint reload를 확인했다. 재로드 RGB 최대 차이는 `1.19e-7`이다.

[GPU 검증 결과](../../logs/diffreg_validation/gpu_check.json)의 native 1920×1280 합성 material-map microbenchmark는 loss forward+backward 중앙값 약 2.95ms, 추가 peak 약 66.4MiB다. 이 측정은 renderer·teacher 정렬·데이터 로딩을 제외하므로 전체 학습시간 증가율로 해석하지 않는다. 장면 품질과 전체 비용은 위 paired run의 완료 후 평가한다.

## 시작한 실행

`scene788_from_scratch_v1` queue는 터미널과 분리된 사용자 systemd service `3ablegs-diffreg-scene788-v1.service`로 시작했다. 서비스는 B0 학습·평가 후 R 학습·평가와 paired temporal 진단을 자동 실행한다. 실험 완료 여부는 출력 루트의 `state.json`으로 확인한다.

```bash
systemctl --user status 3ablegs-diffreg-scene788-v1.service
tail -f output/pbr/diffreg_material/scene788_from_scratch_v1/logs/B0_s0_train_attempt_001.log
```

서비스 전체 로그는 `logs/diffreg_material_scene788_v1.log`, 세부 학습·평가 로그는 queue의 `logs/`에 있다. 같은 queue를 다시 실행하기 전 서비스가 종료됐는지 확인한다.
