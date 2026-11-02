# 실험 2: Base v4 + sparse material consistency

이번 작업은 **구현과 검증만** 수행한다. 사용자 요청에 따라 학습·평가 실험과 background service는 실행하지 않는다. 실행기의 기본 호출 역시 preview이며 `--execute`를 명시해야 프로세스가 시작된다.

## 목적과 연구 근거

실험 1(DiffReg)은 현재 프레임의 teacher G-buffer가 알려주는 재질 유사성으로 공간 regularization을 걸었다. 실험 2는 **서로 다른 학습 프레임에서 같은 정적 표면의 렌더된 재질을 직접 비교**한다. 기존 Base v4 RGB·G-buffer supervision은 유지하고, 실험 1의 DiffReg 항은 켜지 않는다.

근거 연구는 [MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference, NeurIPS 2025](https://arxiv.org/html/2510.11387v1)이다. 논문은 Gaussian 속성이 공유돼도 시점별 alpha blending으로 렌더된 material map이 불일치할 수 있음을 설명하고, depth·normal 기반 평면 homography로 7×7 패치를 대응시켜 consistency를 부과한다. 논문 전체에는 2DGS ray tracing과 reflection prior도 포함된다.

여기서는 Base v4에 맞춰 **현재 표면의 depth 재투영, 단일 detached 이웃, 희소 point 샘플, L1 loss**를 사용한다. 원 논문의 multi-source 7×7 평면 패치/MSE·렌더러 전체 재현은 아니다. 논문의 scalar albedo는 specular 모델에서 쓰이고, consistency는 diffuse color·roughness·metallic에 적용된다. 우리 RGB base-color albedo consistency는 이 원리를 기존 PBR 채널에 옮긴 응용이다.

## 학습 방법

원래 `TemporalMaterialTraining` 경로에 opt-in sparse 모드를 추가한다. `num_samples=0`은 기존 dense 경로이며, 새 T 설정은 `num_samples=4096`으로 sparse 경로를 선택한다.

1. 3000 step 이후 **8 step마다** 현재 이미지와 같은 카메라의 이전 train 이미지를 한 장 선택한다. 최대 frame gap은 2이며, stride-8 test 프레임은 건너뛴다. 예를 들어 train frame 9는 test frame 8 대신 train frame 7을 사용한다. 허용된 이전 이미지가 없으면 해당 update의 temporal 항을 생략한다.
2. 이웃은 현재 model weight로 `no_grad` 렌더한다. 현재 step마다 새 target을 만들며, 오래된 weight로 만든 history cache는 사용하지 않는다. 이 렌더는 현재 main render 전에 수행하고, RNG·renderer 상태를 보존한다.
3. 현재 프레임의 유효한 정적 표면에서 최대 4096개 샘플을 별도의 RNG로 선택한다. 후보가 더 많으면 균등 복원추출하고, 적으면 모두 사용한다. Depth와 camera transform으로 이웃에 재투영하고, 가중치가 양수인 bilinear footprint의 validity, depth·normal 정합, 왕복 재투영을 검사한다.
4. 통과한 위치에서 현재 material과 detached 이웃 material을 비교한다. Albedo는 linear 공간에서 비교하며 **이웃 albedo는 선형화한 뒤 bilinear 보간**한다. Roughness·metallic은 기존 scalar 공간을 유지한다.

```text
L = L_base_v4 + lambda(step) * L_material_pair
L_material_pair = (
    mean_valid |Alinear_current - warp(stopgrad(Alinear_previous))|
  + mean_valid |roughness_current - warp(stopgrad(roughness_previous))|
  + mean_valid |metallic_current - warp(stopgrad(metallic_previous))|
) / 3
```

Albedo RGB의 채널 평균을 먼저 계산해 세 재질 항의 비중을 맞춘다. 현재 material 및 그 rasterization 경로에 gradient가 흐르고, 이웃 target·대응 좌표·acceptance mask는 detach한다. RGB 자체를 시간에 따라 같게 만들지 않으므로, 실제 시점 의존 반사는 현재 PBR shading이 표현한다.

| 설정 | T 기본값 |
|---|---:|
| Temporal weight | 0.05 |
| 시작 / ramp | 3000 / 1000 step |
| 계산 주기 | 8 step |
| 이웃 수 / 최대 frame gap | 1 / 2 |
| 최대 current valid samples | 4096 |
| Sample RNG seed | 20261101 |
| Opacity 최소값 | 0.95 |
| Depth 허용오차 | 0.05 + 0.02 × projected depth |
| Normal 기준 | absolute cosine ≥ 0.9 |
| 왕복 허용오차 | 1 pixel |
| Static mask erosion | 1 pixel |
| Depth edge 상대 기준 | 0.1 |

Weight는 3000에서 0, 4000부터 0.05이다. 매 8 step 실행한다고 weight를 8배로 보상하지 않는다. 따라서 이 설정은 full-frequency temporal loss보다 계산 빈도와 전체 regularization budget을 함께 줄인 고정 실험이다. 누락된 이웃이나 유효 대응이 없는 경우를 진단 로그에 기록한다.

기본 0..30000 학습에서는 첫 활성 step이 3008이고 최대 3375번의 추가 렌더 기회가 있다. 활성 step의 coefficient 합은 165.65이며, 이웃이나 대응이 없는 경우 실제 유효 loss budget은 더 작다. 이는 loss 값의 합이나 성능 향상을 예측한 수치가 아니다.

Sky·dynamic objects·ego car는 양쪽 프레임에서 제외한다. 움직이는 물체에 카메라 변환만 적용하면 잘못된 대응이 되므로, 이 실험에서 동적 객체의 temporal 개선을 직접 주장하지 않는다. 추론은 원래 Base v4 renderer를 그대로 사용하며 history나 필터가 추가되지 않는다.

## 처음부터 학습하는 비교

| Arm | 초기화 | 학습 loss | 업데이트 |
|---|---|---|---|
| B0 | 원래 dataset 초기화 | Base v4 | 0..30000 |
| T | B0와 같은 seed·초기화 | Base v4 + sparse material pair | 0..30000 |

실행기는 기존 `output/pbr/rebuild/scene_788_100f/base_v4_raw/config.yaml`의 **설정만** 읽는다. 해당 checkpoint 가중치를 로드하지 않는다. 두 arm은 scene 788, camera 0, frame 000–099, seed 0, train 88/test 12를 유지한다. 원래 loop가 inclusive이므로 각 arm의 업데이트는 30,001회다.

기존 optimizer·density·resolution·teacher·LPIPS schedule은 유지한다. 저장 공간을 위해 두 arm 모두 최종 checkpoint만 저장하고, 학습 성공 뒤 독립된 image-evaluation stage에서 원래 `tools.eval`을 호출한다. 평가 실패 때문에 완료된 학습을 재실행하지 않도록 단계를 나눈 것이다.

추가 비용은 scheduled step의 이웃 렌더 한 번과 희소 정합/loss이다. 샘플 수를 줄여도 이웃 renderer는 전체 이미지를 계산한다. **추가 렌더 빈도가 1/8이라는 것이 전체 학습시간 증가율 12.5%를 보장하지는 않는다.** Native 해상도와 실제 Gaussian 수에서 두 arm의 wall time을 비교해야 한다.

## 실행기와 결과

실험을 시작하지 않는 preview:

```bash
cd /home/user/Desktop/3ableGS
.conda/bin/python -m tools.run_material_consistency_experiments
```

향후 사용자가 실행을 요청한 뒤 사용할 명령:

```bash
cd /home/user/Desktop/3ableGS
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .conda/bin/python -u -m tools.run_material_consistency_experiments \
  --output-dir output/pbr/material_consistency/scene788_from_scratch_v1 \
  --execute
```

```bash
.conda/bin/python -m tools.run_material_consistency_experiments \
  --output-dir output/pbr/material_consistency/scene788_from_scratch_v1 --status
```

기본 순서는 `B0_s0_train → B0_s0_images → T_s0_train → T_s0_images → T_s0_temporal`이다. 짧은 `--steps` 설정은 실행기 smoke용이며, 3000 이하에서는 새 항이 작동하지 않으므로 품질 실험으로 해석하지 않는다.

이미지 평가는 원래 Base v4 test 12장, 1920×1280, 같은 affine 정책과 PSNR·SSIM·LPIPS 및 occupied/dynamic/human/vehicle 집계를 유지한다. Temporal은 [기존 evaluator](../../tools/eval_gbuffer_attention.py)의 같은 reference flow·mask로 residual, raw warp, edge, coverage를 평가한다. 전체 경로에는 train frame이 포함되므로 연속 held-out 영상 평가가 아니며, train→test pair 진단도 따로 보고한다.

출력 루트에 `manifest.json`, `state.json`, `summary.json/csv`, `logs/`, `stages/`, `reference_flow/`를 저장한다. 요약에는 `image_delta_T_minus_B0`, `training_stage_wall_time_ratio_T_over_B0`, temporal residual 감소율, 마지막 temporal support/loss 통계가 포함된다. 시간에는 데이터 초기화·최종 checkpoint 저장도 포함되며 평가 시간은 따로 기록한다.

학습 `metrics.json`의 `temporal_stats/raw/*`에서 sample·valid sample·각 gate 통과 수, support fraction, 각 material loss와 scheduled update·실제 이웃 render·missing pair 수를 확인할 수 있다. Queue 요약은 원래 logger의 마지막 평균 window를 옮긴 값이며 전체 학습 평균을 새로 계산한 값은 아니다.

실행기는 실험 1의 공통 queue 기능을 재사용한다. Lock, 입력/코드/config identity 검사, 완료 artifact hash 확인, 실패 attempt 보존, 같은 명령 재시도 시 완료 stage skip을 유지한다. 미완료 학습은 새 attempt에서 처음부터 시작하며 optimizer/RNG의 exact resume는 아니다.

이번 코드 변경으로 중단된 실험 1 queue의 code identity는 달라진다. 그 기록은 보존하며 자동 재개하지 않는다. 추후 실험 1을 다시 시작하려면 새 output directory를 사용해야 한다. 실험 2는 별도의 출력 경로를 사용한다.

## 구현 검증 결과

2026-11-01 기준 CPU 회귀 테스트 483개 중 479개 통과, 기존 CUDA 필수 항목 4개는 sandbox에서 GPU에 접근할 수 없어 skip했다. 새 항목은 sparse loss 19개, training hook 7개, queue 9개다. 합성 입력으로 알려진 camera 이동·회전과 subpixel 대응, 가림 판정, 현재 material gradient 및 detached target, history 렌더 전후 RNG·cache·downscale 복원, held-out 차단, 원래 loss 보존, 실행기의 실패 복구를 확인했다. 로그는 `logs/material_consistency_validation/cpu_tests.log`에 있다.

기본 preview, Python 문법 검사와 `git diff --check`도 통과했다. 실제 scene 학습·평가와 GPU 실험은 실행하지 않았으며 기본 실험 출력 폴더도 생성하지 않았다. 따라서 품질 개선량과 실제 학습시간 증가는 아직 측정되지 않았다.
