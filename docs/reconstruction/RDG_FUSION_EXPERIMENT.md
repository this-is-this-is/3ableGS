# 실험 5: Base v4 + RDG-inspired lightweight G-buffer fusion

이번 작업은 **구현과 CPU 검증만** 진행한다. Scene 학습·평가와 background service는 시작하지 않는다. 실행기의 기본 호출은 읽기 전용 preview이며, `--execute`를 명시해야 실제 프로세스를 실행한다.

## 가설과 연구 근거

실험 5는 G-buffer를 기존 material loss의 target으로 사용하는 것과 함께, **현재·이전 프레임의 생성된 재질을 정렬하고 얼마나 섞을지 판단하는 입력**으로 사용한다. Normal·BRDF 정보를 읽는 appearance branch와 depth·왕복 재투영·normal 정합을 읽는 geometry branch를 분리하고, 작은 attention head가 현재·이전 두 후보의 가중치를 예측한다.

근거는 [RDG, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zheng_Efficient_Video_Super-Resolution_for_Real-time_Rendering_with_Decoupled_G-buffer_Guidance_CVPR_2025_paper.html)와 [공식 코드](https://github.com/sunny2109/RDG)다. 원 연구는 normal·BRDF를 이용한 detail 복원과 depth·motion 제약을 이용한 temporal refinement를 분리한 실시간 super-resolution 모델이다. 여기서는 그 분리 원리를 **한 프레임 history와 저해상도 material attention**으로 옮긴다. 원 논문의 UNet·DFM·HFB·CTR 전체나 super-resolution 성능을 재현하는 구현은 아니다.

## Forward와 gradient

1. 활성화된 update에서 같은 카메라의 이전 **train** frame을 하나 고른다. 최대 실제 timestamp gap은 2이며, 예를 들어 frame 9의 학습에서는 test frame 8을 건너뛰고 train frame 7을 사용한다. 유효 이전 frame이 없으면 원본 material을 유지한다.
2. 현재 모델로 history의 normal·albedo·roughness·metallic·depth·opacity를 가로·세로 1/4 해상도에서 `no_grad` 렌더한다. Teacher와 reference RGB를 context forward에 전달하지 않는다. 원본 material을 사용하며 history 안에서 또 다른 history를 호출하지 않는다.
3. 현재 material을 저해상도로 줄이고 실제 resize 비율에 맞춰 K를 조정한다. Albedo 평균과 transport는 linear 공간에서 수행한다. Depth와 camera pose로 history를 재투영하고, opacity·static support·depth·normal·왕복 오차로 가림과 잘못된 정합을 제외한다.
4. 저해상도 appearance/geometry feature로 현재·history 두 후보 attention을 계산한다. 최대 history blend는 0.35이며, low-frequency linear-albedo 변화량을 ±0.05로 제한한다.
5. 작은 residual만 원본 해상도로 올려 현재 albedo에 더한다. Native static·유효 표면 mask를 다시 적용해 sky·동적 물체·geometry 경계에는 보정이 번지지 않게 한다. Native의 고주파 성분을 직접 저해상도 history로 교체하지 않는다. 색 범위 clamp가 걸리는 곳에서는 detail이 변할 수 있다.
6. 수정된 albedo와 현재 normal·roughness·metallic·view direction·lighting으로 현재 시점에서 shading한다. 원래 G-buffer loss는 보정 전 material을 감독한다. 따라서 기존 normal/material supervision과 view-dependent specular shading 경로를 유지한다.

Geometry·history·attention routing 입력은 detach한다. 현재 albedo와 attention parameter에는 RGB loss와 보조 loss의 gradient가 전달되어 원래 3D scene과 함께 학습한다. History 렌더는 RNG, Gaussian refinement 정보, camera·dataset downscale, normal regularizer cache 등 현재 update 상태를 보존한다. Optimizer update 사이에 history를 cache하지 않으므로 이전 parameter 상태에서 생성된 material을 재사용하지 않는다.

Generated static mask는 annotation을 읽지 않고 현재 opacity와 dynamic-only opacity로 구한다. Dynamic Gaussian이 있는 경우 현재·history에 추가 opacity raster pass가 필요하다. 투영 함수의 공통 packet 형식을 맞추는 `rgb` 항목은 생성된 albedo의 alias이며, 별도 RGB guide shading이나 reference image가 아니다.

## 학습 설정과 비용

| 설정 | D 기본값 |
|---|---:|
| Feature/history 해상도 | 가로·세로 1/4 |
| Hidden width | 16 |
| History 수 / train 최대 gap | 1 / 2 |
| Warmup / ramp | 5000 / 1000 step |
| 학습 중 처리 간격 | 8 step |
| 최대 history blend / linear 변화량 | 0.35 / ±0.05 |
| Anchor / temporal weight | 0.01 / 0.02 |
| Head learning rate / weight decay | 0.001 / 0 |
| Head gradient clip | 1.0 |

기본 width 16의 fusion head는 학습 parameter 2,116개다.

Warmup 중에는 기존 Base v4 forward를 수행한다. Step 5000의 strength는 0이고, 이후 1000 step 동안 증가한다. 활성 간격에서만 추가 history·attention·loss를 계산하며 다른 update는 원래 renderer를 사용한다. `--steps 5000` 이하의 짧은 실행은 fusion 품질 실험이 아니다.

Anchor는 보정이 원본 linear albedo에서 과도하게 벗어나지 않도록 제한한다. Temporal 항은 유효하게 정합된 low-resolution history와 보정된 현재 linear albedo 사이의 robust residual이다. **Temporal loss support는 고정 geometry mask**이며 학습된 attention weight로 loss 자체를 축소하지 않는다. 각 loss는 설정 가중치와 ramp strength를 적용한다. 기존 RGB·G-buffer·geometry·lighting loss 가중치는 그대로 둔다.

추론에서는 checkpoint step의 ramp strength를 사용하고 학습용 interval을 적용하지 않는다. 테스트를 frame 8, 16처럼 띄엄띄엄 평가해도 각각 실제 frame 7, 15를 현재 모델로 생성해 사용한다. 이전에 호출된 test frame을 history로 사용하는 stateful 방식이 아니므로 평가 순서에 영향을 받지 않는다. Novel camera나 weather transfer는 이 recorded raw-camera 비교 범위에 포함하지 않는다.

이 설계는 attention과 geometry를 저해상도에서 계산하고 학습 호출 빈도를 줄이지만, **추론에도 low-resolution context render·attention·static mask 비용이 추가된다**. Native residual과 mask 연산도 필요하다. 추가 학습시간·추론 latency·peak memory는 아직 측정하지 않았으며, 기존 A2보다 빠르다는 수치 주장은 하지 않는다. 원 RDG 논문의 FPS는 이 구현에 적용되지 않는다.

## Base v4와 동일한 비교

| Arm | 초기화 | Forward | Optimizer update |
|---|---|---|---|
| B0 | Dataset에서 처음부터 | 원본 Base v4 | 0..30000 |
| D | B0와 동일 seed로 처음부터 | 저해상도 history attention + 제한된 albedo 보정 | 0..30000 |

기본은 scene 788, camera 0, frame 000–099, train 88/test 12, seed 0이다. 원래 inclusive loop를 유지해 각각 **30,001번** 업데이트한다. 실행기는 기존 `output/pbr/rebuild/scene_788_100f/base_v4_raw/config.yaml`의 설정만 읽으며 기존 checkpoint로 초기화하지 않는다. 두 arm의 설정은 `trainer.rdg_fusion.enabled`만 다르고, 실험 1–4나 기존 A2를 함께 켜지 않는다.

두 arm 모두 최종 checkpoint만 저장하고 독립 image stage에서 원래 `tools.eval`을 실행한다. Test 12장, native 1920×1280, 원래 affine 정책과 PSNR·SSIM·LPIPS 및 occupied/dynamic/human/vehicle 집계를 유지한다. 별도 `tools.eval_gbuffer_attention`으로 동일 reference flow·mask의 paired temporal residual·raw warp·edge·coverage를 비교한다. 연속 경로에는 train frame이 포함되므로 held-out 연속 영상 성능으로 해석하지 않으며 train→test 진단도 함께 확인한다.

Temporal residual 감소와 함께 LPIPS·edge 보존·history coverage·추론 시간을 본다. 저해상도 history의 색을 과도하게 따르거나 어려운 경계의 보정을 모두 꺼도 일부 지표가 낮아질 수 있다. 이 구현의 direct temporal constraint는 정적 표면 대상이며 동적 물체 temporal consistency를 직접 학습하지 않는다.

## 실행기

실험을 시작하지 않는 preview:

```bash
cd /home/user/Desktop/3ableGS
.conda/bin/python -m tools.run_rdg_fusion_experiments
```

향후 사용자가 실행을 요청한 뒤 사용할 명령:

```bash
cd /home/user/Desktop/3ableGS
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .conda/bin/python -u -m tools.run_rdg_fusion_experiments \
  --output-dir output/pbr/rdg_fusion/scene788_from_scratch_v1 \
  --execute
```

```bash
.conda/bin/python -m tools.run_rdg_fusion_experiments \
  --output-dir output/pbr/rdg_fusion/scene788_from_scratch_v1 --status
```

순서는 `B0_s0_train → B0_s0_images → D_s0_train → D_s0_images → D_s0_temporal`다. Queue는 `manifest.json`, `state.json`, `summary.json/csv`, `logs/`, `stages/`, `reference_flow/`를 기록한다. Summary에 `image_delta_D_minus_B0`, `training_stage_wall_time_ratio_D_over_B0`, temporal residual 감소율, 마지막 `rdg_stats/raw/*`, `losses/raw/rdg_anchor`, `losses/raw/rdg_temporal`을 담는다. Training wall time에는 초기화·최종 저장이 포함되며 마지막 logger window 값은 전체 학습 평균이 아니다.

`active_update`는 모든 update에서 기록하고, strength·context_count·history_coverage·history_weight는 fusion이 활성인 update마다 기록한다. 후자의 logger window는 활성 update 기준이다. Print 주기와 fusion 간격의 불일치로 실제 동작이 누락되지 않도록 한다. Temporal 평가의 forward 시간에는 history 데이터 준비·resize·device transfer·렌더가 포함되며, 현재 view의 사전 로딩·전송, flow 계산과 파일 저장은 제외된다. RDG에는 history cache를 사용하지 않으므로 `--streaming-cache`를 함께 지정하면 실행 전에 거부한다.

Checkpoint의 config·step과 fusion head key·shape·dtype·finite 값, B0의 fusion state 부재를 검증한다. Code/config/input identity, artifact hash, queue lock, 완료 stage skip과 실패 attempt 보존은 기존 실행기를 재사용한다. 중단된 학습 stage는 새 attempt에서 처음부터 시작하며 optimizer/RNG exact resume가 아니다. 앞선 중단 실험을 재개하거나 삭제하지 않는다.

## 검증 범위

Core·runtime·PBR integration과 가짜 child 기반 queue의 CPU 검증으로 geometry·경계 rejection·linear 색공간·gradient·state 복원·checkpoint와 평가 프로토콜을 확인한다. 기본 preview는 output directory나 child process를 생성하지 않는다. 실제 scene 품질, 학습시간, 추론 latency 및 GPU peak memory는 학습·평가 실행 후 확인해야 한다.

구현 시 전체 회귀 테스트 616개 중 612개 통과, CUDA 환경이 필요한 기존 테스트 4개 skip을 확인했다. 실험 5 전용 core/runtime/integration/queue 검증은 46개다. 로그는 `logs/rdg_fusion_validation/cpu_tests.log`에 있으며 실제 학습·평가 결과가 아니다.
