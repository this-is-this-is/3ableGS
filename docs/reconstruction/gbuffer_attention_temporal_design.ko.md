**Base v4 기반 G-buffer attention temporal consistency 실험 설계**

작성일: 2026-10-26. 설계 후 adapter·trainer 연결·pilot/joint runner를 구현했다. 실제 GPU 학습과 품질 평가는 아직 실행하지 않았다. 기존 실험 결과는 보존한다. 아래 수치 기준과 가설은 실험 설계이며, 마지막 구현 기록에서 현재 실행 방법과 검증 범위를 설명한다.

추천 실험은 **현재 G-buffer로 같은 표면의 과거 재질을 찾아 attention으로 선택하고, 현재 시점의 PBR로 다시 shading하는 방식**이다. 우선 static background의 albedo flicker를 대상으로 한다. 원래 Base v4 이미지 평가를 그대로 유지하고 temporal 지표는 별도 진단으로 추가한다. Attention이 geometry, normal, visibility 문제까지 모두 해결한다고 가정하지 않는다.

**확인한 기준과 선행 결과**

주 실험은 시간 길이가 더 긴 scene 788 / raw / 100 frames를 기준으로 하고, 30 frames는 기존 결과 재현 확인용으로 둔다. 둘의 점수를 섞지 않는다.

| 조건 | 100f: 주 실험 | 30f: 재현 확인 |
|---|---|---|
| 프레임 / 카메라 | 000–099 / camera 0 | 000–029 / camera 0 |
| train / test | 88 / 12 | 27 / 3 |
| test frame IDs | 8,16,24,32,40,48,56,64,72,80,88,96 | 8,16,24 |
| test PSNR ↑ | 30.024253 | 30.124935 |
| test SSIM ↑ | 0.922123 | 0.917821 |
| test LPIPS ↓ | 0.130565 | 0.108272 |

수치는 각각 [100f 평가 JSON](../../output/pbr/rebuild/scene_788_100f/base_v4_raw/metrics/raw/images_test_2026-10-19_00-38-17.json), [30f 평가 JSON](../../output/pbr/rebuild/scene_788_30f/base_v4_raw/metrics/raw/images_test_2026-10-17_12-44-11.json)에서 확인했다. 실행 설정의 기준은 [100f 저장 config](../../output/pbr/rebuild/scene_788_100f/base_v4_raw/config.yaml)이며, 현재 코드로 baseline을 다시 평가한 결과도 함께 보관한다. 저장된 과거 점수와 새 실행 점수가 다르면 원인을 확인한 뒤 비교한다.

이미 [ego_temporal.py](../../models/ego_temporal.py)에 1/4 해상도·width 32·2-block local Transformer와 G-buffer 재투영이 있다. [기존 pilot 결과](../../output/pbr/ego_motion/scene788_v1/summary.csv)는 다음과 같다.

| 기존 pilot | test PSNR ↑ | temporal residual ↓ |
|---|---:|---:|
| Frozen base | 24.540356 | 0.012760730 |
| Current-only CNN | 24.583246 | 0.012725354 |
| Full G-buffer CNN | 24.584477 | 0.012724751 |
| Transformer | 24.583781 | 0.012717500 |

Transformer의 temporal residual 감소는 base 대비 약 0.34%, current-only 대비 약 0.06%다. 따라서 기존 결과는 attention의 뚜렷한 효과를 입증하지 못한다. 이 pilot은 contiguous holdout checkpoint, downscale 2, pooled PSNR을 사용하여 위 Base v4 표와 직접 비교할 수 없다. 기존 runner를 그대로 재실행하는 대신 평가 연결부와 attention의 표면 대응을 보완한다.

검증할 가설은 세 가지다.

- 같은 3D 표면에 대응하는 albedo를 모으면 단일 프레임의 재질 흔들림이 줄어든다.
- Normal·roughness·metallic guide가 geometry-only attention보다 적절한 과거 후보를 고르는 데 도움이 된다.
- 개선은 단순 평균, 추가 네트워크 용량 또는 blur만으로 설명되지 않으며, 원래 test 이미지 품질을 유지한다.

**제안 모듈의 입출력과 위치**

```text
Base v4 Gaussian rasterization → 현재 A / N / R / M / Z / opacity
                                      ↓ query
과거 t−1, t−2, t−4의 모델 생성 G-buffer → geometry reprojection
                                      ↓ 후보별 validity + key/value
                              local cross-attention
                                      ↓ 보정한 linear albedo
현재 N / R / M / view / light → 원래 PBR → 원래 response / affine → RGB
```

삽입 위치는 [scene_graph_pbr.py](../../models/trainers/scene_graph_pbr.py)의 `unpack_gbuffer()` 이후, `shade_gbuffer()` 이전이다. A=albedo, N=world normal, R=roughness, M=metallic, Z=camera-Z depth다. 재질 혼합은 linear albedo에서 하고 기존 shader 입력으로 전달할 때만 sRGB로 되돌린다. 이미 shading한 RGB의 highlight를 그대로 평균하지 않는다.

첫 버전은 **albedo만 보정**한다. Normal·roughness·metallic·depth는 routing과 현재 shading에 사용한다. Head가 이 값을 수정하지 않는다는 뜻이며, 뒤의 joint training에서 원래 scene parameter를 전부 freeze한다는 뜻은 아니다. Roughness 보정은 albedo-only의 효과를 확인한 뒤 독립 실험으로 추가한다.

입력은 모델이 생성한 G-buffer, 모델 RGB, opacity, camera pose/intrinsics와 시간 간격이다. RGB guide를 넣을 경우 모든 arm에서 동일한 pre-affine display RGB를 사용하고 최종 출력에는 원래 affine를 한 번 적용한다. GT RGB와 teacher G-buffer는 학습 supervision으로만 쓴다. GT dynamic/sky mask를 추론 입력으로 넣지 않는다.

현재 pixel p에 대해 다음을 수행한다.

1. 현재 camera-Z와 pose로 world point를 구해 과거 camera에 투영한다. 같은 camera의 과거 1·2·4 프레임만 사용한다. 과거 value는 temporal 보정 전의 base render이므로 재귀적 history drift가 없다.
2. 재투영 위치의 native-pixel 3×3 후보를 모은다. 현재 self 후보 1개를 포함하여 최대 28개다. 1/4-resolution encoder의 feature는 이 좌표에서 sampling한다. 3×3을 1/4 feature grid의 ±4 native pixels와 혼동하지 않는다.
3. 각 후보를 **중앙 query 표면**과 비교한다. 후보 자신의 재투영이 valid라는 이유만으로 이웃 query에 허용하지 않는다. 동일한 static support, 영상 내부, 양의 depth, opacity ≥0.95, depth 오차 ≤0.05m+2%, cycle error ≤1.5 native pixels를 기본 gate로 사용한다. Depth discontinuity와 object 경계는 보수적으로 제외한다.
4. Q/K에 G-buffer feature를 넣고 depth·normal·roughness·age의 차이를 attention logit bias로 더한다. Invalid 후보의 logit은 −∞다. 현재 self 후보는 항상 유효하다.
5. 마지막 attention weight로 실제 linear albedo value를 혼합한다. 기존 구현의 latent attention 후 별도 4-way gate와 달리, 어떤 과거 표면 값이 선택됐는지 직접 기록할 수 있게 한다.

수식으로는 다음과 같다. 아래 계수는 초기 제안값이며 검증된 최적값이 아니다.

```text
e(p,k) = Q(p)·K(k)/sqrt(d)
         − bz·normalized_depth_error(p,k)
         − bn·(1−|N(p)·N(k)|) − br·|R(p)−R(k)| − bt·age(k)
a(p,k) = softmax_k(e(p,k) + validity_mask(p,k))
Aatt(p) = Σk a(p,k) · Alinear(k)
Aout(p) = clip[Alinear(p) + S(p)·s(step)·{
                  β(p)·(Aatt(p)−Alinear(p)) + 0.02·tanh(r(p))}, 0, 1]
```

초기 bias 계수는 `bz=1, bn=2, br=2, bt=0.1`로 고정한다. Depth error는 `|Zsample−Zprojected|/(0.05+0.02·|Zprojected|)`, age는 실제 frame 차이/4로 정규화한다. Self 후보의 각 차이는 0이다. 이 값의 탐색은 별도 development scene에서만 한다.

`S`는 모델 기반 current static surface support이며 학습되지 않는다. `β=0.5·sigmoid(g)`로 history correction을 제한하고 residual head `r`은 0으로 초기화한다. `s(step)`은 학습 초기 bypass/ramp다. Invalid current surface, dynamic object, sky는 base 출력으로 돌아간다. History가 전부 invalid이면 self 값과 작은 current residual만 사용한다. Current-only 대조군은 이 residual의 효과를 측정한다.

Encoder width 32와 2개 local block으로 시작한다. Full-resolution value sampling과 validity 재검사는 spatial chunk로 처리하여 `H×W×28×32` key tensor 전체를 한꺼번에 만들지 않는다. Low-resolution logits를 올려 사용할 때도 native-resolution validity를 다시 적용하고 renormalize한다. 메모리·속도 이득은 구현 후 측정한다.

Geometry-only와 full-guide의 주 비교에서는 **hard geometry gate를 동일하게 유지**한다. Geometry-only는 A/N/R/M 및 N·V 같은 material-derived guide slot을 0으로 만들고 logit의 `bn=br=0`으로 설정한다. 전달할 albedo value 자체는 유지한다. 이 조건과 full guide를 비교하여 routing guide의 효과를 측정한다. Normal hard gate `|cos|≥0.9`를 추가하는 것은 별도 ablation이다. 이를 full-guide에만 조용히 적용하면 guide 효과와 coverage 감소 효과를 분리할 수 없다.

**실험군과 학습 순서**

| ID | 구성 | Temporal loss | 확인하는 효과 |
|---|---|---|---|
| B0 | 원래 Base v4 | 없음 | 기준 |
| H0 | B0 + geometry-gated 고정 가중 albedo transport | 없음 | 학습 없는 시간 평균 |
| C0 | current-only, 같은 입력 폭·residual head | 있음 | 추가 용량/단일 프레임 보정 |
| C1 | full G-buffer temporal CNN | 있음 | attention 이외의 learned fusion |
| A0 | geometry-only temporal attention | 있음 | G-buffer material guide 없이 attention |
| A1 | full G-buffer temporal attention | 없음 | temporal loss의 기여 분리 |
| A2 | full G-buffer temporal attention | 있음 | 제안 방법 |

H0는 A2와 같은 history IDs·geometry gate를 사용한다. 유효한 history의 linear albedo를 균등 평균하고 current 0.7 / history 0.3으로 섞으며 invalid이면 current로 돌아간다. 이는 재귀 EMA가 아닌 고정 윈도 transport다. C0/A0/A1/A2는 가능한 한 같은 network와 parameter 수를 사용한다. C1도 parameter 수를 ±5% 내로 맞추고 실제 FLOPs·latency를 별도로 기록한다. 기존 CNN 104,365 / Transformer 75,181개를 그대로 두고 동일 용량이라고 부르지 않는다.

실행은 다음 순서로 진행한다.

1. **평가 연결 검증:** 기존 30f/100f checkpoint를 원래 경로로 replay한다. Adapter bypass, shuffled evaluation order, sequential/cache rendering에서 같은 scored RGB가 나오는지 검사한다. 기존 test ID·해상도·metric key를 확인한다.
2. **Frozen pilot:** 동일 100f Base v4 checkpoint를 고정하고 C0/C1/A0/A1/A2 head만 각각 2,000 updates 학습한다. Seed 0, 같은 train target 순서, 같은 loss budget, Adam lr=1e-4, gradient clip=1로 시작한다. Head branch는 첫 200 updates 동안 0→1 ramp한다. B0/H0는 추가 학습이 없다. 추가 학습 비용과 frozen adaptation이라는 사실을 별도로 표기한다.
3. **주 실험:** B0/C0/A0/A1/A2를 데이터 초기화부터 각각 nominal 30k로 학습한다. 먼저 seed 0을 완주하고 미리 정한 판정 기준으로 후속 투자를 결정한다. 반복 시 동일 5개 arm을 seeds 0/1/2로 수행한다(총 15개 학습). H0는 각 B0 checkpoint에 추가 적용한다. CNN까지 확인할 여유가 있으면 C1도 동일 조건으로 반복한다.
4. **후속 확인:** history 1개 vs 3개, normal hard gate, albedo+roughness, static 외 rigid motion 대응을 한 축씩 추가한다. 이미 살펴본 scene 788의 수치는 exploratory이며, 일반화 주장은 별도 scene에서 같은 Base v4 protocol로 반복한 뒤 한다.

Pilot의 기존 Base v4 가중치는 freeze하되 보정 albedo를 통한 shader gradient는 보존한다. 모든 head가 원래 `.7 RGB + .3 SSIM + boundary LPIPS(.02)`로 supervised image fitting을 하고, frozen scene에만 걸리는 원래 auxiliary loss는 상수임을 구분한다. 원래 G-buffer teacher loss는 보정 전 scene material에 유지한다. Frozen pilot에서 새로운 teacher를 생성하지 않는다.

Joint training은 [Base v4 overlay](../../configs/base_v4/combo_perceptual_02.yaml), 저장된 resolution·optimizer·density schedule·teacher를 그대로 사용한다. Background split 종료 25k, reset cutoff 20k, RGB/SSIM 0.7/0.3, boundary LPIPS 5k 시작·1k ramp·최대 0.02를 유지한다. 실제 원래 loop는 step 0..30000 inclusive, 즉 30,001 updates다. 추가 head 초기화가 scene 초기화와 sampling RNG를 바꾸지 않도록 RNG stream을 분리한다.

Joint head는 step<5000에서 `s=0`, 5000..6000에서 0→1, 이후 1이다. 이는 추가 실험 설정이며 원래 Base v4의 boundary schedule은 그대로다. 기존 head의 `[2,0,0,0]` gate는 history 3개가 valid하면 초기부터 약 29%를 섞으므로 정확한 baseline 초기화로 재사용하지 않는다.

전체 loss는 다음과 같이 정의한다.

```text
L = L_base_v4(final_RGB, current_unfused_Gbuffer, original_auxiliary_outputs)
    + λtemp · L_material_temporal + 0.01 · L_anchor
L_material_temporal = mean_M Charbonnier(
    Aout_linear(t) − warp(stopgrad(Aout_linear(t−δ))))
L_anchor = mean_S |Aout_linear(t) − stopgrad(Alinear(t))|
```

`Charbonnier(x)=sqrt(x²+1e−6)−0.001`이고 RGB channel 평균을 사용한다. `λtemp`는 joint에서 6000..7000 동안 0→0.05로 ramp하며 pilot에서는 branch ramp에 맞춰 0→0.05로 올린다. A1은 이를 0으로 고정한다. 처음에는 roughness temporal loss를 넣지 않는다.

`M`은 offset/learned gate와 독립인 고정 geometric reprojection support다. Detached tensor라는 것만으로 학습된 sampling 위치와 독립이 되는 것은 아니므로, zero-offset geometry로 loss mask를 별도 계산한다. 유효 pixel 수가 0이면 graph-safe zero loss를 반환한다. Ground-truth RGB로 flow를 추정하는 보조 평가지표를 학습 loss에 재사용하지 않는다.

학습에서는 current와 모든 history를 저장된 train IDs로 제한한다. `{t−1,t−2,t−4}` 중 test이거나 범위를 벗어난 것은 invalid slot으로 둔다. Loss pair `δ`는 남은 유효 offset에서 균등 선택한다. 이전 `Aout` target은 같은 현재 model weight로 no-grad 계산하며, 그 출력의 context도 보정 전 base render와 train IDs만 사용한다. 이 추가 target 계산 비용도 포함한다. Frozen pilot은 scene packet만 cache하고 head output target은 재계산한다. Joint에서는 오래된 weight의 G-buffer cache를 재사용하지 않는다.

Joint의 routing feature/좌표/validity/history value는 detach하지만 **현재 albedo value와 현재 shading 경로는 detach하지 않는다.** 원래 geometry·opacity·normal·material·light·response의 학습 경로는 유지한다. History render는 Gaussian densification용 `means2d/AbsGrad`, `trainer.cur_frame`, affine test 상태, optimizer/RNG 상태를 덮어쓰지 않도록 분리하고 current target의 상태를 복원한다. 이 점을 해결하지 않고 기존 frozen-head 코드를 joint trainer에 붙이면 Base v4 학습이 달라진다.

**Base v4와 동일하게 고정할 이미지 평가**

| 항목 | 고정 조건 |
|---|---|
| 데이터 | 저장 config의 scene·camera·frame·stride 8 split, raw만 평가 |
| 해상도 | native 1920×1280, loading/downscale 1 |
| Renderer | AA off, packed false, 같은 camera/light/response 설정 |
| scored frames | 원래 `dataset.test_image_set`의 12장 또는 3장 |
| 입력/출력 처리 | 원래 undistortion, response, affine, RGB [0,1] clamp |
| PSNR | 이미지별 RGB MSE→PSNR 계산 후 기존 이미지 평균; pooled MSE PSNR로 바꾸지 않음 |
| SSIM | 기존 skimage SSIM, data_range=1, channel_axis=-1 |
| LPIPS | 기존 `trainer.lpips`, normalize=True, AlexNet |
| 평가 mask | 전체 이미지 주지표 그대로; static/history mask로 점수 영역을 좁히지 않음 |
| 추가 기존 항목 | occupied/dynamic/human/vehicle PSNR·SSIM과 기존 유효 이미지 집계 유지 |
| 저장 | 기존 `image_metrics/raw/test/*` key와 metric 경로 유지 |

구현 기준은 [tools/eval.py](../../tools/eval.py) → [models/video_utils.py](../../models/video_utils.py)의 기존 계산 함수다. 일반 test 평가도 이미 mean affine embedding을 사용한다. Shared mean affine를 썼다는 이유만으로 test 점수가 무조건 다른 것은 아니다. 다만 이를 train/full sequence에 강제하면 그쪽 출력은 달라지므로, 여기서는 원래 frame별 정책을 유지한다.

중요하게도 test loader는 `8,16,24,...`를 순회한다. 직전 test를 `t−1` history로 취급하면 안 된다. 각 scored frame t에 대해 **실제 t−1,t−2,t−4 pose의 생성 G-buffer를 deterministic하게 준비하는 adapter**를 둔다. 그러면 원래 test loader와 scoring 함수를 유지할 수 있다. Full chronological render에서 cache를 채운 결과와 random-access 결과가 같아야 한다. Context에는 train pose가 포함될 수 있으나 GT RGB·teacher material·미래 frame을 공급하지 않는다. 이 context 사용 정책을 결과 manifest에 남긴다.

현재 동작하는 baseline replay 명령은 아래와 같다. `--eval_weathers`를 빈 목록으로 주면 raw만 평가한다. 이 문서 작성 중에는 실행하지 않았다.

```bash
cd /home/user/Desktop/3ableGS
PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python tools/eval.py \
  --resume_from output/pbr/rebuild/scene_788_100f/base_v4_raw/checkpoint_final.pth \
  --eval_weathers
```

Attention config와 weight는 checkpoint와 함께 저장·reload하도록 구현했다. 위 명령은 원래 baseline replay이며, 신규 실험은 아래 실행기로 학습한 checkpoint를 같은 평가 코드에 전달한다.

**시간적 안정성의 보조 평가와 판정**

기존 test set은 연속 프레임이 없어서 PSNR/SSIM/LPIPS만으로 flicker 개선을 판정할 수 없다. 데이터 split과 기존 주지표는 그대로 두고 전체 000–099 경로에 아래 진단을 별도 JSON으로 저장한다. 이는 **train frame을 포함한 reconstruction-path temporal 진단**이며 clean contiguous-heldout 성능을 뜻하지 않는다. 20:30/60:70으로 바꾸거나 half resolution으로 렌더하면 별도 benchmark가 되므로 이번 주평가에 섞지 않는다.

기존 [utils/temporal_metrics.py](../../utils/temporal_metrics.py)의 shared GT RGB optical flow와 forward/backward consistency, GT static mask를 사용한다. Flow는 평가용 correspondence 추정치이며 정답 flow가 아니다.

```text
Etemp = mean_valid |(Ihat_t − warp(Ihat_(t−1)))
                    − (Igt_t − warp(Igt_(t−1)))|
```

Ground-truth의 정상적인 시점 변화까지 0으로 만들지 않도록 GT temporal delta를 빼는 residual이다. 모든 방법에 같은 flow·GT mask·frame pair를 사용한다. Attention acceptance로 평가 mask를 바꾸지 않으며 flow coverage와 유효 pixel 수도 함께 기록한다. Geometry가 달라지는 joint arm에서도 평가 mask는 공통이다.

- Full-sequence adjacent residual과 train→test pair(7→8, 15→16, …) residual을 구분한다. 뒤의 값 역시 한쪽이 train인 보조 진단이다.
- GT edge gradient error와 edge strength ratio를 같이 보고한다. 흐려져서 temporal error만 좋아지는 모델은 성공으로 판정하지 않는다.
- Static/background, dynamic/object boundary, roughness가 낮은 영역의 결과를 분리해 원인을 본다. Renderer로 정의하는 영역은 동일 B0에서 고정한다. 현재 모듈은 dynamic temporal 개선을 주장하지 않는다.
- History acceptance, attention entropy, self/history 비중, residual 크기, peak VRAM, 학습시간을 기록한다. Streaming cache latency와 random-access test latency를 구분하며 history rendering 비용을 포함한다.
- 동일 경로·속도의 영상과 확대 ROI를 비교하되 모델별 자동 contrast 조정은 하지 않는다.

기존 temporal evaluator는 frame gap이 있으면 temporal score를 null로 반환한다. `8,16,...`을 넣고 연속 IDs로 바꾸어 계산하지 않는다. 8-frame-gap temporal metric은 이번 최소 설계에서 제외한다.

초기 진행 기준은 **A2의 전체 경로 temporal residual이 재평가 B0보다 5% 이상 감소**, 기존 test PSNR 감소 ≤0.05 dB, SSIM 감소 ≤0.001, LPIPS 증가 ≤0.002, edge error ≤1.02× baseline, edge strength ≥0.98× baseline으로 제안한다. 이 값들은 실행 전 고정하는 연구 판단 기준이며 성능 보장이 아니다. Attention의 기여를 주장하려면 H0/C0를, G-buffer guide의 기여를 주장하려면 A0를, temporal loss의 기여를 주장하려면 A1을 각각 넘어야 한다.

Base v4의 test 12장을 validation으로 재분할하거나 test 최저값으로 checkpoint·loss weight를 고르지 않는다. 고정된 final step과 위 설정으로 모든 arm을 보고한다. 탐색이 필요하면 별도 development scene에서 결정한다. Pilot의 test를 확인해 설정을 바꾸었다면 후속 동일 scene 결과도 exploratory로 표기한다. 반복 seed의 mean±std는 학습 변동성만 나타내며 한 씬의 일반화 신뢰구간으로 해석하지 않는다.

**구현 대상과 완료 확인**

재사용할 것은 `ego_temporal.py`의 좌표/linear-color helper, trainer의 G-buffer 추출·현재 PBR shading, 기존 image evaluator와 independent-flow evaluator다. `run_ego_motion_experiments.py`는 contiguous split을 요구하고 기존 head loss도 다르므로 새 runner의 대체물이 아니다.

구현 범위는 candidate별 query-surface validity와 direct material attention, stride-split용 deterministic history provider, Base v4 loss를 보존하는 pilot/joint trainer 연결, checkpoint 설정 복원, 기존 metric을 호출하는 paired runner다. 결과에는 config/code/checkpoint/teacher identity, seed, train/test/context IDs, 업데이트 수, 실제 시간, frame별 metric을 남긴다.

구현 완료 시에는 identity bypass, 모든 history invalid일 때의 finite 출력, 순서 독립적 렌더, candidate 경계 누출, GT/test 입력 차단, current material/attention gradient, history의 densification state 불변, checkpoint roundtrip, 원래 metric aggregation을 확인한다. 학습 성능은 이 검증과 별개로 위 paired 실험을 완료한 후 판단한다.

설계의 참고 근거는 [RVRT: guided deformable attention](https://proceedings.neurips.cc/paper_files/paper/2022/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html)의 alignment와 temporal feature aggregation, [SVGF](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/)의 temporal accumulation과 detail 보존 문제다. 여기의 G-buffer-guided material attention은 이 개념들을 Base v4에 맞게 조합한 제안이며, 해당 논문이 이 저장소에서의 효과를 입증한 것은 아니다.

**구현 기록 — 2026-10-26**

- [핵심 모듈](../../models/gbuffer_attention.py): 7개 실험군, native 3×3 후보 검증, direct material attention, checkpointed chunk 계산, 고정 geometry temporal loss. Width 32 기준 C0/A0/A1/A2는 53,380개, C1은 53,315개 parameter다.
- [History/학습 연결](../../models/gbuffer_attention_runtime.py): train-only context, renderer·RNG·downscale 상태 복원, optional CPU cache, optimizer 등록과 checkpoint metadata. 기존 PBR trainer의 최종 RGB와 보정 전 material teacher를 분리한다.
- [Frozen pilot](../../tools/train_gbuffer_attention.py): 원래 scene step과 loss를 유지하며 head만 학습한다. Optimizer/RNG를 포함한 `--resume`을 지원한다.
- [추가 temporal 평가](../../tools/eval_gbuffer_attention.py): 전체 경로와 train→test pair를 별도로 집계한다. GPU 동시 상주를 줄이기 위해 baseline float32 출력을 임시 저장하며, 공식 이미지 지표는 기존 `eval.py`에 맡긴다.
- [실행기](../../tools/run_gbuffer_attention_experiments.py), [설정](../../configs/gbuffer_attention/base.yaml): 기본은 read-only preview다. `--execute`에서만 순차 실행하며 기존 결과와 실패한 attempt를 보존한다.

```bash
cd /home/user/Desktop/3ableGS
# 전체 pilot 계획 확인: 디렉터리 생성·GPU 실행 없음
.conda/bin/python -m tools.run_gbuffer_attention_experiments \
  --mode pilot --output-dir output/pbr/gbuffer_attention/scene788_pilot_v1

# A2와 B0만 먼저 실행
.conda/bin/python -m tools.run_gbuffer_attention_experiments \
  --mode pilot --variants A2 \
  --output-dir output/pbr/gbuffer_attention/scene788_pilot_a2_v1 --execute

# 전체 joint 비교: seed별 B0/C0/A0/A1/A2 학습 + 해당 B0의 H0 평가
.conda/bin/python -m tools.run_gbuffer_attention_experiments \
  --mode joint --seeds 0 1 2 \
  --output-dir output/pbr/gbuffer_attention/scene788_joint_v1 --execute
```

원래 100f checkpoint로 dry-run을 확인했으며 train88/test12를 유지한다. 새 결과는 `summary.csv/json`, `state.json`, `stages/`, `logs/`에 기록된다. Queue 재시도는 미완료 학습을 새 attempt에서 시작하고 완료 artifact를 검증하여 재사용한다. Joint optimizer의 정확한 이어하기는 제공하지 않는다. Source/code/checkpoint는 content hash, 입력 image/teacher 자산은 size/mtime으로 식별한다.

현재 환경에서는 NVIDIA 드라이버에 연결되지 않아 실제 CUDA rasterization·native-resolution peak memory·실행 시간·품질 개선은 검증하지 못했다. CPU 검증은 synthetic raster를 사용하면서 실제 PBR forward·loss hook·optimizer·checkpoint 경로와 runner 실패/재시도를 실행한다. Fine-grained roughness/instance 영역 진단과 roughness 보정·rigid motion 확장은 후속 범위로 남긴다.

최종 회귀 검증은 총 **390개 중 386개 통과, CUDA 전용 4개 skip**이다. 신규 attention 관련 42개는 모두 통과했다. [검증 요약](../../logs/gbuffer_attention_validation_20261026/summary.json)과 [전체 로그](../../logs/gbuffer_attention_validation_20261026/tests.log)에 실행 범위를 기록했다.
