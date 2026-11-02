# D1: Base v5 동적 객체별 G-buffer supervision

D1은 Base v5의 RGB 보정망과 OmniRe 계열 객체 표현을 유지하면서 **동적 객체별로
균형 잡힌 G-buffer teacher loss**를 추가하는 실험이다. 배경 위주의 전체 픽셀
평균에서 작은 객체의 기여가 작아지는 문제와, 경계에서 여러 표면이 섞이는 문제를
분리해서 검증한다. 새 attention이나 시간별 history 모듈은 추가하지 않는다.

현재 구현은 성능 가설을 검증하기 위한 것이다. 아래 실행 명령을 실행하기 전에는
학습 결과가 없으며, 테스트 통과를 PSNR 또는 temporal 품질 향상으로 해석하지 않는다.

## 비교군

| Arm | 설정 | 추가 loss |
|---|---|---|
| B0 | `enabled: false` | 없음. 원래 Base v5 |
| D1B | `mode: balanced` | 객체별 평균 G-buffer residual |
| D1P | `mode: purity` | 같은 객체별 평균에 가시 기여도 기반 purity 가중치 |

각 설정은 `configs/base_v5/scene788_100f.yaml`을 상속한다. Scene 788,
camera 0, frames 0~99, 88 train / 12 test, seed 0, **30,001 camera updates**를
기본으로 사용한다. test frame은 8, 16, …, 96이다. 각 arm은 dataset seed points에서
처음부터 따로 학습한다. Base v5의 RGB residual warmup 2,000/ramp 1,000,
material/geometry/pose optimizer, densification, LiDAR·normal·material loss,
teacher 데이터와 pure-PBR 보조 loss를 그대로 유지한다.

## 가시 기여도와 객체 소속

Gaussian에 저장된 `point_ids`는 객체 index이다. Rigid/SMPL/Deformable 노드의
index를 namespace해 충돌하지 않는 객체 ID로 사용한다. Background ID는 0이다.
각 객체 indicator를 auxiliary feature로 렌더하되 **장면의 모든 Gaussian과 opacity,
깊이 순서**를 유지한다. 그러면 객체 k의 채널은 전체 장면의 transmittance를 반영한
가시 기여도 `a_k`가 된다. 다른 객체 opacity를 제거한 isolated render는 사용하지 않는다.

기본 guide raster는 현재 영상의 가로·세로 각각 1/2 크기다.
`visibility_downscale: 1`이면 원래 크기로 계산한다. 각 pass에 최대 8개 객체와
전체 alpha를 구하는 ones 채널을 넣는다. 객체가 8개보다 많아도 생략하지 않고 다음
chunk를 처리한다. 연속값인 기여도와 같은 raster의 전체 alpha를 원래 영상 크기로
복원한 뒤, 기존 teacher loss와 동일한 crop/resize를 적용한다. 낮은 해상도 guide의
기여도를 다른 해상도 raster의 alpha로 나누지 않는다.

객체 ID 숫자 자체를 렌더링하거나 보간하지 않는다. 모든 chunk의 기여도를 모아
dynamic 객체 중 가장 큰 기여도를 찾고, background 기여도
`alpha - sum(dynamic a_k)`와도 비교한다. Background와 동률이면 background를
선택한다. 같은 기여도의 동적 객체끼리는 작은 ID를 선택해 결정성을 유지한다.

guide와 ownership은 detach한다. 추가 loss의 gradient는 현재 예측 재질·normal과
그 예측에 기여한 geometry/pose로 전달되지만, 가림 가중치 자체를 바꾸는 방향으로
역전파하지 않는다. 학습 전용 guide이므로 추론에서는 추가 raster와 teacher 입력이 없다.

## Loss

기존 Base v5 loss는 바꾸지 않고 아래 항을 더한다.

```text
L = L_BaseV5 + lambda(step) * mean_over_eligible_instances(
      sum_over_owned_valid_pixels(purity_weight * material_normal_residual)
      / count_of_owned_valid_pixels
    )
```

`material_normal_residual`은 기존 albedo/roughness/metallic L1과 robust normal
residual을 사용하며 기존 채널별 가중치를 그대로 곱한다. Material teacher는 detach하고,
normal은 기존 camera/OpenGL 변환을 유지한다. 새 loss는 진단값과 분리한
`dynamic_gbuffer_*_loss` 항으로 기록한다.

- Sky·ego를 제외한 기존 teacher 유효 영역을 사용한다.
- 전체 alpha가 0.1 이상이고 동적 객체가 가장 큰 기여를 가진 픽셀을 사용한다.
- 유효 픽셀이 16개 미만인 객체는 해당 프레임의 객체 평균에서 제외한다.
- D1B의 추가 가중치는 1이다.
- D1P는 `purity = a_owner / alpha`로부터 floor와 power를 적용한 가중치를 사용한다.
- **분모는 purity 가중치 합이 아니라 원래 eligible pixel 수이다.** 따라서 관측 전체의
  purity가 낮으면 추가 supervision도 약해지며 정규화로 상쇄되지 않는다.
- 가려진 객체나 객체가 없는 프레임은 추가 loss가 0이다. 이를 유효 관측처럼 집계하지 않는다.

기본 추가 가중치 `w=0.25`, `start_step=5000`, `ramp_steps=1000`, `interval=1`이다.
이는 기존 Base v5 residual schedule과 별개다. 자세한 step 경계는
`models/dynamic_gbuffer.py`의 schedule 함수를 따른다. `interval`을 늘려 guide 계산
빈도를 줄일 수 있지만 학습 목적의 적용 빈도도 바뀌므로 별도 설정의 실험으로 기록해야 한다.

추가 연산은 객체 수와 영상 크기에 따라 달라진다. 학습시간 10% 이내 증가는 목표 예산이며
검증된 결과가 아니다. 작은 객체의 guide가 사라지거나 경계가 부정확해질 수 있으므로
eligible 객체·픽셀 수, 평균 purity, 가중 coverage를 확인하고 필요하면
`visibility_downscale: 1`을 별도 대조한다.

## 실행

아무 옵션 없이 실행하면 계획과 명령만 출력하며 output 폴더나 child process를 만들지 않는다.

```bash
cd /home/user/Desktop/3ableGS
bash run_scene788_dynamic_gbuffer.sh
```

입력 파일과 기존 상태를 확인하되 학습·평가는 실행하지 않는다.

```bash
bash run_scene788_dynamic_gbuffer.sh --prepare-only
```

세 arm을 순서대로 처음부터 학습하고 동일한 평가를 수행한다. 다음 명령만 실제 학습을 시작한다.

```bash
bash run_scene788_dynamic_gbuffer.sh --execute
```

출력은 `output/pbr/dynamic_gbuffer/scene788_from_scratch_v1/seed_0/{B0,D1B,D1P}/`에
분리된다. 중단 뒤 같은 명령을 반복하면 각 arm의 기존 Base v5 전체 training state로
이어가며 완료된 checkpoint와 평가를 확인해 건너뛴다. `--iterations`는 추가 횟수가 아닌
**누적 목표 업데이트 수**다. 누적 목표를 늘려도 native 30k optimizer/densification
schedule은 바꾸지 않는다. fitting 설정이 달라지면 다른 output directory를 사용해야 한다.
이미 완료한 횟수보다 작은 누적 목표는 거부한다. 따라서 30,001회 baseline을 10,000회
실험의 대조군으로 잘못 재사용하지 않는다. 더 작은 예산은 새 output directory에서 실행한다.

```bash
# 필요한 arm만 수행
bash run_scene788_dynamic_gbuffer.sh --execute --variants D1B D1P

# 여러 seed에서도 같은 비교
bash run_scene788_dynamic_gbuffer.sh --execute --seeds 0 1 2

# 선택적 동적 temporal 진단도 동일하게 수행
bash run_scene788_dynamic_gbuffer.sh --execute --dynamic-temporal

# 상태와 로그
bash run_scene788_dynamic_gbuffer.sh --status
tail -f output/pbr/dynamic_gbuffer/scene788_from_scratch_v1/D1P_s0_train.log
```

`--save-images`를 명시해야 paired PNG와 float NPZ를 저장한다. 기본 평가에서는 이를
생략해 디스크 사용을 줄인다. `--skip-eval`은 학습만 수행하며 summary에 평가 생략을
명시한다. `--python` 또는 launcher의 `D1_PYTHON`으로 Python 환경을 지정할 수 있다.

## 평가와 상태 파일

`tools/eval_base_v5.py --dynamic-metrics`를 모든 arm에 동일하게 사용한다.
원래 전체 이미지 PSNR/SSIM/LPIPS와 pure/hybrid 비교를 유지한다. Pure PBR는 같은
공동 학습 체크포인트에서 RGB residual만 끈 출력이며, 별도로 학습한 Base v4가 아니다.

추가 동적 지표는 reference dynamic/human/vehicle masks를 사용한 PSNR,
유효 7×7 window SSIM, boundary PSNR, reference 연결 성분 crop LPIPS다.
연결 성분은 실제 instance/track과 같지 않으므로 객체별 또는 track별 LPIPS라고 부르지 않는다.
빈 영역은 유효 관측에서 제외하고 count를 함께 보고한다. 학습 때 모델에서 계산한
purity·ownership을 평가 mask로 재사용하지 않는다.

`--dynamic-temporal`은 full chronological sequence에서 모든 방법에 같은 reference RGB
flow와 cycle-valid mask를 적용한다. 전체 인접 쌍 및 train→test 등을 분리해 보고하며,
**학습 프레임을 포함한 재구성 진단**이다. 기존 test 12장은 서로 연속하지 않으므로
이 결과를 연속 heldout 영상 benchmark라고 해석하면 안 된다.

- Suite `plan.json`: 해석된 fitting 설정, 명령, 비교 프로토콜.
- Suite `summary.json`, `report.md`: arm별 완료·실패·중단 상태와 실제 평가 결과.
- `<arm>_train.log`, `<arm>_eval.log`: 각 child process 로그.
- Arm `checkpoint_training_latest.pth`, `checkpoint_final.pth`: native Base v5 resume 상태.
- Arm `training_metrics.jsonl`: 기존 loss와 D1 진단.
- Arm `rgb_joint_eval/evaluation.json`, `report.md`: 원래 지표와 추가 동적 지표.
- Arm `d1_evaluation_request.json`: 평가 대상 checkpoint와 요청 옵션의 cache 식별자.

summary의 `train_attempts[].elapsed_seconds`는 모델 준비·resume·checkpoint IO를 포함한
child-process 벽시계 시간이다. 중단·재개 횟수가 다른 실행은 이 값만으로 학습 overhead를
단정하지 않는다. 구현 확인용 짧은 업데이트 예산과 최종 30,001회 결과도 구분해야 한다.
`last_training_process`가 있으면 마지막 training process의 `process_start_update`,
`loop_elapsed_seconds_this_process`(준비를 제외한 loop 시간),
`peak_allocated_vram_bytes_this_process`(초기화와 학습을 포함한 PyTorch allocated peak)를
기록한다. 이는 전체 resume 이력의 합계나 GPU 프로세스 전체 reserved memory가 아니다.

## 검증

Runner 테스트는 실제 child 대신 임시 파일을 만드는 fake child로 preview, validation,
원래 Base v5 설정 보존, 실패 후 동일 경로 resume, 평가만 재시도, 평가 cache invalidation,
누적 업데이트 예산과 fitting 변경 거부를 검사한다. CUDA 학습이나 전체 데이터 평가는 없다.
현재 환경은 CUDA driver 연결이 되지 않아 실제 CUDA rasterizer 실행과 학습 overhead는
확인하지 못했다. CPU 검증 결과와 실제 GPU 학습 검증을 구분해야 한다.

```bash
.conda/bin/python -m unittest tests.test_dynamic_gbuffer_runner -v
```
