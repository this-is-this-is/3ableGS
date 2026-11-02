# Base v5

Base v5는 Base v4 PBR 장면을 display RGB residual branch와 함께 scratch에서
공동 학습하는 모델이다. 기본 설정은 `configs/base_v5/scene788_100f.yaml`이며,
구현은 이 저장소 내부에 있고 SplatAD_IR 코드나 체크포인트를 읽지 않는다.
Base v4는 residual 없는 비교 기준으로 그대로 유지한다.

## 실행

```bash
cd /home/user/Desktop/3ableGS
bash run_scene788_base_v5.sh --prepare-only
bash run_scene788_base_v5.sh
```

VS Code/터미널을 닫아도 실행하려면:

```bash
cd /home/user/Desktop/3ableGS
mkdir -p logs
nohup bash run_scene788_base_v5.sh \
  > logs/scene788_base_v5.log 2>&1 < /dev/null &
disown
```

```bash
tail -f /home/user/Desktop/3ableGS/logs/scene788_base_v5.log
```

기본 결과 폴더는 `output/pbr/rebuild/scene_788_100f/base_v5/`이다.
`RUN_TAG=다른이름`으로 독립 실행을 만들 수 있다. 같은 명령은 이 실행의 최신
체크포인트를 자동으로 이어 학습한다. OS lock으로 중복 실행을 막는다.
`--prepare-only`는 데이터/config/checkpoint를 검증하고 계획만 출력한다.

기본 예산은 **30,001회 카메라 업데이트**, 원래 Base v4의 step 0..30000
inclusive와 같다. `ITERATIONS=35000`처럼 누적 목표를 바꿀 수 있지만,
기하·환경맵·LR의 기존 30k 일정은 연장하지 않는다. 재개할 때도 ITERATIONS는
추가 반복수가 아닌 누적 목표다. `SKIP_EVAL=1`은 종료 후 평가만 생략한다.

## 구조와 loss

Gaussian별 독립 correction feature 8D, 20채널 CNN 입력, 32채널 BasicBlock
두 개, zero-init 1×1 convolution의 6채널 출력을 사용한다. 입력은 feature8 +
normal3 + roughness1 + log-depth1 + alpha1 + view3 + PBR RGB3이다.
Normal/roughness/depth/alpha/view/PBR 조건 입력을 detach한다. 투영된 correction
feature와 최종 합성식의 PBR RGB에는 gradient가 전달된다.

```text
g = exp(0.3 * tanh(head_rgb_gain))
b = 0.1 * tanh(head_rgb_additive)
s = clamp((completed_camera_updates - 2000) / 1000, 0, 1)
C_hybrid = C_pbr + s * alpha * ((g - 1) * C_pbr + b)

L = L_BaseV4(C_hybrid)
    + 0.3 * s * L_RGB(C_pbr)
    + 0.001 * s * R_gain + 0.001 * s * R_additive
L_RGB = 0.7 * L1 + 0.3 * (1 - SSIM)
```

Camera response와 affine을 적용한 display RGB에 residual을 더하며, 원래
Base v4의 최종 clamp/평가 정책을 따른다. 외부 HDR에서는 residual을 우회한다.
Head 출력층과 correction feature는 0에서 시작하고, 추가 head 초기화가 기본
Gaussian/샘플러 RNG를 진행시키지 않도록 분리했다.

R_gain/R_additive는 한도로 정규화한 head 값의 제곱을 RGB 평균하고,
detached alpha와 기존 ego validity mask로 가중 평균한다. 두 head를 별도로
정규화하므로 gain/additive의 큰 상쇄도 벌점을 받는다. PBR 보조 loss도 residual
강도와 함께 ramp하여 처음에는 원래 Base v4 objective를 사용한다.
Correction feature LR=0.0025, CNN LR=0.001이며 나머지 optimizer 설정은 상속한다.

## 유지하는 Base v4 설정

Material feature/MLP, normal, environment, response/affine, geometry와 actor/SMPL의
학습 정책은 **3ableGS Base v4 설정 그대로**다. 카메라 학습에 투영 LiDAR depth
supervision을 사용하며, SplatAD의 camera/LiDAR 교대 스케줄은 없다. 따라서
SplatAD의 30k mixed + 5k camera-only를 복제하지 않는다. 3ableGS의 actor pose/변형
최적화도 유지하므로 SplatAD_IR과 완전히 같은 학습 프로토콜은 아니다.

기존 material/normal/depth/sky priors와 모델 regularizer를 유지한다.
Boundary LPIPS는 global step 5k 시작/1k ramp/최대0.02, environment LR은
global step 3k부터인 기존 일정을 사용한다. Background split <25k/reset <20k,
actor split <20k를 유지한다. 새 feature는 Background/Rigid/Deformable/SMPL의
split/clone/prune, optimizer moment, checkpoint row와 함께 갱신된다.

## 저장·재개와 평가

별도 `tools/train_base_v5.py`가 모델/카메라 카운터, 단일 Adam,
Gaussian 증식 누적값, SMPL KNN, Python/NumPy/Torch/CUDA RNG와 학습 split을
저장한다. LR 함수는 원래 config와 다음 global step으로 복원한다. 재개는
마지막 완료 step+1부터이며 fitting 설정을 바꾼 재개는 거부한다.
Pretrained initializer 옵션은 없고, 기존 model-only resume와 구분한다.

- `checkpoint_training_latest.pth`: 최근 재개 가능한 전체 상태.
- `checkpoint_final.pth`: 요청한 총 업데이트까지 완료한 상태.
- `training_metrics.jsonl`: loss, metric, camera count, strength, Gaussian 수.
- `rgb_joint_eval/evaluation.json`, `report.md`: 종료 후 자동 평가.
- `rgb_joint_eval/renders/`: GT/pure/hybrid, gain/additive/residual PNG·float NPZ.

평가만 다시 실행할 때:

```bash
PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/3ablegs-mpl .conda/bin/python \
  tools/eval_base_v5.py \
  --checkpoint output/pbr/rebuild/scene_788_100f/base_v5/checkpoint_final.pth \
  --save-images
```

평가는 같은 렌더의 pure PBR과 hybrid를 비교하며 teacher G-buffer를 읽지
않는다. Pure PBR은 **현재 공동 학습된 모델에서 display residual만 끈 결과**다.
Native PSNR, skimage SSIM(7×7 uniform), AlexNet LPIPS를 사용한다. SplatAD의
native SSIM 열과 직접 섞지 않는다. HDR에서 보정을 끄는 것은 재조명 품질을
보장하지 않는다. Smoke 결과는 실행 검증용이며 최종 성능으로 해석하지 않는다.

## 검증 도구

`tests/test_rgb_residual.py`, `test_rgb_joint_trainer.py`,
`test_rgb_joint_training.py`, `test_rgb_joint_evaluation.py`와 관련 Base-v4 회귀
테스트가 gradient/시계/optimizer/density/checkpoint/evaluation 계약을 검사한다.
`tools/check_base_v5.py`는 실제 CUDA rasterizer로 PBR·기하·조명·CNN
gradient, 증식 후 feature/Adam 정렬, strict reload와 external HDR bypass를 확인한다.

2026-10-28 검증: CPU 회귀 테스트 122개 중 120개 통과, CUDA 전용 기존 테스트
2개 skip. 별도 CUDA 검사에서 6회 업데이트, 실제 증식, 모든 학습 branch의
gradient, 저장·복원 및 HDR 우회를 확인했다. 실제 788/100프레임 데이터에서는
검증용 warmup/ramp=1/1로 scratch 6회 + 재개 2회를 완료하고, heldout 12장의
paired 평가와 96개 이미지/배열 파일 저장까지 통과했다. 해당 검증 결과는
`output/pbr/rebuild/scene_788_100f/base_v5_smoke_20261028/rgb_joint_eval/`에 있다.

같은 날 기본 warmup/ramp=2000/1000으로 **30,001회 전체 학습도 완료**했다.
최종 체크포인트와 heldout 평가는 `output/pbr/rebuild/scene_788_100f/base_v5/`에
있으며, 12장 평균은 Pure PBR 27.1122 PSNR / 0.9005 SSIM / 0.1722 LPIPS,
Base v5 hybrid 30.9793 PSNR / 0.9319 SSIM / 0.1141 LPIPS다.
