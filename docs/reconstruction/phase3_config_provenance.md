# Base v3 학습 설정 복원 근거

`configs/base_v3/base.yaml`은 완전히 출력된 과거 baseline YAML과 E3 CameraResponse overlay를 합친 설정이다. 공식 `weathercity.yaml`을 상속하지 않아 불필요한 Sky와 rain/snow/night MLP 설정이 섞이지 않는다. 32D Gaussian 및 SMPL voxel deformer=true/freeze_x=false인 최종 구성을 유지한다.

- 기준 YAML 원문: `/home/user/.codex/sessions/2026/10/10/rollout-2026-10-10T18-43-01-01a11afc-d648-7003-be89-54683c9569e6_01a12531-c52f-7162-a93d-aeb18f3d4d2a.jsonl`, 235행 (`item_completed`). 첫 출력은 `configs/reconstruction_followups/scene114_baseline.yaml`이며 바로 뒤에 Base v3 114/788의 CameraResponse overlay가 출력된다.
- 교차 확인: `/home/user/.codex/sessions/2026/10/08/rollout-2026-10-08T19-06-06-01a11afa-3149-7d42-865c-ed897f1d19d3.jsonl`, 148행 (`custom_tool_call_output`). `configs/base_v3/scene094.yaml`의 resolved config다. 이 진단 실행의 명시적 num_iters=2 override만 30000으로 되돌리면 복원 파일의 **trainer/model/render 전체 mapping이 정확히 일치**한다. 진단용 logging override는 baseline 원문의 print250/save5000/vis0/save_seperate_video=true를 따른다.

복원 시 실행 설정을 명시한 차이는 `vgg: null`이다. PBR raw 학습에는 공식 RGB weather trainer의 별도 VGG gradient를 추가하지 않는다. `base.yaml`의 기본 데이터 scene094/100프레임은 기존 reconstruction/data_100f.yaml과 일치시켰다. Scene/frame 범위 외 모든 G-buffer/load/SMPL/LiDAR 설정은 원문을 유지했다.

`scene788_30f.yaml`은 사용자가 먼저 생성하기로 한 raw 000–029 G-buffer에 맞춘 새 데이터 overlay다. Camera 0, test stride8이므로 train27/test3 (008/016/024)다. G-buffer는 `data/waymo/render_png_788_single_static_30f/gbuffer_frames`에서 읽는다. 모든 iteration·loss·density 설정은 Base v3 그대로이며 학습 실행은 하지 않았다.

Raw material 입력은 Background feature32+detached xyz3, 나머지 feature32다. 모두 LeakyReLU(0.01) decoder, Gaussian-local tangent(0.5, reg0.01, seed15015)를 사용한다. Normal head parameter group은 과거 규칙대로 `mlp.raw` LR 0.0001을 상속해야 한다. Background flatten0.01, split15k, opacity reset interval3000, RGB/SSIM0.8/0.2다. Base v4의 split25k/reset종료20k/SSIM0.3/perceptual0.02는 넣지 않았다.

해석만 확인하는 명령:

```bash
.conda/bin/python tools/resolve_config.py \
  --config_file configs/base_v3/scene788_30f.yaml
```

연결 검토에서 `logging.vis_freq=0`일 때 visualization 조건의 modulo가 먼저 평가되던 공식 실행기의 오류도 수정했다. 0인지 먼저 검사하며 학습 설정 값은 바꾸지 않는다.

LiDAR 투영은 RGB/G-buffer `cv2.undistort` 및 camera ray와 같은 원래 K를 사용하도록 과거 수정도 복원했다. 근거는 `/home/user/.codex/sessions/2026/09/02/rollout-2026-09-02T13-58-47-01a0607b-e562-7772-811a-fcd5b2aae183.jsonl` 478행의 `datasets/driving_dataset.py` diff다. LiDAR만 `getOptimalNewCameraMatrix`로 새 K를 쓰면 depth target과 초기 point 색상이 RGB와 다른 픽셀에 매핑된다. `tests/test_pbr_losses.py`는 실제 projection 함수의 depth/색상 위치, normal 좌표 변환, robust loss, mask, 해상도·crop 계약을 검사하며 5개 CPU 검사를 통과했다.
