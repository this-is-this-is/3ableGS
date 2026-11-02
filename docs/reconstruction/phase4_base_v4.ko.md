# Base v4 적용 — 2026-10-17

`Base v4 = Base v3 + combo_perceptual_02`를 현재 복원 코드에 적용했다. 사용자 요청 범위는 구현·검증과 실행 명령 제공이며, 이번 작업에서 30k 학습은 실행하지 않는다.

## 기록에 따른 변경

| 항목 | Base v3 | Base v4 |
|---|---:|---:|
| Background `stop_split_at` | 15000 | 25000 |
| Background `stop_reset_at` | 없음 | 20000 |
| RGB / SSIM 가중치 | 0.8 / 0.2 | 0.7 / 0.3 |
| Boundary patch LPIPS | 비활성 | 최대 0.02 |

LPIPS의 정확한 설정은 start 5000, ramp 1000, patch 128×128, 4개, RGB/gradient loss 가중치 0, Sobel edge threshold 0.05, dilation radius 2, 최소 유효 비율 0.99다. 가중치는 4999에서 비활성, 5000에서 0, 5500에서 0.01, 6000부터 0.02다.

`models/boundary_losses.py`는 [E04 실제 코드 발췌](evidence/E04_loss_implementation.txt)의 원본 모듈을 복원했다. Target RGB edge 밀도에 0.05를 더한 확률로 eligible crop을 replacement sampling한다. Ego mask 영역은 제외하며 sky와 dynamic을 별도로 제외하지 않는다. Prediction crop의 invalid pixel은 target으로 채운 뒤 두 입력을 [0,1]로 clamp하고 `normalize=True`로 호출한다. 유효 crop이 없으면 gradient가 연결된 0 loss를 반환한다. Late decay와 G-buffer confidence helper는 원본에 있으나 Base v4 설정 및 trainer 경로에서 활성화하지 않는다.

Trainer는 기존 평가 metric 내부의 frozen AlexNet `self.lpips.net`을 직접 호출한다. 추가 network 초기화가 없으므로 baseline의 seed와 초기화 RNG를 바꾸지 않는다. 현재 설치된 TorchMetrics의 `_NoTrainLpips`는 `train()` 호출에서도 eval을 유지하며 prediction으로의 gradient는 보존한다. Stateful metric의 update/compute 경로를 학습 loss로 사용하지 않는다. Raw에만 적용하고, 기존 G-buffer·depth·normal·HDR·camera response loss는 유지한다.

Opacity reset은 기존 `step % 3000 == 100` 일정에 `step < 20000` 조건을 추가한다. Warmup 이후 마지막 Background reset은 **18100**, 21100부터는 생략된다. 20000에서 추가 reset을 만들지 않는다. 증식 종료와 별개로 기존 pruning을 유지하며 reset 이후의 관측 대기 조건도 그대로 둔다. Rigid/Deformable은 기존 20k 증식 종료와 reset 동작을 유지한다. SMPL의 별도 KNN/postprocess 경로도 변경하지 않는다.

출처: [E03 설정](evidence/E03_combo_overlay.txt), [E04 구현](evidence/E04_loss_implementation.txt), [boundary 복원 근거](phase4_boundary_provenance.md), [density 복원 근거](phase4_density_provenance.md).

## 설정 파일과 비교 조건

- [combo_perceptual_02.yaml](../../configs/base_v4/combo_perceptual_02.yaml): 네 가지 변경만 포함하는 공통 overlay.
- [base.yaml](../../configs/base_v4/base.yaml): Base v3 공통 설정 뒤에 overlay를 병합. 기존 094/100프레임 데이터 기본값 유지.
- [scene788_30f.yaml](../../configs/base_v4/scene788_30f.yaml): 기존 Base v3 788/30프레임 설정 뒤에 같은 overlay를 병합.

실제 완료된 Base v3 run의 config와 비교해 data/seed/render/logging이 동일함을 확인했다. 동일한 raw G-buffer를 사용하며, frame000–029 / camera0 / train27 / test008·016·024 / seed0 / 30k 조건이다. Base v3 checkpoint를 이어서 학습하지 않고 새로 초기화한다.

기준값은 `output/pbr/rebuild/scene_788_30f/base_v3_raw/metrics/raw/images_test_2026-10-17_10-51-19.json`의 PSNR **30.251726**, SSIM **0.918945**, LPIPS **0.133204**다. 학습 loop 시간은 1:26:48이었다. Base v4의 학습 성능은 실행 후 같은 test3장으로 평가해야 한다.

## 검증 결과

- 전체 회귀 테스트 **70개 통과**, 실패·skip 없음. [로그](../../logs/reconstruction/phase4_tests.log).
- 같은 seed의 Base v3/v4 초기 state와 parameter 구성이 정확히 같음. 설정 차이는 위 네 항목의 15개 leaf뿐임.
- 실제 pretrained AlexNet에 128×128 crop 4개를 입력해 4999/5000/5500/6000 schedule과 frozen/eval 상태, 평가 metric state 불변을 확인함.
- 실제 CUDA PBR 렌더러에서 **LPIPS loss만 역전파**하여 RGB, feature, material decoder, normal head, geometry, HDR, camera response 및 densification용 AbsGrad가 유한·비영임을 확인함.
- Model checkpoint strict reload 후 RGB 최대 오차 `1.19e-7`. [GPU 통합 결과](../../logs/reconstruction/phase4_integration.json), [검증 범위·실행법](phase4_integration_checks.md).
- Background 24,900 증식 허용 / 25,000 종료, 18,100 reset 적용 / 21,100 생략, pruning 유지와 동적 클래스 경로 보존을 검사함.

위 결과는 코드 동작 검증이다. Base v4의 실제 씬 PSNR/SSIM/LPIPS 개선 여부는 아래 30k 실행 후 판정한다.

후속 실행 완료(2026-10-17): 788 raw30의 30k 학습은 1:40:42가 소요됐고, test3장 평가 결과는 PSNR **30.124935**, SSIM **0.917821**, LPIPS **0.108272**다. Base v3 대비 PSNR −0.126792 dB, SSIM −0.001124, LPIPS 18.7% 감소다. 평가·영상 저장은 14:26에 완료됐다. [평가 JSON](../../output/pbr/rebuild/scene_788_30f/base_v4_raw/metrics/raw/images_test_2026-10-17_12-44-11.json). 이 checkpoint를 다음 [Rain/Snow 단계](phase5_weather.ko.md)의 고정 base로 사용한다.

## 실행 명령

학습 후 test PSNR/SSIM/LPIPS 평가와 렌더 저장까지 실행한다. `--resume_from`과 `--train_weathers`는 지정하지 않는다. 결과는 기존 Base v3와 다른 `base_v4_raw` 폴더에 저장한다.

```bash
cd /home/user/Desktop/3ableGS
mkdir -p logs .cache/matplotlib

nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .conda/bin/python -u tools/train.py \
    --config_file configs/base_v4/scene788_30f.yaml \
    --output_root output/pbr/rebuild \
    --project scene_788_30f \
    --run_name base_v4_raw \
    render.render_test=true \
    render.render_full=false \
    render.render_novel=null \
  > logs/scene788_base_v4_raw30.log 2>&1 < /dev/null &

echo "PID=$!"
```

```bash
tail -f /home/user/Desktop/3ableGS/logs/scene788_base_v4_raw30.log
```

- Checkpoint: `output/pbr/rebuild/scene_788_30f/base_v4_raw/checkpoint_final.pth`
- 평가: `output/pbr/rebuild/scene_788_30f/base_v4_raw/metrics/raw/images_test_*.json`

VS Code 종료 후에도 실행되며, 컴퓨터 종료·재부팅·절전 중에는 실행되지 않는다. 이번 구현의 checkpoint 보장은 model weights roundtrip이며 optimizer/RNG까지 동일하게 이어가는 resume는 후속 단계 범위다.
