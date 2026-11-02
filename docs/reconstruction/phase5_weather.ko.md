# Rain/Snow 재질·조명 복원 — 2026-10-17

완료된 `scene_788_30f/base_v4_raw/checkpoint_final.pth`를 고정하고 Rain과 Snow를 각각 독립 학습하는 경로를 구현했다. 이번 요청에서는 구현·검증만 수행하며, 실제 30k/3k/5k 학습은 아래 명령으로 시작한다.

## 복원한 구성

| 단계 | Iterations | 학습 대상 | LR |
|---|---:|---|---:|
| Material | 30,000 | Background weather material decoder | 0.0001 |
| Environment | 3,000 | 해당 weather HDR log-residual | 0.001 |
| Joint | 5,000 | 위 두 모듈 | 0.00005 / 0.0005 |

Rain의 세 단계를 완료한 뒤 Snow의 세 단계를 실행한다. **두 weather의 material 단계는 같은 raw checkpoint에서 독립적으로 시작한다.** 단계 전환에서는 앞 단계의 모델 가중치만 가져오고 step과 Adam을 새로 시작한다. 같은 단계의 재개는 마지막 완료 iteration 다음부터 optimizer와 RNG 상태를 이어간다.

- Geometry·opacity·feature·raw material decoder·tangent normal·base HDR·camera response·CamPose·affine를 고정한다. Densification, culling, opacity reset 및 SMPL KNN 갱신을 실행하지 않는다. Frozen node의 원래 deformation step은 30,000으로 유지한다.
- Weather material 입력은 feature32 + base A3/M1/R1 + orientation1의 38D이며 detach한다. Decoder는 **38→64→64→4**, hidden ReLU, 마지막 층 0 초기화다.
- Orientation은 tangent 보정·camera-facing flip 전 geometry normal과 world-up의 내적 절댓값이다. `world_up=[0,0,1]`.
- `ΔA=4*tanh(zA)`, `ΔR=5*tanh(zR)`. Raw preactivation logit에 `strength*Δ`를 더한 후 A는 sigmoid, R은 `0.99*sigmoid+0.01`로 활성화한다. Metallic과 normal은 유지한다.
- HDR은 `exp(clamp(base_log_radiance + environment_strength*weather_delta, -8, 8))`를 사용한다. Surface와 sky, 차량·사람의 조명에도 적용된다. 동적 클래스의 재질은 raw다.
- Material/environment strength를 별도로 제어한다. 둘 다 0이면 raw 렌더로 돌아간다. Particle/fog는 포함하지 않는다.

재질 감독은 `gt_relative`다. Pred weather와 frozen raw 렌더 사이의 logit 차이를, weather/raw teacher 사이의 logit 차이에 맞춘다. Roughness logit은 `[0.01,1]`을 `[0,1]`로 되돌려 계산한다. Paired A/R 가중치는 1.0/0.5, absolute A/R은 0.1/0.05, delta regularization은 각각 0.001이다. Charbonnier는 `sqrt(r²+0.001²)-0.001`의 채널 균형 평균을 사용한다.

Mask는 sky/dynamic/ego 제외, opacity≥0.5, **raw/weather teacher normal 간** cosine≥0.5, frozen base albedo 오차≤0.25, roughness 오차≤0.35, erosion 1px다. 유효 픽셀 64 미만이면 오류를 내며 감독을 조용히 생략하지 않는다. Environment/joint는 RGB surface 0.8, sky 0.1, SSIM 0.02 및 HDR residual의 L1 magnitude/TV 각각 0.001을 사용한다. Raw Base v4의 boundary LPIPS·depth·normal loss를 weather 단계에 다시 더하지 않는다.

근거: [E09 overlay](evidence/E09_weather_overlay.txt), [E10 실행 설정](evidence/E10_weather_schedule.txt), [material 실제 코드](phase5_material_provenance.md), [HDR 실제 코드](phase5_lighting_provenance.md), [trainer/loss 실제 코드](phase5_trainer_provenance.md). `models/weather_material.py`는 확인된 원본 전체를 복원했고, paired loss/mask의 여섯 메서드도 확인된 원본과 AST가 같다. 현재 저장소에 맞춘 stage 실행기와 full-state 저장 형식은 새 구현이다.

## 실행·checkpoint 계약

[weather.yaml](../../configs/base_v4/weather.yaml)은 raw checkpoint 옆의 `config.yaml`을 상속한다. 현재 788 frame000–029, camera0, train27/test3, 동일 G-buffer 경로를 유지하고 paired RGB/G-buffer는 CPU에 preload한다. Environment/joint 해상도 schedule은 downscale2/double250이다.

- [train_weather.py](../../tools/train_weather.py): 한 단계 실행, checkpoint 재개, 평가.
- [run_base_v4_weather.py](../../tools/run_base_v4_weather.py): Rain→Snow 6단계 순차 실행. 기본은 preview, `--execute`가 실제 학습이다.
- 각 단계 후 test **PSNR/SSIM/LPIPS**, 렌더 PNG 및 영상을 저장한다. 이는 paired weather RGB 기준 평가다. 논문 WeatherCity의 CLIP-S/CLIP-DS/SEM-CS 평가는 별도 F단계 범위다.
- 중간 저장은 material 5k마다, environment/joint 1k마다 `checkpoint_training_latest.pth`를 atomic replace한다. 마지막에는 다음 단계용 `checkpoint_final.pth`도 저장한다. 모든 과거 iteration checkpoint를 추가로 쌓지 않는다.
- Latest에는 model, Adam, 완료 step, Python/NumPy/PyTorch/CUDA RNG, fitting config를 저장한다. Weather LR은 단계별 상수이고 GradScaler/error-buffer 갱신을 사용하지 않는다. Raw trainer의 model-only resume 동작은 변경하지 않는다.
- 동일 명령을 다시 실행하면 완료·평가된 단계를 건너뛰고 미완료 단계는 latest부터 재개한다. Final 모델만 있고 평가가 실패했다면 학습을 반복하지 않고 평가를 재시도한다. 중복 실행은 OS file lock으로 차단한다.
- 같은 출력 폴더를 다른 raw checkpoint/weather/stage로 재사용하면 오류가 난다. 실패한 단계 뒤의 종속 학습을 실행하지 않는다. 원인을 해결한 뒤 같은 명령으로 이어간다.

## 788/30프레임 전체 실행

```bash
cd /home/user/Desktop/3ableGS
mkdir -p logs .cache/matplotlib

nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .conda/bin/python -u tools/run_base_v4_weather.py \
    --base_ckpt output/pbr/rebuild/scene_788_30f/base_v4_raw/checkpoint_final.pth \
    --output_dir output/pbr/rebuild/scene_788_30f/base_v4_weather \
    --execute \
  >> logs/scene788_base_v4_rain_snow30.log 2>&1 < /dev/null &

echo "PID=$!"
```

```bash
tail -f /home/user/Desktop/3ableGS/logs/scene788_base_v4_rain_snow30.log
```

`--execute`를 빼면 파일 경로 확인과 명령 preview만 수행한다. `--weathers rain` 또는 `--weathers snow`를 추가하면 해당 weather만 실행한다. VS Code 종료 후에도 실행되며 컴퓨터 자체의 절전·종료 중에는 실행되지 않는다.

최종 결과:

- Rain: `output/pbr/rebuild/scene_788_30f/base_v4_weather/03_rain_joint/checkpoint_final.pth`
- Snow: `output/pbr/rebuild/scene_788_30f/base_v4_weather/06_snow_joint/checkpoint_final.pth`
- 각 단계의 `metrics/<weather>/images_test_*.json`, `videos/<weather>/`, `stage_status.json`, `weather_metrics.jsonl`.

한 단계씩 실행하려면 다음과 같다. Environment/joint에 각각 앞 단계의 final checkpoint를 지정한다. Iterations·LR은 stage에 따라 자동 설정된다.

```bash
.conda/bin/python -u tools/train_weather.py \
  --init_from_base output/pbr/rebuild/scene_788_30f/base_v4_raw/checkpoint_final.pth \
  --run_dir output/pbr/rebuild/scene_788_30f/base_v4_weather/01_rain_material \
  --weather rain --weather_stage material
```

평가만 다시 실행:

```bash
.conda/bin/python -u tools/train_weather.py \
  --resume_from output/pbr/rebuild/scene_788_30f/base_v4_weather/03_rain_joint/checkpoint_final.pth \
  --evaluate
```

## 검증 범위

전체 회귀 테스트 **102개 통과**, 실패·skip 없음. [전체 로그](../../logs/reconstruction/phase5_tests.log). 기존 Base v3/v4의 material·normal·HDR·camera response·density·LPIPS 동작도 포함한다.

실제 완료된 raw checkpoint와 두 weather의 RGB/G-buffer를 로드해 단계별 1회, weather당 3회 optimizer update를 메모리에서 수행했다. Joint update는 원본 해상도다. 이 검증은 학습 결과를 만들기 위한 38k fit이 아니다.

- Rain/Snow 모두 base state의 SHA256이 update 전후 동일하고 Gaussian 수가 Background 2,214,291 / Rigid 1,549 / SMPL 82,680으로 유지됐다.
- 실제 frame0/15/29의 1/4 해상도 confidence mask 비율은 Rain 72.4/79.3/75.2%, Snow 79.7/78.3/75.6%였다.
- Held-out frame8/16/24에서 material/environment strength 0의 RGB 오차는 모두 0이었다.
- [실제 장면 검증 결과](../../logs/reconstruction/phase5_real_scene.json), [검증 실행기](../../tools/check_weather_reconstruction.py).

실제 CLI도 임시 폴더에서 검증했다. Rain material 1-step 학습 → Adam/RNG checkpoint 로드 → 이미 완료된 step을 반복하지 않음 → 원본 해상도 test3장 평가·PNG/영상 저장을 통과했다. 검증용 임시 run은 정리했고 정식 weather 출력은 생성하지 않았다. [CLI 검증 결과](../../logs/reconstruction/phase5_cli.json).

Full-state CPU 회귀 검사에서는 연속 실행과 저장·재개 실행의 최종 가중치, Adam state, 다음 난수 값이 정확히 일치한다. CUDA 연산 자체의 비결정성까지 bitwise 재현을 보장한다는 뜻은 아니다. 장시간 날씨 품질·수렴과 최종 평가 수치는 위 실행 후 확인해야 한다.
