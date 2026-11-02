# Base v4 boundary LPIPS 복원 근거 — 2026-10-17

`models/boundary_losses.py`는 [E04 실제 코드 출력](evidence/E04_loss_implementation.txt)의 모듈 시작부터 `class MultiTrainer_pbr` 이전까지 **141행을 그대로 복원**했다. 이 기록의 원본 위치는 다음과 같다.

- `/home/user/.codex/sessions/2026/10/12/rollout-2026-10-12T10-12-30-01a12dab-1a2d-7983-aaf4-071800e4f910.jsonl:135`
- 원본 출력 시각: `2026-10-12T01:21:50.294Z`

API는 `validate_boundary_config(config=None, *, gbuffer=False)`와 `boundary_rgb_losses(pred, target, valid, step, config, perceptual_net=None)`다. 입력 이미지는 HWC, 유효 마스크는 HW이며, 반환은 가중치와 ramp가 적용된 loss dict다. 호출자는 raw에만 연결하고 `self.lpips.net.requires_grad_(False)`를 적용한다. 학습 중 stateful LPIPS metric을 호출하지 않고 feature network를 직접 호출한다.

Base v4에서 활성화할 설정은 `enabled=true`, `start_step=5000`, `ramp_steps=1000`, `rgb_w=0`, `gradient_w=0`, `perceptual_w=0.02`, `edge_threshold=0.05`, `radius=2`, `patch_size=128`, `num_patches=4`, `min_valid_fraction=0.99`다. 모듈 기본값은 당시 다른 ablation에도 사용되었으므로 config에서 이 값을 명시해야 한다.

원본과의 동일성을 유지하기 위해 선택적 LPIPS late-decay와 G-buffer boundary confidence helper도 함께 복원했다. 두 기능은 Base v4에서 활성화하지 않는다. 구현을 복원했다는 것은 해당 선택적 실험을 Base v4에 포함했다는 뜻이 아니다.

## 경계와 patch의 정확한 의미

- Target RGB 모든 채널에 signed Sobel x/y를 적용하고 절댓값 최대를 사용한다. 채널 평균 영상의 경계가 아니다. `magnitude <= 1e-6`는 0으로 처리한다.
- Ego 등 invalid 영역은 먼저 반경 1만큼 확장해 Sobel stencil을 제외한다. Edge에 반경 2 dilation 후 stencil을 다시 곱한다. Invalid RGB를 먼저 0으로 만드는 방식은 사용하지 않는다.
- Patch는 원래 128×128이며 이미지가 더 작으면 각 축 크기에 맞춘다. 어느 축이든 32보다 작으면 graph가 연결된 0 loss다.
- 모든 가능한 patch 좌상단 위치에서 valid box mean이 0.99 이상인 것만 허용한다. Sampling weight는 `(edge box mean + 0.05) * eligible`, 샘플 수 4, 복원 추출이다.
- 전역 Torch RNG를 사용한다. 현재 프로젝트의 optimizer/RNG 완전 resume 복원은 별도 단계이며 이 구현 자체가 정확한 resume를 보장하지 않는다.
- Patch 안의 invalid prediction은 detached target으로 채워 perceptual gradient를 제외한다. RGB를 `[0,1]`로 clamp하고 frozen network를 `eval()` 상태에서 `normalize=True`로 호출한다. Target와 mask는 detach하지만 prediction의 gradient는 보존한다.
- 5,000 step의 ramp는 0, 5,500은 0.5, 6,000 이후는 1이다. Eligible patch가 없을 때도 prediction에 연결된 0을 반환한다.

## 검증

`tests/test_boundary_losses.py`의 10개 테스트 통과:

1. 4,999/5,000/5,500/6,000/30,000 step 스케줄과 활성 loss key.
2. Base v3 disabled 경로에서 경계 연산 및 network 호출 없음.
3. 작은 이미지와 유효 patch 부재의 finite, graph-connected zero.
4. Invalid prediction의 target 대체, invalid gradient 0, target detach.
5. Ego 경계 Sobel stencil 제외 및 인위적인 sampling edge 방지.
6. 독립적인 patch mean 기준의 sampling weight, `+0.05`, eligible 필터, 복원 추출 및 seed 재현.
7. RGB 평균이 같아도 보존되는 색 경계.
8. Integral-image box mean과 직접 crop 평균 일치.
9. 잘못된 설정 및 LPIPS network 누락의 명시적 오류.
10. 실제 로컬 캐시의 pretrained AlexNet LPIPS: prediction finite/nonzero gradient, feature parameter gradient 없음, target gradient 없음, torchmetrics 상태 및 update count 불변.

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .conda/bin/python -m unittest discover -s tests -p test_boundary_losses.py -v
```

위 검사는 CPU에서 약 4초에 완료됐다. 사전학습 AlexNet cache가 없는 다른 환경에서는 10번째 검사만 skip하며 다운로드를 시작하지 않는다. 이 환경에서는 실제 cache로 해당 검사를 실행했다.
