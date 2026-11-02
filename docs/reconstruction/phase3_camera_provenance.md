# Camera response 복원 근거

`models/camera_response.py`는 아래 과거 도구 출력의 파일 1–180행을 줄 번호만 제거하여 그대로 복원했다. 새 곡선 모델이나 새 초기화는 추가하지 않았다.

- 원본: `/home/user/.codex/sessions/2026/10/05/rollout-2026-10-05T20-18-55-01a10bc9-c736-7923-85fe-ce959b458f6e.jsonl`, JSONL 424행 (`custom_tool_call_output`).
- 이 출력에는 최종 `_StableReinhard`, `_ReducedKnotInterpolation`, `MonotonicCameraResponse`와 Base v3 YAML의 knots/LR/regularizer가 함께 기록되어 있다.
- 복원 파일 SHA256: `73dcd0bc6e0d38121f4efaee95d63dfeca7650ea402b79d835cdf8fb916deac8`.

9개 knot, 구간 slope 하한 0.05, 고정점 (0,0)/(.5,.5)/(1,1), 8개 공유 interval logits를 사용한다. Forward 순서는 HDR radiance → Reinhard → monotonic curve → sRGB이며 affine은 trainer에서 그 뒤에 적용한다. 초기값은 기존 Reinhard+sRGB 출력과 정확히 같다.

학습 계약은 `get_param_groups()`의 `CameraResponse#all` (LR 0.001), `compute_reg_loss()`의 `camera_response_identity`/`camera_response_smoothness` (각 가중치 0.001)이다. Curve는 RGB 채널·frame·weather가 공유한다. `n` 인자는 misc-model 생성 호환용이고 frame별 parameter를 만들지 않는다.

당시 성능 수정까지 포함한다. `_ReducedKnotInterpolation`의 custom backward는 9개 knot별 reduction으로 수백만 pixel의 atomic scatter contention을 피한다. `_StableReinhard`는 큰 HDR 값에서 상쇄 오차로 gradient가 음수가 되는 문제를 analytic derivative로 막는다. 시각화 layer의 `diagnostic_display_transform()`에는 학습 중 `no_grad`를 사용하던 trainer 패치도 원본 기록에 존재한다.

`tests/test_camera_response.py`의 6개 CPU 검사 통과: identity output, 밝은 HDR gradient 부호, 원본 interpolation의 값·입력 gradient·parameter gradient 일치, 극단 logit에서 단조성/고정점/slope 하한, 공유 채널과 state roundtrip, regularizer와 잘못된 설정. 실행: `OMP_NUM_THREADS=1 .conda/bin/python -m unittest discover -s tests -p test_camera_response.py -v`.
