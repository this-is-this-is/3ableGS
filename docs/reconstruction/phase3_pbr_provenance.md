# Phase 3 HDR/PBR 복원 근거

`models/pbr.py`는 대화의 수식 설명을 새로 근사한 구현이 아니라, 아래 세션의 실제 소스 출력에서 복원했다. raw Base v3에서 필요한 경로만 포함하며 weather residual과 fog는 다음 단계에서 다룬다.

- 본문 1–619행: `/home/user/.codex/sessions/2026/09/03/rollout-2026-09-03T21-25-09-01a0673a-e8a2-7b01-b718-0d6b1fb15f57.jsonl`, 255행의 도구 출력. 당시 `models/pbr.py`의 docstring부터 `clear_external`까지 완전한 소스가 남아 있다.
- 마지막 `forward`, `get_param_groups`, `compute_reg_loss`: `/home/user/.codex/sessions/2026/09/03/rollout-2026-09-03T15-56-57-01a0660e-7193-7d23-873a-7a6dc49f40eb.jsonl`, 37행 출력 중 당시 코드 574–589행. 후속 255행 출력은 `forward` 선언에서 잘려 있어 이 동일한 boilerplate를 보충했다.
- GGX dense-reference 회귀검사: 첫 번째 세션 261행의 `tools/test_ggx_prefilter.py` 실제 코드에서 복원하여 `tests/test_pbr_reconstruction.py`에 포함했다.
- 현재 남아 있는 `/home/user/Desktop/SplatAD_IR/nerfstudio/model_components/ir_environment.py`는 WeatherCity Base v2 이식본임을 파일 헤더에 명시한다. 이후 weather 인자를 추가한 구현에서도 raw 셰이딩 수학이 일치하는 것을 직접 확인했다. 이를 Base v3의 카메라 응답/재질 구조 근거로 사용하지는 않았다.

복원된 계산은 다음과 같다.

- log HDR cubemap은 0으로 초기화하고 `exp(clamp(log_env, -8, 8))`로 radiance를 계산한다.
- cubemap face 순서는 OpenGL `+X,-X,+Y,-Y,+Z,-Z`; world → cubemap은 `[[1,0,0],[0,0,1],[0,-1,0]]`이다. diffuse, reflection, sky 모두 같은 변환을 사용한다.
- diffuse irradiance는 solid-angle 가중 SH9 적분과 당시 5개 SH irradiance 상수를 사용한다.
- GGX prefilter는 전체 texel 적분의 exact top-k core와 방향별 coarse tail을 더한다. tail을 global mean으로 대체하지 않는다.
- BRDF LUT는 1,024 Hammersley samples, GGX alpha=roughness², Smith IBL k=alpha/2를 사용한다.
- albedo 입력은 sRGB이며 shader 내부에서 linear로 바꾼다. `F0=.04(1-M)+A_linear M`, `S=F0 LUT_A+LUT_B`, `D=max(1-S,0)(1-M) A_linear E(n)/pi`다.
- surface와 sky는 linear radiance로 반환한다. trainer가 alpha 합성 후 Reinhard → CameraResponse → sRGB → Affine을 적용해야 한다. `HDREnvLight.forward`는 이전 RGB sky API 호환용이므로 PBR 합성은 `background`를 직접 호출한다.
- `shade`의 viewdir는 surface→camera이고, `background`의 viewdirs는 camera→scene이다.

검증 결과: 2026-10-16, 작은 합성 입력에서 CPU 검사 4개 및 CUDA texture/역전파 검사 1개 통과. checkpoint strict roundtrip, uniform-light 에너지, cubemap 방향 역변환, GGX directional-tail의 dense 적분 대조, material/normal/HDR gradient의 finite 및 nonzero를 검사했다.

실제 설정(resolution 64 / 5 mips / LUT 64 / tail 16)에서도 이식본과 비교했다. 최대 차이는 RGB/specular `5.96e-8`, diffuse/normal/sky `0`, HDR gradient `7.28e-12`였다. GGX kernel 생성의 CUDA 병렬 합산 때문에 bitwise equality는 요구하지 않았다. 결과는 `logs/reconstruction/phase3_pbr_parity.json`에 저장했다. 장면 학습 또는 평가 지표 복원을 검증한 결과는 아니다.
