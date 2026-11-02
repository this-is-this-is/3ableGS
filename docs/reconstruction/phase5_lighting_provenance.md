# Phase 5 Rain/Snow HDR residual 복원 근거

2026-10-17에 `models/pbr.py`의 기존 raw 경로에 weather 환경광 잔차 API를 복원했다. Base v4 raw HDR·PBR 수학과 초기화는 유지한다.

근거는 다음 실제 소스 출력이다.

- `/home/user/.codex/sessions/2026/09/07/rollout-2026-09-07T10-10-53-01a0796b-0b4a-7293-b96b-23e5e33c4102.jsonl:209`: `models/pbr.py`의 weather API 추가 diff. 잔차 초기화, parameter getter, radiance, SH, GGX mip, surface·sky 전달 인자가 남아 있다.
- `/home/user/.codex/sessions/2026/10/14/rollout-2026-10-14T22-29-03-01a11103-396a-7a23-97df-aeaa138bf2b0_01a13a9a-2657-7870-998f-e2a0673c5f5a.jsonl:135`: 실제 Base v4 weather 실행 당시 `weather_log_residuals`, `learned_radiance`, `shade` 소스.
- `/home/user/.codex/sessions/2026/10/14/rollout-2026-10-14T23-04-59-01a11421-4704-7271-b4f5-a5191a88939b_01a13abb-0d9e-7d03-bc08-355d1a632d0f.jsonl:357`: weather 단계 freeze/optimizer 및 `compute_weather_reg_loss` 소스.
- 남아 있는 `/home/user/Desktop/SplatAD_IR/nerfstudio/model_components/ir_environment.py`에서도 위 weather API와 수식이 일치한다. 해당 메서드의 연속 소스를 이식하되 SplatAD의 `_texture` wrapper는 기존 WeatherCity `dr.texture` 호출로 유지했다. SplatAD 고유 생성자 검사나 다른 기능은 가져오지 않았다.
- 실행 설정과 단계별 LR은 [E09](evidence/E09_weather_overlay.txt), [E10](evidence/E10_weather_schedule.txt)에 근거한다.

weather별 잔차는 base와 동일한 `6×R×R×3` log-radiance tensor이고 모두 0에서 시작한다. 렌더링에 쓰는 radiance는 다음과 같다.

`L(weather,s) = exp(clamp(base_log_env + s * weather_log_residuals[weather], -8, 8))`

이는 이미지마다 달라지는 인코딩이 아닌 scene 전체에 공유되는 환경광이다. `None`/`raw`는 기존 base만 사용한다. `weather_strength=0`은 raw와 정확히 같으며 rain과 snow는 독립된 파라미터를 갖는다. `get_base_parameters()`와 `get_weather_parameters(weather)`로 stage trainer가 base를 고정하고 해당 잔차만 선택한다. 모듈 단독으로는 raw 학습을 막지 않으며, stage freeze는 trainer의 책임이다.

같은 weather/strength를 diffuse SH irradiance, GGX reflection mip, sky cubemap lookup에 전달한다. 이 때문에 동적 물체도 바뀐 조명으로 렌더된다. Material strength와의 분리는 trainer에서 제어하며 HDR 모듈이 재질 잔차를 변경하지 않는다.

GGX의 방향별 적분 kernel은 기존 생성자에서 계산해 buffer로 유지한다. **Radiance와 mip tensor 자체는 각 forward에서 새로 계산한다.** 학습 중 이전 graph를 cache하거나 detach해서 환경광 gradient를 끊지 않는다. 따라서 weather 전환이나 optimizer update 직후에도 최신 radiance를 사용한다.

기존 external HDR가 있으면 학습된 base/weather radiance를 우회한다. 기존 외부 HDR 회전·exposure·고해상도 sky API는 그대로다. `clear_external()` 이후에는 학습된 weather 조명으로 돌아온다.

`EnvLight.reg.weather_residual`의 loss는 선택한 weather 하나에만 적용한다.

- magnitude: `w * mean(abs(delta_log_env))`
- TV: `w * (mean(abs(delta[:,1:]-delta[:,:-1])) + mean(abs(delta[:,:,1:]-delta[:,:,:-1])))`

raw checkpoint에 없는 weather 잔차 key만 0으로 채워 strict load한다. Base tensor 누락이나 알 수 없는 key는 계속 오류다. Weather checkpoint는 `weather_log_residuals.rain` 또는 `.snow`를 그대로 저장한다. weather를 설정하지 않은 raw 모델은 기존 state key와 RNG 상태가 유지된다.

검증: `tests/test_weather_environment.py` 8개가 모두 통과했다(CUDA 포함, 2026-10-17). raw key/RNG 보존, log-radiance 수식과 strength 검증, rain/snow 독립성, optimizer의 base freeze, magnitude/TV 값, 두 번 연속 backward의 새 mip graph, strict checkpoint roundtrip, external 우회/복귀를 검사했다. 실제 nvdiffrast에서 surface·sky 양쪽이 2배 radiance를 반영하고 각각 weather residual까지 유한한 gradient를 전달함을 확인했으며 strength 0에서 raw RGB·sky의 bitwise 일치도 확인했다. 실제 장면의 장기 학습 품질을 검증한 결과는 아니다.
