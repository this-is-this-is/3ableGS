# Rain/Snow material residual 복원 근거

2026-10-17 구현. `models/weather_material.py`와 `models/gaussians/feature.py`의 weather 연결을 담당한다. 실제 Rain/Snow 장기 학습 결과를 뜻하지 않는다.

## 확인한 실제 소스

`conversation_index.json`과 `artifact_mentions.json`에서 관련 JSONL을 찾고, 설명 문장이 아니라 당시 도구가 출력한 Python 소스를 확인했다.

| 근거 | 내용 |
|---|---|
| `/home/user/.codex/sessions/2026/10/07/rollout-2026-10-07T21-32-13-01a11421-4704-7271-b4f5-a5191a88939b_01a11659-997c-78d1-b7cf-59f0a7f99a4e.jsonl:74` (UTC `12:36:48.494`) | `WeatherMaterialDecoder` 전체 123줄 |
| 같은 파일 `:82` (UTC `12:36:56.171`) | `_normal_upness`, logit residual 적용, gbuffer normal 처리 순서 |
| `/home/user/.codex/sessions/2026/10/07/rollout-2026-10-07T13-55-08-01a114b7-2104-7d53-b2d5-835751d6de29.jsonl:150` (UTC `04:55:43.133`) | decoder 초기화, world-up 정규화, 입력 detach, runtime strength/cache |
| `/home/user/.codex/sessions/2026/10/07/rollout-2026-10-07T11-11-27-01a11421-4704-7271-b4f5-a5191a88939b.jsonl:66` (UTC `02:12:30.771`) | material_outputs/materials, residual cache/pop 코드 |
| [E09](evidence/E09_weather_overlay.txt), [E10](evidence/E10_weather_schedule.txt) | 실제 Base v4 실행은 template 기본 32×1 대신 64×2 사용, Background만 학습 |

`models/weather_material.py`는 위 전체 출력과 byte 단위로 같다. SHA256: `8d8667b337cf3c8be1e5d8f045af91089b4f3e4d020e24b281a04e39aa911191`. `feature.py`는 필요한 역사적 메서드의 의미를 현재 복원 코드에 맞게 연결한 것이며 파일 전체를 byte 단위로 복원했다는 뜻은 아니다.

## 수식과 순서

입력은 `feature32 | base A3 | base M1 | base R1 | upness1`로 38차원이다. 실제 설정은 `38 → 64 → 64 → 4`, hidden activation은 `ReLU(inplace=True)`다. 마지막 Linear의 weight와 bias만 0으로 초기화한다. 두 hidden 층은 일반 Linear 초기화를 유지한다.

```
upness = clamp(abs(dot(n_geometry, normalize(world_up))), 0, 1)
delta_A = 4 * tanh(decoder_output[:3])
delta_R = 5 * tanh(decoder_output[3:4])
A_weather = sigmoid(z_A_raw + strength * delta_A)
R_weather = 0.99 * sigmoid(z_R_raw + strength * delta_R) + 0.01
M_weather = M_raw
```

`n_geometry`는 최종 world-space quaternion과 scale에서 구한 Gaussian 최소 scale 축이다. **학습된 tangent normal 보정을 적용하기 전**의 축을 사용한다. normal의 부호에 무관하므로 카메라 방향도 입력하지 않는다. 이후 shading용 normal에 기존 tangent 보정과 camera-facing flip을 차례로 적용한다.

입력 feature, base A/M/R, upness를 decoder 직전에 detach한다. residual을 더하는 raw의 원래 preactivation logits도 detach한다. `logit(sigmoid(z))`로 복원하지 않으므로 zero output 초기화는 base 활성값과 정확히 일치한다. `logit_eps=1e-4`는 trainer의 화면 공간 paired target loss에 쓰이며 이 Gaussian decoder 수식에 추가하지 않는다.

`raw`와 strength 0에서는 residual decoder를 호출하지 않고 base 값을 통과시킨다. 이때 같은 weather의 이전 cached residual도 제거한다. metallic과 shading normal은 weather material decoder가 수정하지 않는다. 환경 조명 residual은 별도 모듈이며 동적 물체와 sky에도 영향을 준다.

## 현재 코드에 맞춘 연결

- `weather_material` 설정이 켜진 `Background`에만 residual `ModuleDict`를 추가한다. Rain/Snow는 별도의 branch다. 기존 raw decoder는 항상 하나이며 색을 직접 예측하는 추가 weather head를 만들지 않는다.
- weather branch 이름은 `weather_material.weathers`에서 가져온다. 원래 Base v3의 `weathers=['raw']` 계약을 유지하기 위해 residual 이름 목록을 별도로 인정한다.
- 설정이 없거나 꺼져 있으면 파라미터/state key/global RNG 소비가 추가되지 않는다. normalized world-up은 checkpoint에 저장하지 않는 buffer이며 설정에 기록한다.
- `material_outputs`, `get/pop/clear_weather_material_outputs`, `get_weather_material_parameters`, `set_weather_material_strength`를 연결했다. trainer는 cached delta 두 개만 regularization에 사용하고 pop하여 autograd 참조를 해제한다.
- 기본 `load_state_dict(..., strict=True)`는 weather branch가 빠져도 실패한다. 명시적인 `allow_new_weather=True` warm start에서만 **전체가 없는 새 branch**의 초기값을 채운다. 부분 손상된 branch나 base 누락은 실패한다. 학습 단계 전환은 trainer가 모델 전체의 metadata/branch를 검사한 뒤 수행한다.
- geometry·feature·base decoder·normal·camera·HDR freeze와 densification 차단은 weather trainer의 책임이다. 이 모듈의 detach만으로 전체 장면이 고정됐다고 판정하지 않는다.

## 검증

`tests/test_weather_material.py`의 9개 CPU 검사를 통과했다.

- 실제 38→64→64→4 구조, zero 출력 및 첫 update에서 finite output-layer gradient
- `cap*tanh` 정확한 값, 잘못된 채널/leading shape 오류
- Rain/Snow zero 초기화 및 strength 0에서 bit-exact raw material identity
- raw preactivation logit에 strength를 곱한 delta 적용, metallic 보존
- nonzero residual에서도 feature/raw decoder/geometry/normal에는 gradient가 없고 weather hidden/output 층에 finite gradient
- world-up 정규화, normal 부호 불변, tangent 보정 및 camera flip 이전 orientation 사용
- raw-only state key/RNG 보존과 optimizer parameter coverage
- raw warm start와 weather checkpoint round trip, 부분 branch/base 손상 검출
- invalid world-up, weather 이름, strength 검출

기존 `tests/test_base_v3_material.py` 12개도 통과했다. 기존 RGB 경로, 네 Gaussian 종류의 normal 처리, raw parameter/state 계약 및 split/clone/cull 검사를 포함한다. 전체 trainer/GPU 단계 검증은 별도 통합 기록에서 다룬다.
