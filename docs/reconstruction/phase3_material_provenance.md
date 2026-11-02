# Phase 3 material / normal 복원 근거

이 문서는 2026-10-16 `3ableGS`의 raw Base v3 재질 경로를 복원할 때 사용한 실제 코드 출력과 patch의 출처다. 과거 파일 전체를 그대로 복사한 것은 아니며, Base v3에 사용된 분기만 현재 공식 WeatherCity 클래스에 이식했다. 과거의 추가 ablation 및 weather residual decoder는 이번 범위에 포함하지 않는다.

## 기록에서 확인한 내용

- `/home/user/.codex/sessions/2026/09/29/rollout-2026-09-29T14-12-58-01a0eb94-9408-7411-8cb4-81678572f34d.jsonl`, line 45 tool output: `feature.py` material 초기화 부분. 먼저 `32→32→5` MLP를 global RNG로 생성하고, Background 공간 입력 분기만 seed `12012`의 격리된 RNG로 `35→32→5` MLP를 새로 생성한다. hidden activation은 Base v3 설정에서 `LeakyReLU(0.01, inplace=True)`이며 dynamic class는 원래 Linear 가중치를 유지하고 activation만 바꾼다.
- `/home/user/.codex/sessions/2026/10/07/rollout-2026-10-07T11-11-27-01a11421-4704-7271-b4f5-a5191a88939b.jsonl`, line 51 tool output: normal head 초기화, 격리 RNG context, material activation. Normal head는 `32→32→3`, 중간 **ReLU**, seed `15015`, 마지막 Linear weight/bias 모두 0이다. 따라서 첫 normal은 geometry normal과 동일하다. A/M은 sigmoid, R은 `0.99*sigmoid(logit)+0.01`이다.
- `/home/user/.codex/sessions/2026/10/05/rollout-2026-10-05T20-18-55-01a10bc9-c736-7923-85fe-ce959b458f6e.jsonl`, line 31 tool output: XYZ 입력, normal correction 및 G-buffer packing. XYZ는 `clamp((means.detach()-scene_origin)/max(scene_scale,1e-6), -1, 1)`이다. Normal correction은 아래 수식으로 계산하고 camera-facing orientation을 마지막에 적용한다.
- `/home/user/.codex/sessions/2026/09/21/rollout-2026-09-21T20-28-14-01a0bf7d-a3fb-79e1-b1b0-6ae60bc9180c_01a0c3b9-4506-7a63-a329-af77b309d7c9.jsonl`, line 231 tool patch: Gaussian local tangent mode의 구현 및 regularization 추가. 이 patch의 다른 저장소 작업은 복원에 사용하지 않았다.
- `/home/user/.codex/sessions/2026/09/02/rollout-2026-09-02T13-58-47-01a0607b-e562-7772-811a-fcd5b2aae183.jsonl`, line 1571 tool output: `basics.py`의 `gaussian_normals` 실제 구현. min scale index에 해당하는 quaternion rotation matrix의 column을 선택하고 normalize한다. scale argmin에는 gradient가 없다.

`d = normal_head(feature)`이고 `e`가 최소 scale 축의 one-hot vector일 때:

```text
b = 0.5 * d / sqrt(1 + sum(d*d))
t = b - e * dot(b, e)
n = normalize(n_geometry + R(q_world) @ t)
normal_regularizer = 0.01 * mean(sum(t*t))
```

`q_world`는 Rigid instance pose, Deformable quaternion delta 및 instance pose, SMPL skinning 이후 각 기존 클래스가 계산한 최종 quaternion이다. Normal을 camera 쪽으로 flip한 뒤 `[albedo3, metallic1, roughness1, normal3]` 8채널을 splat한다.

## 현재 코드에 맞춘 명시적 통합 선택

- 기존 RGB 설정을 유지하기 위해 `raw_material` block이 있는 클래스에서만 PBR 경로를 활성화한다. block이 없으면 원래 `mlp_color` 3채널 경로와 초기화 RNG가 유지된다.
- 이번 단계는 raw 전용이다. PBR 클래스에 raw 이외 weather를 지정하면 명시적으로 오류를 낸다. 이후 weather material decoder 복원 단계에서 확장한다.
- `mlp_material.raw` 및 `raw_normal_residual` state key, `mlp.raw` 및 `mlp.raw.normal_residual` optimizer group 이름을 유지한다.
- 현재 프레임에서 사라진 SMPL node가 이전 frame의 normal graph를 재사용하지 않도록 step 시작과 absent SMPL branch에서 cached normal penalty를 지운다. 이것은 복원 중 발견한 수명 관리 보강이다.
- 별도 ablation의 Fourier 입력, hidden 폭/깊이 변경, split decoder를 조용히 무시하지 않고 지원 범위 밖으로 명시한다.

## 검증

`tests/test_base_v3_material.py`: 12개 CPU test. 기록에 따른 구조/초기화, detached 공간 입력, roughness 하한, RNG 격리, 회전에 대한 normal equivariance 및 bound/gradient, camera-facing 순서, frame 간 graph 초기화, Rigid/Deformable/SMPL 최종 quaternion 전달, optimizer parameter 중복/누락 및 strict checkpoint roundtrip, 기존 RGB 경로를 확인한다. SMPL node integration test는 skinning 결과를 fixture로 제공하므로 실제 SMPL skinning 전체를 검증했다고 해석하지 않는다.

추가 lifecycle 검증은 Background/Rigid/Deformable의 실제 refinement orchestration으로 split 1개, clone 1개, opacity cull 1개를 거쳐 3→5 Gaussian이 되는 경우를 실행한다. 32D feature와 point ID의 row 대응, 기존 Adam moment 유지, 새 row의 zero moment, decoder/normal head 불변 및 다음 Adam step을 확인했다. SMPL KNN regularizer는 RGB 전용 attribute 없이 material albedo와 feature에 gradient가 흐르는 것을 확인했다.
