# Base v3 구현·검증 기록 — 2026-10-16

사용자의 “기록에서 확인된 구성을 기준으로 진행” 지시에 따라 Phase C의 raw Base v3 코드와 실행 설정을 복원했다. **코드 및 기본 검증 완료이며, 재생성 G-buffer를 사용한 학습·정량 성능 재현은 아직이다.** Base v4의 combo 및 weather decoder는 다음 단계다.

## 복원 범위와 출처

- [Material / normal 근거](phase3_material_provenance.md): 32D feature, Background detached xyz 입력, LeakyReLU decoder, 별도 ReLU normal head, Gaussian local tangent·최종 quaternion·camera-facing 순서.
- [HDR/PBR 근거](phase3_pbr_provenance.md): raw shader 소스 복원, 실제 `64/5/64/16` 설정과 보존된 SplatAD_IR 이식본의 shader/gradient 비교.
- [Camera response 근거](phase3_camera_provenance.md): 최종 효율화 backward 포함 원본 180행 복원. `response(linear_hdr)` 자체가 Reinhard→curve→sRGB를 수행한다.
- [최종 config 근거](phase3_config_provenance.md): 실제 Base v3 resolved config의 trainer/model/render mapping 대조. [base.yaml](../../configs/base_v3/base.yaml), [scene788_30f.yaml](../../configs/base_v3/scene788_30f.yaml).

Trainer와 로더는 아래 실제 출력에서 Base v3에 사용된 분기만 복원했다. 후속 ablation/weather/fog 분기는 가져오지 않았다.

| 기록 | 확인한 구현 |
|---|---|
| `/home/user/.codex/sessions/2026/09/02/rollout-2026-09-02T13-58-47-01a0607b-e562-7772-811a-fcd5b2aae183.jsonl:3890` | G-buffer alpha 정규화, shading 방향, 원래 재질·normal loss |
| `/home/user/.codex/sessions/2026/10/05/rollout-2026-10-05T20-18-55-01a10bc9-c736-7923-85fe-ce959b458f6e.jsonl:46` | 최종 trainer의 8채널 unpack 및 refined viewdirs |
| `/home/user/.codex/sessions/2026/10/06/rollout-2026-10-06T10-11-12-01a0c78d-c043-7f90-888f-7a5809f85f34_01a10ec3-bfd7-7d02-901e-94a043482779.jsonl:334` | 최종 G-buffer reduction·normal robust loss |
| `/home/user/.codex/sessions/2026/10/06/rollout-2026-10-06T20-39-46-01a11103-396a-7a23-97df-aeaa138bf2b0.jsonl:401` | `seq.frame.pass.png` 경로와 Waymo loader 설정 전달 |
| `/home/user/.codex/sessions/2026/10/07/rollout-2026-10-07T13-44-25-01a114ad-52bd-7680-8028-9e8cd522826f.jsonl:98` | uint8 native G-buffer 저장, undistort, normal decode |
| `/home/user/.codex/sessions/2026/10/07/rollout-2026-10-07T11-11-27-01a11421-4704-7271-b4f5-a5191a88939b.jsonl:56` | 최종 linear 합성→response→affine, 진단 layer no_grad, class별 PBR 렌더 |

## 확정한 loss·좌표 계약

Rasterizer는 `[albedo3, metallic1, roughness1, world normal3, expected camera-Z1]`을 반환한다. signed normal 채널에 RGB clamp를 적용하지 않는다. 재질은 `opacity.clamp(min=1e-4)`로 나누고, A/M은 `[0,1]`, R은 `[.01,1]`, normal은 unit vector로 정규화한다. Surface와 sky는 같은 HDR로 linear 공간에서 합성한다.

RGB/SSIM/depth/mask/affine loss는 기존 경로를 사용한다. G-buffer는 render와 target 중 더 낮은 해상도에서 비교한다. Sky와 ego 영역은 제외하며, dynamic 영역은 포함한다. Bilinear antialias resize 이후 mask>0.99만 사용한다. A/M/R은 masked L1을 유효 픽셀 수×채널 수로 나눈다. Normal은 실제 render pose의 world→OpenCV camera 회전 후 `[1,-1,-1]`을 곱해 OpenGL로 옮긴다. `r=1-cos`, loss는 `w*mean_masked(r/(1+r/0.5))`다. G-buffer albedo는 sRGB이고 shader에서 linear로 변환한다.

## 현재 공식 코드에 맞춘 연결과 보강

- PBR 전용 trainer의 rasterizer 경로를 추가했다. 기존 RGB trainer의 3채널 경로는 유지한다. 보조 class 렌더는 densification의 full-render `info`를 덮어쓰지 않는다.
- Raw는 `vgg:null`이며, train loop는 VGG 설정이 있을 때만 수동 content backward를 호출한다. `vis_freq=0`의 modulo 오류도 수정했다.
- EnvLight가 있는 경우에도 mask loss를 활성화했다. `mlp.raw.normal_residual` optimizer group은 `mlp.raw` 설정을 상속한다.
- 과거 수정대로 LiDAR 투영의 새 카메라 행렬 생성을 제거해 RGB/G-buffer와 같은 K를 사용한다. 출처는 위 Sep 2 기록의 line 478이다.
- 누락되거나 크기가 다른 teacher PNG는 명시적 오류다. 초기 복원 범위의 G-buffer 시퀀스는 camera 0만 지원한다. RGB loading downscale을 먼저 되돌려 target K를 계산하도록 보강했으며, 과거 기본값 downscale=1의 수식은 동일하다.
- 빈 LiDAR hit 집합은 gradient를 보존하는 0 loss로 처리한다. 모두 cull된 Gaussian class의 빈 평균 regularizer는 건너뛴다.
- 데이터에 없는 node class가 checkpoint에서 빠진 경우 동일하게 제거한 후 strict-load한다. 호출자의 checkpoint dict를 변경하지 않는다. **이번 검증은 model weights roundtrip이며 optimizer/RNG를 포함한 정확한 resume는 후속 Phase F 범위다.**

## 검증 결과

- 전체 회귀 테스트 **50개 통과**(CUDA 포함): [로그](../../logs/reconstruction/phase3_tests.log).
- 합성 CUDA 장면: 실제 material/HDR/response/affine, loss, optimizer schedule, AbsGrad, checkpoint 통합 검사 통과. Step 0 HDR update=0, step 3001 update>0. 모든 활성 parameter group에서 finite/nonzero gradient. 새 trainer에서 strict-load 후 RGB 최대 차이 `1.1920929e-7`. [결과 JSON](../../logs/reconstruction/phase3_integration.json).
- 실제 788 raw000–029 로딩 및 frame015 렌더/역전파 통과. 검사용 1/4 해상도, Background 초기 LiDAR 10k, random seed points 0. 실제 SMPL skinning/voxel deformer를 포함해 Background 9,941 / Rigid 552 / SMPL 82,680개를 초기화했다. 모든 gradient가 finite이고 해당 뷰의 Rigid gradient는 0이었다. Deformable은 해당 데이터에서 제거되어 별도 contract/refinement test로 확인했다. **Optimizer update와 G-buffer loss는 이 실제 장면 검사에서 실행하지 않았다.** [결과 JSON](../../logs/reconstruction/phase3_real_scene_smoke.json).
- G-buffer PNG loading/undistortion/normal decode, normal 회전·robust loss, split/clone/cull 후 32D feature 및 Adam state, camera response의 단조성/고정점/custom backward를 검사했다.

재실행 명령(프로젝트 루트):

```bash
.conda/bin/python -m unittest discover -s tests -v
.conda/bin/python tools/check_base_v3_reconstruction.py
.conda/bin/python tools/check_base_v3_scene.py
.conda/bin/python tools/resolve_config.py --config_file configs/base_v3/scene788_30f.yaml
```

788 실제 장면 검사 결과는 먼저 동일 본문을 `/tmp/check_phase3_scene.py`에서 실행한 뒤 재사용 가능한 `tools/check_base_v3_scene.py`로 정리한 것이다. 본 학습은 시작하지 않았다. 현재 확인한 경로에 완성된 788 raw30 G-buffer PNG 묶음이 없으므로, 다음 실제 PBR 학습에는 앞 단계 생성 결과 120장이 필요하다. 이 검증 수치는 복원 모델의 PSNR/SSIM/LPIPS 성능을 뜻하지 않는다.
