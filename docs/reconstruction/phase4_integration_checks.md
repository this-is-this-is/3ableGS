# Base v4 CUDA 통합 검증

2026-10-17에 `tools/check_base_v4_reconstruction.py`를 실행해 통과했다.
재현 명령은 아래와 같다. 실제 장면 학습이나 저장된 Base v3 결과의 변경은 하지 않는다.

```bash
cd /home/user/Desktop/3ableGS
CUDA_VISIBLE_DEVICES=0 .conda/bin/python -u tools/check_base_v4_reconstruction.py
```

설정은 `configs/base_v4/scene788_30f.yaml`, 비교 기준은
`configs/base_v3/scene788_30f.yaml`이다. 실제 material decoder, local tangent
normal, HDR64/5 mips, camera response, gsplat 및 pretrained AlexNet LPIPS를
사용한다. 장면만 16개 Background Gaussian과 160×160 합성 입력으로 대체했다.
이 검증의 LPIPS 수치는 합성 입력에서의 loss 값이며 장면 평가 지표가 아니다.

## 확인한 내용

- 동일 seed의 새 Base v3/v4 모델은 checkpoint state key와 초기 tensor 값이
  정확하게 일치했다. 등록된 parameter 이름과 `requires_grad`도 같았다.
- `set_train()` 이후에도 LPIPS network는 eval 상태이고 parameter가 모두
  frozen이다. 학습 loss 계산 후 evaluation Metric의 state와 update count가
  변하지 않았다.
- Step 4999에서는 LPIPS network를 호출하지 않았다. Step 5000/5500/6000에서
  동일 crop RNG로 각 4개의 128×128 patch를 계산했다. 원본 LPIPS 값
  `0.8090257645`에 각각 `0 / 0.01 / 0.02`가 곱해져 loss
  `0 / 0.0080902576 / 0.0161805153`이 됐다. Step 5000에서 network 계산은
  수행하지만 ramp가 0이어서 loss와 gradient 기여는 0이다.
- **`boundary_perceptual_loss` 하나만 역전파**했을 때 RGB, xyz, feature,
  opacity, scaling, rotation, material decoder, normal head, HDR, affine,
  camera response에 유한하고 0이 아닌 gradient가 전달됐다. CamPose의
  learning rate는 0으로 유지됐다. LPIPS parameter에는 gradient가 없었다.
- RGB gradient L1 합은 `1.14476156`, main rasterization의 projected Gaussian
  absgrad L1 합은 `0.00690107`이었다. Auxiliary render가 densification에
  사용할 main render gradient 정보를 덮어쓰지 않았다.
- 임시 파일로 저장한 model-only checkpoint를 strict reload한 후 동일
  렌더링 RGB의 최대 차이는 `1.1920929e-7`이었다. 임시 checkpoint는 검사
  종료 시 정리된다. 기존 788씬 Base v3 checkpoint는 열거나 수정하지 않았다.

전체 결과는 `logs/reconstruction/phase4_integration.json`, 실행 로그는
`logs/reconstruction/phase4_integration.log`에 있다. 실제 788씬 Base v4의
학습 안정성과 PSNR/SSIM/LPIPS 개선 여부는 별도의 30k 학습 후 판단해야 한다.
