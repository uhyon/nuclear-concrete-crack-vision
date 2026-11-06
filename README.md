# 원전 콘크리트 균열 원격 감시 머신 비전

장기 운영 원전(계속운전)과 사용후핵연료 건식저장시설의 콘크리트 열화를 로봇·드론 원격 영상으로 감시하기 위한 균열 분할(segmentation) 프로젝트입니다.

> 연구 질문: 일반 시설물 영상으로 학습한 균열 분할 모델은 원전 원격 점검 영상 조건(방사선 노이즈, 저조도, 좁은 시야각)에서 얼마나 성능이 떨어지고, 어떻게 회복할 수 있는가?

## 왜 이 문제인가 (2026년 기준)

- **계속운전 확대**: 2026-10-27 고리 3·4호기 10년 수명 연장 허가, 한빛 1·2호기 등 7기 심사 대기. 오래된 구조물을 더 오래, 더 자주 점검해야 합니다.
- **사용후핵연료 저장 포화**: 습식저장조가 2030년(한빛)부터 차례로 포화되어 부지 내 건식저장시설이 20년 이상 운영될 수 있습니다. 콘크리트 오버팩의 장기 열화 감시가 과제입니다.
- **현업의 방향**: 한국수력원자력은 로봇·드론과 3D 디지털트윈을 결합한 원전 구조물 점검 기술로 2026년 한국콘크리트학회 기술상을 받았습니다. 판독 자동화와 현장 조건 강건성이 다음 단계입니다.

## 접근 방법

1. **베이스라인**: AI Hub「SOC 시설물 균열패턴 이미지」로 U-Net(ResNet34) 균열 분할 모델 학습
2. **원전 조건 강건성 평가**: 같은 검증셋에 원전 촬영 조건을 합성해 IoU/F1 하락 측정
3. **회복**: 원전 조건 증강을 넣어 다시 학습하고 하락 폭이 얼마나 줄어드는지 비교

### 합성하는 원전 촬영 조건 (`src/augment/nuclear_conditions.py`)

| 조건 | 흉내 내는 현장 상황 |
| --- | --- |
| `radiation` | 감마선이 이미지 센서에 닿아 생기는 흰 점·짧은 줄·잡음 |
| `low_light` | 조명이 부족한 밀폐 공간 |
| `uneven_light` | 로봇 탑재 조명 하나로 비출 때의 불균일 조명 |
| `motion_blur` | 이동 중인 로봇·드론 촬영 흔들림 |
| `narrow_fov` | 내시경·소형 로봇 카메라의 좁은 시야, 렌즈 왜곡, 비네팅 |
| `combined` | 위 조건 복합 |

각 조건은 강도 1~3단계로 적용합니다. 이 합성은 실제 방사선 환경 영상의 근사치이며, 실제 영상과의 비교 검증은 후속 과제입니다.

## 결과

> 2단계(2026-11-26) 학습 후 채울 예정입니다.

| 조건 | IoU | F1 |
| --- | --- | --- |
| clean | – | – |
| radiation (L2) | – | – |
| low_light (L2) | – | – |
| combined (L2) | – | – |

## 빠른 시작

Colab 실행 순서는 [docs/colab_quickstart.md](docs/colab_quickstart.md)에 있습니다.

```bash
pip install -r requirements.txt

# 1) AI Hub 라벨(JSON) -> 마스크(PNG)
python -m src.data.aihub_to_masks --labels data/labels --images data/images --out data/masks

# 2) 학습
python -m src.train --images data/images --masks data/masks --out runs/baseline --limit 10000

# 3) 원전 조건 강건성 평가
python -m src.evaluate --ckpt runs/baseline/best.pt --split runs/baseline/split.json \
    --masks data/masks --out runs/baseline/robustness.csv

# 4) 원전 조건 증강으로 재학습(회복 실험)
python -m src.train --images data/images --masks data/masks --out runs/condition_aug --limit 10000 --condition-aug
```

## 구조

```
src/
  data/aihub_to_masks.py        AI Hub 라벨 -> 분할 마스크
  augment/nuclear_conditions.py 원전 촬영 조건 합성
  dataset.py                    데이터셋, 분할
  train.py                      학습
  evaluate.py                   조건별 강건성 평가
  metrics.py                    IoU, Precision, Recall, F1
  preview_conditions.py         조건 비교 그림
tests/                          단위 테스트 (pytest)
```

## 데이터

- [AI Hub SOC 시설물 균열패턴 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71769) (교량·댐·터널 등, 결함 10종, 약 50만 장)
- AI Hub 이용약관에 따라 데이터 자체는 이 저장소에 올리지 않습니다.

## 로드맵

- [x] 1단계: 원전 이슈·선행 연구·데이터 조사, 주제 선정 (2026-11-06)
- [ ] 2단계: 베이스라인 학습, 조건별 성능 하락 측정 (2026-11-26)
- [ ] 3단계: 원전 조건 증강·도메인 적응으로 회복, 건식저장 용기 금속 부식으로 확장 (2026-12 ~ 2027-02)
- [ ] 4단계: 논문·연구 제안서 초안 (2027-03)
