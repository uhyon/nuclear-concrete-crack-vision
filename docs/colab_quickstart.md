# Colab 실행 가이드 (무료 GPU 기준)

무료 Colab은 디스크 약 100GB, 세션 최대 약 12시간입니다. 데이터 전체(119GB)는 들어가지 않으므로 **일부만** 씁니다.

## 0. 준비할 데이터 (AI Hub에서 PC로 받아 Google Drive에 올리기)

AI Hub 다운로드 화면에서 폴더를 펼쳐 아래만 고릅니다.

- **라벨링 데이터**: Training과 Validation 라벨 압축파일 전부 (JSON이라 작음)
- **원천 데이터(이미지)**: Training 이미지 압축파일 1~2개 (합쳐 10GB 안팎)

Google Drive에 `MyDrive/nuclear_crack/raw/` 폴더를 만들어 올립니다. Drive 무료 용량은 15GB입니다.

## 1. 런타임

메뉴 `런타임 > 런타임 유형 변경 > T4 GPU`

## 2. 셀 순서

```python
# 셀 1: Drive 연결, 코드 받기
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/uhyon/nuclear-concrete-crack-vision.git
%cd nuclear-concrete-crack-vision
!pip install -q -r requirements.txt
```

```python
# 셀 2: 압축 풀기 (Drive가 아닌 Colab 로컬 디스크로 풀어야 빠름)
!mkdir -p /content/data/images /content/data/labels
!for f in /content/drive/MyDrive/nuclear_crack/raw/*원천*.zip; do unzip -q -o "$f" -d /content/data/images; done
!for f in /content/drive/MyDrive/nuclear_crack/raw/*라벨*.zip; do unzip -q -o "$f" -d /content/data/labels; done
!find /content/data/images -type f | wc -l
!find /content/data/labels -name "*.json" | head -3
```

압축파일 이름은 AI Hub 실제 이름에 맞춰 바꿔 주세요(`TS_`, `TL_` 등으로 시작할 수 있습니다).

```python
# 셀 3: 라벨 JSON 하나 열어 구조 확인 (변환 전에 꼭 보기)
import json, glob
p = glob.glob('/content/data/labels/**/*.json', recursive=True)[0]
print(json.dumps(json.load(open(p, encoding='utf-8')), ensure_ascii=False, indent=1)[:2000])
```

`image.annotations[].label / points / shape` 구조가 아니면 결과를 Claude에게 보여 주세요. 변환 코드를 맞추겠습니다.

```python
# 셀 4: 마스크 변환
!python -m src.data.aihub_to_masks --labels /content/data/labels --images /content/data/images --out /content/data/masks
```

```python
# 셀 5: 조건 비교 그림 (README용)
import glob
img = glob.glob('/content/data/images/**/*.jpg', recursive=True)[0]
import os; stem = os.path.splitext(os.path.basename(img))[0]
!python -m src.preview_conditions --image "{img}" --mask "/content/data/masks/{stem}.png" --out docs/conditions.png
from IPython.display import Image; Image('docs/conditions.png')
```

```python
# 셀 6: 베이스라인 학습 (T4 기준 1만 장, 15 epoch는 약 2~3시간 예상)
!python -m src.train --images /content/data/images --masks /content/data/masks \
    --out /content/drive/MyDrive/nuclear_crack/runs/baseline --limit 10000 --epochs 15
```

처음에는 `--limit 2000 --epochs 3`으로 30분 안에 끝까지 도는지 먼저 확인하는 것을 권합니다.

```python
# 셀 7: 원전 조건 강건성 평가
!python -m src.evaluate --ckpt /content/drive/MyDrive/nuclear_crack/runs/baseline/best.pt \
    --split /content/drive/MyDrive/nuclear_crack/runs/baseline/split.json \
    --masks /content/data/masks --out /content/drive/MyDrive/nuclear_crack/runs/baseline/robustness.csv
```

결과(`history.json`, `robustness.csv`)는 Drive에 저장되므로 세션이 끊겨도 남습니다.
