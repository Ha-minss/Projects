# House Prices — 누수 없는 Stacking / GitHub용 재현 가능한 Kaggle 파이프라인

## 프로젝트 개요

이 저장소는 원래의 실험용 주피터 노트북을 **재현 가능하고, 평가 누수를 줄인 탭ुल러 머신러닝 프로젝트**로 다시 정리한 버전입니다.

대상 문제는 Kaggle의 **House Prices: Advanced Regression Techniques**이고, 목표는 Ames 주택 데이터의 다양한 구조적 속성을 이용해 `SalePrice`를 예측하는 것입니다.

원래 노트북에도 좋은 요소는 이미 있었습니다.

- 수치형 + 범주형 혼합 데이터 처리
- 도메인 기반 결측치 처리
- 로그 변환
- 여러 회귀 모델
- stacking / blending

문제는 모델 아이디어가 아니라, **최종 blended CV 점수 계산 방식**이었습니다.  
원본 노트북은 마지막 CV 단계에서 **이미 전체 데이터로 학습된 전역 모델**을 재사용하고 있었기 때문에, 대표 성과로 쓰기엔 낙관적 편향 가능성이 있었습니다.

이 저장소는 그 부분을 고친 버전입니다.

---

## 우리가 무엇을 했는가

### 문제 정의
Ames Housing 데이터에서 `SalePrice`를 예측합니다.

설명 변수는 다음처럼 매우 다양합니다.

- 주택 크기와 품질
- 대지 크기와 동네 특성
- 지하실 / 차고 / 리모델링 정보
- 판매 시기와 판매 유형
- 희소한 범주형 특성들

즉, 이 프로젝트는 전형적인 **혼합형 탭ुल러 회귀 문제**입니다.

### 데이터사이언스 목표
이 프로젝트의 핵심 목표는 4가지입니다.

1. **재현 가능성**
2. **평가의 정직함**
3. **Kaggle 제출 가능성**
4. **GitHub / 포트폴리오 설명력**

---

## 원본 노트북 대비 가장 중요한 개선점

### 기존 문제
원본 노트북의 마지막 `BlendEstimator`는:

- 이미 전체 학습 데이터로 fit된 base model
- 전역적으로 만들어진 stacked meta model
- `fit()`에서 실제 재학습을 하지 않는 estimator

를 사용한 채 `cross_val_score`를 돌리는 구조였습니다.

즉, 최종 blended CV 단계에서는 검증 fold가 완전히 격리되지 않았습니다.

### 이번 버전에서 고친 점
이 저장소는 다음 기준으로 다시 작성했습니다.

- 전처리 통계는 fold별 train 데이터로만 추정
- base model OOF 예측을 fold별로 생성
- meta model은 해당 train portion의 OOF 예측만으로 학습
- 최종 ensemble OOF는 nested stacking 방식으로 계산

즉, **“이미 fit된 전역 블렌드 모델을 CV에 재사용하는 방식”을 없앴습니다.**

이 방식은 더 느리지만, 훨씬 더 방어적이고 신뢰할 수 있습니다.

---

## 전처리 전략

Ames 데이터셋의 결측은 단순히 "모름"이 아니라, 실제로는 **해당 시설이 없음**을 의미하는 경우가 많습니다.

### 도메인 기반 결측치 처리
예시:
- garage 없음 → garage 관련 필드 `"None"` 또는 `0`
- basement 없음 → basement 관련 범주형 `"None"`
- `Functional` → `"Typ"`
- `Electrical` → `"SBrkr"`
- `KitchenQual` → `"TA"`
- `Exterior1st`, `Exterior2nd`, `SaleType` → train fold 최빈값
- `LotFrontage` → train fold 기준 neighborhood 중앙값

### 숫자 코드형 변수 범주화
다음 변수는 숫자처럼 보여도 범주형에 더 가깝기 때문에 문자열로 변환합니다.

- `MSSubClass`
- `YrSold`
- `MoSold`

### 왜도 보정
오른쪽 꼬리가 긴 연속형 변수에 대해 로그 변환을 적용합니다.

조건:
- 음수가 아님
- 연속형 변수로 보는 것이 타당함
- `abs(skew) > threshold`

### 타깃 변환
타깃은 로그 스케일로 학습합니다.

```python
y = np.log1p(SalePrice)
```

예측 후에는 다시 원래 스케일로 되돌립니다.

```python
SalePrice = np.expm1(pred_log)
```

---

## 모델링 전략

Base learner:
- RidgeCV
- SVR
- GradientBoostingRegressor
- RandomForestRegressor
- XGBoost (설치되어 있으면 사용)
- LightGBM (설치되어 있으면 사용)

최종 예측 구조:
1. base model OOF 예측 생성
2. 그 예측으로 meta model 학습
3. 전체 학습 데이터로 base model 재학습
4. test의 base prediction 생성
5. stack prediction 생성
6. base + stack을 가중합으로 blend

즉, 이 프로젝트는 **클래식 탭ुल러 앙상블 회귀 파이프라인**입니다.

---

## 검증 설계

이 프로젝트에는 검증이 두 층으로 있습니다.

### 1) Base model OOF CV
각 base learner의 성능을 개별적으로 확인하기 위한 단계입니다.

### 2) Leakage-free nested stacking OOF
최종 ensemble의 성능을 더 정직하게 보기 위한 단계입니다.

왜 nested가 필요할까요?

stacking도 결국 **학습되는 모델(meta model)** 이기 때문입니다.  
OOF base prediction을 다 만든 뒤, 그 전체 OOF 위에 meta model을 fit하고 같은 데이터에서 성능을 보면 meta model 입장에서 여전히 in-sample 성격이 생깁니다.

그래서 이 저장소는:

- outer fold → 최종 검증
- inner fold → outer-train 내부에서 meta model용 OOF feature 생성

구조를 사용합니다.

이게 최종 stacked ensemble OOF를 더 신뢰할 수 있게 만드는 방식입니다.

---

## 폴더 구조

```text
house-prices-leakage-free/
├─ README.md
├─ README_ko.md
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ data/
│  ├─ raw/
│  │  ├─ train.csv
│  │  └─ test.csv
│  └─ processed/
├─ notebooks/
│  └─ house_prices_kaggle_leakage_free.ipynb
├─ scripts/
│  ├─ train.py
│  └─ predict.py
└─ src/
   └─ house_prices/
      ├─ __init__.py
      ├─ config.py
      ├─ data.py
      ├─ preprocess.py
      ├─ models.py
      ├─ ensemble.py
      └─ utils.py
```

---

## 실행 방법

### 1) Kaggle 데이터 배치

```text
data/raw/train.csv
data/raw/test.csv
```

### 2) 설치

```bash
python -m venv .venv
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 3) 학습 요약 실행

```bash
python scripts/train.py
```

결과:
- base model OOF 결과
- leakage-free nested ensemble OOF 결과
- `artifacts/training_summary.json`

### 4) submission 생성

```bash
python scripts/predict.py
```

결과 파일:
```text
artifacts/submission.csv
```

---

## 이 프로젝트를 면접 / 포트폴리오에서 어떻게 설명하면 좋은가

이렇게 설명하면 좋습니다.

> 이 프로젝트는 Ames Housing 데이터를 활용한 탭ुल러 회귀 앙상블 프로젝트다.  
> 원래는 Kaggle용 실험 노트북이었지만, GitHub로 정리하는 과정에서 최종 blended CV 계산이 prefit 모델을 재사용하는 구조라는 점을 발견했다.  
> 그래서 fold별 전처리, OOF base prediction, nested stacking evaluation을 적용한 leakage-aware pipeline으로 리팩터링했다.  
> 핵심은 단순히 점수가 아니라, 실험용 분석을 재현 가능하고 방어적인 ML 프로젝트로 바꿨다는 점이다.

---

## 왜 이게 더 좋은 프로젝트인가

좋은 데이터사이언스 프로젝트는 단지 숫자 하나가 아닙니다.

중요한 것은:
- 실험용 노트북을 코드베이스로 바꿀 수 있는가
- 전처리 / 모델링 / 검증을 분리할 수 있는가
- suspicious score를 그냥 믿지 않고 점검할 수 있는가
- 모델링 선택을 설명할 수 있는가
- 무엇이 신뢰 가능하고 무엇이 아닌지 솔직하게 말할 수 있는가

이 저장소는 바로 그 점을 보여줍니다.

---

## 참고
- leakage-free nested evaluation은 느립니다.
- 빠른 실험이 필요하면 `Config`에서 fold 수를 줄일 수 있습니다.
- Kaggle 제출용으로는 함께 제공한 단일 노트북을 그대로 업로드하면 됩니다.
