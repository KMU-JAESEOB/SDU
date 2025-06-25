# 핵심 SDA-U 알고리즘

🎯 **핵심 기능만 담은 간소화된 SDA-U (Selective Domain Adaptation with Unlearning) 구현**

## 📋 알고리즘 구조

1. **타겟 도메인 선별** (타겟 도메인 데이터셋의 5%)
2. **학습 진행** (+ L2 정규화)
3. **10에포크 정체 시 언러닝** (소스도메인 샘플 중 선별하여 머신 언러닝)
4. **300에포크 반복**

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Office-31 실험 실행
```bash
# 단일 실험
python main.py --dataset Office31 --source_domain Amazon --target_domain Webcam

# 전체 실험 (6개 조합)
python office31_full_experiments.py
```

### 3. Office-Home 실험 실행
```bash
# 단일 실험  
python main.py --dataset OfficeHome --source_domain Art --target_domain Clipart

# 전체 실험 (12개 조합)
python officehome_full_experiments.py
```

## 📁 파일 구조

```
📂 SDA/
├── 🎯 main.py                        # 핵심 SDA-U 알고리즘
├── 🏢 office31_full_experiments.py   # Office-31 전체 실험
├── 🏠 officehome_full_experiments.py # Office-Home 전체 실험
├── 📊 office31_loader.py             # Office-31 데이터 로더
├── 📊 officehome_loader.py           # Office-Home 데이터 로더
├── ⚙️ gpu_config.py                  # GPU 최적화
├── 📋 requirements.txt               # 의존성
├── 📁 results/                       # 실험 결과
└── 📁 models/                        # 저장된 모델
```

## ⚙️ 주요 하이퍼파라미터

- **배치 크기**: 32
- **학습률**: 2e-4 (Adam + L2 정규화)
- **타겟 샘플 비율**: 5% (Budget)
- **정체 임계값**: 10에포크
- **총 훈련 에포크**: 300에포크
- **언러닝 강도**: -0.05 (매우 부드러운)

## 📊 지원 데이터셋

- **Office-31**: Amazon, Webcam, DSLR (31개 클래스)
- **Office-Home**: Art, Clipart, Product, Real World (65개 클래스)

## 📈 결과 확인

실험 결과는 `results/` 디렉토리에 JSON 형태로 저장됩니다:
- `results/office31/`
- `results/officehome/` 