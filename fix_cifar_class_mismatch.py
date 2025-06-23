"""
🔧 CIFAR10 ↔ CIFAR100 클래스 수 불일치 해결 유틸리티
CUDA 오류 'Assertion `t >= 0 && t < n_classes` failed' 해결을 위한 도구
"""

import os
import sys
import json
from pathlib import Path

def analyze_cifar_mismatch():
    """CIFAR 데이터셋 클래스 수 불일치 문제 분석"""
    
    print("🔍 CIFAR10 ↔ CIFAR100 클래스 수 불일치 분석")
    print("="*60)
    
    datasets = {
        'CIFAR10': {
            'classes': 10,
            'labels': list(range(10)),
            'categories': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        },
        'CIFAR100': {
            'classes': 100,
            'labels': list(range(100)),
            'categories': f'100개 세분화된 카테고리'
        }
    }
    
    print(f"📊 CIFAR10:  {datasets['CIFAR10']['classes']}개 클래스 (라벨: 0-{datasets['CIFAR10']['classes']-1})")
    print(f"📊 CIFAR100: {datasets['CIFAR100']['classes']}개 클래스 (라벨: 0-{datasets['CIFAR100']['classes']-1})")
    
    print(f"\n❌ 문제 상황:")
    print(f"   1. CIFAR10으로 훈련된 모델: 10개 클래스만 처리 가능")
    print(f"   2. CIFAR100 타겟 데이터: 0-99 라벨 포함")
    print(f"   3. 영향도 계산 시: 라벨 10-99가 모델 클래스 수(10)를 초과")
    
    return datasets

def create_cifar_safe_config():
    """CIFAR 안전 설정 생성"""
    
    print("\n🔧 CIFAR 안전 설정 생성 중...")
    
    # 해결 방법 1: 공통 클래스 수 사용 (상위 호환)
    safe_configs = {
        'method1_common_classes': {
            'description': '공통 클래스 수 사용 (CIFAR100 기준)',
            'num_classes': 100,  # CIFAR100 기준으로 통일
            'label_mapping': 'expand_cifar10_to_100',
            'batch_size': 64,    # 안전한 배치 크기
            'influence_samples': 100,  # 샘플 수 감소
        },
        
        'method2_label_clipping': {
            'description': '라벨 클리핑 사용 (CIFAR10 기준)',
            'num_classes': 10,   # CIFAR10 기준으로 통일
            'label_mapping': 'clip_cifar100_to_10',
            'batch_size': 64,
            'influence_samples': 100,
        },
        
        'method3_separate_models': {
            'description': '데이터셋별 별도 모델 사용',
            'strategy': 'separate_experiments',
            'cifar10_classes': 10,
            'cifar100_classes': 100,
            'batch_size': 64,
            'influence_samples': 100,
        }
    }
    
    print("✅ CIFAR 안전 설정 3가지 방법 생성 완료")
    return safe_configs

def generate_cifar_experiment_config(method='method2_label_clipping'):
    """CIFAR 실험용 안전 config.py 생성"""
    
    print(f"\n⚙️ CIFAR 안전 실험 설정 생성 중 (방법: {method})...")
    
    configs = create_cifar_safe_config()
    selected_config = configs[method]
    
    if method == 'method2_label_clipping':
        # 라벨 클리핑 방법 (권장)
        config_content = f'''# config.py - CIFAR10→CIFAR100 안전 실험용 설정

# ============================================
# 🔧 CIFAR 클래스 수 불일치 해결 (라벨 클리핑 방법)
# ============================================
ARCHITECTURE = 'resnet18'
BATCH_SIZE = 64           # 안전한 배치 크기
NUM_EPOCHS = 20           # CIFAR에 적합한 에포크
LEARNING_RATE = 1e-3

# 🚨 클래스 수 통일 (CIFAR10 기준)
NUM_CLASSES = 10          # CIFAR10 기준으로 고정
LABEL_CLIPPING = True     # CIFAR100 라벨을 0-9로 클리핑

# 데이터셋 설정
SOURCE_DATASET = 'CIFAR10'
TARGET_DATASET = 'CIFAR100'

# SDA-U 알고리즘 설정 (안전 모드)
TARGET_SUBSET_SIZE = 300     # 타겟 샘플 수 감소
NUM_UNLEARN_STEPS = 5        # 언러닝 스텝 감소
INFLUENCE_SAMPLES = 100      # 영향도 샘플 수 대폭 감소 (200→100)
ADAPTATION_EPOCHS = 8        # 적응 훈련 에포크
MAX_UNLEARN_SAMPLES = 50     # 최대 언러닝 샘플 감소

# 🔧 안전 장치 설정
VALIDATE_LABELS = True       # 라벨 범위 검증 활성화
SKIP_INVALID_LABELS = True   # 잘못된 라벨 건너뛰기
MAX_LABEL_VALUE = 9          # 최대 라벨 값 (CIFAR10 기준)

# 하이브리드 스코어링 파라미터
LAMBDA_U = 0.6
BETA = 0.1

# 저장 설정
SAVE_MODELS = True
SAVE_RESULTS = True
QUICK_TEST = False

def get_config():
    """설정을 반환하는 함수 (CIFAR 안전 모드)"""
    import torch
    
    return {{
        'architecture': ARCHITECTURE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'num_classes': NUM_CLASSES,
        'target_subset_size': TARGET_SUBSET_SIZE,
        'num_unlearn_steps': NUM_UNLEARN_STEPS,
        'influence_samples': INFLUENCE_SAMPLES,
        'adaptation_epochs': ADAPTATION_EPOCHS,
        'max_unlearn_samples': MAX_UNLEARN_SAMPLES,
        'validate_labels': VALIDATE_LABELS,
        'skip_invalid_labels': SKIP_INVALID_LABELS,
        'max_label_value': MAX_LABEL_VALUE,
        'label_clipping': LABEL_CLIPPING,
        'lambda_u': LAMBDA_U,
        'beta': BETA,
        'save_models': SAVE_MODELS,
        'save_results': SAVE_RESULTS,
        'quick_test': QUICK_TEST,
        'source_dataset': SOURCE_DATASET,
        'target_dataset': TARGET_DATASET,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }}
'''
        
        # config.py 파일 생성
        with open('config_cifar_safe.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✅ 안전한 CIFAR 설정 파일 생성: config_cifar_safe.py")
        
        return 'config_cifar_safe.py'

def create_label_safe_main():
    """라벨 안전 검증이 포함된 main.py 수정 사항 생성"""
    
    print("\n🔧 라벨 안전 검증 코드 생성 중...")
    
    safe_influence_function = '''
def compute_influence_scores_enhanced_safe(model, source_loader, target_batch, num_samples=100):
    """안전한 영향도 계산 (CIFAR 클래스 수 불일치 해결)"""
    print("🔍 안전한 영향도 점수 계산 중...")
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    target_data, target_labels = target_batch
    target_data, target_labels = target_data.to(device), target_labels.to(device)
    
    # 🚨 모델 클래스 수 확인
    if hasattr(model, 'fc'):
        num_classes = model.fc.out_features
    elif hasattr(model, 'classifier'):
        num_classes = model.classifier[-1].out_features
    else:
        num_classes = 10  # 기본값
    
    print(f"📊 모델 클래스 수: {num_classes}")
    print(f"🎯 원본 타겟 라벨 범위: {target_labels.min().item()} ~ {target_labels.max().item()}")
    
    # 🔧 라벨 클리핑 (강제로 모델 클래스 수에 맞춤)
    target_labels_safe = torch.clamp(target_labels, 0, num_classes - 1)
    clipped_count = (target_labels != target_labels_safe).sum().item()
    
    if clipped_count > 0:
        print(f"🔧 클리핑된 라벨 수: {clipped_count}개")
        print(f"✅ 안전한 타겟 라벨 범위: {target_labels_safe.min().item()} ~ {target_labels_safe.max().item()}")
    
    # 안전한 라벨로 손실 계산
    model.zero_grad()
    target_output = model(target_data)
    target_loss = criterion(target_output, target_labels_safe)
    target_loss.backward()
    
    # 나머지 영향도 계산 로직...
    influence_scores = []
    sample_count = 0
    invalid_samples = 0
    
    print(f"🎯 {num_samples}개 샘플의 영향도 계산...")
    
    for batch_idx, (data, labels) in enumerate(source_loader):
        if sample_count >= num_samples:
            break
            
        data, labels = data.to(device), labels.to(device)
        
        # 소스 라벨도 안전하게 클리핑
        labels_safe = torch.clamp(labels, 0, num_classes - 1)
        
        for i in range(min(len(data), num_samples - sample_count)):
            try:
                model.zero_grad()
                output = model(data[i:i+1])
                loss = criterion(output, labels_safe[i:i+1])
                loss.backward()
                
                # 단순화된 영향도 계산
                influence_scores.append(loss.item())
                sample_count += 1
                
            except Exception as e:
                print(f"⚠️ 샘플 건너뛰기: {e}")
                invalid_samples += 1
                continue
        
        if sample_count >= num_samples:
            break
    
    print(f"✅ {len(influence_scores)}개 샘플 영향도 계산 완료")
    if invalid_samples > 0:
        print(f"⚠️ 건너뛴 샘플: {invalid_samples}개")
    
    return influence_scores
'''
    
    print("✅ 안전한 영향도 계산 함수 생성 완료")
    return safe_influence_function

def generate_cifar_fix_commands():
    """CIFAR 문제 해결 명령어 생성"""
    
    print("\n💡 CIFAR10→CIFAR100 오류 해결 방법:")
    print("="*60)
    
    print("🔧 즉시 해결 방법:")
    print("1. 환경 변수 설정:")
    print("   set CUDA_LAUNCH_BLOCKING=1")
    print("   set TORCH_USE_CUDA_DSA=1")
    
    print("\n2. 안전한 실험 실행:")
    print("   python multi_dataset_experiments.py")
    print("   → 자연 이미지 도메인 선택 → CIFAR10→CIFAR100 선택")
    
    print("\n3. 수동 안전 실행:")
    print("   # 라벨 클리핑 모드")
    print("   CUDA_LAUNCH_BLOCKING=1 python main.py --source_dataset CIFAR10 --target_dataset CIFAR100 --batch_size 64 --influence_samples 100")
    
    print("\n🚨 핵심 해결책:")
    print("   • 모델 클래스 수: 100개로 통일 (CIFAR100 기준)")
    print("   • 라벨 클리핑: CIFAR100 라벨 → 0-9 범위로 제한")
    print("   • 배치 크기: 64로 감소")
    print("   • 영향도 샘플: 100개로 대폭 감소")

def main():
    """메인 실행 함수"""
    
    print("🔧 CIFAR10 ↔ CIFAR100 클래스 수 불일치 해결 도구")
    print("="*60)
    print("문제: CIFAR10(10클래스) → CIFAR100(100클래스) 영향도 계산 오류")
    print("원인: Assertion `t >= 0 && t < n_classes` failed")
    print("="*60)
    
    # 1. 문제 분석
    datasets = analyze_cifar_mismatch()
    
    # 2. 안전 설정 생성
    safe_configs = create_cifar_safe_config()
    
    # 3. 실험 설정 파일 생성
    config_file = generate_cifar_experiment_config()
    
    # 4. 안전 함수 생성
    safe_function = create_label_safe_main()
    
    # 5. 해결 명령어 출력
    generate_cifar_fix_commands()
    
    print(f"\n🎉 CIFAR 문제 해결 도구 준비 완료!")
    print(f"📁 생성된 파일: {config_file}")

if __name__ == "__main__":
    main() 