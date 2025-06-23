# config.py - SDA-U 프레임워크 설정 파일

# ============================================
# 🎯 아키텍처 선택 (원하는 것으로 변경하세요!)
# ============================================
# 옵션:
# - 'resnet18': ResNet-18 (빠름, 11M 파라미터)
# - 'resnet50': ResNet-50 (균형, 25M 파라미터) 
# - 'vit_b_16': Vision Transformer Base (고성능, 86M 파라미터)
# - 'vit_l_16': Vision Transformer Large (최고성능, 304M 파라미터, GPU 메모리 많이 필요)
# - 'custom_cnn': 커스텀 CNN (가벼움, 0.1M 파라미터)

ARCHITECTURE = 'resnet50'  # 👈 여기를 변경하세요!

# ============================================
# 📊 데이터셋 선택 (새로 추가!)
# ============================================
# 소스 → 타겟 도메인 적응 쌍 선택
# 사용 가능한 데이터셋: MNIST, SVHN, CIFAR10, CIFAR100, FashionMNIST, KMNIST, EMNIST
# Office-31: Office31_Amazon, Office31_Webcam, Office31_DSLR

SOURCE_DATASET = 'Office31_Amazon'     # 소스 도메인 (고품질 제품 이미지)
TARGET_DATASET = 'Office31_Webcam'    # 타겟 도메인 (저품질 웹캠 이미지)

# 추천 조합들:
# - SVHN → MNIST: 컬러 거리번호 → 흑백 손글씨 (기본)
# - CIFAR10 → FashionMNIST: 일반사물 → 패션아이템
# - MNIST → KMNIST: 아라비아 숫자 → 일본 문자
# - CIFAR10 → CIFAR100: 10클래스 → 100클래스
# - Office31_Amazon → Office31_Webcam: 고품질 → 저품질 (도메인 적응 벤치마크!)
# - Office31_Amazon → Office31_DSLR: 인공적 → 자연스러운

# ============================================
# 🔧 훈련 설정
# ============================================
BATCH_SIZE = 64           # 배치 크기 (GPU 메모리에 따라 조정)
NUM_EPOCHS = 3            # 훈련 에포크 수
LEARNING_RATE = 1e-3      # 학습률

# ============================================
# 🎯 SDA-U 알고리즘 설정
# ============================================
TARGET_SUBSET_SIZE = 1000    # 타겟 서브셋 크기
NUM_UNLEARN_STEPS = 5        # 언러닝 스텝 수
INFLUENCE_SAMPLES = 300      # 영향도 계산 샘플 수
ADAPTATION_EPOCHS = 100      # 타겟 도메인 적응 훈련 에포크 수 (충분한 적응을 위해)
MAX_UNLEARN_SAMPLES = 200    # 언러닝할 최대 샘플 수 (실험용으로 조정 가능)

# 하이브리드 스코어링 파라미터
LAMBDA_U = 0.6              # 영향도 가중치
BETA = 0.1                  # 불확실성 가중치

# ============================================
# 💾 저장 설정
# ============================================
SAVE_MODELS = True          # 모델 저장 여부
SAVE_RESULTS = True         # 결과 저장 여부

# ============================================
# 🚀 실행 모드 설정
# ============================================
QUICK_TEST = False          # 빠른 테스트 모드 (True: 30배치만, False: 전체 데이터)

# ============================================
# 🚀 GPU 최적화 설정
# ============================================
def get_config():
    """설정을 반환하는 함수"""
    import torch
    
    # GPU 감지 및 최적화
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # A100 최적화
        if "A100" in gpu_name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("🚀 A100 최적화 활성화!")
            
            # A100용 고성능 설정
            if gpu_memory >= 70:
                return {
                    'architecture': 'vit_l_16',  # A100에서는 ViT Large 추천
                    'batch_size': 128,
                    'num_epochs': 5,
                    'target_subset_size': 2000,
                    'num_unlearn_steps': 10,
                    'source_dataset': SOURCE_DATASET,
                    'target_dataset': TARGET_DATASET,
                    'gpu_name': gpu_name,
                    'gpu_memory': f"{gpu_memory:.1f}GB"
                }
    
    # 기본 설정 반환
    return {
        'architecture': ARCHITECTURE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'target_subset_size': TARGET_SUBSET_SIZE,
        'num_unlearn_steps': NUM_UNLEARN_STEPS,
        'influence_samples': INFLUENCE_SAMPLES,
        'adaptation_epochs': ADAPTATION_EPOCHS,
        'max_unlearn_samples': MAX_UNLEARN_SAMPLES,
        'lambda_u': LAMBDA_U,
        'beta': BETA,
        'save_models': SAVE_MODELS,
        'save_results': SAVE_RESULTS,
        'quick_test': QUICK_TEST,
        'source_dataset': SOURCE_DATASET,
        'target_dataset': TARGET_DATASET,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }

if __name__ == "__main__":
    # 설정 테스트
    config = get_config()
    print("🔧 현재 설정:")
    for key, value in config.items():
        print(f"   {key}: {value}")
        
    # 데이터셋 정보 표시
    from dataset_manager import DatasetManager
    manager = DatasetManager()
    
    print(f"\n📊 선택된 도메인 적응:")
    source_info = manager.get_dataset_info(SOURCE_DATASET)
    target_info = manager.get_dataset_info(TARGET_DATASET)
    
    if source_info and target_info:
        print(f"   📤 소스: {SOURCE_DATASET} - {source_info['description']}")
        print(f"   📥 타겟: {TARGET_DATASET} - {target_info['description']}")
    else:
        print(f"   ⚠️ 지원하지 않는 데이터셋입니다.") 