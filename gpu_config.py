# gpu_config.py
# A100 80GB 환경 최적화 설정

import torch

class A100Config:
    """A100 80GB GPU 최적화 설정"""
    
    # 기본 설정
    BATCH_SIZE = 512  # A100에서 더 큰 배치 크기 사용 가능
    NUM_EPOCHS = 25   # 큰 모델에서는 더 적은 에포크로도 충분
    LEARNING_RATE = 1e-4  # 사전 훈련된 모델에는 낮은 학습률
    
    # 데이터 설정
    TARGET_SUBSET_SIZE = 5000  # 더 많은 타겟 데이터 사용
    NUM_UNLEARN_STEPS = 20     # 더 많은 언러닝 스텝
    
    # 추천 아키텍처 (A100 80GB)
    ARCHITECTURES = {
        'high_performance': 'vit_l_16',      # 최고 성능 (Vision Transformer Large)
        'balanced': 'resnet152',             # 균형잡힌 성능
        'fast_training': 'resnet101',        # 빠른 훈련
        'memory_efficient': 'resnet50',      # 메모리 효율적
        'lightweight': 'efficientnet_b3'    # 경량 모델
    }
    
    # 사전 훈련 가중치 사용 권장
    USE_PRETRAINED = True
    
    # A100 특화 최적화
    ENABLE_TENSOR_CORE = True      # Tensor Core 활성화
    ENABLE_MIXED_PRECISION = True  # Mixed Precision 훈련
    ENABLE_BENCHMARK = True        # CUDNN 벤치마크 모드
    
    @staticmethod
    def apply_optimizations():
        """A100 최적화 설정 적용"""
        if torch.cuda.is_available():
            # Tensor Core 최적화
            if A100Config.ENABLE_TENSOR_CORE:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("🚀 A100 Tensor Core (TF32) 활성화!")
            
            # CUDNN 벤치마크
            if A100Config.ENABLE_BENCHMARK:
                torch.backends.cudnn.benchmark = True
                print("⚡ CUDNN 벤치마크 모드 활성화!")
            
            print("✅ A100 최적화 설정 적용 완료!")
    
    @staticmethod
    def get_model_config(performance_level: str = 'balanced'):
        """성능 레벨에 따른 모델 설정 반환"""
        
        configs = {
            'high_performance': {
                'architecture': 'vit_l_16',
                'batch_size': 256,  # ViT는 상대적으로 큰 배치 크기
                'num_epochs': 15,   # 큰 모델은 빠르게 수렴
                'learning_rate': 5e-5
            },
            'balanced': {
                'architecture': 'resnet152',
                'batch_size': 384,
                'num_epochs': 20,
                'learning_rate': 1e-4
            },
            'fast_training': {
                'architecture': 'resnet101',
                'batch_size': 512,
                'num_epochs': 25,
                'learning_rate': 1e-4
            },
            'memory_efficient': {
                'architecture': 'resnet50',
                'batch_size': 768,
                'num_epochs': 30,
                'learning_rate': 2e-4
            },
            'lightweight': {
                'architecture': 'efficientnet_b3',
                'batch_size': 1024,
                'num_epochs': 35,
                'learning_rate': 3e-4
            }
        }
        
        if performance_level not in configs:
            print(f"⚠️ 알 수 없는 성능 레벨: {performance_level}")
            print(f"사용 가능한 옵션: {list(configs.keys())}")
            performance_level = 'balanced'
        
        config = configs[performance_level]
        print(f"🎯 선택된 성능 레벨: {performance_level}")
        print(f"🏗️ 아키텍처: {config['architecture']}")
        print(f"📦 배치 크기: {config['batch_size']}")
        print(f"🔄 에포크 수: {config['num_epochs']}")
        print(f"📚 학습률: {config['learning_rate']}")
        
        return config

class GeneralConfig:
    """일반 GPU 환경 설정"""
    
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    TARGET_SUBSET_SIZE = 2000
    NUM_UNLEARN_STEPS = 10
    
    ARCHITECTURES = {
        'recommended': 'resnet50',
        'lightweight': 'resnet18',
        'custom': 'custom_cnn'
    }

def get_optimal_config():
    """현재 GPU 환경에 최적화된 설정 반환"""
    
    if not torch.cuda.is_available():
        print("💻 CPU 환경 감지: 기본 설정 사용")
        return {
            'architecture': 'custom_cnn',
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-3,
            'target_subset_size': 1000,
            'num_unlearn_steps': 5
        }
    
    # GPU 정보 확인
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"🔍 GPU 감지: {gpu_name}")
    print(f"💾 GPU 메모리: {gpu_memory_gb:.1f}GB")
    
    if "A100" in gpu_name and gpu_memory_gb >= 70:
        print("🔥 A100 80GB 환경 감지!")
        A100Config.apply_optimizations()
        return A100Config.get_model_config('balanced')  # 기본값
    
    elif gpu_memory_gb >= 40:
        print("🚀 고성능 GPU 환경 (40GB+)")
        return {
            'architecture': 'resnet101',
            'batch_size': 256,
            'num_epochs': 25,
            'learning_rate': 1e-4,
            'target_subset_size': 3000,
            'num_unlearn_steps': 15
        }
    
    elif gpu_memory_gb >= 20:
        print("⚡ 중고성능 GPU 환경 (20GB+)")
        return {
            'architecture': 'resnet50',
            'batch_size': 128,
            'num_epochs': 30,
            'learning_rate': 1e-4,
            'target_subset_size': 2500,
            'num_unlearn_steps': 12
        }
    
    else:
        print("💻 일반 GPU 환경")
        return {
            'architecture': 'resnet18',
            'batch_size': 64,
            'num_epochs': 40,
            'learning_rate': 2e-4,
            'target_subset_size': 2000,
            'num_unlearn_steps': 10
        }

if __name__ == "__main__":
    # 설정 테스트
    config = get_optimal_config()
    print("\n📋 최적화된 설정:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # A100 특화 설정들 출력
    if torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0):
        print("\n🎯 A100 성능 레벨별 설정:")
        for level in ['high_performance', 'balanced', 'fast_training', 'memory_efficient', 'lightweight']:
            print(f"\n🔸 {level}:")
            A100Config.get_model_config(level) 