# experiment_configs.py - SDA-U 실험 설정들

"""
🧪 다양한 언러닝 샘플 수로 실험하기 위한 설정들
사용법: config.py에서 원하는 실험 설정을 import해서 사용
"""

# ============================================
# 🧪 실험 1: 언러닝 샘플 수 비교
# ============================================

# 적은 언러닝 (기본)
EXPERIMENT_1_SMALL = {
    'name': 'Small_Unlearning',
    'max_unlearn_samples': 50,
    'adaptation_epochs': 100,
    'target_subset_size': 1000,
    'description': '적은 언러닝 샘플로 실험'
}

# 중간 언러닝
EXPERIMENT_1_MEDIUM = {
    'name': 'Medium_Unlearning', 
    'max_unlearn_samples': 200,
    'adaptation_epochs': 100,
    'target_subset_size': 1000,
    'description': '중간 언러닝 샘플로 실험'
}

# 많은 언러닝
EXPERIMENT_1_LARGE = {
    'name': 'Large_Unlearning',
    'max_unlearn_samples': 500,
    'adaptation_epochs': 100,
    'target_subset_size': 1000,
    'description': '많은 언러닝 샘플로 실험'
}

# 매우 많은 언러닝
EXPERIMENT_1_XLARGE = {
    'name': 'XLarge_Unlearning',
    'max_unlearn_samples': 1000,
    'adaptation_epochs': 100,
    'target_subset_size': 1000,
    'description': '매우 많은 언러닝 샘플로 실험'
}

# ============================================
# 🧪 실험 2: 타겟 서브셋 크기 비교
# ============================================

EXPERIMENT_2_SMALL_TARGET = {
    'name': 'Small_Target_Subset',
    'max_unlearn_samples': 200,
    'adaptation_epochs': 100,
    'target_subset_size': 500,
    'description': '작은 타겟 서브셋으로 실험'
}

EXPERIMENT_2_LARGE_TARGET = {
    'name': 'Large_Target_Subset',
    'max_unlearn_samples': 200,
    'adaptation_epochs': 100,
    'target_subset_size': 2000,
    'description': '큰 타겟 서브셋으로 실험'
}

# ============================================
# 🧪 실험 3: 적응 에포크 비교
# ============================================

EXPERIMENT_3_SHORT_ADAPTATION = {
    'name': 'Short_Adaptation',
    'max_unlearn_samples': 200,
    'adaptation_epochs': 50,
    'target_subset_size': 1000,
    'description': '짧은 적응 훈련으로 실험'
}

EXPERIMENT_3_LONG_ADAPTATION = {
    'name': 'Long_Adaptation',
    'max_unlearn_samples': 200,
    'adaptation_epochs': 200,
    'target_subset_size': 1000,
    'description': '긴 적응 훈련으로 실험'
}

# ============================================
# 🧪 모든 실험 설정 리스트
# ============================================

ALL_EXPERIMENTS = [
    EXPERIMENT_1_SMALL,
    EXPERIMENT_1_MEDIUM, 
    EXPERIMENT_1_LARGE,
    EXPERIMENT_1_XLARGE,
    EXPERIMENT_2_SMALL_TARGET,
    EXPERIMENT_2_LARGE_TARGET,
    EXPERIMENT_3_SHORT_ADAPTATION,
    EXPERIMENT_3_LONG_ADAPTATION
]

def get_experiment_config(experiment_name):
    """실험 이름으로 설정을 가져옵니다."""
    for exp in ALL_EXPERIMENTS:
        if exp['name'] == experiment_name:
            return exp
    return None

def print_all_experiments():
    """모든 실험 설정을 출력합니다."""
    print("🧪 사용 가능한 실험 설정들:")
    print("=" * 60)
    for i, exp in enumerate(ALL_EXPERIMENTS, 1):
        print(f"{i}. {exp['name']}")
        print(f"   📊 언러닝 샘플: {exp['max_unlearn_samples']}개")
        print(f"   🎯 타겟 서브셋: {exp['target_subset_size']}개")
        print(f"   🏋️ 적응 에포크: {exp['adaptation_epochs']}회")
        print(f"   📝 설명: {exp['description']}")
        print()

if __name__ == "__main__":
    print_all_experiments() 