# multi_dataset_experiments.py
"""
🎯 다중 데이터셋 SDA-U 실험 시스템
- SVHN ↔ MNIST
- CIFAR-10 ↔ STL-10  
- CIFAR-10 ↔ CIFAR-100
- Fashion-MNIST ↔ MNIST
등 다양한 도메인 적응 실험 지원
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import torch
import torchvision

class MultiDatasetExperiments:
    """다중 데이터셋 SDA-U 실험 관리자"""
    
    def __init__(self):
        self.results_dir = Path('multi_dataset_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # 🎯 지원 데이터셋 조합 정의
        self.dataset_combinations = {
            # 🔢 숫자 인식 도메인
            'digit_recognition': [
                ('SVHN', 'MNIST', 'SVHN→MNIST: 자연환경→손글씨'),
                ('MNIST', 'SVHN', 'MNIST→SVHN: 손글씨→자연환경'),
                ('MNIST', 'FashionMNIST', 'MNIST→Fashion: 숫자→패션'),
                ('FashionMNIST', 'MNIST', 'Fashion→MNIST: 패션→숫자'),
            ],
            
            # 🖼️ 자연 이미지 도메인
            'natural_images': [
                ('CIFAR10', 'STL10', 'CIFAR-10→STL-10: 저해상도→고해상도'),
                ('STL10', 'CIFAR10', 'STL-10→CIFAR-10: 고해상도→저해상도'),
                ('CIFAR10', 'CIFAR100', 'CIFAR-10→CIFAR-100: 10클래스→100클래스'),
            ],
            
            # 🔄 교차 도메인
            'cross_domain': [
                ('SVHN', 'FashionMNIST', 'SVHN→Fashion: 숫자→패션'),
                ('CIFAR10', 'MNIST', 'CIFAR-10→MNIST: 자연이미지→숫자'),
            ]
        }
        
        print("🎯 다중 데이터셋 SDA-U 실험 시스템 초기화 완료!")
        print(f"📁 결과 저장 위치: {self.results_dir}")
    
    def create_dataset_config(self, source_dataset, target_dataset, experiment_name):
        """특정 데이터셋 조합을 위한 config.py 생성"""
        
        # 🎯 데이터셋별 최적 설정
        dataset_settings = {
            'SVHN': {
                'image_size': 32,
                'channels': 3,
                'num_classes': 10,
                'batch_size': 128,
                'epochs': 20,
                'learning_rate': 1e-3,
                'architecture': 'resnet18'
            },
            'MNIST': {
                'image_size': 28,
                'channels': 1,
                'num_classes': 10,
                'batch_size': 256,
                'epochs': 15,
                'learning_rate': 1e-3,
                'architecture': 'custom_cnn'
            },
            'FashionMNIST': {
                'image_size': 28,
                'channels': 1,
                'num_classes': 10,
                'batch_size': 256,
                'epochs': 15,
                'learning_rate': 1e-3,
                'architecture': 'custom_cnn'
            },
            'CIFAR10': {
                'image_size': 32,
                'channels': 3,
                'num_classes': 10,
                'batch_size': 128,
                'epochs': 25,
                'learning_rate': 1e-3,
                'architecture': 'resnet18'
            },
            'STL10': {
                'image_size': 96,
                'channels': 3,
                'num_classes': 10,
                'batch_size': 64,
                'epochs': 30,
                'learning_rate': 5e-4,
                'architecture': 'resnet18'
            },
            'CIFAR100': {
                'image_size': 32,
                'channels': 3,
                'num_classes': 100,
                'batch_size': 128,
                'epochs': 30,
                'learning_rate': 1e-3,
                'architecture': 'resnet18'
            }
        }
        
        source_settings = dataset_settings.get(source_dataset, dataset_settings['CIFAR10'])
        target_settings = dataset_settings.get(target_dataset, dataset_settings['CIFAR10'])
        
        # 🔧 하이브리드 설정 (소스와 타겟의 중간값)
        hybrid_batch_size = min(source_settings['batch_size'], target_settings['batch_size'])
        hybrid_epochs = max(source_settings['epochs'], target_settings['epochs'])
        hybrid_lr = (source_settings['learning_rate'] + target_settings['learning_rate']) / 2
        
        # 🏗️ 아키텍처 선택 (더 복잡한 데이터셋 기준)
        if source_settings['channels'] == 3 or target_settings['channels'] == 3:
            architecture = 'resnet18'
        else:
            architecture = 'custom_cnn'
        
        config_content = f'''# config.py - {experiment_name} 실험용 설정

# ============================================
# 🎯 다중 데이터셋 실험: {experiment_name}
# ============================================
ARCHITECTURE = '{architecture}'
BATCH_SIZE = {hybrid_batch_size}
NUM_EPOCHS = {hybrid_epochs}
LEARNING_RATE = {hybrid_lr}

# 데이터셋 설정
SOURCE_DATASET = '{source_dataset}'
TARGET_DATASET = '{target_dataset}'

# SDA-U 알고리즘 설정 (데이터셋 최적화)
TARGET_SUBSET_SIZE = 500     # 타겟 샘플 수
NUM_UNLEARN_STEPS = 6        # 언러닝 스텝
INFLUENCE_SAMPLES = 200      # 영향도 계산 샘플
ADAPTATION_EPOCHS = {target_settings['epochs'] // 2}  # 적응 훈련 에포크
MAX_UNLEARN_SAMPLES = 100    # 최대 언러닝 샘플

# 🔄 다중 라운드 SDA-U 설정
SDA_U_ROUNDS = 1
PROGRESSIVE_UNLEARNING = True
DYNAMIC_THRESHOLD = True

# 🎯 사전 훈련 가중치 설정
USE_PRETRAINED = {'True' if architecture == 'resnet18' else 'False'}
FREEZE_BACKBONE = False

# 🔧 훈련 설정
SCHEDULER_TYPE = 'cosine'
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

# 하이브리드 스코어링 파라미터
LAMBDA_U = 0.6
BETA = 0.1

# 저장 설정
SAVE_MODELS = True
SAVE_RESULTS = True
QUICK_TEST = False

def get_config():
    """설정을 반환하는 함수"""
    import torch
    
    return {{
        'architecture': ARCHITECTURE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'target_subset_size': TARGET_SUBSET_SIZE,
        'num_unlearn_steps': NUM_UNLEARN_STEPS,
        'influence_samples': INFLUENCE_SAMPLES,
        'adaptation_epochs': ADAPTATION_EPOCHS,
        'max_unlearn_samples': MAX_UNLEARN_SAMPLES,
        'sda_u_rounds': SDA_U_ROUNDS,
        'progressive_unlearning': PROGRESSIVE_UNLEARNING,
        'dynamic_threshold': DYNAMIC_THRESHOLD,
        'use_pretrained': USE_PRETRAINED,
        'freeze_backbone': FREEZE_BACKBONE,
        'scheduler_type': SCHEDULER_TYPE,
        'warmup_epochs': WARMUP_EPOCHS,
        'weight_decay': WEIGHT_DECAY,
        'gradient_clip': GRADIENT_CLIP,
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
        
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
    
    def run_single_experiment(self, source_dataset, target_dataset, description):
        """단일 데이터셋 조합 실험 실행"""
        
        experiment_name = f"{source_dataset}2{target_dataset}"
        
        print(f"\n{'='*80}")
        print(f"🧪 실험 시작: {experiment_name}")
        print(f"📊 {description}")
        print(f"📤 소스: {source_dataset}")
        print(f"📥 타겟: {target_dataset}")
        print(f"{'='*80}")
        
        # 🗂️ 실험별 디렉토리 생성
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        models_dir = Path('models') / experiment_name
        models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 실험 디렉토리: {experiment_dir}")
        print(f"📁 모델 디렉토리: {models_dir}")
        
        # config.py 백업
        if os.path.exists('config.py.backup'):
            os.remove('config.py.backup')
        if os.path.exists('config.py'):
            os.rename('config.py', 'config.py.backup')
        
        try:
            # 실험용 config.py 생성
            self.create_dataset_config(source_dataset, target_dataset, experiment_name)
            
            # 실험 실행
            start_time = time.time()
            print("🚀 SDA-U 실험 실행 중...")
            
            process = subprocess.Popen(['python', 'main.py'], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=1)
            
            # 실시간 출력 및 로그 저장
            output_lines = []
            log_file = experiment_dir / f"{experiment_name}_execution_log.txt"
            
            with open(log_file, 'w', encoding='utf-8') as log:
                while True:
                    if process.stdout is None:
                        break
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
                        log.write(output)
                        log.flush()
                        output_lines.append(output.strip())
            
            return_code = process.poll()
            end_time = time.time()
            execution_time = end_time - start_time
            
            if return_code == 0:
                print(f"✅ 실험 완료! (실행시간: {execution_time:.1f}초)")
                
                # 🗂️ 결과 파일 관리
                result_file = 'results/sda_u_comprehensive_results.json'
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    # 실험 메타데이터 추가
                    result_data['experiment_info']['experiment_name'] = experiment_name
                    result_data['experiment_info']['source_dataset'] = source_dataset
                    result_data['experiment_info']['target_dataset'] = target_dataset
                    result_data['experiment_info']['description'] = description
                    result_data['experiment_info']['execution_time_seconds'] = execution_time
                    
                    # 결과 저장
                    main_result_file = experiment_dir / f"{experiment_name}_results.json"
                    with open(main_result_file, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"📋 결과 저장: {main_result_file}")
                    return True
                else:
                    print("⚠️ 결과 파일을 찾을 수 없습니다.")
                    return False
            else:
                print(f"❌ 실험 실패! (종료 코드: {return_code})")
                return False
                
        except Exception as e:
            print(f"❌ 실험 중 오류 발생: {str(e)}")
            return False
        finally:
            # config.py 복원
            if os.path.exists('config.py.backup'):
                if os.path.exists('config.py'):
                    os.remove('config.py')
                os.rename('config.py.backup', 'config.py')
    
    def run_category_experiments(self, category):
        """특정 카테고리의 모든 실험 실행"""
        
        if category not in self.dataset_combinations:
            print(f"❌ 지원하지 않는 카테고리: {category}")
            return
        
        experiments = self.dataset_combinations[category]
        total_experiments = len(experiments)
        successful_experiments = 0
        
        print(f"\n🎯 {category.upper()} 카테고리 실험 시작!")
        print(f"📊 총 {total_experiments}개 실험 예정")
        
        start_time = time.time()
        
        for i, (source, target, description) in enumerate(experiments, 1):
            print(f"\n🔢 진행상황: {i}/{total_experiments}")
            
            if self.run_single_experiment(source, target, description):
                successful_experiments += 1
                print(f"✅ {i}번째 실험 성공!")
            else:
                print(f"❌ {i}번째 실험 실패!")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"🎉 {category.upper()} 카테고리 실험 완료!")
        print(f"✅ 성공한 실험: {successful_experiments}/{total_experiments}")
        print(f"⏱️ 총 소요시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"{'='*80}")
        
        # 성능 요약 테이블 생성
        self.create_performance_table(category)
    
    def create_performance_table(self, category):
        """카테고리별 성능 요약 테이블 생성"""
        
        print(f"\n📊 {category.upper()} 성능 요약 테이블 생성 중...")
        
        performance_data = []
        
        for source, target, description in self.dataset_combinations[category]:
            experiment_name = f"{source}2{target}"
            result_file = self.results_dir / experiment_name / f"{experiment_name}_results.json"
            
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    performance_data.append({
                        'experiment': experiment_name,
                        'source': source,
                        'target': target,
                        'description': description,
                        'final_target_accuracy': data.get('final_results', {}).get('target_accuracy', 0),
                        'best_target_accuracy': data.get('training_progress', {}).get('best_target_accuracy', 0),
                        'execution_time': data.get('experiment_info', {}).get('execution_time_seconds', 0)
                    })
                except Exception as e:
                    print(f"⚠️ {experiment_name} 결과 로딩 실패: {e}")
        
        # CSV 형태로 저장
        csv_file = self.results_dir / f"{category}_performance_summary.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("실험명,소스,타겟,설명,최종정확도,최고정확도,실행시간(초)\n")
            for data in performance_data:
                f.write(f"{data['experiment']},{data['source']},{data['target']},"
                       f"{data['description']},{data['final_target_accuracy']:.2f}%,"
                       f"{data['best_target_accuracy']:.2f}%,{data['execution_time']:.1f}\n")
        
        print(f"📊 성능 요약 테이블 저장: {csv_file}")
        
        # 콘솔 출력
        print(f"\n📈 {category.upper()} 성능 요약:")
        print("-" * 100)
        print(f"{'실험명':<20} {'소스':<12} {'타겟':<12} {'최종정확도':<12} {'최고정확도':<12} {'실행시간':<10}")
        print("-" * 100)
        for data in performance_data:
            print(f"{data['experiment']:<20} {data['source']:<12} {data['target']:<12} "
                  f"{data['final_target_accuracy']:>10.2f}% {data['best_target_accuracy']:>10.2f}% "
                  f"{data['execution_time']:>8.1f}s")
        print("-" * 100)

def main():
    """메인 실행 함수"""
    
    print("🎯 다중 데이터셋 SDA-U 실험 시스템")
    print("=" * 60)
    
    experiments = MultiDatasetExperiments()
    
    print("\n실행할 실험 카테고리를 선택하세요:")
    print("1. digit_recognition (숫자 인식 도메인)")
    print("2. natural_images (자연 이미지 도메인)")
    print("3. cross_domain (교차 도메인)")
    print("4. 개별 실험 선택")
    print("5. 전체 실험 실행")
    
    choice = input("\n선택 (1-5): ").strip()
    
    if choice == '1':
        experiments.run_category_experiments('digit_recognition')
    elif choice == '2':
        experiments.run_category_experiments('natural_images')
    elif choice == '3':
        experiments.run_category_experiments('cross_domain')
    elif choice == '4':
        # 개별 실험 선택
        print("\n📋 사용 가능한 실험:")
        all_experiments = []
        for category, exps in experiments.dataset_combinations.items():
            all_experiments.extend(exps)
        
        for i, (source, target, desc) in enumerate(all_experiments, 1):
            print(f"{i}. {source}→{target}: {desc}")
        
        try:
            exp_choice = int(input(f"\n실험 선택 (1-{len(all_experiments)}): ")) - 1
            if 0 <= exp_choice < len(all_experiments):
                source, target, desc = all_experiments[exp_choice]
                experiments.run_single_experiment(source, target, desc)
            else:
                print("❌ 잘못된 선택입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    elif choice == '5':
        # 전체 실험 실행
        print("\n⚠️ 전체 실험은 매우 오래 걸릴 수 있습니다.")
        confirm = input("계속하시겠습니까? (y/n): ").lower()
        if confirm == 'y':
            for category in experiments.dataset_combinations.keys():
                experiments.run_category_experiments(category)
        else:
            print("❌ 실험이 취소되었습니다.")
    
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 