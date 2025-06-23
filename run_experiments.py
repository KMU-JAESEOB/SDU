# run_experiments.py - SDA-U 실험 실행 스크립트

"""
🧪 다양한 설정으로 SDA-U 실험을 실행하고 결과를 비교합니다.
"""

import os
import json
import subprocess
import time
from datetime import datetime
from experiment_configs import ALL_EXPERIMENTS, print_all_experiments

def run_single_experiment(experiment_config):
    """단일 실험을 실행합니다."""
    
    print(f"\n🧪 실험 시작: {experiment_config['name']}")
    print(f"📊 설정: {experiment_config['description']}")
    print(f"🔧 파라미터:")
    print(f"   - 언러닝 샘플: {experiment_config['max_unlearn_samples']}개")
    print(f"   - 타겟 서브셋: {experiment_config['target_subset_size']}개") 
    print(f"   - 적응 에포크: {experiment_config['adaptation_epochs']}회")
    print("=" * 60)
    
    # config.py 백업
    if os.path.exists('config.py.backup'):
        os.remove('config.py.backup')
    os.rename('config.py', 'config.py.backup')
    
    try:
        # 실험용 config.py 생성
        print("📝 실험용 config.py 생성 중...")
        create_experiment_config(experiment_config)
        print("✅ config.py 생성 완료")
        
        # 실험 실행 (실시간 출력)
        start_time = time.time()
        print("🚀 실험 실행 중... (실시간 출력)")
        print("-" * 60)
        
        # 실시간 출력을 위한 subprocess 실행
        process = subprocess.Popen(['python', 'main.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True, 
                                 bufsize=1, 
                                 universal_newlines=True)
        
        # 실시간 출력 읽기
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # 실시간 출력
                output_lines.append(output.strip())
        
        return_code = process.poll()
        end_time = time.time()
        
        # 실행 시간 계산
        execution_time = end_time - start_time
        
        # 결과 처리
        
        if return_code == 0:
            print("-" * 60)
            print(f"✅ 실험 완료! (실행시간: {execution_time:.1f}초)")
            
            # 결과 파일 이름 변경
            if os.path.exists('results/sda_u_comprehensive_results.json'):
                result_filename = f"results/experiment_{experiment_config['name']}.json"
                os.rename('results/sda_u_comprehensive_results.json', result_filename)
                
                # 실행 시간 추가
                with open(result_filename, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                result_data['experiment_info']['execution_time_seconds'] = execution_time
                
                with open(result_filename, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                print(f"📋 결과 저장: {result_filename}")
                return True, result_filename
            else:
                print("⚠️ 결과 파일을 찾을 수 없습니다.")
                return False, None
        else:
            print("-" * 60)
            print(f"❌ 실험 실패! (반환코드: {return_code})")
            print("마지막 출력:")
            for line in output_lines[-10:]:  # 마지막 10줄만 표시
                print(f"  {line}")
            return False, None
            
    except Exception as e:
        print(f"❌ 실험 중 오류 발생: {e}")
        return False, None
        
    finally:
        # config.py 복원
        if os.path.exists('config.py.backup'):
            if os.path.exists('config.py'):
                os.remove('config.py')
            os.rename('config.py.backup', 'config.py')

def create_experiment_config(experiment_config):
    """실험용 config.py를 생성합니다."""
    
    config_content = f'''# config.py - 실험용 설정 ({experiment_config['name']})

# ============================================
# 🧪 실험 설정: {experiment_config['name']}
# ============================================
ARCHITECTURE = 'resnet50'
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3

# 실험 파라미터
TARGET_SUBSET_SIZE = {experiment_config['target_subset_size']}
NUM_UNLEARN_STEPS = 5
INFLUENCE_SAMPLES = 300
ADAPTATION_EPOCHS = {experiment_config['adaptation_epochs']}
MAX_UNLEARN_SAMPLES = {experiment_config['max_unlearn_samples']}

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
    
    # GPU 최적화
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A100" in gpu_name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    
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
        'lambda_u': LAMBDA_U,
        'beta': BETA,
        'save_models': SAVE_MODELS,
        'save_results': SAVE_RESULTS,
        'quick_test': QUICK_TEST,
        'source_dataset': 'Office31_Amazon',  # Office-31 기본 설정
        'target_dataset': 'Office31_Webcam',
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }}
'''
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)

def compare_results(result_files):
    """실험 결과들을 비교합니다."""
    
    if not result_files:
        print("⚠️ 비교할 결과가 없습니다.")
        return
    
    print("\n📊 실험 결과 비교")
    print("=" * 80)
    
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"⚠️ {file_path} 로드 실패: {e}")
    
    if not results:
        return
    
    # 결과 테이블 출력
    print(f"{'실험명':<20} {'언러닝샘플':<10} {'타겟서브셋':<10} {'적응에포크':<10} {'최종정확도':<10} {'실행시간':<10}")
    print("-" * 80)
    
    for result in results:
        exp_info = result.get('experiment_info', {})
        exp_settings = exp_info.get('experiment_settings', {})
        final_perf = result.get('final_performance', {})
        
        name = exp_settings.get('max_unlearn_samples', 'N/A')
        unlearn_samples = exp_settings.get('max_unlearn_samples', 'N/A')
        target_subset = exp_settings.get('target_subset_size', 'N/A')
        adaptation_epochs = exp_settings.get('adaptation_epochs', 'N/A')
        final_acc = final_perf.get('target_subset_accuracy', 0)
        exec_time = exp_info.get('execution_time_seconds', 0)
        
        print(f"{name:<20} {unlearn_samples:<10} {target_subset:<10} {adaptation_epochs:<10} {final_acc:<10.2f} {exec_time:<10.1f}s")

def main():
    """메인 실험 실행 함수"""
    
    print("🧪 SDA-U 실험 실행기")
    print("=" * 60)
    
    # 결과 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    
    # 실험 목록 출력
    print_all_experiments()
    
    print("\n실행할 실험을 선택하세요:")
    print("1. 모든 실험 실행")
    print("2. 언러닝 샘플 수 비교 실험만")
    print("3. 특정 실험 선택")
    
    choice = input("\n선택 (1-3): ").strip()
    
    result_files = []
    
    if choice == '1':
        # 모든 실험 실행
        print(f"\n🚀 총 {len(ALL_EXPERIMENTS)}개 실험을 순차적으로 실행합니다...")
        for i, exp in enumerate(ALL_EXPERIMENTS, 1):
            print(f"\n[{i}/{len(ALL_EXPERIMENTS)}] 실험 진행 중...")
            success, result_file = run_single_experiment(exp)
            if success and result_file:
                result_files.append(result_file)
                
    elif choice == '2':
        # 언러닝 샘플 수 비교 실험만
        unlearning_experiments = [exp for exp in ALL_EXPERIMENTS 
                                if 'Unlearning' in exp['name']]
        print(f"\n🚀 언러닝 비교 실험 {len(unlearning_experiments)}개를 실행합니다...")
        for i, exp in enumerate(unlearning_experiments, 1):
            print(f"\n[{i}/{len(unlearning_experiments)}] 실험 진행 중...")
            success, result_file = run_single_experiment(exp)
            if success and result_file:
                result_files.append(result_file)
                
    elif choice == '3':
        # 특정 실험 선택
        print("\n실험을 선택하세요:")
        for i, exp in enumerate(ALL_EXPERIMENTS, 1):
            print(f"{i}. {exp['name']} - {exp['description']}")
        
        try:
            exp_idx = int(input(f"\n선택 (1-{len(ALL_EXPERIMENTS)}): ")) - 1
            if 0 <= exp_idx < len(ALL_EXPERIMENTS):
                success, result_file = run_single_experiment(ALL_EXPERIMENTS[exp_idx])
                if success and result_file:
                    result_files.append(result_file)
            else:
                print("❌ 잘못된 선택입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    # 결과 비교
    if result_files:
        compare_results(result_files)
        print(f"\n🎉 실험 완료! 총 {len(result_files)}개 결과가 저장되었습니다.")
    else:
        print("\n⚠️ 완료된 실험이 없습니다.")

if __name__ == "__main__":
    main() 