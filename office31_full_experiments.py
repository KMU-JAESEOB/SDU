#!/usr/bin/env python3
# office31_full_experiments.py - Office-31 전체 도메인 조합 실험

"""
🏢 Office-31 전체 도메인 조합 SDA-U 실험

Office-31의 6가지 도메인 조합에 대해 SDA-U 실험을 수행하고
소스와 타겟 도메인 성능을 종합적으로 분석합니다.

실험 조합:
1. Amazon → Webcam (고품질 → 저품질)
2. Amazon → DSLR (인공적 → 자연스러운)  
3. Webcam → Amazon (저품질 → 고품질)
4. Webcam → DSLR (저품질 → 고품질)
5. DSLR → Amazon (자연스러운 → 인공적)
6. DSLR → Webcam (고품질 → 저품질)
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

class Office31FullExperiments:
    """Office-31 전체 도메인 조합 실험 관리자"""
    
    def __init__(self):
        self.results_dir = Path('office31_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Office-31 도메인 조합 정의
        self.domain_combinations = [
            ('Office31_Amazon', 'Office31_Webcam', 'Amazon→Webcam: 고품질→저품질'),
            ('Office31_Amazon', 'Office31_DSLR', 'Amazon→DSLR: 인공적→자연스러운'),
            ('Office31_Webcam', 'Office31_Amazon', 'Webcam→Amazon: 저품질→고품질'),
            ('Office31_Webcam', 'Office31_DSLR', 'Webcam→DSLR: 저품질→고품질'),
            ('Office31_DSLR', 'Office31_Amazon', 'DSLR→Amazon: 자연스러운→인공적'),
            ('Office31_DSLR', 'Office31_Webcam', 'DSLR→Webcam: 고품질→저품질')
        ]
        
        # 실험 결과 저장용
        self.all_results = []
        
    def create_experiment_config(self, source_dataset, target_dataset, experiment_name):
        """특정 도메인 조합을 위한 config.py 생성 (Office-31 호환성 강화)"""
        
        # 🎯 도메인별 적응 에포크 딕셔너리를 문자열로 안전하게 변환
        domain_epochs_dict = {
            'Amazon2Webcam': 10,     # 중간 난이도
            'Amazon2DSLR': 8,        # 상대적으로 쉬움
            'Webcam2Amazon': 12,     # 어려움 (작은→큰 데이터셋)
            'Webcam2DSLR': 8,        # 중간
            'DSLR2Amazon': 12,       # 어려움 (작은→큰 데이터셋)  
            'DSLR2Webcam': 10        # 중간 난이도
        }
        
        # 딕셔너리를 안전한 문자열로 변환
        domain_epochs_str = "{\n"
        for key, value in domain_epochs_dict.items():
            domain_epochs_str += f"    '{key}': {value},     # 도메인별 설정\n"
        domain_epochs_str += "}"
        
        config_content = f'''# config.py - Office-31 실험용 설정 ({experiment_name}) - 성능 최적화

# ============================================
# 🏢 Office-31 실험: {experiment_name} (고성능 설정)
# ============================================
ARCHITECTURE = 'resnet50'  # Office-31에 최적화된 아키텍처 (고정)
BATCH_SIZE = 64           # A100에 최적화된 배치 크기 (32→64)
NUM_EPOCHS = 15           # 충분한 학습을 위한 에포크 (5→15)
LEARNING_RATE = 2e-4      # 사전 훈련 모델에 적합한 학습률 (1e-4→2e-4)

# 데이터셋 설정
SOURCE_DATASET = '{source_dataset}'
TARGET_DATASET = '{target_dataset}'

# SDA-U 알고리즘 설정 (성능 최적화)
TARGET_SUBSET_SIZE = 600     # 더 많은 타겟 샘플 사용 (500→600)
NUM_UNLEARN_STEPS = 8        # 더 정교한 언러닝 (5→8)
INFLUENCE_SAMPLES = 300      # 더 많은 영향도 샘플 (200→300)
ADAPTATION_EPOCHS = 10       # 적응 훈련 에포크 (8→10, 도메인별 조정 가능)
MAX_UNLEARN_SAMPLES = 150    # 더 많은 언러닝 샘플 (100→150)

# 🔄 다중 라운드 SDA-U 설정
SDA_U_ROUNDS = 1            # SDA-U 라운드 수
PROGRESSIVE_UNLEARNING = True  # 점진적 언러닝 여부
DYNAMIC_THRESHOLD = True    # 동적 임계값 조정 여부

# 🎯 사전 훈련 가중치 설정 (새로 추가!)
USE_PRETRAINED = True       # ImageNet 사전 훈련 가중치 사용
FREEZE_BACKBONE = False     # 백본 고정 여부 (False=fine-tuning)

# 🔧 고급 훈련 설정 (새로 추가!)
SCHEDULER_TYPE = 'cosine'   # 학습률 스케줄러 ('cosine', 'step', 'none')
WARMUP_EPOCHS = 2          # 워밍업 에포크
WEIGHT_DECAY = 1e-4        # 가중치 감쇠
GRADIENT_CLIP = 1.0        # 그래디언트 클리핑

# 하이브리드 스코어링 파라미터
LAMBDA_U = 0.6
BETA = 0.1

# 저장 설정
SAVE_MODELS = True
SAVE_RESULTS = True
QUICK_TEST = False  # 전체 데이터셋으로 실제 성능 측정

# 🎯 도메인별 적응 에포크 설정 (새로 추가!)
DOMAIN_SPECIFIC_EPOCHS = {domain_epochs_str}

def get_config():
    """설정을 반환하는 함수 (Office-31 고성능 최적화)"""
    import torch
    
    # GPU 최적화 (A100에서도 ResNet50 강제 사용)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A100" in gpu_name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("🚀 A100 최적화 활성화! (고성능 설정)")
    
    # 🚨 중요: 고성능 설정으로 업그레이드
    return {{
        'architecture': 'resnet50',
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
        'domain_specific_epochs': DOMAIN_SPECIFIC_EPOCHS,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }}
'''
        
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
            
    def run_single_experiment(self, source_dataset, target_dataset, description):
        """단일 도메인 조합 실험 실행 (체계적 파일 관리)"""
        
        experiment_name = f"{source_dataset.split('_')[1]}2{target_dataset.split('_')[1]}"
        
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
            self.create_experiment_config(source_dataset, target_dataset, experiment_name)
            
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
                
                # 🗂️ 결과 파일 체계적 관리
                result_file = 'results/sda_u_comprehensive_results.json'
                if os.path.exists(result_file):
                    # 결과 로드 및 추가 정보 삽입
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    # 실험 메타데이터 추가
                    result_data['experiment_info']['experiment_name'] = experiment_name
                    result_data['experiment_info']['source_dataset'] = source_dataset
                    result_data['experiment_info']['target_dataset'] = target_dataset
                    result_data['experiment_info']['description'] = description
                    result_data['experiment_info']['execution_time_seconds'] = execution_time
                    result_data['experiment_info']['log_file'] = str(log_file)
                    result_data['experiment_info']['models_directory'] = str(models_dir)
                    
                    # 🗂️ 다양한 형태로 결과 저장
                    # 1. 메인 결과 파일
                    main_result_file = experiment_dir / f"{experiment_name}_results.json"
                    with open(main_result_file, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, indent=2, ensure_ascii=False)
                    
                    # 2. 성능 요약 파일
                    performance_summary = self.extract_performance_summary(result_data)
                    summary_file = experiment_dir / f"{experiment_name}_performance_summary.json"
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(performance_summary, f, indent=2, ensure_ascii=False)
                    
                    # 3. 빠른 참조용 텍스트 파일
                    quick_ref_file = experiment_dir / f"{experiment_name}_quick_reference.txt"
                    with open(quick_ref_file, 'w', encoding='utf-8') as f:
                        f.write(f"실험: {experiment_name}\n")
                        f.write(f"소스 → 타겟: {source_dataset} → {target_dataset}\n")
                        f.write(f"설명: {description}\n")
                        f.write(f"실행시간: {execution_time:.1f}초\n")
                        f.write(f"최종 타겟 정확도: {performance_summary.get('target_subset_accuracy', 0):.2f}%\n")
                        f.write(f"전체 타겟 정확도: {performance_summary.get('full_target_accuracy', 0):.2f}%\n")
                        f.write(f"최고 성능: {performance_summary.get('best_target_accuracy', 0):.2f}%\n")
                    
                    print(f"📋 메인 결과: {main_result_file}")
                    print(f"📊 성능 요약: {summary_file}")
                    print(f"⚡ 빠른 참조: {quick_ref_file}")
                    print(f"📝 실행 로그: {log_file}")
                    
                    # 전체 결과에 추가
                    self.all_results.append({
                        'experiment_name': experiment_name,
                        'source_dataset': source_dataset,
                        'target_dataset': target_dataset,
                        'description': description,
                        'execution_time': execution_time,
                        'result_data': result_data,
                        'files': {
                            'main_result': str(main_result_file),
                            'performance_summary': str(summary_file),
                            'quick_reference': str(quick_ref_file),
                            'execution_log': str(log_file),
                            'models_directory': str(models_dir)
                        }
                    })
                    
                    return True, result_data
                else:
                    print("⚠️ 결과 파일을 찾을 수 없습니다.")
                    return False, None
            else:
                print(f"❌ 실험 실패! (반환코드: {return_code})")
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
    
    def extract_performance_summary(self, result_data):
        """결과에서 핵심 성능 지표 추출"""
        
        try:
            final_perf = result_data.get('final_performance', {})
            exp_info = result_data.get('experiment_info', {})
            
            return {
                'source_accuracy': final_perf.get('source_accuracy', 0),
                'target_subset_accuracy': final_perf.get('target_subset_accuracy', 0),
                'full_target_accuracy': final_perf.get('full_target_accuracy', 0),
                'improvement': final_perf.get('improvement_over_baseline', 0),
                'execution_time': exp_info.get('execution_time_seconds', 0),
                'best_target_accuracy': exp_info.get('best_target_accuracy', 0)
            }
        except Exception as e:
            print(f"⚠️ 성능 지표 추출 실패: {e}")
            return {
                'source_accuracy': 0,
                'target_subset_accuracy': 0,
                'full_target_accuracy': 0,
                'improvement': 0,
                'execution_time': 0,
                'best_target_accuracy': 0
            }
    
    def create_performance_table(self):
        """성능 결과를 테이블 형태로 정리 (pandas 없이)"""
        
        if not self.all_results:
            print("⚠️ 실험 결과가 없습니다.")
            return None
        
        # 성능 데이터 추출
        performance_data = []
        for result in self.all_results:
            perf = self.extract_performance_summary(result['result_data'])
            performance_data.append({
                '실험명': result['experiment_name'],
                '소스→타겟': f"{result['source_dataset'].split('_')[1]}→{result['target_dataset'].split('_')[1]}",
                '설명': result['description'],
                '소스 정확도(%)': f"{perf['source_accuracy']:.2f}",
                '타겟 서브셋 정확도(%)': f"{perf['target_subset_accuracy']:.2f}",
                '전체 타겟 정확도(%)': f"{perf['full_target_accuracy']:.2f}",
                '최고 타겟 정확도(%)': f"{perf['best_target_accuracy']:.2f}",
                '개선도(%)': f"{perf['improvement']:.2f}",
                '실행시간(초)': f"{perf['execution_time']:.1f}"
            })
        
        # CSV 파일로 저장 (pandas 없이)
        csv_file = self.results_dir / 'office31_performance_summary.csv'
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            if performance_data:
                # 헤더 작성
                headers = list(performance_data[0].keys())
                f.write(','.join(headers) + '\n')
                
                # 데이터 작성
                for row in performance_data:
                    f.write(','.join(str(row[header]) for header in headers) + '\n')
        
        print(f"📊 성능 요약 테이블 저장: {csv_file}")
        
        return performance_data
    
    def print_performance_summary(self):
        """성능 요약을 콘솔에 출력"""
        
        if not self.all_results:
            print("⚠️ 실험 결과가 없습니다.")
            return
        
        print(f"\n{'='*100}")
        print("📊 Office-31 전체 도메인 조합 실험 결과 요약")
        print(f"{'='*100}")
        
        # 테이블 헤더
        print(f"{'실험명':<15} {'도메인 조합':<20} {'소스 정확도':<12} {'타겟 정확도':<12} {'최고 정확도':<12} {'실행시간':<10}")
        print("-" * 100)
        
        # 각 실험 결과 출력
        total_time = 0
        best_experiment = None
        best_accuracy = 0
        
        for result in self.all_results:
            perf = self.extract_performance_summary(result['result_data'])
            
            print(f"{result['experiment_name']:<15} "
                  f"{result['source_dataset'].split('_')[1]}→{result['target_dataset'].split('_')[1]:<19} "
                  f"{perf['source_accuracy']:<11.2f}% "
                  f"{perf['full_target_accuracy']:<11.2f}% "
                  f"{perf['best_target_accuracy']:<11.2f}% "
                  f"{perf['execution_time']:<9.1f}s")
            
            total_time += perf['execution_time']
            
            if perf['best_target_accuracy'] > best_accuracy:
                best_accuracy = perf['best_target_accuracy']
                best_experiment = result['experiment_name']
        
        print("-" * 100)
        print(f"🏆 최고 성능: {best_experiment} ({best_accuracy:.2f}%)")
        print(f"⏱️ 총 실행시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"📁 상세 결과: {self.results_dir}/")
    
    def save_comprehensive_report(self):
        """종합 보고서 저장"""
        
        report = {
            'experiment_info': {
                'total_experiments': len(self.all_results),
                'timestamp': datetime.now().isoformat(),
                'framework': 'SDA-U Office-31 Full Domain Adaptation',
                'domain_combinations': len(self.domain_combinations)
            },
            'performance_summary': [],
            'detailed_results': self.all_results
        }
        
        # 성능 요약 추가
        for result in self.all_results:
            perf = self.extract_performance_summary(result['result_data'])
            report['performance_summary'].append({
                'experiment_name': result['experiment_name'],
                'source_dataset': result['source_dataset'],
                'target_dataset': result['target_dataset'],
                'description': result['description'],
                'performance': perf
            })
        
        # 종합 보고서 저장
        report_file = self.results_dir / 'office31_comprehensive_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 종합 보고서 저장: {report_file}")
    
    def run_all_experiments(self):
        """모든 도메인 조합에 대해 실험 실행"""
        
        print("🏢 Office-31 전체 도메인 조합 SDA-U 실험 시작!")
        print(f"📊 총 {len(self.domain_combinations)}개 실험 예정")
        print(f"📁 결과 저장 위치: {self.results_dir}/")
        
        start_time = time.time()
        successful_experiments = 0
        
        for i, (source, target, description) in enumerate(self.domain_combinations, 1):
            print(f"\n🔢 진행상황: {i}/{len(self.domain_combinations)}")
            
            success, result_data = self.run_single_experiment(source, target, description)
            
            if success:
                successful_experiments += 1
                print(f"✅ {i}번째 실험 성공!")
            else:
                print(f"❌ {i}번째 실험 실패!")
            
            # 중간 결과 저장 (실험이 중단되어도 결과 보존)
            if self.all_results:
                self.create_performance_table()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print("🎉 Office-31 전체 실험 완료!")
        print(f"✅ 성공한 실험: {successful_experiments}/{len(self.domain_combinations)}")
        print(f"⏱️ 총 소요시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"{'='*80}")
        
        # 최종 결과 정리
        if self.all_results:
            self.print_performance_summary()
            self.create_performance_table()
            self.save_comprehensive_report()
        else:
            print("❌ 성공한 실험이 없습니다.")

def main():
    """메인 함수"""
    
    print("🏢 Office-31 전체 도메인 조합 SDA-U 실험")
    print("=" * 60)
    
    # 실험 관리자 생성
    experiment_manager = Office31FullExperiments()
    
    print("\n실행할 작업을 선택하세요:")
    print("1. 전체 실험 실행 (6개 도메인 조합)")
    print("2. 개별 실험 선택")
    print("3. 기존 결과 분석")
    
    try:
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1":
            # 전체 실험 실행
            print("\n⚠️ 전체 실험은 시간이 오래 걸릴 수 있습니다.")
            confirm = input("계속하시겠습니까? (y/n): ").lower()
            if confirm == 'y':
                experiment_manager.run_all_experiments()
            else:
                print("실험을 취소했습니다.")
                
        elif choice == "2":
            # 개별 실험 선택
            print("\n도메인 조합을 선택하세요:")
            for i, (source, target, desc) in enumerate(experiment_manager.domain_combinations, 1):
                print(f"{i}. {desc}")
            
            exp_choice = int(input(f"\n선택 (1-{len(experiment_manager.domain_combinations)}): ")) - 1
            if 0 <= exp_choice < len(experiment_manager.domain_combinations):
                source, target, description = experiment_manager.domain_combinations[exp_choice]
                success, result_data = experiment_manager.run_single_experiment(source, target, description)
                
                if success:
                    print("✅ 실험 완료!")
                    experiment_manager.print_performance_summary()
                else:
                    print("❌ 실험 실패!")
            else:
                print("❌ 잘못된 선택입니다.")
                
        elif choice == "3":
            # 기존 결과 분석
            results_dir = Path('office31_results')
            if results_dir.exists():
                result_files = list(results_dir.glob('*_results.json'))
                if result_files:
                    print(f"\n📊 발견된 결과 파일: {len(result_files)}개")
                    
                    # 기존 결과 로드
                    for result_file in result_files:
                        try:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                result_data = json.load(f)
                            
                            exp_info = result_data.get('experiment_info', {})
                            experiment_manager.all_results.append({
                                'experiment_name': exp_info.get('experiment_name', 'Unknown'),
                                'source_dataset': exp_info.get('source_dataset', 'Unknown'),
                                'target_dataset': exp_info.get('target_dataset', 'Unknown'),
                                'description': exp_info.get('description', 'Unknown'),
                                'execution_time': exp_info.get('execution_time_seconds', 0),
                                'result_data': result_data
                            })
                        except Exception as e:
                            print(f"⚠️ {result_file} 로드 실패: {e}")
                    
                    if experiment_manager.all_results:
                        experiment_manager.print_performance_summary()
                        experiment_manager.create_performance_table()
                    else:
                        print("❌ 유효한 결과를 찾을 수 없습니다.")
                else:
                    print("❌ 결과 파일을 찾을 수 없습니다.")
            else:
                print("❌ 결과 디렉토리가 없습니다.")
        else:
            print("❌ 잘못된 선택입니다.")
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main() 