# officehome_full_experiments.py
"""
🏠 Office-Home 전체 도메인 조합 실험 시스템
- 4개 도메인: Art, Clipart, Product, Real World
- 12개 도메인 조합 (4×3 = 12)
- SDA-U 알고리즘 적용
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
import subprocess
import argparse

class OfficeHomeExperimentRunner:
    """Office-Home 전체 실험 실행기"""
    
    def __init__(self, base_config=None):
        """
        Args:
            base_config (dict): 기본 설정 (옵션)
        """
        self.domains = ['art', 'clipart', 'product', 'real_world']
        self.domain_names = {
            'art': 'Art',
            'clipart': 'Clipart', 
            'product': 'Product',
            'real_world': 'Real World'
        }
        
        # 기본 설정
        self.base_config = base_config or self._get_default_config()
        
        # 결과 저장 경로
        self.results_dir = Path('results/officehome')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장 경로
        self.models_dir = Path('models/officehome')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print("🏠 Office-Home 실험 러너 초기화 완료!")
        print(f"📁 결과 저장: {self.results_dir}")
        print(f"💾 모델 저장: {self.models_dir}")
    
    def _get_default_config(self):
        """기본 실험 설정"""
        
        return {
            # 데이터셋 설정
            'dataset': 'OfficeHome',
            'data_root': './data',
            'num_classes': 65,
            'image_size': 224,
            'channels': 3,
            
            # 훈련 설정
            'num_epochs': 15,
            'adaptation_epochs': 10,
            'batch_size': 32,
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            
            # SDA-U 설정
            'influence_samples': 500,  # Office-Home은 더 많은 샘플
            'unlearn_ratio': 0.25,
            'max_unlearn_samples': 200,
            'dos_steps': 8,
            'target_samples': 800,  # Office-Home은 더 많은 타겟 샘플
            
            # 모델 설정
            'architecture': 'resnet50',  # Office-Home은 더 복잡한 모델 사용
            'pretrained': True,
            'use_scheduler': True,
            'gradient_clipping': 1.0,
            
            # GPU 설정
            'device': 'cuda',
            'mixed_precision': True,
            'pin_memory': True,
            'num_workers': 4
        }
    
    def get_domain_combinations(self):
        """모든 도메인 조합 생성"""
        
        combinations = []
        for source in self.domains:
            for target in self.domains:
                if source != target:
                    combinations.append((source, target))
        
        return combinations
    
    def create_experiment_config(self, source_domain, target_domain):
        """특정 도메인 조합을 위한 실험 설정 생성"""
        
        config = self.base_config.copy()
        
        # 도메인 설정
        config['source_domain'] = source_domain
        config['target_domain'] = target_domain
        config['source_name'] = self.domain_names[source_domain]
        config['target_name'] = self.domain_names[target_domain]
        
        # 실험 이름
        experiment_name = f"{config['source_name']}2{config['target_name']}"
        config['experiment_name'] = experiment_name
        
        # 파일 경로
        config['model_save_dir'] = str(self.models_dir / experiment_name)
        config['results_file'] = str(self.results_dir / f"{experiment_name}_results.json")
        config['log_file'] = str(self.results_dir / f"{experiment_name}_log.txt")
        
        return config
    
    def run_single_experiment(self, source_domain, target_domain, verbose=True):
        """단일 도메인 조합 실험 실행"""
        
        config = self.create_experiment_config(source_domain, target_domain)
        experiment_name = config['experiment_name']
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🚀 실험 시작: {experiment_name}")
            print(f"📊 소스 도메인: {config['source_name']} ({source_domain})")
            print(f"🎯 타겟 도메인: {config['target_name']} ({target_domain})")
            print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # main.py 실행 명령어 구성
            cmd = [
                sys.executable, 'main.py',
                '--dataset', 'OfficeHome',
                '--source_domain', source_domain,
                '--target_domain', target_domain,
                '--num_epochs', str(config['num_epochs']),
                '--adaptation_epochs', str(config['adaptation_epochs']),
                '--batch_size', str(config['batch_size']),
                '--learning_rate', str(config['learning_rate']),
                '--influence_samples', str(config['influence_samples']),
                '--target_samples', str(config['target_samples']),
                '--model_save_dir', config['model_save_dir'],
                '--results_file', config['results_file']
            ]
            
            # 로그 파일로 출력 리다이렉션 + 실시간 출력
            with open(config['log_file'], 'w', encoding='utf-8') as log_file:
                if verbose:
                    print(f"📝 로그 파일: {config['log_file']}")
                    print(f"🚀 실험 실행 중... (실시간 진행 상황은 로그 파일 참조)")
                    print(f"📊 예상 소요 시간: Art 도메인 ~15-20분, 타 도메인 ~20-30분")
                    print(f"💡 진행 확인: tail -f {config['log_file']}")
                
                # 실험 실행
                result = subprocess.run(
                    cmd, 
                    stdout=log_file, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.getcwd()
                )
            
            # 실행 시간 계산
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                if verbose:
                    print(f"✅ 실험 완료: {experiment_name}")
                    print(f"⏱️ 실행 시간: {elapsed_time/60:.1f}분")
                
                # 결과 로드 및 반환
                if os.path.exists(config['results_file']):
                    with open(config['results_file'], 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    results['elapsed_time'] = elapsed_time
                    return results
                else:
                    print(f"⚠️ 결과 파일을 찾을 수 없음: {config['results_file']}")
                    return {'error': 'results_file_not_found', 'elapsed_time': elapsed_time}
            else:
                if verbose:
                    print(f"❌ 실험 실패: {experiment_name}")
                    print(f"💥 종료 코드: {result.returncode}")
                    print(f"📝 로그 확인: {config['log_file']}")
                
                return {'error': f'experiment_failed_code_{result.returncode}', 'elapsed_time': elapsed_time}
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"💥 예외 발생: {experiment_name}")
                print(f"❌ 오류: {str(e)}")
            
            return {'error': str(e), 'elapsed_time': elapsed_time}
    
    def run_all_experiments(self, skip_existing=True, max_parallel=1):
        """모든 도메인 조합 실험 실행"""
        
        combinations = self.get_domain_combinations()
        total_experiments = len(combinations)
        
        print(f"\n🏠 Office-Home 전체 실험 시작!")
        print(f"📊 총 실험 수: {total_experiments}개")
        print(f"🔄 도메인 조합: {len(self.domains)}개 도메인 × {len(self.domains)-1}개 타겟")
        print(f"⚡ 병렬 실행: {max_parallel}개")
        print(f"⏭️ 기존 결과 건너뛰기: {skip_existing}")
        
        # 전체 결과 저장
        all_results = {}
        successful_experiments = 0
        failed_experiments = 0
        skipped_experiments = 0
        
        start_time = time.time()
        
        for i, (source, target) in enumerate(combinations, 1):
            experiment_name = f"{self.domain_names[source]}2{self.domain_names[target]}"
            
            print(f"\n[{i}/{total_experiments}] {experiment_name}")
            
            # 기존 결과 확인
            results_file = self.results_dir / f"{experiment_name}_results.json"
            if skip_existing and results_file.exists():
                print(f"⏭️ 기존 결과 존재, 건너뛰기: {results_file}")
                
                # 기존 결과 로드
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                    all_results[experiment_name] = existing_results
                    skipped_experiments += 1
                    continue
                except:
                    print("⚠️ 기존 결과 파일 손상, 재실행")
            
            # 실험 실행
            result = self.run_single_experiment(source, target, verbose=True)
            all_results[experiment_name] = result
            
            if 'error' in result:
                failed_experiments += 1
            else:
                successful_experiments += 1
            
            # 진행률 출력
            progress = (i / total_experiments) * 100
            print(f"📈 진행률: {progress:.1f}% ({i}/{total_experiments})")
        
        # 전체 실행 시간
        total_elapsed = time.time() - start_time
        
        # 결과 요약
        print(f"\n{'='*80}")
        print(f"🏠 Office-Home 전체 실험 완료!")
        print(f"{'='*80}")
        print(f"⏱️ 총 실행 시간: {total_elapsed/3600:.1f}시간")
        print(f"✅ 성공한 실험: {successful_experiments}개")
        print(f"❌ 실패한 실험: {failed_experiments}개")
        print(f"⏭️ 건너뛴 실험: {skipped_experiments}개")
        print(f"📊 총 실험 수: {total_experiments}개")
        
        # 전체 결과 저장
        summary_results = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments,
            'skipped_experiments': skipped_experiments,
            'total_elapsed_time': total_elapsed,
            'experiments': all_results
        }
        
        summary_file = self.results_dir / 'officehome_full_results.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 전체 결과 저장: {summary_file}")
        
        # 성능 요약 테이블 생성
        self.create_performance_summary(all_results)
        
        return summary_results
    
    def create_performance_summary(self, all_results):
        """성능 요약 테이블 생성"""
        
        print(f"\n📊 성능 요약 테이블 생성 중...")
        
        # 데이터 수집
        summary_data = []
        
        for experiment_name, result in all_results.items():
            if 'error' in result:
                summary_data.append({
                    'Experiment': experiment_name,
                    'Source Domain': experiment_name.split('2')[0],
                    'Target Domain': experiment_name.split('2')[1],
                    'Final Target Accuracy': 'FAILED',
                    'Overall Target Accuracy': 'FAILED',
                    'Source Accuracy': 'FAILED',
                    'Unlearned Samples': 'FAILED',
                    'Selected Target Samples': 'FAILED',
                    'Training Time (min)': f"{result.get('elapsed_time', 0)/60:.1f}",
                    'Status': 'FAILED'
                })
            else:
                summary_data.append({
                    'Experiment': experiment_name,
                    'Source Domain': experiment_name.split('2')[0],
                    'Target Domain': experiment_name.split('2')[1],
                    'Final Target Accuracy': f"{result.get('final_target_accuracy', 0):.2f}%",
                    'Overall Target Accuracy': f"{result.get('overall_target_accuracy', 0):.2f}%",
                    'Source Accuracy': f"{result.get('source_accuracy', 0):.2f}%",
                    'Unlearned Samples': result.get('unlearned_samples', 0),
                    'Selected Target Samples': result.get('selected_target_samples', 0),
                    'Training Time (min)': f"{result.get('elapsed_time', 0)/60:.1f}",
                    'Status': 'SUCCESS'
                })
        
        # DataFrame 생성
        df = pd.DataFrame(summary_data)
        
        # CSV 저장
        csv_file = self.results_dir / 'officehome_performance_summary.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ CSV 요약 저장: {csv_file}")
        
        # 콘솔 출력
        print(f"\n📊 Office-Home 성능 요약:")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        # 통계 요약
        successful_results = [r for r in all_results.values() if 'error' not in r]
        if successful_results:
            final_accuracies = [r.get('final_target_accuracy', 0) for r in successful_results]
            overall_accuracies = [r.get('overall_target_accuracy', 0) for r in successful_results]
            
            print(f"\n📈 성능 통계:")
            print(f"   🎯 최종 타겟 정확도 - 평균: {sum(final_accuracies)/len(final_accuracies):.2f}%, 최고: {max(final_accuracies):.2f}%, 최저: {min(final_accuracies):.2f}%")
            print(f"   📊 전체 타겟 정확도 - 평균: {sum(overall_accuracies)/len(overall_accuracies):.2f}%, 최고: {max(overall_accuracies):.2f}%, 최저: {min(overall_accuracies):.2f}%")
        
        return df

def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description='Office-Home 전체 실험 실행')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='기존 결과 건너뛰기 (기본: True)')
    parser.add_argument('--max_parallel', type=int, default=1,
                       help='최대 병렬 실행 수 (기본: 1)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='사용자 정의 설정 파일 (JSON)')
    
    args = parser.parse_args()
    
    # 사용자 정의 설정 로드
    base_config = None
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r', encoding='utf-8') as f:
            base_config = json.load(f)
        print(f"📄 사용자 정의 설정 로드: {args.config_file}")
    
    # 실험 러너 생성
    runner = OfficeHomeExperimentRunner(base_config=base_config)
    
    # 전체 실험 실행
    results = runner.run_all_experiments(
        skip_existing=args.skip_existing,
        max_parallel=args.max_parallel
    )
    
    print(f"\n🎉 모든 Office-Home 실험 완료!")
    print(f"📁 결과 확인: {runner.results_dir}")

if __name__ == "__main__":
    main() 