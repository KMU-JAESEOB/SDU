# extract_officehome_performance.py
"""
🏠 Office-Home 실험 결과 성능 추출 및 분석 도구
- 개별 실험 결과 분석
- 전체 실험 성능 요약
- 도메인별 성능 비교
- 시각화 및 통계 분석
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class OfficeHomePerformanceExtractor:
    """Office-Home 성능 추출기"""
    
    def __init__(self, results_dir='results/officehome'):
        """
        Args:
            results_dir (str): 결과 디렉토리 경로
        """
        self.results_dir = Path(results_dir)
        self.domains = ['art', 'clipart', 'product', 'real_world']
        self.domain_names = {
            'art': 'Art',
            'clipart': 'Clipart',
            'product': 'Product', 
            'real_world': 'Real World'
        }
        
        print(f"🏠 Office-Home 성능 추출기 초기화")
        print(f"📁 결과 디렉토리: {self.results_dir}")
        
        if not self.results_dir.exists():
            print(f"⚠️ 결과 디렉토리가 존재하지 않습니다: {self.results_dir}")
            self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_single_experiment_result(self, source_domain, target_domain):
        """단일 실험 결과 로드"""
        
        source_name = self.domain_names.get(source_domain, source_domain)
        target_name = self.domain_names.get(target_domain, target_domain)
        experiment_name = f"{source_name}2{target_name}"
        
        result_file = self.results_dir / f"{experiment_name}_results.json"
        
        if not result_file.exists():
            print(f"❌ 결과 파일을 찾을 수 없습니다: {result_file}")
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            print(f"✅ {experiment_name} 결과 로드 성공")
            return result
            
        except Exception as e:
            print(f"❌ {experiment_name} 결과 로드 실패: {e}")
            return None
    
    def load_all_experiment_results(self):
        """모든 실험 결과 로드"""
        
        print(f"📊 모든 Office-Home 실험 결과 로드 중...")
        
        all_results = {}
        loaded_count = 0
        
        # 개별 결과 파일 로드
        for source in self.domains:
            for target in self.domains:
                if source != target:
                    result = self.load_single_experiment_result(source, target)
                    if result:
                        experiment_name = f"{self.domain_names[source]}2{self.domain_names[target]}"
                        all_results[experiment_name] = result
                        loaded_count += 1
        
        # 전체 결과 파일 확인
        full_results_file = self.results_dir / 'officehome_full_results.json'
        if full_results_file.exists():
            try:
                with open(full_results_file, 'r', encoding='utf-8') as f:
                    full_results = json.load(f)
                
                # 전체 결과에서 개별 실험 추출
                if 'experiments' in full_results:
                    for exp_name, exp_result in full_results['experiments'].items():
                        if exp_name not in all_results:
                            all_results[exp_name] = exp_result
                            loaded_count += 1
                
                print(f"✅ 전체 결과 파일도 로드: {full_results_file}")
                
            except Exception as e:
                print(f"⚠️ 전체 결과 파일 로드 실패: {e}")
        
        print(f"📈 총 {loaded_count}개 실험 결과 로드 완료")
        return all_results
    
    def extract_performance_metrics(self, results):
        """성능 지표 추출"""
        
        print(f"📊 성능 지표 추출 중...")
        
        performance_data = []
        
        for experiment_name, result in results.items():
            # 실험 이름 파싱
            if '2' in experiment_name:
                source_domain, target_domain = experiment_name.split('2')
            else:
                continue
            
            # 오류 확인
            if 'error' in result:
                performance_data.append({
                    'experiment': experiment_name,
                    'source_domain': source_domain,
                    'target_domain': target_domain,
                    'final_target_accuracy': 0.0,
                    'overall_target_accuracy': 0.0,
                    'source_accuracy': 0.0,
                    'unlearned_samples': 0,
                    'selected_target_samples': 0,
                    'training_time_minutes': result.get('elapsed_time', 0) / 60,
                    'status': 'FAILED',
                    'error': result.get('error', 'Unknown error')
                })
                continue
            
            # 성공한 실험 데이터 추출
            performance_data.append({
                'experiment': experiment_name,
                'source_domain': source_domain,
                'target_domain': target_domain,
                'final_target_accuracy': result.get('final_target_accuracy', 0.0),
                'overall_target_accuracy': result.get('overall_target_accuracy', 0.0),
                'source_accuracy': result.get('source_accuracy', 0.0),
                'unlearned_samples': result.get('unlearned_samples', 0),
                'selected_target_samples': result.get('selected_target_samples', 0),
                'training_time_minutes': result.get('elapsed_time', 0) / 60,
                'status': 'SUCCESS',
                'error': None
            })
        
        print(f"✅ {len(performance_data)}개 실험 성능 지표 추출 완료")
        return performance_data
    
    def create_performance_summary_table(self, performance_data):
        """성능 요약 테이블 생성"""
        
        print(f"\n📊 Office-Home 성능 요약 테이블")
        print("="*120)
        
        # 헤더
        header = f"{'실험':<20} {'소스':<12} {'타겟':<12} {'최종정확도':<12} {'전체정확도':<12} {'소스정확도':<12} {'언러닝':<8} {'타겟선택':<8} {'시간(분)':<8} {'상태':<8}"
        print(header)
        print("-"*120)
        
        # 성공한 실험들
        successful_experiments = [exp for exp in performance_data if exp['status'] == 'SUCCESS']
        failed_experiments = [exp for exp in performance_data if exp['status'] == 'FAILED']
        
        # 성공한 실험 출력
        for exp in successful_experiments:
            row = f"{exp['experiment']:<20} {exp['source_domain']:<12} {exp['target_domain']:<12} {exp['final_target_accuracy']:<12.2f} {exp['overall_target_accuracy']:<12.2f} {exp['source_accuracy']:<12.2f} {exp['unlearned_samples']:<8} {exp['selected_target_samples']:<8} {exp['training_time_minutes']:<8.1f} {exp['status']:<8}"
            print(row)
        
        # 실패한 실험 출력
        if failed_experiments:
            print("\n❌ 실패한 실험들:")
            for exp in failed_experiments:
                row = f"{exp['experiment']:<20} {exp['source_domain']:<12} {exp['target_domain']:<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<8} {'FAILED':<8} {exp['training_time_minutes']:<8.1f} {exp['status']:<8}"
                print(row)
        
        print("="*120)
        
        # 통계 요약
        if successful_experiments:
            final_accuracies = [exp['final_target_accuracy'] for exp in successful_experiments]
            overall_accuracies = [exp['overall_target_accuracy'] for exp in successful_experiments]
            source_accuracies = [exp['source_accuracy'] for exp in successful_experiments]
            training_times = [exp['training_time_minutes'] for exp in successful_experiments]
            
            print(f"\n📈 성능 통계 요약:")
            print(f"   🎯 최종 타겟 정확도: 평균 {np.mean(final_accuracies):.2f}%, 최고 {np.max(final_accuracies):.2f}%, 최저 {np.min(final_accuracies):.2f}%, 표준편차 {np.std(final_accuracies):.2f}")
            print(f"   📊 전체 타겟 정확도: 평균 {np.mean(overall_accuracies):.2f}%, 최고 {np.max(overall_accuracies):.2f}%, 최저 {np.min(overall_accuracies):.2f}%, 표준편차 {np.std(overall_accuracies):.2f}")
            print(f"   🔄 소스 도메인 정확도: 평균 {np.mean(source_accuracies):.2f}%, 최고 {np.max(source_accuracies):.2f}%, 최저 {np.min(source_accuracies):.2f}%, 표준편차 {np.std(source_accuracies):.2f}")
            print(f"   ⏱️ 훈련 시간: 평균 {np.mean(training_times):.1f}분, 최대 {np.max(training_times):.1f}분, 최소 {np.min(training_times):.1f}분")
            print(f"   ✅ 성공률: {len(successful_experiments)}/{len(performance_data)} ({len(successful_experiments)/len(performance_data)*100:.1f}%)")
    
    def create_domain_analysis(self, performance_data):
        """도메인별 성능 분석"""
        
        print(f"\n🏷️ 도메인별 성능 분석")
        print("="*80)
        
        successful_experiments = [exp for exp in performance_data if exp['status'] == 'SUCCESS']
        
        if not successful_experiments:
            print("❌ 성공한 실험이 없어 도메인 분석을 수행할 수 없습니다.")
            return
        
        # 소스 도메인별 분석
        print("\n📤 소스 도메인별 성능 (타겟으로 전이할 때):")
        source_performance = {}
        for domain in self.domains:
            domain_name = self.domain_names[domain]
            domain_experiments = [exp for exp in successful_experiments if exp['source_domain'] == domain_name]
            
            if domain_experiments:
                final_accs = [exp['final_target_accuracy'] for exp in domain_experiments]
                overall_accs = [exp['overall_target_accuracy'] for exp in domain_experiments]
                
                source_performance[domain_name] = {
                    'count': len(domain_experiments),
                    'final_acc_mean': np.mean(final_accs),
                    'final_acc_std': np.std(final_accs),
                    'overall_acc_mean': np.mean(overall_accs),
                    'overall_acc_std': np.std(overall_accs)
                }
                
                print(f"   {domain_name:>12}: {len(domain_experiments)}개 실험, 최종정확도 {np.mean(final_accs):.2f}±{np.std(final_accs):.2f}%, 전체정확도 {np.mean(overall_accs):.2f}±{np.std(overall_accs):.2f}%")
        
        # 타겟 도메인별 분석
        print("\n📥 타겟 도메인별 성능 (소스에서 적응받을 때):")
        target_performance = {}
        for domain in self.domains:
            domain_name = self.domain_names[domain]
            domain_experiments = [exp for exp in successful_experiments if exp['target_domain'] == domain_name]
            
            if domain_experiments:
                final_accs = [exp['final_target_accuracy'] for exp in domain_experiments]
                overall_accs = [exp['overall_target_accuracy'] for exp in domain_experiments]
                
                target_performance[domain_name] = {
                    'count': len(domain_experiments),
                    'final_acc_mean': np.mean(final_accs),
                    'final_acc_std': np.std(final_accs),
                    'overall_acc_mean': np.mean(overall_accs),
                    'overall_acc_std': np.std(overall_accs)
                }
                
                print(f"   {domain_name:>12}: {len(domain_experiments)}개 실험, 최종정확도 {np.mean(final_accs):.2f}±{np.std(final_accs):.2f}%, 전체정확도 {np.mean(overall_accs):.2f}±{np.std(overall_accs):.2f}%")
        
        return source_performance, target_performance
    
    def find_best_and_worst_experiments(self, performance_data):
        """최고/최악 성능 실험 찾기"""
        
        successful_experiments = [exp for exp in performance_data if exp['status'] == 'SUCCESS']
        
        if not successful_experiments:
            print("❌ 성공한 실험이 없습니다.")
            return
        
        print(f"\n🏆 최고/최악 성능 실험")
        print("="*80)
        
        # 최종 타겟 정확도 기준
        best_final = max(successful_experiments, key=lambda x: x['final_target_accuracy'])
        worst_final = min(successful_experiments, key=lambda x: x['final_target_accuracy'])
        
        print(f"🥇 최고 최종 타겟 정확도: {best_final['experiment']} ({best_final['final_target_accuracy']:.2f}%)")
        print(f"   📊 전체 타겟 정확도: {best_final['overall_target_accuracy']:.2f}%")
        print(f"   🔄 소스 정확도: {best_final['source_accuracy']:.2f}%")
        print(f"   ⏱️ 훈련 시간: {best_final['training_time_minutes']:.1f}분")
        
        print(f"\n🥉 최저 최종 타겟 정확도: {worst_final['experiment']} ({worst_final['final_target_accuracy']:.2f}%)")
        print(f"   📊 전체 타겟 정확도: {worst_final['overall_target_accuracy']:.2f}%")
        print(f"   🔄 소스 정확도: {worst_final['source_accuracy']:.2f}%")
        print(f"   ⏱️ 훈련 시간: {worst_final['training_time_minutes']:.1f}분")
        
        # 전체 타겟 정확도 기준
        best_overall = max(successful_experiments, key=lambda x: x['overall_target_accuracy'])
        worst_overall = min(successful_experiments, key=lambda x: x['overall_target_accuracy'])
        
        print(f"\n🥇 최고 전체 타겟 정확도: {best_overall['experiment']} ({best_overall['overall_target_accuracy']:.2f}%)")
        print(f"🥉 최저 전체 타겟 정확도: {worst_overall['experiment']} ({worst_overall['overall_target_accuracy']:.2f}%)")
    
    def save_performance_csv(self, performance_data, filename=None):
        """성능 데이터를 CSV로 저장"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"officehome_performance_{timestamp}.csv"
        
        csv_path = self.results_dir / filename
        
        # CSV 헤더
        headers = [
            'Experiment', 'Source Domain', 'Target Domain',
            'Final Target Accuracy (%)', 'Overall Target Accuracy (%)', 'Source Accuracy (%)',
            'Unlearned Samples', 'Selected Target Samples', 'Training Time (min)',
            'Status', 'Error'
        ]
        
        try:
            with open(csv_path, 'w', encoding='utf-8-sig') as f:
                # 헤더 작성
                f.write(','.join(headers) + '\n')
                
                # 데이터 작성
                for exp in performance_data:
                    row = [
                        exp['experiment'],
                        exp['source_domain'],
                        exp['target_domain'],
                        f"{exp['final_target_accuracy']:.2f}" if exp['status'] == 'SUCCESS' else 'FAILED',
                        f"{exp['overall_target_accuracy']:.2f}" if exp['status'] == 'SUCCESS' else 'FAILED',
                        f"{exp['source_accuracy']:.2f}" if exp['status'] == 'SUCCESS' else 'FAILED',
                        str(exp['unlearned_samples']) if exp['status'] == 'SUCCESS' else 'FAILED',
                        str(exp['selected_target_samples']) if exp['status'] == 'SUCCESS' else 'FAILED',
                        f"{exp['training_time_minutes']:.1f}",
                        exp['status'],
                        exp['error'] or ''
                    ]
                    f.write(','.join(row) + '\n')
            
            print(f"💾 성능 데이터 CSV 저장: {csv_path}")
            return csv_path
            
        except Exception as e:
            print(f"❌ CSV 저장 실패: {e}")
            return None
    
    def run_full_analysis(self):
        """전체 성능 분석 실행"""
        
        print(f"\n🏠 Office-Home 전체 성능 분석 시작")
        print("="*80)
        
        # 1. 모든 결과 로드
        all_results = self.load_all_experiment_results()
        
        if not all_results:
            print("❌ 분석할 결과가 없습니다.")
            return
        
        # 2. 성능 지표 추출
        performance_data = self.extract_performance_metrics(all_results)
        
        # 3. 성능 요약 테이블
        self.create_performance_summary_table(performance_data)
        
        # 4. 도메인별 분석
        self.create_domain_analysis(performance_data)
        
        # 5. 최고/최악 실험 찾기
        self.find_best_and_worst_experiments(performance_data)
        
        # 6. CSV 저장
        csv_path = self.save_performance_csv(performance_data)
        
        print(f"\n🎉 Office-Home 성능 분석 완료!")
        print(f"📁 결과 디렉토리: {self.results_dir}")
        if csv_path:
            print(f"💾 CSV 파일: {csv_path}")

def main():
    """메인 실행 함수"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Office-Home 실험 성능 추출 및 분석')
    parser.add_argument('--results_dir', type=str, default='results/officehome',
                       help='결과 디렉토리 경로 (기본: results/officehome)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='특정 실험만 분석 (예: Art2Clipart)')
    parser.add_argument('--save_csv', action='store_true',
                       help='결과를 CSV로 저장')
    
    args = parser.parse_args()
    
    # 성능 추출기 생성
    extractor = OfficeHomePerformanceExtractor(results_dir=args.results_dir)
    
    if args.experiment:
        # 특정 실험 분석
        print(f"🔍 특정 실험 분석: {args.experiment}")
        
        if '2' in args.experiment:
            source_name, target_name = args.experiment.split('2')
            
            # 도메인 이름을 키로 변환
            source_key = None
            target_key = None
            for key, name in extractor.domain_names.items():
                if name == source_name:
                    source_key = key
                if name == target_name:
                    target_key = key
            
            if source_key and target_key:
                result = extractor.load_single_experiment_result(source_key, target_key)
                if result:
                    performance_data = extractor.extract_performance_metrics({args.experiment: result})
                    extractor.create_performance_summary_table(performance_data)
                    
                    if args.save_csv:
                        extractor.save_performance_csv(performance_data, f"{args.experiment}_performance.csv")
            else:
                print(f"❌ 유효하지 않은 실험 이름: {args.experiment}")
        else:
            print(f"❌ 실험 이름 형식이 잘못되었습니다. 예: Art2Clipart")
    else:
        # 전체 분석
        extractor.run_full_analysis()

if __name__ == "__main__":
    main() 