#!/usr/bin/env python3
# office31_full_experiments.py - Office31 종합 실험 도구

"""
🎯 Office31 종합 실험 도구

기능:
1. 모든 소스-타겟 도메인 조합 실험 
2. 타겟 샘플 수별 성능 비교 (100~600개)
3. λ_u, β 하이퍼파라미터 그리드 서치
4. 언러닝 효과 시각화

사용법:
python office31_full_experiments.py --experiment [domain_pairs|sample_sizes|hyperparams|all]
"""

import os
import sys
import json
import copy
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess

# 시각화를 위한 라이브러리
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.font_manager as fm
    
    # 한글 폰트 설정 (에러 방지)
    try:
        # 사용 가능한 한글 폰트 찾기
        font_list = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Arial Unicode MS']
        available_korean_font = None
        
        for font in korean_fonts:
            if font in font_list:
                available_korean_font = font
                break
        
        if available_korean_font:
            plt.rcParams['font.family'] = [available_korean_font, 'DejaVu Sans']
        else:
            # 한글 폰트가 없으면 영어만 사용
            plt.rcParams['font.family'] = ['DejaVu Sans']
            
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 폰트 설정 실패 시 기본값 유지
        pass
    
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ 시각화 라이브러리 없음. pip install matplotlib seaborn pandas 실행하세요.")

# main.py의 SDAUAlgorithm 클래스 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from main import SDAUAlgorithm, SDAUConfig
    MAIN_AVAILABLE = True
except ImportError:
    MAIN_AVAILABLE = False
    print("⚠️ main.py import 실패. 환경 설정 문제일 수 있습니다.")

class Office31ExperimentRunner:
    """Office31 종합 실험 실행기"""
    
    def __init__(self, base_config_path: str = "config.json"):
        self.base_config_path = base_config_path
        self.results_dir = Path("results/office31_comprehensive")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Office31 도메인 정의
        self.domains = ["Amazon", "Webcam", "DSLR"]
        self.domain_pairs = [(s, t) for s in self.domains for t in self.domains if s != t]
        
        # 실험 설정
        self.sample_sizes = [100, 200, 300, 400, 500, 600]
        self.lambda_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.beta_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        print(f"🎯 Office31 종합 실험 초기화 완료")
        print(f"📂 결과 저장: {self.results_dir}")
    
    def create_experiment_config(self, modifications: Dict[str, Any]) -> str:
        """실험용 설정 파일 생성"""
        # 기본 설정 로딩
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 수정사항 적용
        for key_path, value in modifications.items():
            keys = key_path.split('.')
            current = config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
        
        # 임시 설정 파일 생성
        temp_config_path = self.results_dir / f"temp_config_{int(time.time())}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return str(temp_config_path)
    
    def run_single_experiment(self, source_domain: str, target_domain: str, 
                            config_path: str, experiment_name: str) -> Dict[str, Any]:
        """단일 실험 실행"""
        print(f"🔬 실험 시작: {experiment_name} ({source_domain} → {target_domain})")
        
        try:
            if MAIN_AVAILABLE:
                # 직접 실행 방식
                sda_u = SDAUAlgorithm(config_path=config_path)
                results = sda_u.run_experiment("Office31", source_domain, target_domain)
                
                return {
                    'source_domain': source_domain,
                    'target_domain': target_domain,
                    'experiment_name': experiment_name,
                    'initial_target_acc': results.initial_target_acc,
                    'final_target_acc': results.final_target_acc,
                    'best_target_acc': results.best_target_acc,
                    'best_epoch': results.best_epoch,
                    'improvement': results.improvement,
                    'best_improvement': results.best_improvement,
                    'unlearning_count': results.unlearning_count,
                    'total_epochs': results.total_epochs,
                    'success': True,
                    'timestamp': time.time()
                }
            else:
                # subprocess 실행 방식
                cmd = [
                    sys.executable, 'main.py',
                    '--dataset', 'Office31',
                    '--source_domain', source_domain,
                    '--target_domain', target_domain,
                    '--config', config_path,
                    '--results_file', f'{experiment_name}_results.json'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0:
                    print(f"✅ 성공: {experiment_name}")
                    return {
                        'source_domain': source_domain,
                        'target_domain': target_domain,
                        'experiment_name': experiment_name,
                        'success': True,
                        'timestamp': time.time()
                    }
                else:
                    print(f"❌ 실패: {experiment_name}")
                    return {
                        'source_domain': source_domain,
                        'target_domain': target_domain,
                        'experiment_name': experiment_name,
                        'success': False,
                        'error': result.stderr,
                        'timestamp': time.time()
                    }
                    
        except Exception as e:
            print(f"❌ 오류: {experiment_name} - {e}")
            return {
                'source_domain': source_domain,
                'target_domain': target_domain,
                'experiment_name': experiment_name,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def run_domain_pairs_experiment(self) -> Dict[str, Any]:
        """모든 도메인 쌍 실험"""
        print(f"\n🎯 도메인 쌍 실험 시작 (총 {len(self.domain_pairs)}개)")
        
        results = []
        for i, (source, target) in enumerate(self.domain_pairs, 1):
            print(f"\n진행률: {i}/{len(self.domain_pairs)}")
            
            # 기본 설정으로 실험
            config_path = self.create_experiment_config({})
            experiment_name = f"domain_pairs_{source}2{target}"
            
            result = self.run_single_experiment(source, target, config_path, experiment_name)
            results.append(result)
            
            # 임시 설정 파일 삭제
            os.remove(config_path)
        
        # 결과 저장
        output_file = self.results_dir / "domain_pairs_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 도메인 쌍 실험 완료! 결과: {output_file}")
        return {'results': results, 'output_file': str(output_file)}
    
    def run_sample_sizes_experiment(self) -> Dict[str, Any]:
        """타겟 샘플 수별 실험"""
        print(f"\n📊 타겟 샘플 수별 실험 시작")
        print(f"샘플 크기: {self.sample_sizes}")
        
        results = []
        # 대표적인 도메인 쌍 선택 (Amazon → Webcam)
        source, target = "Amazon", "Webcam"
        
        for i, num_samples in enumerate(self.sample_sizes, 1):
            print(f"\n진행률: {i}/{len(self.sample_sizes)} - 샘플 {num_samples}개")
            
            # 타겟 샘플 수 설정
            config_path = self.create_experiment_config({
                'target_selection.num_samples': num_samples
            })
            
            experiment_name = f"sample_size_{num_samples}"
            result = self.run_single_experiment(source, target, config_path, experiment_name)
            result['num_samples'] = num_samples
            results.append(result)
            
            # 임시 설정 파일 삭제
            os.remove(config_path)
        
        # 결과 저장
        output_file = self.results_dir / "sample_sizes_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 샘플 수별 실험 완료! 결과: {output_file}")
        return {'results': results, 'output_file': str(output_file)}
    
    def run_hyperparameter_experiment(self) -> Dict[str, Any]:
        """하이퍼파라미터 그리드 서치"""
        print(f"\n🔍 하이퍼파라미터 그리드 서치 시작")
        print(f"λ_u 값: {self.lambda_values}")
        print(f"β 값: {self.beta_values}")
        print(f"총 조합: {len(self.lambda_values) * len(self.beta_values)}개")
        
        results = []
        # 대표적인 도메인 쌍 선택 (Amazon → Webcam)
        source, target = "Amazon", "Webcam"
        
        total_combinations = len(self.lambda_values) * len(self.beta_values)
        current_combo = 0
        
        for lambda_u in self.lambda_values:
            for beta in self.beta_values:
                current_combo += 1
                print(f"\n진행률: {current_combo}/{total_combinations} - λ_u={lambda_u}, β={beta}")
                
                # 하이퍼파라미터 설정
                config_path = self.create_experiment_config({
                    'target_selection.lambda_utility': lambda_u,
                    'target_selection.beta_uncertainty': beta
                })
                
                experiment_name = f"hyperparam_lambda{lambda_u}_beta{beta}"
                result = self.run_single_experiment(source, target, config_path, experiment_name)
                result['lambda_utility'] = lambda_u
                result['beta_uncertainty'] = beta
                results.append(result)
                
                # 임시 설정 파일 삭제
                os.remove(config_path)
        
        # 결과 저장
        output_file = self.results_dir / "hyperparameter_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 하이퍼파라미터 실험 완료! 결과: {output_file}")
        return {'results': results, 'output_file': str(output_file)}
    
    def create_visualizations(self) -> None:
        """실험 결과 시각화"""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ 시각화 라이브러리가 없어 건너뛰기")
            return
        
        print(f"\n📈 시각화 생성 중...")
        
        # 시각화 디렉토리 생성
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. 도메인 쌍 결과 시각화
        self._plot_domain_pairs_results(viz_dir)
        
        # 2. 샘플 수별 결과 시각화
        self._plot_sample_sizes_results(viz_dir)
        
        # 3. 하이퍼파라미터 히트맵
        self._plot_hyperparameter_heatmap(viz_dir)
        
        # 4. 언러닝 효과 시각화
        self._plot_unlearning_effects(viz_dir)
        
        print(f"✅ 시각화 완료! 저장 위치: {viz_dir}")
    
    def _plot_domain_pairs_results(self, viz_dir: Path) -> None:
        """도메인 쌍 결과 시각화"""
        try:
            with open(self.results_dir / "domain_pairs_results.json", 'r') as f:
                results = json.load(f)
            
            # 성공한 실험만 필터링
            successful_results = [r for r in results if r.get('success', False)]
            
            if not successful_results:
                print("⚠️ 성공한 도메인 쌍 실험 결과가 없음")
                return
            
            # 데이터 준비
            domain_pairs = [f"{r['source_domain']} → {r['target_domain']}" for r in successful_results]
            final_accs = [r.get('final_target_acc', 0) for r in successful_results]
            improvements = [r.get('improvement', 0) for r in successful_results]
            
            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 최종 성능
            ax1.bar(range(len(domain_pairs)), final_accs, color='skyblue')
            ax1.set_title('Office31 Domain Pairs Final Performance')
            ax1.set_xlabel('Domain Pairs')
            ax1.set_ylabel('Target Accuracy (%)')
            ax1.set_xticks(range(len(domain_pairs)))
            ax1.set_xticklabels(domain_pairs, rotation=45)
            
            # 성능 개선
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            ax2.bar(range(len(domain_pairs)), improvements, color=colors)
            ax2.set_title('Office31 Domain Pairs Performance Improvement')
            ax2.set_xlabel('Domain Pairs')
            ax2.set_ylabel('Performance Improvement (%)')
            ax2.set_xticks(range(len(domain_pairs)))
            ax2.set_xticklabels(domain_pairs, rotation=45)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "domain_pairs_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 도메인 쌍 시각화 실패: {e}")
    
    def _plot_sample_sizes_results(self, viz_dir: Path) -> None:
        """샘플 수별 결과 시각화"""
        try:
            with open(self.results_dir / "sample_sizes_results.json", 'r') as f:
                results = json.load(f)
            
            successful_results = [r for r in results if r.get('success', False)]
            
            if not successful_results:
                print("⚠️ 성공한 샘플 수 실험 결과가 없음")
                return
            
            # 데이터 준비
            sample_sizes = [r['num_samples'] for r in successful_results]
            final_accs = [r.get('final_target_acc', 0) for r in successful_results]
            improvements = [r.get('improvement', 0) for r in successful_results]
            unlearning_counts = [r.get('unlearning_count', 0) for r in successful_results]
            
            # 시각화
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 최종 성능 vs 샘플 수
            ax1.plot(sample_sizes, final_accs, 'o-', color='blue', linewidth=2, markersize=8)
            ax1.set_title('Final Performance vs Target Sample Size')
            ax1.set_xlabel('Target Sample Size')
            ax1.set_ylabel('Target Accuracy (%)')
            ax1.grid(True, alpha=0.3)
            
            # 성능 개선 vs 샘플 수
            ax2.plot(sample_sizes, improvements, 'o-', color='green', linewidth=2, markersize=8)
            ax2.set_title('Performance Improvement vs Target Sample Size')
            ax2.set_xlabel('Target Sample Size')
            ax2.set_ylabel('Performance Improvement (%)')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            # 언러닝 횟수 vs 샘플 수
            ax3.bar(sample_sizes, unlearning_counts, color='orange', alpha=0.7)
            ax3.set_title('Unlearning Count vs Target Sample Size')
            ax3.set_xlabel('Target Sample Size')
            ax3.set_ylabel('Unlearning Count')
            ax3.grid(True, alpha=0.3)
            
            # 효율성 (개선/샘플수)
            efficiency = [imp/size*100 for imp, size in zip(improvements, sample_sizes)]
            ax4.plot(sample_sizes, efficiency, 'o-', color='purple', linewidth=2, markersize=8)
            ax4.set_title('Sample Efficiency (Improvement/Size × 100)')
            ax4.set_xlabel('Target Sample Size')
            ax4.set_ylabel('Efficiency')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "sample_sizes_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 샘플 수 시각화 실패: {e}")
    
    def _plot_hyperparameter_heatmap(self, viz_dir: Path) -> None:
        """하이퍼파라미터 히트맵"""
        try:
            with open(self.results_dir / "hyperparameter_results.json", 'r') as f:
                results = json.load(f)
            
            successful_results = [r for r in results if r.get('success', False)]
            
            if not successful_results:
                print("⚠️ 성공한 하이퍼파라미터 실험 결과가 없음")
                return
            
            # 데이터 준비
            df = pd.DataFrame(successful_results)
            pivot_final = df.pivot(index='beta_uncertainty', columns='lambda_utility', values='final_target_acc')
            pivot_improvement = df.pivot(index='beta_uncertainty', columns='lambda_utility', values='improvement')
            
            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 최종 성능 히트맵
            sns.heatmap(pivot_final, annot=True, cmap='Blues', ax=ax1, fmt='.2f')
            ax1.set_title('Hyperparameter Final Performance (%)')
            ax1.set_xlabel('λ_u (Utility Weight)')
            ax1.set_ylabel('β (Uncertainty Weight)')
            
            # 성능 개선 히트맵
            sns.heatmap(pivot_improvement, annot=True, cmap='RdYlGn', center=0, ax=ax2, fmt='.2f')
            ax2.set_title('Hyperparameter Performance Improvement (%)')
            ax2.set_xlabel('λ_u (Utility Weight)')
            ax2.set_ylabel('β (Uncertainty Weight)')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "hyperparameter_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 하이퍼파라미터 시각화 실패: {e}")
    
    def _plot_unlearning_effects(self, viz_dir: Path) -> None:
        """언러닝 효과 시각화"""
        try:
            # 모든 실험 결과에서 언러닝 효과 분석
            all_results = []
            
            for file_name in ["domain_pairs_results.json", "sample_sizes_results.json", "hyperparameter_results.json"]:
                file_path = self.results_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        results = json.load(f)
                        all_results.extend([r for r in results if r.get('success', False)])
            
            if not all_results:
                print("⚠️ 언러닝 효과 분석할 데이터 없음")
                return
            
            # 데이터 준비
            unlearning_counts = [r.get('unlearning_count', 0) for r in all_results]
            improvements = [r.get('improvement', 0) for r in all_results]
            
            # 언러닝 횟수별 그룹화
            unlearning_groups = {}
            for count, improvement in zip(unlearning_counts, improvements):
                if count not in unlearning_groups:
                    unlearning_groups[count] = []
                unlearning_groups[count].append(improvement)
            
            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 언러닝 횟수 vs 성능 개선 산점도
            ax1.scatter(unlearning_counts, improvements, alpha=0.6, s=50)
            ax1.set_title('Unlearning Count vs Performance Improvement')
            ax1.set_xlabel('Unlearning Count')
            ax1.set_ylabel('Performance Improvement (%)')
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)
            
            # 언러닝 횟수별 평균 개선량
            counts = sorted(unlearning_groups.keys())
            avg_improvements = [np.mean(unlearning_groups[count]) for count in counts]
            
            ax2.bar(counts, avg_improvements, color='lightgreen', alpha=0.7)
            ax2.set_title('Average Performance Improvement by Unlearning Count')
            ax2.set_xlabel('Unlearning Count')
            ax2.set_ylabel('Average Performance Improvement (%)')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "unlearning_effects.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 언러닝 효과 시각화 실패: {e}")
    
    def generate_summary_report(self) -> None:
        """종합 결과 리포트 생성"""
        print(f"\n📋 종합 결과 리포트 생성 중...")
        
        report = []
        report.append("# Office31 SDA-U 종합 실험 리포트\n")
        report.append(f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 각 실험별 요약
        for experiment_name, file_name in [
            ("도메인 쌍 실험", "domain_pairs_results.json"),
            ("샘플 수 실험", "sample_sizes_results.json"),
            ("하이퍼파라미터 실험", "hyperparameter_results.json")
        ]:
            file_path = self.results_dir / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    results = json.load(f)
                
                successful = [r for r in results if r.get('success', False)]
                total = len(results)
                success_rate = len(successful) / total * 100 if total > 0 else 0
                
                report.append(f"## {experiment_name}\n")
                report.append(f"- 총 실험: {total}개\n")
                report.append(f"- 성공: {len(successful)}개 ({success_rate:.1f}%)\n")
                
                if successful:
                    avg_improvement = np.mean([r.get('improvement', 0) for r in successful])
                    max_improvement = max([r.get('improvement', 0) for r in successful])
                    avg_unlearning = np.mean([r.get('unlearning_count', 0) for r in successful])
                    
                    report.append(f"- 평균 성능 개선: {avg_improvement:.2f}%\n")
                    report.append(f"- 최대 성능 개선: {max_improvement:.2f}%\n")
                    report.append(f"- 평균 언러닝 횟수: {avg_unlearning:.1f}회\n")
                
                report.append("\n")
        
        # 리포트 저장
        report_file = self.results_dir / "comprehensive_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"✅ 종합 리포트 생성 완료: {report_file}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Office31 종합 실험 도구')
    parser.add_argument('--experiment', type=str, 
                       choices=['domain_pairs', 'sample_sizes', 'hyperparams', 'all'],
                       default='all', help='실행할 실험 종류')
    parser.add_argument('--config', type=str, default='config.json', help='기본 설정 파일')
    parser.add_argument('--visualize', action='store_true', help='시각화 생성')
    
    args = parser.parse_args()
    
    print("🎯 Office31 종합 실험 도구 시작!")
    print("="*60)
    
    # 실험 러너 초기화
    runner = Office31ExperimentRunner(args.config)
    
    # 실험 실행
    if args.experiment == 'domain_pairs' or args.experiment == 'all':
        runner.run_domain_pairs_experiment()
    
    if args.experiment == 'sample_sizes' or args.experiment == 'all':
        runner.run_sample_sizes_experiment()
    
    if args.experiment == 'hyperparams' or args.experiment == 'all':
        runner.run_hyperparameter_experiment()
    
    # 시각화 및 리포트 생성
    if args.visualize or args.experiment == 'all':
        runner.create_visualizations()
    
    runner.generate_summary_report()
    
    print("\n🎉 Office31 종합 실험 완료!")
    print(f"📂 결과 확인: {runner.results_dir}")

if __name__ == "__main__":
    main() 