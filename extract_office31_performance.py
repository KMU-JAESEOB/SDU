#!/usr/bin/env python3
# extract_office31_performance.py - Office-31 성능 추출 스크립트

"""
🏢 Office-31 성능 추출 도구

Office-31 실험 결과에서 핵심 성능 지표를 추출하고 정리합니다.
"""

import json
import os
from pathlib import Path

def extract_single_result(result_file):
    """단일 결과 파일에서 성능 지표 추출"""
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 실험 정보
        exp_info = data.get('experiment_info', {})
        final_perf = data.get('final_performance', {})
        
        # 핵심 지표 추출
        performance = {
            'experiment_name': exp_info.get('experiment_name', 'Unknown'),
            'source_dataset': exp_info.get('source_dataset', 'Unknown'),
            'target_dataset': exp_info.get('target_dataset', 'Unknown'),
            'description': exp_info.get('description', 'Unknown'),
            'source_accuracy': final_perf.get('source_accuracy', 0),
            'target_subset_accuracy': final_perf.get('target_subset_accuracy', 0),
            'full_target_accuracy': final_perf.get('full_target_accuracy', 0),
            'best_target_accuracy': exp_info.get('best_target_accuracy', 0),
            'improvement': final_perf.get('improvement_over_baseline', 0),
            'execution_time': exp_info.get('execution_time_seconds', 0)
        }
        
        return performance
        
    except Exception as e:
        print(f"⚠️ {result_file} 처리 실패: {e}")
        return None

def print_performance_table(performances):
    """성능 결과를 테이블 형태로 출력"""
    
    if not performances:
        print("❌ 표시할 성능 데이터가 없습니다.")
        return
    
    print("\n" + "="*120)
    print("📊 Office-31 도메인 적응 성능 결과")
    print("="*120)
    
    # 테이블 헤더
    header = f"{'실험명':<15} {'도메인 조합':<25} {'소스 정확도':<12} {'타겟 정확도':<12} {'최고 정확도':<12} {'개선도':<10} {'실행시간':<10}"
    print(header)
    print("-" * 120)
    
    # 각 실험 결과
    total_time = 0
    best_result = None
    best_accuracy = 0
    
    for perf in performances:
        source_short = perf['source_dataset'].replace('Office31_', '')
        target_short = perf['target_dataset'].replace('Office31_', '')
        domain_combo = f"{source_short}→{target_short}"
        
        row = (f"{perf['experiment_name']:<15} "
               f"{domain_combo:<25} "
               f"{perf['source_accuracy']:<11.1f}% "
               f"{perf['full_target_accuracy']:<11.1f}% "
               f"{perf['best_target_accuracy']:<11.1f}% "
               f"{perf['improvement']:<9.1f}% "
               f"{perf['execution_time']:<9.0f}s")
        
        print(row)
        
        total_time += perf['execution_time']
        if perf['best_target_accuracy'] > best_accuracy:
            best_accuracy = perf['best_target_accuracy']
            best_result = perf
    
    print("-" * 120)
    
    # 요약 통계
    if best_result:
        print(f"🏆 최고 성능: {best_result['experiment_name']} ({best_accuracy:.1f}%)")
    print(f"⏱️ 총 실행시간: {total_time:.0f}초 ({total_time/60:.1f}분)")
    print(f"📊 평균 타겟 정확도: {sum(p['full_target_accuracy'] for p in performances)/len(performances):.1f}%")

def save_performance_csv(performances, output_file='office31_performance.csv'):
    """성능 결과를 CSV 파일로 저장"""
    
    if not performances:
        print("❌ 저장할 성능 데이터가 없습니다.")
        return
    
    try:
        # CSV 헤더
        csv_lines = [
            "실험명,소스도메인,타겟도메인,설명,소스정확도(%),타겟서브셋정확도(%),전체타겟정확도(%),최고타겟정확도(%),개선도(%),실행시간(초)"
        ]
        
        # 데이터 행들
        for perf in performances:
            line = (f"{perf['experiment_name']},"
                   f"{perf['source_dataset']},"
                   f"{perf['target_dataset']},"
                   f"\"{perf['description']}\","
                   f"{perf['source_accuracy']:.2f},"
                   f"{perf['target_subset_accuracy']:.2f},"
                   f"{perf['full_target_accuracy']:.2f},"
                   f"{perf['best_target_accuracy']:.2f},"
                   f"{perf['improvement']:.2f},"
                   f"{perf['execution_time']:.1f}")
            csv_lines.append(line)
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(csv_lines))
        
        print(f"💾 성능 결과 CSV 저장: {output_file}")
        
    except Exception as e:
        print(f"❌ CSV 저장 실패: {e}")

def main():
    """메인 함수"""
    
    print("🏢 Office-31 성능 추출 도구")
    print("=" * 50)
    
    # 결과 디렉토리 확인
    results_dir = Path('office31_results')
    if not results_dir.exists():
        # 기본 results 디렉토리도 확인
        results_dir = Path('results')
        if not results_dir.exists():
            print("❌ 결과 디렉토리를 찾을 수 없습니다.")
            print("   다음 디렉토리를 확인하세요:")
            print("   - office31_results/")
            print("   - results/")
            return
    
    # 결과 파일 찾기
    result_files = []
    
    # Office-31 전용 결과 파일들
    office31_files = list(results_dir.glob('*2*_results.json'))  # Amazon2Webcam_results.json 등
    result_files.extend(office31_files)
    
    # 일반 결과 파일들
    general_files = list(results_dir.glob('*results*.json'))
    result_files.extend(general_files)
    
    # 중복 제거
    result_files = list(set(result_files))
    
    if not result_files:
        print(f"❌ {results_dir}에서 결과 파일을 찾을 수 없습니다.")
        print("   다음 패턴의 파일을 찾고 있습니다:")
        print("   - *2*_results.json (예: Amazon2Webcam_results.json)")
        print("   - *results*.json (예: sda_u_comprehensive_results.json)")
        return
    
    print(f"📊 발견된 결과 파일: {len(result_files)}개")
    for f in result_files:
        print(f"   - {f.name}")
    
    # 성능 데이터 추출
    performances = []
    for result_file in result_files:
        perf = extract_single_result(result_file)
        if perf:
            performances.append(perf)
    
    if not performances:
        print("❌ 유효한 성능 데이터를 찾을 수 없습니다.")
        return
    
    print(f"\n✅ {len(performances)}개 실험 결과 처리 완료!")
    
    # 성능 테이블 출력
    print_performance_table(performances)
    
    # CSV 저장
    save_performance_csv(performances)
    
    # 상세 분석 옵션
    print(f"\n{'='*50}")
    print("📋 추가 분석 옵션:")
    print("1. 도메인별 성능 분석")
    print("2. 상세 결과 보기")
    
    try:
        choice = input("\n선택 (1-2, 또는 Enter로 종료): ").strip()
        
        if choice == "1":
            # 도메인별 성능 분석
            print("\n📊 도메인별 성능 분석:")
            print("-" * 40)
            
            # 소스 도메인별 평균 성능
            source_stats = {}
            for perf in performances:
                source = perf['source_dataset'].replace('Office31_', '')
                if source not in source_stats:
                    source_stats[source] = []
                source_stats[source].append(perf['full_target_accuracy'])
            
            print("📤 소스 도메인별 평균 타겟 정확도:")
            for source, accuracies in source_stats.items():
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"   {source}: {avg_acc:.1f}% (실험 {len(accuracies)}개)")
            
            # 타겟 도메인별 평균 성능
            target_stats = {}
            for perf in performances:
                target = perf['target_dataset'].replace('Office31_', '')
                if target not in target_stats:
                    target_stats[target] = []
                target_stats[target].append(perf['full_target_accuracy'])
            
            print("\n📥 타겟 도메인별 평균 정확도:")
            for target, accuracies in target_stats.items():
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"   {target}: {avg_acc:.1f}% (실험 {len(accuracies)}개)")
        
        elif choice == "2":
            # 상세 결과 보기
            print("\n📋 상세 실험 결과:")
            for i, perf in enumerate(performances, 1):
                print(f"\n{i}. {perf['experiment_name']} ({perf['description']})")
                print(f"   📤 소스 도메인 정확도: {perf['source_accuracy']:.2f}%")
                print(f"   📥 타겟 서브셋 정확도: {perf['target_subset_accuracy']:.2f}%")
                print(f"   🎯 전체 타겟 정확도: {perf['full_target_accuracy']:.2f}%")
                print(f"   🏆 최고 타겟 정확도: {perf['best_target_accuracy']:.2f}%")
                print(f"   📈 성능 개선도: {perf['improvement']:.2f}%")
                print(f"   ⏱️ 실행 시간: {perf['execution_time']:.1f}초")
        
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main() 