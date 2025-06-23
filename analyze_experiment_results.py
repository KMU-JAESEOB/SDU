# analyze_experiment_results.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from datetime import datetime

# 폰트 설정 - 시스템 호환성 개선
import matplotlib.font_manager as fm
import warnings

# 한글 폰트 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# 사용 가능한 폰트 확인 및 설정
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 우선순위 폰트 리스트 (한글 지원 + 영어 폰트)
font_candidates = [
    'Arial Unicode MS',  # macOS
    'NanumGothic',       # 나눔고딕
    'NanumBarunGothic',  # 나눔바른고딕  
    'DejaVu Sans',       # 기본 영어 폰트
    'Arial',             # Windows 기본
    'Helvetica',         # macOS 기본
    'Liberation Sans',   # Linux
    'sans-serif'         # 폴백
]

# 사용 가능한 첫 번째 폰트 선택
selected_font = 'DejaVu Sans'  # 기본값
for font in font_candidates:
    if font in available_fonts:
        selected_font = font
        break

plt.rcParams['font.family'] = [selected_font]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

print(f"🎨 사용 폰트: {selected_font}")

class ExperimentResultAnalyzer:
    """실험 결과 분석 및 시각화 클래스"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.office_home_results = {}
        self.office31_results = {}
        self.other_results = {}
        
        print("📊 실험 결과 분석기 초기화")
        print(f"📁 결과 디렉토리: {self.results_dir}")
        
    def load_all_results(self):
        """모든 실험 결과 로드"""
        print("📦 실험 결과 로딩 중...")
        
        # Office-Home 결과 로드
        self._load_officehome_results()
        
        # 기타 JSON 결과 로드
        self._load_json_results()
        
        print("✅ 모든 결과 로딩 완료!")
        
    def _load_officehome_results(self):
        """Office-Home 결과 로드"""
        officehome_dir = self.results_dir / 'officehome'
        
        if not officehome_dir.exists():
            print("⚠️ Office-Home 결과 디렉토리가 없습니다.")
            return
            
        # JSON 결과 파일 로드
        json_file = officehome_dir / 'officehome_full_results.json'
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                self.office_home_results = json.load(f)
                
        # CSV 결과 파일 로드
        csv_file = officehome_dir / 'officehome_performance_summary.csv'
        if csv_file.exists():
            self.office_home_df = pd.read_csv(csv_file)
            
        # 로그 파일에서 세부 정보 추출
        self._parse_officehome_logs(officehome_dir)
        
        print(f"✅ Office-Home 결과 로드 완료: {len(self.office_home_results)}개 실험")
        
    def _parse_officehome_logs(self, log_dir):
        """Office-Home 로그 파일 파싱"""
        log_files = list(log_dir.glob('*_log.txt'))
        detailed_results = {}
        
        for log_file in log_files:
            domain_pair = log_file.stem.replace('_log', '')
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 정확도 추출
                acc_matches = re.findall(r'정확도: ([\d.]+)%', content)
                if acc_matches:
                    detailed_results[domain_pair] = {
                        'accuracies': [float(acc) for acc in acc_matches],
                        'final_accuracy': float(acc_matches[-1]) if acc_matches else 0,
                        'initial_accuracy': float(acc_matches[0]) if acc_matches else 0,
                        'improvement': float(acc_matches[-1]) - float(acc_matches[0]) if len(acc_matches) >= 2 else 0
                    }
                    
            except Exception as e:
                print(f"⚠️ {log_file} 파싱 오류: {e}")
                
        self.office_home_detailed = detailed_results
        
    def _load_json_results(self):
        """기타 JSON 결과 파일들 로드"""
        json_files = list(self.results_dir.glob('*.json'))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.other_results[json_file.stem] = data
            except Exception as e:
                print(f"⚠️ {json_file} 로드 오류: {e}")
                
    def create_officehome_comparison_table(self):
        """Office-Home 도메인 비교 테이블 생성"""
        print("\n📊 Office-Home 도메인 비교 테이블 생성 중...")
        
        if not self.office_home_detailed:
            print("❌ Office-Home 상세 결과가 없습니다.")
            return None
            
        # 데이터 준비
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        table_data = []
        
        for source in domains:
            row = [source]
            for target in domains:
                if source == target:
                    row.append('-')
                else:
                    # 도메인 페어 이름 생성
                    pair_name = f"{source}2{target}"
                    if pair_name in self.office_home_detailed:
                        final_acc = self.office_home_detailed[pair_name]['final_accuracy']
                        improvement = self.office_home_detailed[pair_name]['improvement']
                        row.append(f"{final_acc:.1f}% (+{improvement:.1f})")
                    else:
                        row.append('N/A')
            table_data.append(row)
            
        # DataFrame 생성
        columns = ['Source\\Target'] + domains
        df = pd.DataFrame(table_data, columns=columns)
        
        print("✅ Office-Home 비교 테이블:")
        print(df.to_string(index=False))
        
        # CSV로 저장
        df.to_csv(self.results_dir / 'officehome_comparison_table.csv', index=False)
        print(f"💾 테이블 저장: {self.results_dir / 'officehome_comparison_table.csv'}")
        
        return df
        
    def create_performance_heatmap(self):
        """성능 히트맵 생성"""
        print("\n🔥 성능 히트맵 생성 중...")
        
        if not self.office_home_detailed:
            print("❌ Office-Home 상세 결과가 없습니다.")
            return
            
        # 히트맵 데이터 준비
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        heatmap_data = np.zeros((4, 4))
        
        for i, source in enumerate(domains):
            for j, target in enumerate(domains):
                if source != target:
                    pair_name = f"{source}2{target}"
                    if pair_name in self.office_home_detailed:
                        heatmap_data[i, j] = self.office_home_detailed[pair_name]['final_accuracy']
                else:
                    heatmap_data[i, j] = np.nan
                    
        # 히트맵 생성
        plt.figure(figsize=(10, 8))
        mask = np.isnan(heatmap_data)
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.1f',
                   xticklabels=domains,
                   yticklabels=domains,
                   cmap='YlOrRd',
                   mask=mask,
                   cbar_kws={'label': 'Final Accuracy (%)'})
                   
        plt.title('Office-Home Domain Adaptation Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Target Domain', fontsize=12)
        plt.ylabel('Source Domain', fontsize=12)
        plt.tight_layout()
        
        # 저장
        plt.savefig(self.results_dir / 'officehome_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'officehome_performance_heatmap.pdf', bbox_inches='tight')
        print(f"💾 히트맵 저장: {self.results_dir / 'officehome_performance_heatmap.png'}")
        plt.show()
        
    def create_improvement_analysis(self):
        """개선도 분석 차트"""
        print("\n📈 개선도 분석 차트 생성 중...")
        
        if not self.office_home_detailed:
            print("❌ Office-Home 상세 결과가 없습니다.")
            return
            
        # 개선도 데이터 수집
        improvements = []
        domain_pairs = []
        
        for pair_name, data in self.office_home_detailed.items():
            improvements.append(data['improvement'])
            domain_pairs.append(pair_name.replace('2', ' → '))
            
        # 차트 생성
        plt.figure(figsize=(15, 8))
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        bars = plt.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        
        plt.title('SDA-U Performance Improvement by Domain Pair', fontsize=16, fontweight='bold')
        plt.xlabel('Domain Adaptation Task', fontsize=12)
        plt.ylabel('Accuracy Improvement (%)', fontsize=12)
        plt.xticks(range(len(domain_pairs)), domain_pairs, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # 값 표시
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if imp > 0 else -0.3), 
                    f'{imp:.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        plt.savefig(self.results_dir / 'officehome_improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'officehome_improvement_analysis.pdf', bbox_inches='tight')
        print(f"💾 개선도 차트 저장: {self.results_dir / 'officehome_improvement_analysis.png'}")
        plt.show()
        
    def create_latex_table(self):
        """LaTeX 형식 테이블 생성 (논문용)"""
        print("\n📝 LaTeX 테이블 생성 중...")
        
        if not self.office_home_detailed:
            print("❌ Office-Home 상세 결과가 없습니다.")
            return
            
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        
        latex_content = """
\\begin{table*}[t]
\\centering
\\caption{Office-Home Domain Adaptation Results using SDA-U Framework}
\\label{tab:officehome_results}
\\begin{tabular}{l|cccc}
\\toprule
\\textbf{Source $\\backslash$ Target} & \\textbf{Art} & \\textbf{Clipart} & \\textbf{Product} & \\textbf{Real World} \\\\
\\midrule
"""
        
        for source in domains:
            row = f"\\textbf{{{source}}}"
            for target in domains:
                if source == target:
                    row += " & -"
                else:
                    pair_name = f"{source}2{target}"
                    if pair_name in self.office_home_detailed:
                        final_acc = self.office_home_detailed[pair_name]['final_accuracy']
                        improvement = self.office_home_detailed[pair_name]['improvement']
                        if improvement > 0:
                            row += f" & \\textbf{{{final_acc:.1f}\\%}} (+{improvement:.1f})"
                        else:
                            row += f" & {final_acc:.1f}\\% ({improvement:.1f})"
                    else:
                        row += " & N/A"
            row += " \\\\\n"
            latex_content += row
            
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
        
        # 저장
        with open(self.results_dir / 'officehome_latex_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_content)
            
        print("✅ LaTeX 테이블 생성 완료!")
        print(f"💾 저장 위치: {self.results_dir / 'officehome_latex_table.tex'}")
        print("\n📋 LaTeX 테이블 내용:")
        print(latex_content)
        
    def create_summary_statistics(self):
        """요약 통계 생성"""
        print("\n📊 요약 통계 생성 중...")
        
        if not self.office_home_detailed:
            print("❌ Office-Home 상세 결과가 없습니다.")
            return
            
        # 통계 계산
        all_final_acc = [data['final_accuracy'] for data in self.office_home_detailed.values()]
        all_improvements = [data['improvement'] for data in self.office_home_detailed.values()]
        
        stats = {
            'Total Experiments': len(self.office_home_detailed),
            'Average Final Accuracy': np.mean(all_final_acc),
            'Max Final Accuracy': np.max(all_final_acc),
            'Min Final Accuracy': np.min(all_final_acc),
            'Average Improvement': np.mean(all_improvements),
            'Max Improvement': np.max(all_improvements),
            'Min Improvement': np.min(all_improvements),
            'Positive Improvements': sum(1 for imp in all_improvements if imp > 0),
            'Negative Improvements': sum(1 for imp in all_improvements if imp < 0)
        }
        
        print("✅ Office-Home 실험 요약 통계:")
        print("="*50)
        for key, value in stats.items():
            if 'Accuracy' in key or 'Improvement' in key:
                if 'Average' in key or 'Max' in key or 'Min' in key:
                    print(f"{key:25}: {value:.2f}%")
                else:
                    print(f"{key:25}: {value}")
            else:
                print(f"{key:25}: {value}")
        print("="*50)
        
        # JSON으로 저장
        with open(self.results_dir / 'summary_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        return stats
        
    def generate_full_report(self):
        """전체 보고서 생성"""
        print("\n📋 전체 실험 보고서 생성 중...")
        
        # 모든 분석 수행
        comparison_table = self.create_officehome_comparison_table()
        self.create_performance_heatmap()
        self.create_improvement_analysis()
        self.create_latex_table()
        stats = self.create_summary_statistics()
        
        # 보고서 작성
        report_content = f"""
# SDA-U 실험 결과 분석 보고서

## 생성 일시
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 실험 개요
- **데이터셋**: Office-Home (65 classes, 4 domains)
- **총 실험 수**: {stats['Total Experiments']}개
- **도메인**: Art, Clipart, Product, Real World

## 주요 결과

### 성능 요약
- **평균 최종 정확도**: {stats['Average Final Accuracy']:.2f}%
- **최고 정확도**: {stats['Max Final Accuracy']:.2f}%
- **최저 정확도**: {stats['Min Final Accuracy']:.2f}%

### 개선도 분석
- **평균 개선도**: {stats['Average Improvement']:.2f}%
- **최대 개선도**: {stats['Max Improvement']:.2f}%
- **성능 향상 실험**: {stats['Positive Improvements']}개
- **성능 저하 실험**: {stats['Negative Improvements']}개

## 생성된 파일들
1. `officehome_comparison_table.csv` - 도메인 비교 테이블
2. `officehome_performance_heatmap.png/pdf` - 성능 히트맵
3. `officehome_improvement_analysis.png/pdf` - 개선도 분석
4. `officehome_latex_table.tex` - 논문용 LaTeX 테이블
5. `summary_statistics.json` - 요약 통계

## 사용 방법
- PNG/PDF 파일: 논문 Figure로 직접 사용 가능
- LaTeX 테이블: 논문에 복사하여 사용
- CSV 파일: 추가 분석 또는 스프레드시트에서 사용

## 논문 작성 시 참고사항
1. 히트맵은 전체 성능 비교에 효과적
2. 개선도 차트는 SDA-U의 효과를 보여줌
3. LaTeX 테이블은 정확한 수치 제공
"""

        # 보고서 저장
        with open(self.results_dir / 'experiment_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print("✅ 전체 보고서 생성 완료!")
        print(f"💾 보고서 위치: {self.results_dir / 'experiment_analysis_report.md'}")
        
        print("\n🎉 모든 분석 완료! 논문 작성에 필요한 모든 자료가 준비되었습니다.")

def main():
    """메인 함수"""
    print("📊 SDA-U 실험 결과 분석기")
    print("="*60)
    
    # 분석기 생성
    analyzer = ExperimentResultAnalyzer()
    
    # 결과 로드
    analyzer.load_all_results()
    
    # 전체 보고서 생성
    analyzer.generate_full_report()

if __name__ == "__main__":
    main() 