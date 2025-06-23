# monitor_experiment.py
"""
🔍 Office-Home 실험 진행 상황 모니터링
- 실시간 로그 출력
- 진행률 추정
- 성능 지표 추적
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import re
import json
from datetime import datetime

class ExperimentMonitor:
    """실험 진행 상황 모니터링 클래스"""
    
    def __init__(self, log_file_path=None):
        self.log_file_path = log_file_path
        self.results_dir = Path('results/officehome')
        self.start_time = time.time()
        
        # 단계별 진행률 패턴
        self.progress_patterns = {
            'data_loading': r'데이터 로딩|데이터셋.*로딩|Office-Home.*로딩',
            'model_creation': r'모델 생성|ResNet.*로드|분류 레이어',
            'source_training': r'소스 도메인.*훈련|에포크.*\d+/\d+',
            'target_evaluation': r'타겟 도메인.*평가|초기 평가',
            'influence_computation': r'영향도.*계산|influence.*score',
            'hybrid_scoring': r'하이브리드.*스코어링|hybrid.*score',
            'unlearning': r'언러닝.*수행|DOS.*unlearning',
            'final_evaluation': r'최종.*평가|final.*evaluation',
            'results_saving': r'결과.*저장|Results saved'
        }
        
        print("🔍 Office-Home 실험 모니터 시작!")
        print(f"📁 결과 디렉토리: {self.results_dir}")
    
    def find_active_log_file(self):
        """활성 로그 파일 찾기"""
        
        if self.log_file_path and Path(self.log_file_path).exists():
            return Path(self.log_file_path)
        
        # 가장 최근 로그 파일 찾기
        log_files = list(self.results_dir.glob('*_log.txt'))
        
        if not log_files:
            print("❌ 활성 로그 파일을 찾을 수 없습니다.")
            print("💡 실험이 아직 시작되지 않았거나 로그 파일이 생성되지 않았을 수 있습니다.")
            return None
        
        # 가장 최근 수정된 파일
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        print(f"📝 활성 로그 파일: {latest_log}")
        return latest_log
    
    def analyze_progress(self, log_content):
        """로그 내용 분석하여 진행률 계산"""
        
        progress_info = {
            'current_stage': 'unknown',
            'progress_percent': 0,
            'stage_details': {},
            'estimated_time_remaining': 'unknown'
        }
        
        lines = log_content.split('\n')
        
        # 각 단계별 진행 상황 확인
        for stage, pattern in self.progress_patterns.items():
            matches = [line for line in lines if re.search(pattern, line, re.IGNORECASE)]
            
            if matches:
                progress_info['stage_details'][stage] = {
                    'completed': True,
                    'last_message': matches[-1].strip(),
                    'count': len(matches)
                }
            else:
                progress_info['stage_details'][stage] = {
                    'completed': False,
                    'last_message': '',
                    'count': 0
                }
        
        # 현재 단계 결정
        stage_order = list(self.progress_patterns.keys())
        completed_stages = [s for s in stage_order if progress_info['stage_details'][s]['completed']]
        
        if completed_stages:
            current_stage_idx = len(completed_stages) - 1
            if current_stage_idx < len(stage_order) - 1:
                progress_info['current_stage'] = stage_order[current_stage_idx + 1]
            else:
                progress_info['current_stage'] = 'completed'
            
            # 진행률 계산 (각 단계를 동일 비중으로 가정)
            progress_info['progress_percent'] = (len(completed_stages) / len(stage_order)) * 100
        
        # 에포크 진행률 상세 분석
        epoch_matches = re.findall(r'에포크\s+(\d+)/(\d+)', log_content)
        if epoch_matches:
            current_epoch, total_epochs = map(int, epoch_matches[-1])
            epoch_progress = (current_epoch / total_epochs) * 100
            
            # 소스 훈련 단계에서 에포크 진행률 반영
            if progress_info['current_stage'] == 'source_training':
                stage_progress = (1 / len(stage_order)) * (epoch_progress / 100) * 100
                progress_info['progress_percent'] += stage_progress
            
            progress_info['epoch_info'] = {
                'current': current_epoch,
                'total': total_epochs,
                'progress': epoch_progress
            }
        
        return progress_info
    
    def extract_performance_metrics(self, log_content):
        """성능 지표 추출"""
        
        metrics = {}
        
        # 정확도 추출
        acc_matches = re.findall(r'정확도[:\s]*(\d+\.?\d*)%?', log_content)
        if acc_matches:
            metrics['accuracies'] = [float(acc) for acc in acc_matches]
            metrics['latest_accuracy'] = float(acc_matches[-1])
        
        # 손실 추출
        loss_matches = re.findall(r'손실[:\s]*(\d+\.?\d+)', log_content)
        if loss_matches:
            metrics['losses'] = [float(loss) for loss in loss_matches]
            metrics['latest_loss'] = float(loss_matches[-1])
        
        # 영향도 점수 추출
        influence_matches = re.findall(r'영향도.*점수[:\s]*(\d+\.?\d+)', log_content)
        if influence_matches:
            metrics['influence_scores'] = [float(score) for score in influence_matches]
        
        return metrics
    
    def format_progress_display(self, progress_info, metrics, log_content):
        """진행 상황 디스플레이 포맷팅"""
        
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print(f"🔍 Office-Home 실험 진행 상황 ({datetime.now().strftime('%H:%M:%S')})")
        print("="*80)
        
        # 전체 진행률
        progress_bar_length = 50
        filled_length = int(progress_bar_length * progress_info['progress_percent'] / 100)
        bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
        
        print(f"📊 전체 진행률: |{bar}| {progress_info['progress_percent']:.1f}%")
        print(f"⏱️ 경과 시간: {elapsed_time/60:.1f}분")
        
        # 현재 단계
        stage_names = {
            'data_loading': '📦 데이터 로딩',
            'model_creation': '🏗️ 모델 생성',
            'source_training': '🏋️ 소스 도메인 훈련',
            'target_evaluation': '📊 타겟 도메인 평가',
            'influence_computation': '🎯 영향도 계산',
            'hybrid_scoring': '🔢 하이브리드 스코어링',
            'unlearning': '🔄 DOS 언러닝',
            'final_evaluation': '📈 최종 평가',
            'results_saving': '💾 결과 저장',
            'completed': '✅ 완료',
            'unknown': '❓ 알 수 없음'
        }
        
        current_stage_name = stage_names.get(progress_info['current_stage'], progress_info['current_stage'])
        print(f"🎯 현재 단계: {current_stage_name}")
        
        # 에포크 정보
        if 'epoch_info' in progress_info:
            epoch_info = progress_info['epoch_info']
            print(f"📚 에포크 진행: {epoch_info['current']}/{epoch_info['total']} ({epoch_info['progress']:.1f}%)")
        
        # 최신 성능 지표
        if metrics:
            if 'latest_accuracy' in metrics:
                print(f"🎯 최신 정확도: {metrics['latest_accuracy']:.2f}%")
            if 'latest_loss' in metrics:
                print(f"📉 최신 손실: {metrics['latest_loss']:.4f}")
        
        # 단계별 상세 진행 상황
        print(f"\n📋 단계별 진행 상황:")
        for stage, name in stage_names.items():
            if stage in ['completed', 'unknown']:
                continue
                
            if stage in progress_info['stage_details']:
                details = progress_info['stage_details'][stage]
                status = "✅" if details['completed'] else "⏳"
                print(f"   {status} {name}")
                
                if details['completed'] and details['last_message']:
                    # 메시지가 너무 길면 줄임
                    msg = details['last_message']
                    if len(msg) > 80:
                        msg = msg[:77] + "..."
                    print(f"      💬 {msg}")
        
        # 시간 추정
        if progress_info['progress_percent'] > 10:
            estimated_total_time = elapsed_time / (progress_info['progress_percent'] / 100)
            remaining_time = estimated_total_time - elapsed_time
            print(f"\n⏰ 예상 잔여 시간: {remaining_time/60:.1f}분")
            print(f"⏰ 예상 완료 시간: {(time.time() + remaining_time)}")
        
        print("="*80)
    
    def monitor_realtime(self, update_interval=10):
        """실시간 모니터링"""
        
        log_file = self.find_active_log_file()
        if not log_file:
            return
        
        print(f"🚀 실시간 모니터링 시작 (업데이트 간격: {update_interval}초)")
        print(f"📝 모니터링 파일: {log_file}")
        print("💡 종료하려면 Ctrl+C를 누르세요.")
        
        last_size = 0
        
        try:
            while True:
                if log_file.exists():
                    current_size = log_file.stat().st_size
                    
                    # 파일이 업데이트되었거나 처음 읽기
                    if current_size != last_size:
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                log_content = f.read()
                            
                            progress_info = self.analyze_progress(log_content)
                            metrics = self.extract_performance_metrics(log_content)
                            
                            # 콘솔 클리어 (선택사항)
                            os.system('cls' if os.name == 'nt' else 'clear')
                            
                            self.format_progress_display(progress_info, metrics, log_content)
                            
                            last_size = current_size
                            
                            # 완료 확인
                            if progress_info['current_stage'] == 'completed':
                                print("\n🎉 실험 완료!")
                                break
                                
                        except Exception as e:
                            print(f"⚠️ 로그 파일 읽기 오류: {e}")
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print(f"\n👋 모니터링 종료")
    
    def show_latest_logs(self, num_lines=20):
        """최신 로그 출력"""
        
        log_file = self.find_active_log_file()
        if not log_file:
            return
        
        print(f"📄 최신 로그 ({num_lines}줄):")
        print("-" * 80)
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
                
                for line in recent_lines:
                    print(line.rstrip())
                    
        except Exception as e:
            print(f"❌ 로그 파일 읽기 실패: {e}")

def main():
    """메인 실행 함수"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Office-Home 실험 모니터링')
    parser.add_argument('--log-file', type=str, help='모니터링할 로그 파일 경로')
    parser.add_argument('--interval', type=int, default=10, help='업데이트 간격 (초)')
    parser.add_argument('--tail', type=int, help='최신 N줄만 출력')
    parser.add_argument('--realtime', action='store_true', help='실시간 모니터링')
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor(args.log_file)
    
    if args.tail:
        monitor.show_latest_logs(args.tail)
    elif args.realtime:
        monitor.monitor_realtime(args.interval)
    else:
        # 기본: 현재 상태 한 번 출력
        log_file = monitor.find_active_log_file()
        if log_file and log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            progress_info = monitor.analyze_progress(log_content)
            metrics = monitor.extract_performance_metrics(log_content)
            monitor.format_progress_display(progress_info, metrics, log_content)
        else:
            print("❌ 활성 로그 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main() 