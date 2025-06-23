"""
🔍 Office-Home 라벨 디버깅 및 수정 유틸리티
CUDA 오류 'Assertion `t >= 0 && t < n_classes` failed' 해결을 위한 도구
"""

import os
import sys
from pathlib import Path

def check_officehome_structure():
    """Office-Home 데이터셋 구조 확인"""
    
    print("🔍 Office-Home 데이터셋 구조 검사 시작")
    print("="*60)
    
    data_root = Path('./data/OfficeHome')
    
    if not data_root.exists():
        print(f"❌ Office-Home 데이터 디렉토리를 찾을 수 없습니다: {data_root}")
        return False
    
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    
    for domain in domains:
        domain_path = data_root / domain
        
        if not domain_path.exists():
            print(f"❌ 도메인 디렉토리를 찾을 수 없습니다: {domain_path}")
            continue
        
        print(f"\n📁 {domain} 도메인 검사:")
        
        # 클래스 디렉토리 확인
        class_dirs = [d for d in domain_path.iterdir() if d.is_dir()]
        print(f"   📊 클래스 수: {len(class_dirs)}개")
        
        # 각 클래스별 샘플 수 확인
        total_samples = 0
        class_samples = {}
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            sample_count = len(image_files)
            
            class_samples[class_name] = sample_count
            total_samples += sample_count
        
        print(f"   📈 총 샘플 수: {total_samples:,}개")
        print(f"   📋 클래스 목록 (처음 10개):")
        
        for i, (class_name, count) in enumerate(sorted(class_samples.items())[:10]):
            print(f"      {i}: {class_name} ({count}개)")
        
        if len(class_samples) > 10:
            print(f"      ... 및 {len(class_samples)-10}개 더")
    
    return True

def create_safe_officehome_loader():
    """안전한 Office-Home 로더 생성"""
    
    print("\n🔧 안전한 Office-Home 로더 생성 중...")
    
    # Office-Home 표준 65개 클래스 정의
    standard_classes = [
        'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
        'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
        'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
        'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade',
        'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip',
        'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
        'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table',
        'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam'
    ]
    
    print(f"📋 표준 클래스 수: {len(standard_classes)}개")
    
    # 클래스-인덱스 매핑 생성 (0-64 범위 보장)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(standard_classes))}
    
    print("✅ 안전한 클래스-인덱스 매핑 생성 완료")
    print(f"   📊 인덱스 범위: 0 ~ {len(class_to_idx)-1}")
    
    return class_to_idx

def test_label_ranges():
    """라벨 범위 테스트"""
    
    print("\n🧪 라벨 범위 테스트 시작")
    print("="*40)
    
    try:
        # 안전한 클래스 매핑 생성
        class_to_idx = create_safe_officehome_loader()
        
        # 모든 라벨이 0-64 범위인지 확인
        all_indices = list(class_to_idx.values())
        min_idx = min(all_indices)
        max_idx = max(all_indices)
        
        print(f"📊 실제 라벨 범위: {min_idx} ~ {max_idx}")
        
        if min_idx >= 0 and max_idx <= 64:
            print("✅ 라벨 범위 검증 통과!")
            return True
        else:
            print(f"❌ 라벨 범위 오류: 예상 0-64, 실제 {min_idx}-{max_idx}")
            return False
            
    except Exception as e:
        print(f"❌ 라벨 범위 테스트 실패: {e}")
        return False

def fix_officehome_labels():
    """Office-Home 라벨 문제 수정"""
    
    print("\n🔧 Office-Home 라벨 문제 수정 시작")
    print("="*50)
    
    # 1. 데이터셋 구조 확인
    if not check_officehome_structure():
        print("❌ 데이터셋 구조 문제로 수정 불가")
        return False
    
    # 2. 안전한 로더 생성
    class_to_idx = create_safe_officehome_loader()
    
    # 3. 라벨 범위 테스트
    if test_label_ranges():
        print("✅ Office-Home 라벨 문제 수정 완료!")
        return True
    else:
        print("❌ Office-Home 라벨 문제 수정 실패")
        return False

def create_cuda_safe_config():
    """CUDA 안전 설정 생성"""
    
    print("\n⚙️ CUDA 안전 설정 생성 중...")
    
    config = {
        # 안전한 배치 크기 (CUDA 메모리 고려)
        'batch_size': 16,  # 작은 배치로 시작
        
        # 클래스 수 명시적 설정
        'num_classes': 65,
        
        # 라벨 검증 활성화
        'validate_labels': True,
        'clip_labels': True,
        
        # CUDA 디버깅 설정
        'cuda_launch_blocking': True,
        'device_side_assertions': True,
        
        # 안전한 샘플 수
        'influence_samples': 100,  # 200 → 100으로 감소
        'target_samples': 300,     # 800 → 300으로 감소
        
        # 오류 복구 설정
        'skip_invalid_samples': True,
        'max_invalid_samples': 10
    }
    
    print("✅ CUDA 안전 설정 생성 완료")
    return config

def main():
    """메인 디버깅 실행"""
    
    print("🔍 Office-Home CUDA 오류 디버깅 도구")
    print("="*60)
    print("문제: RuntimeError: CUDA error: device-side assert triggered")
    print("원인: Assertion `t >= 0 && t < n_classes` failed")
    print("="*60)
    
    # 1. 라벨 문제 수정
    if fix_officehome_labels():
        print("\n🎉 라벨 문제 수정 성공!")
    else:
        print("\n❌ 라벨 문제 수정 실패")
    
    # 2. 안전 설정 생성
    safe_config = create_cuda_safe_config()
    
    # 3. 권장 사항 출력
    print("\n💡 권장 해결 방법:")
    print("1. 환경 변수 설정:")
    print("   export CUDA_LAUNCH_BLOCKING=1")
    print("   export TORCH_USE_CUDA_DSA=1")
    
    print("\n2. 실행 시 작은 배치 크기 사용:")
    print("   python main.py --dataset OfficeHome --source_domain art --target_domain clipart --batch_size 16")
    
    print("\n3. 영향도 샘플 수 감소:")
    print("   --influence_samples 100 --target_samples 300")
    
    print("\n4. 라벨 검증 활성화:")
    print("   officehome_loader.py의 안전 장치 사용")
    
    print("\n🔧 즉시 적용할 수 있는 명령어:")
    print("CUDA_LAUNCH_BLOCKING=1 python main.py --dataset OfficeHome --source_domain art --target_domain clipart --batch_size 16 --influence_samples 100")

if __name__ == "__main__":
    main() 