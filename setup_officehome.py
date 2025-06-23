# setup_officehome.py
"""
🏠 Office-Home 데이터셋 통합 설정 스크립트
- 공식 데이터셋 자동 다운로드 및 설정
- 데이터 검증 및 실험 준비
- 원클릭 설정 지원
"""

import os
import sys
from pathlib import Path
import json

def print_banner():
    """배너 출력"""
    print("🏠 Office-Home 데이터셋 설정")
    print("="*60)
    print("Office-Home 데이터셋을 자동으로 다운로드하고 설정합니다.")
    print("4개 도메인 (Art, Clipart, Product, Real World)")
    print("65개 클래스, 15,588개 이미지")
    print("="*60)

def check_dependencies():
    """필요한 패키지 확인"""
    
    print("\n🔍 의존성 패키지 확인 중...")
    
    required_packages = ['torch', 'torchvision', 'PIL', 'tqdm', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 모든 의존성 패키지가 설치되어 있습니다!")
    return True

def download_officehome():
    """Office-Home 데이터셋 다운로드"""
    
    print("\n📥 Office-Home 데이터셋 다운로드")
    print("-"*40)
    
    try:
        from download_officehome import OfficeHomeDownloader
        
        downloader = OfficeHomeDownloader()
        success = downloader.download_official_dataset()
        
        if success:
            print("✅ Office-Home 데이터셋 다운로드 완료!")
            return True
        else:
            print("❌ 데이터셋 다운로드 실패")
            return False
            
    except ImportError:
        print("❌ download_officehome.py 파일이 필요합니다.")
        return False
    except Exception as e:
        print(f"❌ 다운로드 중 오류: {e}")
        return False

def test_dataset():
    """데이터셋 테스트"""
    
    print("\n🧪 데이터셋 테스트")
    print("-"*40)
    
    try:
        from officehome_loader import OfficeHomeLoader
        
        # 로더 생성
        loader = OfficeHomeLoader()
        
        # 각 도메인 테스트
        domains = ['art', 'clipart', 'product', 'real_world']
        
        for domain in domains:
            try:
                print(f"📦 {domain.upper()} 도메인 테스트 중...")
                
                train_dataset, test_dataset = loader.load_domain_data(domain)
                
                print(f"  ✅ 훈련 데이터: {len(train_dataset):,}개")
                print(f"  ✅ 테스트 데이터: {len(test_dataset):,}개")
                
                # 첫 번째 샘플 테스트
                sample_img, sample_label = train_dataset[0]
                print(f"  📊 샘플 크기: {sample_img.shape}")
                print(f"  🏷️ 라벨 범위: 0-{max([train_dataset[i][1] for i in range(min(100, len(train_dataset)))])}")
                
            except Exception as e:
                print(f"  ❌ {domain} 테스트 실패: {e}")
                return False
        
        print("✅ 모든 도메인 테스트 통과!")
        return True
        
    except ImportError:
        print("❌ officehome_loader.py 파일이 필요합니다.")
        return False
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        return False

def verify_experiment_setup():
    """실험 환경 검증"""
    
    print("\n🔧 실험 환경 검증")
    print("-"*40)
    
    # 필수 파일 확인
    required_files = [
        'main.py',
        'officehome_loader.py', 
        'officehome_full_experiments.py',
        'download_officehome.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"  ❌ {file}")
        else:
            print(f"  ✅ {file}")
    
    if missing_files:
        print(f"\n⚠️ 누락된 파일: {', '.join(missing_files)}")
        return False
    
    # 디렉토리 확인
    required_dirs = ['data', 'results', 'models']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"  📁 {dir_name} 디렉토리 생성")
        else:
            print(f"  ✅ {dir_name} 디렉토리 존재")
    
    print("✅ 실험 환경 검증 완료!")
    return True

def create_quick_start_guide():
    """빠른 시작 가이드 생성"""
    
    guide_content = """# Office-Home 실험 빠른 시작 가이드

## 🚀 단일 실험 실행
```bash
# Art → Clipart 실험
python main.py --dataset OfficeHome --source_domain art --target_domain clipart

# Product → Real World 실험  
python main.py --dataset OfficeHome --source_domain product --target_domain real_world
```

## 🏠 전체 실험 실행 (12개 도메인 조합)
```bash
python officehome_full_experiments.py
```

## 📊 결과 확인
- 결과 파일: `results/officehome/`
- 모델 파일: `models/officehome/`
- 로그 파일: `results/officehome/*_log.txt`

## 🎯 실험 파라미터
- 모델: ResNet50 (ImageNet 사전훈련)
- 배치 크기: 32
- 학습률: 2e-4
- 에포크: 15
- 영향도 샘플: 500개
- 타겟 샘플: 800개

## 📋 도메인 정보
- **Art**: 예술적 이미지 (2,427개)
- **Clipart**: 클립아트 이미지 (4,365개)  
- **Product**: 제품 이미지 (4,439개)
- **Real World**: 실제 환경 이미지 (4,357개)

## 🔧 문제 해결
1. CUDA 메모리 부족 시: `--batch_size 16`
2. 영향도 계산 오류 시: `--influence_samples 200`
3. 데이터 재다운로드: `python download_officehome.py`
"""
    
    with open('OFFICEHOME_QUICKSTART.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("📋 빠른 시작 가이드 생성: OFFICEHOME_QUICKSTART.md")

def main():
    """메인 설정 함수"""
    
    print_banner()
    
    # 1. 의존성 확인
    if not check_dependencies():
        print("\n❌ 설정 실패: 필요한 패키지를 설치해주세요.")
        return False
    
    # 2. 실험 환경 검증
    if not verify_experiment_setup():
        print("\n❌ 설정 실패: 필요한 파일이 누락되었습니다.")
        return False
    
    # 3. 데이터셋 다운로드
    print("\n" + "="*60)
    print("📥 데이터셋 설정")
    print("="*60)
    
    data_dir = Path('./data/OfficeHome')
    if data_dir.exists() and len(list(data_dir.rglob('*.jpg'))) > 1000:
        print("✅ Office-Home 데이터셋이 이미 존재합니다.")
        
        response = input("다시 다운로드하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("⏭️ 기존 데이터 사용")
        else:
            if not download_officehome():
                print("❌ 설정 실패: 데이터셋 다운로드에 실패했습니다.")
                return False
    else:
        if not download_officehome():
            print("❌ 설정 실패: 데이터셋 다운로드에 실패했습니다.")
            return False
    
    # 4. 데이터셋 테스트
    if not test_dataset():
        print("❌ 설정 실패: 데이터셋 테스트에 실패했습니다.")
        return False
    
    # 5. 가이드 생성
    create_quick_start_guide()
    
    # 완료 메시지
    print("\n" + "="*60)
    print("🎉 Office-Home 설정 완료!")
    print("="*60)
    print("✅ 공식 데이터셋 다운로드 완료")
    print("✅ 데이터 검증 완료") 
    print("✅ 실험 환경 준비 완료")
    print("\n🚀 이제 실험을 시작할 수 있습니다!")
    print("📋 빠른 시작: OFFICEHOME_QUICKSTART.md 참조")
    print("\n💡 추천 실행 명령어:")
    print("   python officehome_full_experiments.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 