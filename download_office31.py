#!/usr/bin/env python3
"""
Office-31 데이터셋 다운로드 스크립트

이 스크립트는 Office-31 데이터셋을 자동으로 다운로드하고 설정합니다.
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path
import subprocess

def install_gdown():
    """gdown 패키지를 설치합니다."""
    try:
        import gdown
        return True
    except ImportError:
        print("📦 gdown 패키지 설치 중...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            return True
        except Exception as e:
            print(f"❌ gdown 설치 실패: {e}")
            return False

def download_from_google_drive():
    """Google Drive에서 Office-31 다운로드"""
    if not install_gdown():
        return False
    
    import gdown
    
    print("🔄 Google Drive에서 Office-31 다운로드 중...")
    try:
        # Office-31 Google Drive 링크
        url = "https://drive.google.com/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
        output = "./data/office_31.tar.gz"
        
        # 폴더 생성
        os.makedirs("./data", exist_ok=True)
        
        # 다운로드
        gdown.download(url, output, quiet=False)
        return True
        
    except Exception as e:
        print(f"❌ Google Drive 다운로드 실패: {e}")
        return False

def download_from_alternative():
    """대체 소스에서 다운로드"""
    urls = [
        "https://github.com/jindongwang/transferlearning/raw/master/data/office31.tar.gz",
        "http://www.eecs.berkeley.edu/~jhoffman/domainadapt/office_31.tar.gz"
    ]
    
    for url in urls:
        try:
            print(f"🔄 다운로드 시도: {url}")
            urllib.request.urlretrieve(url, "./data/office_31.tar.gz")
            return True
        except Exception as e:
            print(f"❌ 실패: {e}")
            continue
    
    return False

def extract_dataset():
    """데이터셋 압축 해제"""
    print("📦 압축 해제 중...")
    try:
        with tarfile.open("./data/office_31.tar.gz", 'r:gz') as tar:
            tar.extractall("./data")
        
        # 압축 파일 삭제
        os.remove("./data/office_31.tar.gz")
        print("✅ 압축 해제 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 압축 해제 실패: {e}")
        return False

def create_dummy_data():
    """테스트용 더미 데이터 생성"""
    print("🔧 더미 데이터 생성 중...")
    
    from PIL import Image
    import random
    
    # Office-31 클래스들
    classes = [
        'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator',
        'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet',
        'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
        'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook',
        'pen', 'phone', 'printer', 'projector', 'punchers', 'ring_binder',
        'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can'
    ]
    
    domains = ['amazon', 'webcam', 'dslr']
    
    for domain in domains:
        for class_name in classes:
            # 폴더 생성
            class_dir = Path(f"./data/office31/{domain}/images/{class_name}")
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # 더미 이미지 생성 (각 클래스당 5개)
            for i in range(5):
                img_path = class_dir / f"dummy_{i:03d}.jpg"
                if not img_path.exists():
                    # 랜덤 색상 이미지
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    img = Image.new('RGB', (224, 224), color=color)
                    img.save(img_path)
    
    print("✅ 더미 데이터 생성 완료!")

def main():
    """메인 함수"""
    print("🏢 Office-31 데이터셋 다운로드 스크립트")
    print("=" * 50)
    
    # 이미 데이터가 있는지 확인
    if Path("./data/office31").exists():
        print("✅ Office-31 데이터셋이 이미 존재합니다!")
        return
    
    print("\n다운로드 방법을 선택하세요:")
    print("1. Google Drive에서 다운로드 (추천)")
    print("2. 대체 소스에서 다운로드")
    print("3. 더미 데이터 생성 (테스트용)")
    print("4. 수동 다운로드 가이드")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == "1":
        if download_from_google_drive():
            extract_dataset()
        else:
            print("❌ Google Drive 다운로드 실패")
            
    elif choice == "2":
        if download_from_alternative():
            extract_dataset()
        else:
            print("❌ 대체 소스 다운로드 실패")
            
    elif choice == "3":
        create_dummy_data()
        
    elif choice == "4":
        print("\n📋 수동 다운로드 방법:")
        print("1. https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view 방문")
        print("2. office_31.tar.gz 다운로드")
        print("3. 이 폴더에 data/office_31.tar.gz로 저장")
        print("4. 이 스크립트를 다시 실행하여 압축 해제")
        
        # 수동 다운로드한 파일이 있는지 확인
        if Path("./data/office_31.tar.gz").exists():
            print("\n✅ office_31.tar.gz 파일을 찾았습니다!")
            if input("압축을 해제하시겠습니까? (y/n): ").lower() == 'y':
                extract_dataset()
    
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 