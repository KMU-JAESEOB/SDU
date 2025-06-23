# office31_loader.py - Office-31 데이터셋 로더

"""
🏢 Office-31 데이터셋 자동 다운로드 및 처리
- Amazon (A): 2817개 이미지
- Webcam (W): 795개 이미지  
- DSLR (D): 498개 이미지
- 31개 클래스 (백팩, 자전거, 계산기, 헤드폰, 키보드, 노트북, 마우스, 머그컵, 프로젝터 등)
"""

import os
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class Office31Dataset(Dataset):
    """Office-31 데이터셋 클래스"""
    
    def __init__(self, root, domain, transform=None, download=True):
        """
        Args:
            root: 데이터 저장 경로
            domain: 'amazon', 'webcam', 'dslr' 중 하나
            transform: 이미지 변환
            download: 자동 다운로드 여부
        """
        self.root = Path(root)
        self.domain = domain.lower()
        self.transform = transform
        
        # 도메인 매핑
        self.domain_mapping = {
            'amazon': 'amazon',
            'webcam': 'webcam', 
            'dslr': 'dslr',
            'a': 'amazon',
            'w': 'webcam',
            'd': 'dslr'
        }
        
        if self.domain not in self.domain_mapping:
            raise ValueError(f"지원하지 않는 도메인: {domain}. 'amazon', 'webcam', 'dslr' 중 선택하세요.")
        
        self.domain = self.domain_mapping[self.domain]
        
        # 클래스 이름들 (31개)
        self.classes = [
            'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator',
            'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet',
            'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
            'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook',
            'pen', 'phone', 'printer', 'projector', 'punchers', 'ring_binder',
            'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can'
        ]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        if download:
            self.download()
        
        # 이미지 경로와 라벨 로드
        self.samples = self._load_samples()
        
    def download(self):
        """Office-31 데이터셋을 다운로드합니다."""
        
        data_dir = self.root / 'office31'
        
        if data_dir.exists() and len(list(data_dir.glob('*'))) > 0:
            print(f"✅ Office-31 데이터셋이 이미 존재합니다: {data_dir}")
            return
        
        print("📥 Office-31 데이터셋 다운로드 중...")
        
        # Office-31 다운로드 URL들 (여러 미러 사이트)
        download_urls = [
            # Kaggle 미러 (가장 안정적)
            "https://www.kaggle.com/datasets/gepuro/office31",
            # GitHub 릴리즈
            "https://github.com/jindongwang/transferlearning/raw/master/data/office31.tar.gz",
            # 대체 미러들
            "https://drive.google.com/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE",
            "https://pan.baidu.com/s/14JEGQ56LJX7LMbd7GTlMFA",
            # 백업 URL
            "http://www.eecs.berkeley.edu/~jhoffman/domainadapt/office_31.tar.gz"
        ]
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드 시도
        success = False
        for i, download_url in enumerate(download_urls):
            try:
                print(f"🔄 다운로드 시도 {i+1}/{len(download_urls)}: {download_url}")
                
                if 'kaggle.com' in download_url:
                    print("⚠️ Kaggle 데이터셋은 Kaggle API가 필요합니다.")
                    print("   pip install kaggle 후 API 키 설정 필요")
                    continue
                elif 'drive.google.com' in download_url:
                    # Google Drive 다운로드 (gdown 사용)
                    try:
                        import gdown
                        file_id = download_url.split('id=')[1] if 'id=' in download_url else download_url.split('/')[-2]
                        gdown.download(f"https://drive.google.com/uc?id={file_id}", 
                                     str(data_dir / "office_31.tar.gz"), quiet=False)
                        success = True
                        break
                    except ImportError:
                        print("⚠️ gdown 패키지가 필요합니다: pip install gdown")
                        continue
                    except Exception as e:
                        print(f"⚠️ Google Drive 다운로드 실패: {e}")
                        continue
                elif 'github.com' in download_url:
                    # GitHub 다운로드
                    try:
                        urllib.request.urlretrieve(download_url, data_dir / "office_31.tar.gz")
                        success = True
                        break
                    except Exception as e:
                        print(f"⚠️ GitHub 다운로드 실패: {e}")
                        continue
                else:
                    # 일반 HTTP 다운로드
                    try:
                        urllib.request.urlretrieve(download_url, data_dir / "office_31.tar.gz")
                        success = True
                        break
                    except Exception as e:
                        print(f"⚠️ 다운로드 실패: {e}")
                        continue
                    
            except Exception as e:
                print(f"❌ 다운로드 실패: {e}")
                continue
        
        if not success:
            print("❌ 자동 다운로드에 실패했습니다.")
            print("\n📋 수동 다운로드 방법 (아래 중 하나 선택):")
            print("\n🔹 방법 1: Google Drive (추천)")
            print("1. https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view 방문")
            print("2. office_31.tar.gz 다운로드")
            print(f"3. 다운로드한 파일을 {data_dir}/office_31.tar.gz 위치에 저장")
            print("\n🔹 방법 2: 직접 생성 (빠른 해결)")
            print(f"mkdir -p {data_dir}/office31/{{amazon,dslr,webcam}}/images")
            print("# 각 폴더에 해당 도메인의 이미지들을 넣어주세요")
            print("\n🔹 방법 3: 다른 Office-31 데이터셋 사용")
            print(f"기존에 다운로드한 Office-31이 있다면 {data_dir}/office31/ 폴더로 복사하세요")
            
            # 임시로 빈 데이터셋 생성하여 에러 방지
            print("\n⚠️ 임시 해결: 빈 데이터셋으로 진행합니다 (테스트용)")
            self._create_dummy_dataset(data_dir)
            return
        
        # 압축 해제
        print("📦 압축 해제 중...")
        try:
            with tarfile.open(data_dir / "office_31.tar.gz", 'r:gz') as tar:
                tar.extractall(data_dir)
            print("✅ Office-31 데이터셋 다운로드 완료!")
            
            # 압축 파일 삭제
            (data_dir / "office_31.tar.gz").unlink()
            
        except Exception as e:
            print(f"❌ 압축 해제 실패: {e}")
    
    def _create_dummy_dataset(self, data_dir):
        """테스트용 더미 데이터셋을 생성합니다."""
        print("🔧 더미 데이터셋 생성 중...")
        
        # 기본 폴더 구조 생성
        for domain in ['amazon', 'webcam', 'dslr']:
            domain_dir = data_dir / 'office31' / domain / 'images'
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            # 각 클래스별 폴더 생성
            for class_name in self.classes:
                class_dir = domain_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                # 더미 이미지 생성 (1개씩만)
                dummy_img_path = class_dir / f"dummy_{class_name}.jpg"
                if not dummy_img_path.exists():
                    # 간단한 더미 이미지 생성
                    from PIL import Image
                    import random
                    
                    # 랜덤 색상의 더미 이미지
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    dummy_img = Image.new('RGB', (224, 224), color=color)
                    dummy_img.save(dummy_img_path)
        
        print("✅ 더미 데이터셋 생성 완료! (테스트용)")
        print("⚠️ 실제 실험을 위해서는 진짜 Office-31 데이터셋을 다운로드하세요.")
    
    def _load_samples(self):
        """이미지 경로와 라벨을 로드합니다."""
        
        # 사용자의 실제 데이터 구조에 맞는 경로들 시도
        possible_paths = [
            # 사용자의 실제 구조: data/amazon, data/webcam, data/dslr
            self.root / self.domain,
            # 기존 표준 구조들
            self.root / 'office31' / self.domain / 'images',
            self.root / 'office31' / f'{self.domain}_images',
            self.root / 'office31' / f'Office31' / self.domain / 'images',
            self.root / 'office31' / f'office31' / self.domain / 'images',
            # 추가 가능한 구조들
            self.root / self.domain / 'images',
            self.root / f'{self.domain}_images'
        ]
        
        domain_dir = None
        for path in possible_paths:
            if path.exists():
                domain_dir = path
                print(f"✅ 데이터 경로 발견: {domain_dir}")
                break
        
        if domain_dir is None:
            print(f"❌ 다음 경로들에서 데이터를 찾을 수 없습니다:")
            for path in possible_paths:
                print(f"   - {path}")
            raise FileNotFoundError(f"도메인 디렉토리를 찾을 수 없습니다. 확인된 경로: {possible_paths}")
        
        samples = []
        
        for class_name in self.classes:
            # 클래스 디렉토리 가능한 경로들
            class_possible_paths = [
                domain_dir / class_name,  # 직접 클래스 폴더
                domain_dir / 'images' / class_name,  # images 하위 클래스 폴더
            ]
            
            class_dir = None
            for class_path in class_possible_paths:
                if class_path.exists():
                    class_dir = class_path
                    break
            
            if class_dir is None:
                print(f"⚠️ 클래스 디렉토리가 없습니다: {class_name}")
                print(f"   시도한 경로들: {[str(p) for p in class_possible_paths]}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # 이미지 파일들 찾기 (더 많은 확장자 지원)
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            class_images = 0
            
            for ext in image_extensions:
                for img_path in class_dir.glob(ext):
                    samples.append((str(img_path), class_idx))
                    class_images += 1
            
            if class_images > 0:
                print(f"   📁 {class_name}: {class_images}개 이미지")
            else:
                print(f"   ⚠️ {class_name}: 이미지 없음")
        
        print(f"✅ {self.domain.upper()} 도메인: {len(samples)}개 이미지 로드")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {img_path}, {e}")
            # 기본 이미지 생성
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class Office31Manager:
    """Office-31 데이터셋 관리자"""
    
    def __init__(self, root='./data'):
        self.root = Path(root)
        
        # 도메인 정보
        self.domains = {
            'amazon': {
                'name': 'Amazon',
                'description': '아마존 제품 이미지 (2817개)',
                'size': 2817,
                'characteristics': '깨끗한 배경, 고품질'
            },
            'webcam': {
                'name': 'Webcam', 
                'description': '웹캠으로 촬영한 이미지 (795개)',
                'size': 795,
                'characteristics': '낮은 해상도, 노이즈'
            },
            'dslr': {
                'name': 'DSLR',
                'description': 'DSLR 카메라로 촬영한 이미지 (498개)', 
                'size': 498,
                'characteristics': '고해상도, 자연스러운 배경'
            }
        }
        
        # 추천 도메인 적응 쌍들
        self.recommended_pairs = [
            ('amazon', 'webcam', 'Amazon → Webcam: 고품질 → 저품질'),
            ('amazon', 'dslr', 'Amazon → DSLR: 인공적 → 자연스러운'),
            ('webcam', 'dslr', 'Webcam → DSLR: 저품질 → 고품질'),
            ('dslr', 'amazon', 'DSLR → Amazon: 자연스러운 → 인공적'),
            ('webcam', 'amazon', 'Webcam → Amazon: 저품질 → 고품질'),
            ('dslr', 'webcam', 'DSLR → Webcam: 고품질 → 저품질')
        ]
    
    def print_domain_info(self):
        """도메인 정보를 출력합니다."""
        print("🏢 Office-31 데이터셋 정보:")
        print("=" * 60)
        
        for domain, info in self.domains.items():
            print(f"📊 {info['name']} ({domain.upper()})")
            print(f"   📝 {info['description']}")
            print(f"   🔍 특징: {info['characteristics']}")
            print()
    
    def print_recommended_pairs(self):
        """추천 도메인 적응 쌍들을 출력합니다."""
        print("🎯 추천 Office-31 도메인 적응 쌍들:")
        print("=" * 60)
        
        for i, (source, target, desc) in enumerate(self.recommended_pairs, 1):
            source_info = self.domains[source]
            target_info = self.domains[target]
            
            print(f"{i}. {source_info['name']} → {target_info['name']}")
            print(f"   📝 {desc}")
            print(f"   📊 {source_info['size']}개 → {target_info['size']}개")
            print()
    
    def create_data_loaders(self, source_domain, target_domain, batch_size=32, 
                          image_size=224, num_workers=2):
        """Office-31 데이터 로더들을 생성합니다."""
        
        print(f"🎯 Office-31 도메인 적응 데이터 로더 생성:")
        print(f"   📤 소스: {source_domain.upper()}")
        print(f"   📥 타겟: {target_domain.upper()}")
        
        # 변환 설정
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet 정규화
        ])
        
        # 데이터셋 생성
        source_dataset = Office31Dataset(
            root=self.root, domain=source_domain, transform=transform, download=True
        )
        target_dataset = Office31Dataset(
            root=self.root, domain=target_domain, transform=transform, download=True
        )
        
        # 데이터 로더 생성
        source_loader = DataLoader(
            source_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        target_loader = DataLoader(
            target_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        print(f"✅ 데이터 로더 생성 완료!")
        print(f"   📊 소스 배치 수: {len(source_loader)}")
        print(f"   📊 타겟 배치 수: {len(target_loader)}")
        
        return source_loader, target_loader, source_dataset, target_dataset

def main():
    """Office-31 로더 테스트"""
    
    manager = Office31Manager()
    
    print("🏢 Office-31 데이터셋 관리자")
    print("=" * 60)
    
    # 도메인 정보 출력
    manager.print_domain_info()
    
    # 추천 쌍 출력
    manager.print_recommended_pairs()
    
    print("\n테스트할 도메인 적응 쌍을 선택하세요:")
    for i, (source, target, desc) in enumerate(manager.recommended_pairs, 1):
        print(f"{i}. {source.upper()} → {target.upper()}")
    
    try:
        choice = int(input(f"\n선택 (1-{len(manager.recommended_pairs)}): ")) - 1
        if 0 <= choice < len(manager.recommended_pairs):
            source, target, desc = manager.recommended_pairs[choice]
            print(f"\n선택: {desc}")
            
            # 데이터 로더 생성 테스트
            source_loader, target_loader, source_dataset, target_dataset = \
                manager.create_data_loaders(source, target)
            
            print(f"\n🎉 {source.upper()} → {target.upper()} 데이터 로더 준비 완료!")
            
            # 샘플 이미지 확인
            sample_batch = next(iter(source_loader))
            print(f"📊 샘플 배치 형태: {sample_batch[0].shape}")
            
        else:
            print("❌ 잘못된 선택입니다.")
    except ValueError:
        print("❌ 숫자를 입력해주세요.")
    except KeyboardInterrupt:
        print("\n👋 종료합니다.")

if __name__ == "__main__":
    main() 