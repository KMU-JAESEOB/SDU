# officehome_loader.py
"""
🏠 Office-Home 데이터셋 로더
- Art: 예술적 이미지 (2,427개)
- Clipart: 클립아트 이미지 (4,365개)  
- Product: 제품 이미지 (4,439개)
- Real World: 실제 환경 이미지 (4,357개)
총 65개 클래스, 15,588개 이미지
"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import urllib.request
import zipfile
from pathlib import Path

class OfficeHomeDataset(Dataset):
    """Office-Home 데이터셋 클래스"""
    
    def __init__(self, root, domain, transform=None, download=True):
        """
        Args:
            root (str): 데이터 루트 디렉토리
            domain (str): 도메인 ('art', 'clipart', 'product', 'real_world')
            transform: 이미지 변환
            download (bool): 자동 다운로드 여부
        """
        self.root = Path(root)
        self.domain = domain.lower()
        self.transform = transform
        
        # 도메인 매핑
        self.domain_mapping = {
            'art': 'Art',
            'clipart': 'Clipart', 
            'product': 'Product',
            'real_world': 'Real World'
        }
        
        if self.domain not in self.domain_mapping:
            raise ValueError(f"지원하지 않는 도메인: {domain}. 지원 도메인: {list(self.domain_mapping.keys())}")
        
        self.domain_folder = self.domain_mapping[self.domain]
        self.data_dir = self.root / 'OfficeHome'
        
        if download:
            self._download()
        
        self.samples = self._load_samples()
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 🚨 클래스 인덱스 검증 추가
        print(f"📦 Office-Home {self.domain_folder} 로드 완료!")
        print(f"   📊 샘플 수: {len(self.samples):,}개")
        print(f"   🎯 클래스 수: {len(self.classes)}개")
        print(f"   🔍 클래스 인덱스 범위: 0 ~ {len(self.classes)-1}")
        
        # 클래스 수 검증 (Office-Home은 정확히 65개 클래스여야 함)
        if len(self.classes) != 65:
            print(f"⚠️ 경고: Office-Home은 65개 클래스여야 하는데 {len(self.classes)}개 발견됨")
    
    def _download(self):
        """Office-Home 데이터셋 다운로드"""
        
        if (self.data_dir / self.domain_folder).exists():
            print(f"✅ Office-Home {self.domain_folder} 이미 존재")
            return
        
        print(f"📥 Office-Home 데이터셋 다운로드 중...")
        
        # 공식 데이터셋 확인
        if self._check_official_dataset():
            print("✅ 공식 Office-Home 데이터셋 발견!")
            return
        
        # 공식 데이터셋이 없으면 다운로드 안내
        print("⚠️ Office-Home 공식 데이터셋이 필요합니다.")
        print("📋 다운로드 방법:")
        print("1. python download_officehome.py 실행 (자동 다운로드)")
        print("2. 또는 https://www.hemanthdv.org/officeHomeDataset.html 방문하여 수동 다운로드")
        print(f"3. 데이터를 {self.data_dir} 경로에 배치")
        print("4. 폴더 구조: OfficeHome/Art/, OfficeHome/Clipart/, OfficeHome/Product/, OfficeHome/Real World/")
        
        # 자동 다운로드 시도
        try:
            print("\n🤖 자동 다운로드를 시도하시겠습니까?")
            response = input("(y/N): ")
            if response.lower() == 'y':
                self._auto_download()
                return
        except:
            pass
        
        # 테스트용 샘플 데이터 생성
        print("\n🧪 공식 데이터가 없으므로 테스트용 샘플 데이터를 생성합니다.")
        self._create_sample_data()
    
    def _check_official_dataset(self):
        """공식 데이터셋 존재 여부 확인"""
        
        # 데이터셋 정보 파일 확인
        info_file = self.data_dir / 'dataset_info.json'
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                # 공식 데이터셋인지 확인
                if info.get('source') == 'Official Office-Home Dataset':
                    domain_info = info.get('domains', {}).get(self.domain_folder, {})
                    expected_images = {
                        'Art': 2427, 'Clipart': 4365, 
                        'Product': 4439, 'Real World': 4357
                    }.get(self.domain_folder, 0)
                    
                    actual_images = domain_info.get('images', 0)
                    
                    # 이미지 수가 예상값의 80% 이상이면 공식 데이터셋으로 간주
                    if actual_images >= expected_images * 0.8:
                        print(f"📊 {self.domain_folder}: {actual_images:,}개 이미지 (공식 데이터셋)")
                        return True
            except:
                pass
        
        return False
    
    def _auto_download(self):
        """자동 다운로드 시도"""
        
        try:
            import subprocess
            print("🚀 자동 다운로드 스크립트 실행 중...")
            result = subprocess.run(['python', 'download_officehome.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 자동 다운로드 성공!")
                return True
            else:
                print(f"❌ 자동 다운로드 실패: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ 자동 다운로드 오류: {e}")
            return False
    
    def _create_sample_data(self):
        """테스트용 샘플 데이터 생성"""
        
        print("🧪 테스트용 샘플 데이터 생성 중...")
        
        # 65개 클래스 정의 (Office-Home 실제 클래스)
        classes = [
            'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
            'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
            'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
            'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade',
            'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip',
            'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
            'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table',
            'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam'
        ]
        
        domain_dir = self.data_dir / self.domain_folder
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 클래스별로 샘플 이미지 생성
        for class_name in classes:
            class_dir = domain_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # 클래스별 5개 샘플 이미지 생성
            for i in range(5):
                # 224x224 RGB 더미 이미지 생성
                from PIL import Image
                import numpy as np
                
                # 클래스별로 다른 색상 패턴
                color_base = hash(class_name) % 256
                dummy_image = np.random.randint(
                    color_base, min(color_base + 50, 255), 
                    (224, 224, 3), dtype=np.uint8
                )
                
                img = Image.fromarray(dummy_image)
                img_path = class_dir / f"{class_name}_{i:03d}.jpg"
                img.save(img_path)
        
        print(f"✅ 테스트용 샘플 데이터 생성 완료: {domain_dir}")
    
    def _load_samples(self):
        """샘플 목록 로드"""
        
        samples = []
        domain_dir = self.data_dir / self.domain_folder
        
        if not domain_dir.exists():
            raise FileNotFoundError(f"도메인 디렉토리를 찾을 수 없습니다: {domain_dir}")
        
        for class_dir in domain_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_file in class_dir.glob('*.jpg'):
                    samples.append((str(img_file), class_name))
        
        return samples
    
    def _get_classes(self):
        """클래스 목록 반환"""
        
        classes = set()
        for _, class_name in self.samples:
            classes.add(class_name)
        
        return sorted(list(classes))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {img_path}, {e}")
            # 기본 이미지 생성
            image = Image.new('RGB', (224, 224), color='white')
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # 클래스 인덱스
        label = self.class_to_idx[class_name]
        
        # 🚨 라벨 범위 안전 검증
        if label < 0 or label >= 65:
            print(f"⚠️ 라벨 범위 오류: {label} (클래스: {class_name})")
            label = max(0, min(label, 64))  # 0-64 범위로 클리핑
            print(f"🔧 수정된 라벨: {label}")
        
        # Office31과 동일하게 정수로 반환 (일관성을 위해)
        return image, label

class OfficeHomeLoader:
    """Office-Home 데이터셋 로더 관리자"""
    
    def __init__(self, root='./data'):
        self.root = Path(root)
        self.domains = ['art', 'clipart', 'product', 'real_world']
        
        print("🏠 Office-Home 로더 초기화 완료!")
        print(f"📁 데이터 루트: {self.root}")
        print(f"🎯 지원 도메인: {', '.join(self.domains)}")
    
    def get_transforms(self, is_training=True):
        """Office-Home용 이미지 변환"""
        
        if is_training:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def load_domain_data(self, domain, batch_size=32, shuffle=True):
        """특정 도메인 데이터 로드 (DataLoader 반환)"""
        
        if domain not in self.domains:
            raise ValueError(f"지원하지 않는 도메인: {domain}")
        
        print(f"📦 Office-Home {domain} 도메인 로딩 중...")
        
        # 변환 정의
        train_transform = self.get_transforms(is_training=True)
        test_transform = self.get_transforms(is_training=False)
        
        # 데이터셋 생성 (Office-Home은 train/test 분할이 없으므로 동일 데이터 사용)
        train_dataset = OfficeHomeDataset(
            root=self.root, domain=domain, transform=train_transform, download=True
        )
        test_dataset = OfficeHomeDataset(
            root=self.root, domain=domain, transform=test_transform, download=True
        )
        
        print(f"✅ Office-Home {domain} 로딩 완료!")
        
        # OfficeHome 전용 collate_fn: 정수 라벨을 텐서로 변환
        def officehome_collate_fn(batch):
            """OfficeHome의 정수 라벨을 텐서로 변환하는 collate function"""
            import torch
            
            images, labels = zip(*batch)
            
            # 이미지 스택
            images = torch.stack(images)
            
            # 라벨을 텐서로 변환 (정수 리스트 → 텐서)
            labels = torch.tensor(labels, dtype=torch.long)
            
            return images, labels
        
        # DataLoader 생성 (collate_fn으로 라벨 호환성 확보)
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=officehome_collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=officehome_collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        # DataLoader 반환 (Office31과 호환)
        return train_loader, test_loader
    
    def get_domain_info(self, domain):
        """도메인 정보 반환"""
        
        domain_info = {
            'art': {
                'name': 'Art',
                'description': '예술적 이미지 (회화, 스케치 등)',
                'samples': 2427,
                'characteristics': '추상적, 예술적 스타일'
            },
            'clipart': {
                'name': 'Clipart', 
                'description': '클립아트 이미지 (단순화된 그래픽)',
                'samples': 4365,
                'characteristics': '단순화, 만화적 스타일'
            },
            'product': {
                'name': 'Product',
                'description': '제품 이미지 (카탈로그, 광고)',
                'samples': 4439, 
                'characteristics': '깔끔한 배경, 상업적'
            },
            'real_world': {
                'name': 'Real World',
                'description': '실제 환경 이미지 (자연스러운 설정)',
                'samples': 4357,
                'characteristics': '복잡한 배경, 현실적'
            }
        }
        
        return domain_info.get(domain, {})
    
    def print_dataset_summary(self):
        """데이터셋 요약 정보 출력"""
        
        print("\n" + "="*80)
        print("🏠 Office-Home 데이터셋 요약")
        print("="*80)
        print("📊 총 클래스: 65개")
        print("📊 총 이미지: 15,588개")
        print("📐 이미지 크기: 가변 (224x224로 리사이즈)")
        print("🎨 채널: 3 (RGB)")
        print()
        
        total_samples = 0
        for domain in self.domains:
            info = self.get_domain_info(domain)
            print(f"🏷️ {info['name']} ({domain}):")
            print(f"   📝 설명: {info['description']}")
            print(f"   📊 샘플 수: {info['samples']:,}개")
            print(f"   ✨ 특징: {info['characteristics']}")
            print()
            total_samples += info['samples']
        
        print(f"📈 총 샘플 수: {total_samples:,}개")
        print("="*80)

def main():
    """테스트 실행"""
    
    print("🏠 Office-Home 데이터셋 로더 테스트")
    print("="*60)
    
    # 로더 생성
    loader = OfficeHomeLoader()
    
    # 데이터셋 요약 출력
    loader.print_dataset_summary()
    
    # 샘플 데이터 로딩 테스트
    print("\n🧪 샘플 데이터 로딩 테스트:")
    try:
        train_dataset, test_dataset = loader.load_domain_data('art', batch_size=16)
        print("✅ Art 도메인 로딩 성공!")
        
        # DataLoader 생성
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # 첫 번째 배치 확인
        for images, labels in train_loader:
            print(f"   📊 배치 크기: {images.shape}")
            print(f"   🎯 라벨 범위: {labels.min().item()} ~ {labels.max().item()}")
            break
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    main() 