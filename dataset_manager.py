# dataset_manager.py - 다양한 데이터셋 지원

"""
🎯 SDA-U를 위한 다양한 데이터셋 지원
- 소스/타겟 도메인 조합 제공
- 자동 다운로드 및 전처리
- 도메인 적응에 적합한 데이터셋 쌍 추천
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

class DatasetManager:
    """다양한 데이터셋을 통합 관리하는 클래스"""
    
    def __init__(self):
        self.data_dir = './data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 🎯 지원 데이터셋 목록 확장
        self.supported_datasets = {
            # 기존 데이터셋
            'MNIST', 'SVHN', 'CIFAR10', 'CIFAR100',
            # 새로 추가된 데이터셋
            'FashionMNIST', 'STL10', 
            # Office-31 데이터셋
            'Office31_Amazon', 'Office31_Webcam', 'Office31_DSLR',
            # Office-Home 데이터셋
            'OfficeHome'
        }
        
        print(f"📦 데이터셋 매니저 초기화 완료!")
        print(f"🎯 지원 데이터셋: {', '.join(sorted(self.supported_datasets))}")
    
    def get_dataset_info(self, dataset_name):
        """데이터셋 정보 반환"""
        
        dataset_info = {
            'MNIST': {
                'num_classes': 10,
                'image_size': 28,
                'channels': 1,
                'mean': [0.1307],
                'std': [0.3081],
                'description': '손글씨 숫자 (0-9)'
            },
            'FashionMNIST': {
                'num_classes': 10,
                'image_size': 28,
                'channels': 1,
                'mean': [0.2860],
                'std': [0.3530],
                'description': '패션 아이템 (10개 카테고리)'
            },
            'SVHN': {
                'num_classes': 10,
                'image_size': 32,
                'channels': 3,
                'mean': [0.4377, 0.4438, 0.4728],
                'std': [0.1980, 0.2010, 0.1970],
                'description': '자연환경 숫자 (Street View)'
            },
            'CIFAR10': {
                'num_classes': 10,
                'image_size': 32,
                'channels': 3,
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010],
                'description': '자연 이미지 (10개 클래스)'
            },
            'CIFAR100': {
                'num_classes': 100,
                'image_size': 32,
                'channels': 3,
                'mean': [0.5071, 0.4867, 0.4408],
                'std': [0.2675, 0.2565, 0.2761],
                'description': '자연 이미지 (100개 클래스)'
            },
            'STL10': {
                'num_classes': 10,
                'image_size': 96,
                'channels': 3,
                'mean': [0.4467, 0.4398, 0.4066],
                'std': [0.2603, 0.2566, 0.2713],
                'description': '고해상도 자연 이미지'
            },
            'Office31_Amazon': {
                'num_classes': 31,
                'image_size': 224,
                'channels': 3,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'description': 'Amazon 제품 이미지'
            },
            'Office31_Webcam': {
                'num_classes': 31,
                'image_size': 224,
                'channels': 3,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'description': 'Webcam 제품 이미지'
            },
            'Office31_DSLR': {
                'num_classes': 31,
                'image_size': 224,
                'channels': 3,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'description': 'DSLR 제품 이미지'
            },
            'OfficeHome': {
                'num_classes': 65,
                'image_size': 224,
                'channels': 3,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'description': 'Office-Home 데이터셋'
            }
        }
        
        return dataset_info.get(dataset_name, {
            'num_classes': 10,
            'image_size': 32,
            'channels': 3,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'description': '알 수 없는 데이터셋'
        })
    
    def get_transforms(self, dataset_name, is_training=True):
        """데이터셋별 최적화된 변환 반환"""
        
        info = self.get_dataset_info(dataset_name)
        image_size = info['image_size']
        channels = info['channels']
        mean = info['mean']
        std = info['std']
        
        if is_training:
            if channels == 1:  # 흑백 이미지 (MNIST, Fashion-MNIST)
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            else:  # 컬러 이미지
                if image_size <= 32:  # 작은 이미지 (CIFAR-10, SVHN)
                    transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomRotation(10),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                    ])
                else:  # 큰 이미지 (STL-10, Office-31)
                    transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomRotation(15),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                    ])
        else:
            # 테스트용 변환 (데이터 증강 없음)
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        return transform
    
    def load_dataset(self, dataset_name, batch_size=32, shuffle=True, num_workers=2):
        """데이터셋 로드 및 DataLoader 반환"""
        
        print(f"📦 {dataset_name} 데이터셋 로딩 중...")
        
        train_transform = self.get_transforms(dataset_name, is_training=True)
        test_transform = self.get_transforms(dataset_name, is_training=False)
        
        try:
            if dataset_name == 'MNIST':
                train_dataset = torchvision.datasets.MNIST(
                    root=self.data_dir, train=True, download=True, transform=train_transform)
                test_dataset = torchvision.datasets.MNIST(
                    root=self.data_dir, train=False, download=True, transform=test_transform)
                    
            elif dataset_name == 'FashionMNIST':
                train_dataset = torchvision.datasets.FashionMNIST(
                    root=self.data_dir, train=True, download=True, transform=train_transform)
                test_dataset = torchvision.datasets.FashionMNIST(
                    root=self.data_dir, train=False, download=True, transform=test_transform)
                    
            elif dataset_name == 'SVHN':
                train_dataset = torchvision.datasets.SVHN(
                    root=self.data_dir, split='train', download=True, transform=train_transform)
                test_dataset = torchvision.datasets.SVHN(
                    root=self.data_dir, split='test', download=True, transform=test_transform)
                    
            elif dataset_name == 'CIFAR10':
                train_dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir, train=True, download=True, transform=train_transform)
                test_dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir, train=False, download=True, transform=test_transform)
                    
            elif dataset_name == 'CIFAR100':
                train_dataset = torchvision.datasets.CIFAR100(
                    root=self.data_dir, train=True, download=True, transform=train_transform)
                test_dataset = torchvision.datasets.CIFAR100(
                    root=self.data_dir, train=False, download=True, transform=test_transform)
                    
            elif dataset_name == 'STL10':
                train_dataset = torchvision.datasets.STL10(
                    root=self.data_dir, split='train', download=True, transform=train_transform)
                test_dataset = torchvision.datasets.STL10(
                    root=self.data_dir, split='test', download=True, transform=test_transform)
                    
            elif dataset_name.startswith('Office31_'):
                # Office-31 데이터셋은 별도 로더 사용
                from office31_loader import Office31Loader
                office31_loader = Office31Loader()
                domain = dataset_name.split('_')[1].lower()
                train_dataset, test_dataset = office31_loader.load_domain_data(domain)
                
            elif dataset_name == 'OfficeHome':
                # Office-Home 데이터셋은 별도 로더 사용
                from officehome_loader import OfficeHomeLoader
                officehome_loader = OfficeHomeLoader(root=self.data_dir)
                train_dataset, test_dataset = officehome_loader.load_domain_data(domain)
                
            else:
                raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
            
            # DataLoader 생성
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle, 
                num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=True)
            
            info = self.get_dataset_info(dataset_name)
            print(f"✅ {dataset_name} 로딩 완료!")
            print(f"   📊 훈련 샘플: {len(train_dataset):,}개")
            print(f"   📊 테스트 샘플: {len(test_dataset):,}개")
            print(f"   🎯 클래스 수: {info['num_classes']}개")
            print(f"   📐 이미지 크기: {info['image_size']}x{info['image_size']}x{info['channels']}")
            
            return train_loader, test_loader
            
        except Exception as e:
            print(f"❌ {dataset_name} 로딩 실패: {str(e)}")
            raise
    
    def get_subset_loader(self, dataset_name, subset_size=500, batch_size=32, is_training=True):
        """데이터셋의 부분집합 로더 반환"""
        
        print(f"🎯 {dataset_name} 부분집합 생성 중 (크기: {subset_size})")
        
        transform = self.get_transforms(dataset_name, is_training=is_training)
        
        try:
            if dataset_name == 'MNIST':
                dataset = torchvision.datasets.MNIST(
                    root=self.data_dir, train=is_training, download=True, transform=transform)
            elif dataset_name == 'FashionMNIST':
                dataset = torchvision.datasets.FashionMNIST(
                    root=self.data_dir, train=is_training, download=True, transform=transform)
            elif dataset_name == 'SVHN':
                split = 'train' if is_training else 'test'
                dataset = torchvision.datasets.SVHN(
                    root=self.data_dir, split=split, download=True, transform=transform)
            elif dataset_name == 'CIFAR10':
                dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir, train=is_training, download=True, transform=transform)
            elif dataset_name == 'CIFAR100':
                dataset = torchvision.datasets.CIFAR100(
                    root=self.data_dir, train=is_training, download=True, transform=transform)
            elif dataset_name == 'STL10':
                split = 'train' if is_training else 'test'
                dataset = torchvision.datasets.STL10(
                    root=self.data_dir, split=split, download=True, transform=transform)
            elif dataset_name.startswith('Office31_'):
                from office31_loader import Office31Loader
                office31_loader = Office31Loader()
                domain = dataset_name.split('_')[1].lower()
                train_dataset, test_dataset = office31_loader.load_domain_data(domain)
                dataset = train_dataset if is_training else test_dataset
            elif dataset_name == 'OfficeHome':
                from officehome_loader import OfficeHomeLoader
                officehome_loader = OfficeHomeLoader(root=self.data_dir)
                domain = dataset_name.split('_')[1].lower()
                train_dataset, test_dataset = officehome_loader.load_domain_data(domain)
                dataset = train_dataset if is_training else test_dataset
            else:
                raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
            
            # 랜덤 부분집합 생성
            total_size = len(dataset)
            subset_size = min(subset_size, total_size)
            indices = np.random.choice(total_size, subset_size, replace=False)
            subset_dataset = Subset(dataset, indices)
            
            # DataLoader 생성
            subset_loader = DataLoader(
                subset_dataset, batch_size=batch_size, shuffle=True, 
                num_workers=2, pin_memory=True)
            
            print(f"✅ {dataset_name} 부분집합 생성 완료! ({subset_size}/{total_size} 샘플)")
            
            return subset_loader
            
        except Exception as e:
            print(f"❌ {dataset_name} 부분집합 생성 실패: {str(e)}")
            raise
    
    def print_dataset_summary(self):
        """지원 데이터셋 요약 출력"""
        
        print("\n" + "="*80)
        print("📦 지원 데이터셋 요약")
        print("="*80)
        
        categories = {
            '🔢 숫자 인식': ['MNIST', 'FashionMNIST', 'SVHN'],
            '🖼️ 자연 이미지': ['CIFAR10', 'CIFAR100', 'STL10'],
            '🏢 Office-31': ['Office31_Amazon', 'Office31_Webcam', 'Office31_DSLR'],
            '🏠 Office-Home': ['OfficeHome']
        }
        
        for category, datasets in categories.items():
            print(f"\n{category}:")
            print("-" * 60)
            for dataset_name in datasets:
                if dataset_name in self.supported_datasets:
                    info = self.get_dataset_info(dataset_name)
                    print(f"  📊 {dataset_name:<20} | "
                          f"클래스: {info['num_classes']:>3}개 | "
                          f"크기: {info['image_size']:>3}x{info['image_size']} | "
                          f"채널: {info['channels']} | "
                          f"{info['description']}")
        
        print("\n" + "="*80)

    def get_unified_transforms(self, source_dataset, target_dataset, is_training=True):
        """도메인 적응을 위한 통일된 변환 반환 (채널 및 크기 통일)"""
        
        source_info = self.get_dataset_info(source_dataset)
        target_info = self.get_dataset_info(target_dataset)
        
        # 🔧 통일된 설정 결정
        # 더 큰 이미지 크기 사용
        unified_size = max(source_info['image_size'], target_info['image_size'])
        # 더 많은 채널 수 사용 (3채널로 통일)
        unified_channels = max(source_info['channels'], target_info['channels'])
        
        print(f"🔧 도메인 적응 변환 설정:")
        print(f"   📐 통일 이미지 크기: {unified_size}x{unified_size}")
        print(f"   🎨 통일 채널 수: {unified_channels}")
        
        # ImageNet 정규화 사용 (사전 훈련 모델 호환)
        if unified_channels == 3:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5]
            std = [0.5]
        
        if is_training:
            transform = transforms.Compose([
                transforms.Resize((unified_size, unified_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                # 🎯 핵심: 1채널→3채널 변환
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((unified_size, unified_size)),
                transforms.ToTensor(),
                # 🎯 핵심: 1채널→3채널 변환
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        return transform
    
    def load_dataset_for_domain_adaptation(self, source_dataset, target_dataset, batch_size=32, shuffle=True, num_workers=2):
        """도메인 적응용 데이터셋 로더 (채널 및 크기 통일)"""
        
        print(f"🔄 도메인 적응용 데이터셋 로딩: {source_dataset} → {target_dataset}")
        
        # 통일된 변환 생성
        train_transform = self.get_unified_transforms(source_dataset, target_dataset, is_training=True)
        test_transform = self.get_unified_transforms(source_dataset, target_dataset, is_training=False)
        
        try:
            # 소스 데이터셋 로드
            source_train_dataset, source_test_dataset = self._load_single_dataset(
                source_dataset, train_transform, test_transform)
            
            # 타겟 데이터셋 로드
            target_train_dataset, target_test_dataset = self._load_single_dataset(
                target_dataset, train_transform, test_transform)
            
            # DataLoader 생성
            source_train_loader = DataLoader(
                source_train_dataset, batch_size=batch_size, shuffle=shuffle, 
                num_workers=num_workers, pin_memory=True)
            source_test_loader = DataLoader(
                source_test_dataset, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=True)
            target_train_loader = DataLoader(
                target_train_dataset, batch_size=batch_size, shuffle=shuffle, 
                num_workers=num_workers, pin_memory=True)
            target_test_loader = DataLoader(
                target_test_dataset, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=True)
            
            print(f"✅ 도메인 적응용 데이터 로더 생성 완료!")
            print(f"   📊 소스 훈련: {len(source_train_dataset):,}개")
            print(f"   📊 타겟 훈련: {len(target_train_dataset):,}개")
            
            return source_train_loader, target_train_loader, source_test_loader, target_test_loader
            
        except Exception as e:
            print(f"❌ 도메인 적응용 데이터 로더 생성 실패: {str(e)}")
            raise
    
    def _load_single_dataset(self, dataset_name, train_transform, test_transform):
        """단일 데이터셋 로드 (내부 함수)"""
        
        if dataset_name == 'MNIST':
            train_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=test_transform)
                
        elif dataset_name == 'FashionMNIST':
            train_dataset = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=False, download=True, transform=test_transform)
                
        elif dataset_name == 'SVHN':
            train_dataset = torchvision.datasets.SVHN(
                root=self.data_dir, split='train', download=True, transform=train_transform)
            test_dataset = torchvision.datasets.SVHN(
                root=self.data_dir, split='test', download=True, transform=test_transform)
                
        elif dataset_name == 'CIFAR10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=test_transform)
                
        elif dataset_name == 'CIFAR100':
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=False, download=True, transform=test_transform)
                
        elif dataset_name == 'STL10':
            train_dataset = torchvision.datasets.STL10(
                root=self.data_dir, split='train', download=True, transform=train_transform)
            test_dataset = torchvision.datasets.STL10(
                root=self.data_dir, split='test', download=True, transform=test_transform)
                
        elif dataset_name.startswith('Office31_'):
            from office31_loader import Office31Loader
            office31_loader = Office31Loader()
            domain = dataset_name.split('_')[1].lower()
            train_dataset, test_dataset = office31_loader.load_domain_data(domain)
            
        elif dataset_name == 'OfficeHome':
            from officehome_loader import OfficeHomeLoader
            officehome_loader = OfficeHomeLoader(root=self.data_dir)
            domain = dataset_name.split('_')[1].lower()
            train_dataset, test_dataset = officehome_loader.load_domain_data(domain)
            
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
        
        return train_dataset, test_dataset
    
    def get_subset_loader_unified(self, source_dataset, target_dataset, dataset_name, subset_size=500, batch_size=32, is_training=True):
        """도메인 적응용 통일된 부분집합 로더 반환"""
        
        print(f"🎯 {dataset_name} 통일 부분집합 생성 중 (크기: {subset_size})")
        
        # 통일된 변환 사용
        transform = self.get_unified_transforms(source_dataset, target_dataset, is_training=is_training)
        
        try:
            dataset, _ = self._load_single_dataset(dataset_name, transform, transform)
            if not is_training:
                _, dataset = self._load_single_dataset(dataset_name, transform, transform)
            
            # 랜덤 부분집합 생성
            total_size = len(dataset)
            subset_size = min(subset_size, total_size)
            indices = np.random.choice(total_size, subset_size, replace=False)
            subset_dataset = Subset(dataset, indices)
            
            # DataLoader 생성
            subset_loader = DataLoader(
                subset_dataset, batch_size=batch_size, shuffle=True, 
                num_workers=2, pin_memory=True)
            
            print(f"✅ {dataset_name} 통일 부분집합 생성 완료! ({subset_size}/{total_size} 샘플)")
            
            return subset_loader
            
        except Exception as e:
            print(f"❌ {dataset_name} 통일 부분집합 생성 실패: {str(e)}")
            raise

# 전역 데이터셋 매니저 인스턴스
dataset_manager = DatasetManager()

def get_data_loaders(source_dataset, target_dataset, batch_size=32):
    """소스와 타겟 데이터셋 로더 반환 (호환성 함수)"""
    
    print(f"🔄 데이터 로더 생성: {source_dataset} → {target_dataset}")
    
    source_train_loader, source_test_loader = dataset_manager.load_dataset(
        source_dataset, batch_size=batch_size)
    target_train_loader, target_test_loader = dataset_manager.load_dataset(
        target_dataset, batch_size=batch_size)
    
    return source_train_loader, source_test_loader, target_train_loader, target_test_loader

def load_officehome_domains(source_domain, target_domain, data_root='./data', batch_size=32, num_workers=2):
    """Office-Home 도메인 로딩"""
    
    print(f"🏠 Office-Home 도메인 로딩: {source_domain} → {target_domain}")
    
    try:
        from officehome_loader import OfficeHomeLoader
        
        # Office-Home 로더 생성
        loader = OfficeHomeLoader(root=data_root)
        
        # 도메인 이름 매핑
        domain_mapping = {
            'Art': 'art',
            'Clipart': 'clipart', 
            'Product': 'product',
            'Real World': 'real_world',
            'RealWorld': 'real_world'  # 공백 없는 버전도 지원
        }
        
        # 소스/타겟 도메인 키 변환
        source_key = domain_mapping.get(source_domain, source_domain.lower())
        target_key = domain_mapping.get(target_domain, target_domain.lower())
        
        # 소스 도메인 데이터 로드
        print(f"📤 소스 도메인 로딩: {source_domain} ({source_key})")
        source_train_loader, source_test_loader = loader.load_domain_data(
            source_key, batch_size=batch_size, shuffle=True
        )
        
        # 타겟 도메인 데이터 로드
        print(f"📥 타겟 도메인 로딩: {target_domain} ({target_key})")
        target_train_loader, target_test_loader = loader.load_domain_data(
            target_key, batch_size=batch_size, shuffle=True
        )
        
        # 클래스 수 확인 (Office-Home은 65개 클래스)
        num_classes = 65
        
        print(f"✅ Office-Home 도메인 로딩 완료!")
        print(f"   📊 클래스 수: {num_classes}")
        print(f"   📦 소스 배치: {len(source_train_loader)}개")
        print(f"   🎯 타겟 배치: {len(target_train_loader)}개")
        
        return {
            'source_train_loader': source_train_loader,
            'source_test_loader': source_test_loader,
            'target_train_loader': target_train_loader,
            'target_test_loader': target_test_loader,
            'num_classes': num_classes,
            'source_domain': source_domain,
            'target_domain': target_domain
        }
        
    except ImportError as e:
        print(f"❌ Office-Home 로더를 찾을 수 없습니다: {e}")
        print("💡 officehome_loader.py 파일이 필요합니다.")
        raise
    
    except Exception as e:
        print(f"❌ Office-Home 도메인 로딩 실패: {e}")
        raise

if __name__ == "__main__":
    # 테스트 실행
    manager = DatasetManager()
    manager.print_dataset_summary()
    
    # 샘플 데이터 로딩 테스트
    print("\n🧪 샘플 데이터 로딩 테스트:")
    try:
        train_loader, test_loader = manager.load_dataset('SVHN', batch_size=64)
        print("✅ SVHN 데이터셋 로딩 성공!")
        
        subset_loader = manager.get_subset_loader('MNIST', subset_size=100)
        print("✅ MNIST 부분집합 생성 성공!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}") 