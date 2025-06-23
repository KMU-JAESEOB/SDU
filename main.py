#!/usr/bin/env python3
# main.py - SDA-U 프레임워크 메인 실행 스크립트

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings('ignore')

print("🚀 SDA-U 프레임워크 메인 실행!")
print("=" * 60)

# 설정 로드
try:
    from config import get_config
    config = get_config()
    print(f"🚀 설정 로드 완료: {config['gpu_name']}")
except ImportError:
    # 폴백 설정
    config = {
        'batch_size': 64,
        'num_epochs': 3,
        'architecture': 'resnet50',  # 기존 모델로 변경
        'target_subset_size': 1000,
        'num_unlearn_steps': 5,
        'influence_samples': 300,
        'lambda_u': 0.6,
        'beta': 0.1,
        'learning_rate': 1e-3,
        'save_models': True,
        'save_results': True,
        'gpu_name': 'Default'
    }
    print("⚙️ 기본 설정 사용 (ResNet50)")

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 사용 디바이스: {device}")

# 결과 저장 디렉토리 생성
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 전역 변수
best_target_accuracy = 0.0
performance_history = []

# 1. 데이터 변환 - Office-31 호환성 개선 (중복 제거)
def get_transforms(architecture):
    """아키텍처에 맞는 데이터 변환을 반환합니다 (Office-31 호환)"""
    if 'vit' in architecture.lower():
        # Vision Transformer는 224x224 크기 사용
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet 정규화
        ])
    elif 'resnet' in architecture.lower():
        # ResNet은 Office-31에 맞게 224x224, 3채널 사용
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet 정규화
        ])
    else:
        # 기본 변환 (Office-31 호환)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

# 2. CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=31):  # Office-31에 맞게 기본값 변경
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Office-31은 RGB 이미지
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 3. 모델 생성 (Office-31 호환성 강화)
def create_model(architecture, num_classes=31):
    """Office-31 호환 모델 생성 (사전 훈련 가중치 + 채널 최적화)"""
    print(f"🏗️ 모델 생성 중: {architecture}")
    
    # 사전 훈련 가중치 사용 여부 확인
    use_pretrained = config.get('use_pretrained', True)
    weights = 'IMAGENET1K_V1' if use_pretrained else None
    
    try:
        if architecture == 'custom_cnn':
            model = SimpleCNN(num_classes=num_classes)
            print("✅ 커스텀 CNN 생성 완료 (3채널 RGB 입력)")
            return model
            
        elif 'resnet' in architecture.lower():
            # ResNet 계열 모델 생성 (사전 훈련 가중치 사용)
            if architecture == 'resnet18':
                model = torchvision.models.resnet18(weights=weights)
                print(f"📦 ResNet18 로드 (사전훈련: {use_pretrained})")
            elif architecture == 'resnet50':
                model = torchvision.models.resnet50(weights=weights)
                print(f"📦 ResNet50 로드 (사전훈련: {use_pretrained})")
            else:
                # 기본값으로 ResNet50 사용
                model = torchvision.models.resnet50(weights=weights)
                print(f"📦 알 수 없는 ResNet 변형 {architecture}, ResNet50 사용 (사전훈련: {use_pretrained})")
            
            # 🚨 핵심: 첫 번째 convolution은 사전훈련시 이미 3채널이므로 그대로 사용
            if use_pretrained:
                print(f"✅ 사전 훈련된 conv1 사용: {model.conv1}")
            else:
                # 사전 훈련을 사용하지 않을 때만 수동으로 3채널 설정
                model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                print(f"✅ 새로운 conv1: {model.conv1} (3채널 → 64채널)")
            
            # ⚡ 백본 동결 설정 (fine-tuning vs full training)
            freeze_backbone = config.get('freeze_backbone', False)
            if freeze_backbone and use_pretrained:
                for param in model.parameters():
                    param.requires_grad = False
                # 분류 레이어만 학습 가능하게 설정
                model.fc.requires_grad_(True)
                print("🔒 백본 동결, 분류 레이어만 학습")
            else:
                print("🔓 전체 모델 fine-tuning")
            
            # 마지막 분류 레이어를 Office-31 클래스 수에 맞게 조정
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            print(f"✅ 분류 레이어 조정: → {num_classes}개 클래스")
            
            return model
            
        elif 'vit' in architecture.lower():
            # Vision Transformer 모델 생성
            if architecture == 'vit_b_16':
                model = torchvision.models.vit_b_16(weights=weights)
                model.heads = nn.Linear(model.heads.head.in_features, num_classes)
            elif architecture == 'vit_l_16':
                model = torchvision.models.vit_l_16(weights=weights)
                model.heads = nn.Linear(model.heads.head.in_features, num_classes)
            else:
                model = torchvision.models.vit_b_16(weights=weights)
                model.heads = nn.Linear(model.heads.head.in_features, num_classes)
            
            print(f"✅ ViT 모델 생성 완료 (224x224 3채널 입력, 사전훈련: {use_pretrained})")
            return model
            
        else:
            # 알 수 없는 아키텍처는 ResNet50으로 대체
            print(f"⚠️ 알 수 없는 아키텍처: {architecture}, ResNet50으로 대체")
            model = torchvision.models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            print(f"✅ ResNet50 대체 모델 생성 완료 (3채널 RGB 입력, 사전훈련: {use_pretrained})")
            return model
            
    except Exception as e:
        print(f"❌ 모델 생성 실패: {e}")
        print("🔄 기본 CNN으로 대체")
        model = SimpleCNN(num_classes=num_classes)
        print("✅ 기본 CNN 생성 완료 (3채널 RGB 입력)")
        return model

# 4. 데이터 로더 (통합된 데이터셋 관리자 사용)
def get_data_loaders(batch_size, architecture, source_dataset_name='SVHN', target_dataset_name='MNIST'):
    """통합된 데이터셋 관리자를 사용하는 데이터 로더 생성 (조건부 채널 통일)"""
    
    from dataset_manager import dataset_manager
    
    print(f"🎯 도메인 적응 설정:")
    print(f"   📤 소스: {source_dataset_name}")
    print(f"   📥 타겟: {target_dataset_name}")
    
    # 🔧 채널 통일이 필요한 조합 정의
    channel_unification_needed = [
        ('SVHN', 'MNIST'),          # 3채널 → 1채널
        ('MNIST', 'SVHN'),          # 1채널 → 3채널  
        ('SVHN', 'FashionMNIST'),   # 3채널 → 1채널
        ('FashionMNIST', 'SVHN'),   # 1채널 → 3채널
        ('CIFAR10', 'MNIST'),       # 3채널 → 1채널
        ('MNIST', 'CIFAR10'),       # 1채널 → 3채널
        ('CIFAR10', 'FashionMNIST'), # 3채널 → 1채널
        ('FashionMNIST', 'CIFAR10'), # 1채널 → 3채널
    ]
    
    current_combination = (source_dataset_name, target_dataset_name)
    needs_unification = current_combination in channel_unification_needed
    
    try:
        if needs_unification:
            print(f"🔧 채널 통일 적용: {source_dataset_name}({dataset_manager.get_dataset_info(source_dataset_name)['channels']}ch) ↔ {target_dataset_name}({dataset_manager.get_dataset_info(target_dataset_name)['channels']}ch)")
            
            # 🔄 통일된 도메인 적응 로더 사용
            source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
                dataset_manager.load_dataset_for_domain_adaptation(
                    source_dataset_name, target_dataset_name, 
                    batch_size=batch_size, shuffle=True
                )
        else:
            print(f"📦 일반 로더 사용: 채널 통일 불필요")
            
            # 🔄 일반 데이터셋 로더 사용 (기존 방식)
            source_train_loader, source_test_loader = dataset_manager.load_dataset(
                source_dataset_name, batch_size=batch_size, shuffle=True)
            target_train_loader, target_test_loader = dataset_manager.load_dataset(
                target_dataset_name, batch_size=batch_size, shuffle=True)
        
        print(f"✅ 데이터 로더 생성 완료!")
        print(f"   📊 소스 훈련 배치 수: {len(source_train_loader)}")
        print(f"   📊 타겟 훈련 배치 수: {len(target_train_loader)}")
        
        return source_train_loader, target_train_loader, source_test_loader, target_test_loader
        
    except Exception as e:
        print(f"❌ 데이터 로더 생성 실패: {str(e)}")
        raise

# 5. 모델 저장 함수 (도메인별 구분 저장)
def save_model(model, filename, additional_info=None, source_domain=None, target_domain=None):
    """모델과 추가 정보를 도메인별 디렉토리에 저장합니다."""
    
    # 도메인 정보 추출 (파라미터 우선, 없으면 config에서)
    source_dataset = source_domain if source_domain else config.get('source_dataset', 'Unknown')
    target_dataset = target_domain if target_domain else config.get('target_dataset', 'Unknown')
    
    # 도메인 이름 정리 (Office31_Amazon → Amazon)
    source_name = source_dataset.split('_')[-1] if '_' in source_dataset else source_dataset
    target_name = target_dataset.split('_')[-1] if '_' in target_dataset else target_dataset
    
    # 실험 이름 생성
    experiment_name = f"{source_name}2{target_name}"
    
    # 도메인별 디렉토리 생성
    domain_dir = f'models/{experiment_name}'
    os.makedirs(domain_dir, exist_ok=True)
    
    # 타임스탬프 추가 (중복 방지)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = filename.replace('.pt', '')
    timestamped_filename = f"{base_name}_{timestamp}.pt"
    
    # 저장할 정보 구성
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'experiment_name': experiment_name,
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'timestamp': datetime.now().isoformat(),
        'filename': filename
    }
    if additional_info:
        save_dict.update(additional_info)
    
    # 두 가지 방식으로 저장
    full_path = f'{domain_dir}/{filename}'
    timestamped_path = f'{domain_dir}/{timestamped_filename}'
    
    # 1. 최신 버전 (덮어쓰기)
    torch.save(save_dict, full_path)
    
    # 2. 타임스탬프 버전 (보존용)
    torch.save(save_dict, timestamped_path)
    
    print(f"💾 모델 저장: {full_path}")
    print(f"💾 백업 저장: {timestamped_path}")
    
    return full_path, timestamped_path

# 6. 성능 추적 함수
def track_performance(epoch, source_acc, target_acc, loss):
    """성능을 추적하고 최고 성능 모델을 저장합니다."""
    global best_target_accuracy, performance_history
    
    performance_history.append({
        'epoch': epoch,
        'source_accuracy': source_acc,
        'target_accuracy': target_acc,
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    })
    
    if target_acc > best_target_accuracy:
        best_target_accuracy = target_acc
        return True  # 새로운 최고 성능
    return False

# 7. 향상된 훈련 함수 (고급 설정 적용)
def train_model_with_evaluation(model, source_loader, target_loader, num_epochs=3, source_domain=None, target_domain=None):
    print(f"🏋️ 고성능 모델 훈련 시작 ({num_epochs} 에포크)")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # 🔧 고급 옵티마이저 설정
    learning_rate = config.get('learning_rate', 2e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(f"⚙️ 옵티마이저: AdamW (lr={learning_rate}, wd={weight_decay})")
    
    # 🔥 학습률 스케줄러 설정
    scheduler_type = config.get('scheduler_type', 'cosine')
    warmup_epochs = config.get('warmup_epochs', 2)
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        print("📈 코사인 어닐링 스케줄러 사용")
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
        print("📉 스텝 스케줄러 사용")
    else:
        scheduler = None
        print("📊 고정 학습률 사용")
    
    # 🎯 그래디언트 클리핑 설정
    gradient_clip = config.get('gradient_clip', 1.0)
    print(f"✂️ 그래디언트 클리핑: {gradient_clip}")
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # 워밍업 학습률 조정
        if epoch < warmup_epochs and scheduler is not None:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"🔥 워밍업 학습률: {warmup_lr:.6f}")
        
        pbar = tqdm(source_loader, desc=f"에포크 {epoch+1}/{num_epochs}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 🎯 그래디언트 클리핑 적용
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 현재 학습률 표시
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # 빠른 테스트 모드 (설정에서 제어)
            if config.get('quick_test', False) and batch_count >= 30:
                break
        
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        
        # 학습률 스케줄러 업데이트 (워밍업 이후)
        if epoch >= warmup_epochs and scheduler is not None:
            scheduler.step()
            print(f"📊 학습률 업데이트: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 중간 평가 (동적 데이터셋 이름 표시)
        print(f"\n📊 에포크 {epoch+1} 중간 평가:")
        source_dataset_name = source_domain if source_domain else 'Unknown'
        target_dataset_name = target_domain if target_domain else 'Unknown'
        source_acc = evaluate_model(model, source_loader, max_batches=15, domain_name=f"소스({source_dataset_name})")
        target_acc = evaluate_model(model, target_loader, max_batches=15, domain_name=f"타겟({target_dataset_name})")
        
        # 성능 추적 및 최고 성능 모델 저장
        is_best = track_performance(epoch+1, source_acc, target_acc, avg_loss)
        if is_best:
            save_model(model, 'best_model.pt', {
                'epoch': epoch+1,
                'source_accuracy': source_acc,
                'target_accuracy': target_acc,
                'loss': avg_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, source_domain=source_domain, target_domain=target_domain)
            print(f"🏆 새로운 최고 성능! 타겟 정확도: {target_acc:.2f}%")
        
        print(f"에포크 {epoch+1} 완료, 평균 손실: {avg_loss:.4f}")
    
    # 소스 모델 저장
    save_model(model, 'source_model.pt', {
        'final_source_accuracy': source_acc,
        'final_target_accuracy': target_acc,
        'training_losses': losses,
        'final_learning_rate': optimizer.param_groups[0]['lr']
    }, source_domain=source_domain, target_domain=target_domain)
    
    print(f"✅ 훈련 완료! 최종 학습률: {optimizer.param_groups[0]['lr']:.6f}")
    return losses

# 8. 향상된 영향도 계산 (논문 공식 기반) - 안전 장치 추가
def compute_influence_scores_enhanced(model, source_loader, target_batch, num_samples=300):
    print("🔍 향상된 영향도 점수 계산 중... (Pseudo-label 사용)")
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    target_data, _ = target_batch  # ❌ 실제 라벨 무시 (Unsupervised)
    target_data = target_data.to(device)
    
    # 🎯 Pseudo-label 생성 (실제 라벨 대신 사용)
    with torch.no_grad():
        target_output = model(target_data)
        target_pseudo_labels = torch.argmax(target_output, dim=1)
        confidence_scores = torch.max(torch.softmax(target_output, dim=1), dim=1)[0]
    
    print(f"🎯 Pseudo-label 생성 완료: 평균 확신도 {confidence_scores.mean().item():.3f}")
    
    # 🚨 클래스 인덱스 안전 검증 추가
    num_classes = model.fc.out_features if hasattr(model, 'fc') else model.classifier[-1].out_features
    print(f"📊 모델 클래스 수: {num_classes}")
    print(f"🎯 Pseudo-label 범위: {target_pseudo_labels.min().item()} ~ {target_pseudo_labels.max().item()}")
    
    # 라벨 범위 검증 (Pseudo-label은 이미 모델 출력이므로 안전해야 함)
    if target_pseudo_labels.max().item() >= num_classes or target_pseudo_labels.min().item() < 0:
        print(f"❌ Pseudo-label 범위 오류! 범위: [{target_pseudo_labels.min().item()}, {target_pseudo_labels.max().item()}]")
        target_pseudo_labels = torch.clamp(target_pseudo_labels, 0, num_classes - 1)
        print(f"✅ 수정된 Pseudo-label 범위: {target_pseudo_labels.min().item()} ~ {target_pseudo_labels.max().item()}")
    
    # 타겟 배치 그래디언트 계산 (Pseudo-label 사용)
    model.zero_grad()
    target_output = model(target_data)
    target_loss = criterion(target_output, target_pseudo_labels)  # ✅ Pseudo-label 사용
    target_loss.backward()
    
    target_grads = []
    for param in model.parameters():
        if param.grad is not None:
            target_grads.append(param.grad.data.flatten())
    target_grad = torch.cat(target_grads) if target_grads else torch.zeros(1).to(device)
    
    # 헤시안 근사 (단순화된 버전)
    hessian_inv_approx = 1.0  # 실제로는 더 복잡한 계산이 필요
    
    influence_scores = []
    sample_count = 0
    invalid_samples = 0
    
    print(f"🎯 {num_samples}개 샘플의 영향도 계산...")
    
    for batch_idx, (data, labels) in enumerate(source_loader):
        if sample_count >= num_samples:
            break
            
        data, labels = data.to(device), labels.to(device)
        
        # 🚨 소스 라벨 범위 검증 및 수정
        original_labels = labels.clone()
        labels = torch.clamp(labels, 0, num_classes - 1)
        
        # 수정된 라벨 수 카운트
        modified_count = (original_labels != labels).sum().item()
        if modified_count > 0:
            invalid_samples += modified_count
        
        for i in range(min(len(data), num_samples - sample_count)):
            try:
                model.zero_grad()
                output = model(data[i:i+1])
                
                # 추가 안전 검증
                if labels[i] >= num_classes or labels[i] < 0:
                    print(f"⚠️ 건너뛰는 샘플: 라벨 {labels[i].item()}, 클래스 수 {num_classes}")
                    continue
                
                loss = criterion(output, labels[i:i+1])
                loss.backward()
                
                sample_grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        sample_grads.append(param.grad.data.flatten())
                sample_grad = torch.cat(sample_grads) if sample_grads else torch.zeros(1).to(device)
                
                # 논문 공식: I_up(z_i, D_T^batch) = -∇_θ L(D_T^batch, θ)^T H_θ^(-1) ∇_θ L(z_i, θ)
                influence = -torch.dot(target_grad, sample_grad).item() * hessian_inv_approx
                influence_scores.append(influence)
                sample_count += 1
                
            except RuntimeError as e:
                if "assert" in str(e).lower() or "index" in str(e).lower():
                    print(f"⚠️ 샘플 건너뛰기 (라벨 오류): {e}")
                    invalid_samples += 1
                    continue
                else:
                    raise e
        
        if sample_count >= num_samples:
            break
    
    if invalid_samples > 0:
        print(f"⚠️ 수정/건너뛴 샘플: {invalid_samples}개")
    
    print(f"✅ {len(influence_scores)}개 샘플 영향도 계산 완료 (Pseudo-label 기반)")
    return influence_scores

# 9. 하이브리드 스코어링 (논문 구현)
def unsupervised_target_sample_selection(model, target_loader, num_samples=800):
    """타겟 라벨을 모른다고 가정하고 불확실성 기반으로 타겟 샘플 선별"""
    print(f"🎯 1단계: 불확실성 기반 타겟 샘플 선별 ({num_samples}개)")
    
    model.eval()
    uncertainty_scores = []
    all_samples = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(target_loader):
            # 배치 데이터 처리 (라벨 유무에 관계없이)
            if len(batch_data) == 2:
                data, labels = batch_data
            else:
                data = batch_data
                labels = None
            
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            
            # 불확실성 계산 (엔트로피)
            for i in range(data.size(0)):
                sample_probs = probs[i]
                entropy = -torch.sum(sample_probs * torch.log(sample_probs + 1e-8)).item()
                
                uncertainty_scores.append(entropy)
                if labels is not None:
                    all_samples.append((data[i].cpu(), labels[i].cpu()))
                else:
                    all_samples.append((data[i].cpu(), torch.tensor(0)))  # 더미 라벨
    
    # 불확실성이 높은 순으로 정렬하여 상위 num_samples개 선택
    sorted_indices = sorted(range(len(uncertainty_scores)), 
                           key=lambda i: uncertainty_scores[i], reverse=True)
    
    selected_samples = [all_samples[i] for i in sorted_indices[:num_samples]]
    selected_scores = [uncertainty_scores[i] for i in sorted_indices[:num_samples]]
    
    print(f"✅ 불확실성 기반 타겟 샘플 선별 완료: {len(selected_samples)}개")
    print(f"📊 평균 불확실성: {np.mean(selected_scores):.4f}")
    
    return selected_samples, selected_scores

def compute_influence_and_select_harmful(model, source_loader, target_samples, num_harmful=50):
    """선별된 타겟 샘플과 비교하여 유해한 소스 샘플 선별"""
    print(f"🎯 2단계: 유해 소스 샘플 선별 ({num_harmful}개)")
    
    # 타겟 샘플들에 대한 영향도 계산
    target_batch_for_influence = []
    for i, (data, label) in enumerate(target_samples[:100]):  # 처음 100개만 사용
        target_batch_for_influence.append((data, label))
    
    # 영향도 점수 계산
    influence_scores = compute_influence_scores_enhanced(
        model, source_loader, target_batch_for_influence, num_samples=500)
    
    # 영향도가 가장 높은 (유해한) 소스 샘플들 선별
    harmful_indices = sorted(range(len(influence_scores)), 
                           key=lambda i: abs(influence_scores[i]), reverse=True)[:num_harmful]
    
    # 유해한 소스 샘플들 수집
    harmful_samples = []
    harmful_scores = []
    
    sample_count = 0
    for batch_idx, (data, labels) in enumerate(source_loader):
        for i in range(data.size(0)):
            if sample_count in harmful_indices:
                harmful_samples.append((data[i].cpu(), labels[i].cpu()))
                harmful_scores.append(influence_scores[sample_count])
            sample_count += 1
            
            if len(harmful_samples) >= num_harmful:
                break
        if len(harmful_samples) >= num_harmful:
            break
    
    print(f"✅ 유해 소스 샘플 선별 완료: {len(harmful_samples)}개")
    print(f"📊 평균 영향도: {np.mean([abs(s) for s in harmful_scores]):.4f}")
    
    return harmful_samples, harmful_scores

def compute_hybrid_scores(model, target_samples, influence_scores, lambda_u=0.6, beta=0.1):
    """논문의 하이브리드 스코어링: S(x) = λ_u·I(x;θ') + (1-λ_u)·div(x, D_T^sub) + β·H(x)"""
    print("🎯 하이브리드 스코어 계산 중...")
    
    model.eval()
    hybrid_scores = []
    
    with torch.no_grad():
        for i, sample in enumerate(target_samples):
            if i >= len(influence_scores):
                break
                
            # 샘플 데이터 처리 - 유연한 언패킹
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                data, _ = sample[0], sample[1]
            else:
                data = sample
            
            # 데이터를 디바이스로 이동하고 배치 차원 추가
            data = data.to(device)
            if len(data.shape) == 3:  # [C, H, W] -> [1, C, H, W]
                data = data.unsqueeze(0)
            
            output = model(data)
            probs = torch.softmax(output, dim=1)
            
            # 1. 영향도 스코어 (정규화)
            influence_score = influence_scores[i]
            
            # 2. 다양성 스코어 (단순화된 버전)
            diversity_score = torch.max(probs).item()  # 최대 확률의 역수로 근사
            
            # 3. 불확실성 스코어 (엔트로피)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            
            # 하이브리드 스코어 계산
            hybrid_score = (lambda_u * influence_score + 
                          (1 - lambda_u) * (1 - diversity_score) + 
                          beta * entropy)
            
            hybrid_scores.append(hybrid_score)
    
    print(f"✅ {len(hybrid_scores)}개 하이브리드 스코어 계산 완료")
    return hybrid_scores

# 10. 향상된 언러닝 (DOS 알고리즘) - BatchNorm 안전 버전
def perform_dos_unlearning(model, harmful_data, influence_scores, num_steps=5):
    print("🔄 DOS (Dynamic Orthogonal Scaling) 언러닝 수행 중...")
    
    # BatchNorm 문제 해결: 훈련 모드 유지하되 배치 크기 조정
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 동적 가중치 계산 (영향도에 비례)
    abs_scores = [abs(s) for s in influence_scores]
    max_abs = max(abs_scores) if abs_scores else 1.0
    weights = [s / max_abs for s in abs_scores]  # 정규화된 가중치
    
    unlearn_losses = []
    
    # BatchNorm 안정성을 위한 최소 배치 크기 보장
    min_batch_size = 2  # BatchNorm이 안정적으로 작동하는 최소 크기
    effective_batch_size = max(min_batch_size, min(8, len(harmful_data)))
    
    for step in range(num_steps):
        total_loss = 0.0
        processed_batches = 0
        
        # 데이터를 배치로 그룹핑
        for batch_start in range(0, len(harmful_data), effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, len(harmful_data))
            
            # 배치가 너무 작으면 건너뛰기 (BatchNorm 안정성)
            if batch_end - batch_start < min_batch_size:
                continue
            
            batch_data = []
            batch_labels = []
            batch_weights = []
            
            # 배치 데이터 준비
            for i in range(batch_start, batch_end):
                if i >= len(harmful_data) or i >= len(weights):
                    break
                    
                data, label = harmful_data[i]
                
                # 데이터 타입 확인 및 변환
                data = data.to(device)
                if isinstance(label, int):
                    label = torch.tensor(label).to(device)
                else:
                    label = label.to(device)
                
                batch_data.append(data)
                batch_labels.append(label)
                batch_weights.append(abs(weights[i]))
            
            if len(batch_data) < min_batch_size:
                continue
                
            # 배치 텐서 생성
            batch_data_tensor = torch.stack(batch_data)
            batch_labels_tensor = torch.stack(batch_labels)
            batch_weights_tensor = torch.tensor(batch_weights, device=device)
            
            optimizer.zero_grad()
            
            # 순전파 (BatchNorm이 안전하게 작동)
            output = model(batch_data_tensor)
            
            # 개별 샘플 손실 계산 후 가중치 적용
            batch_loss = 0.0
            for j in range(len(batch_data_tensor)):
                sample_loss = criterion(output[j:j+1], batch_labels_tensor[j:j+1])
                weighted_sample_loss = sample_loss * batch_weights_tensor[j]
                batch_loss += weighted_sample_loss
            
            # 배치 평균 손실
            avg_batch_loss = batch_loss / len(batch_data_tensor)
            
            # 언러닝을 위한 그래디언트 반전 (음의 그래디언트)
            (-avg_batch_loss).backward()
            optimizer.step()
            
            total_loss += avg_batch_loss.item()
            processed_batches += 1
        
        avg_loss = total_loss / max(1, processed_batches)
        unlearn_losses.append(avg_loss)
        
        if (step + 1) % 2 == 0:
            print(f"  DOS 스텝 {step + 1}/{num_steps}, 가중 손실: {avg_loss:.4f}")
    
    print("✅ DOS 언러닝 완료")
    return unlearn_losses

# 11. 타겟 도메인 재학습 (Self-training)
def retrain_with_curated_target_samples(model, curated_samples, adaptation_epochs=5):
    """큐레이션된 타겟 샘플로 Fine-tuning 수행 (실제 라벨 사용)"""
    print(f"🎓 4단계: 선별된 타겟 샘플로 Fine-tuning ({adaptation_epochs}에포크)")
    print("📋 중요: 이 단계에서는 실제 타겟 라벨을 사용합니다!")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(adaptation_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # 배치 단위로 처리
        batch_size = 16
        for i in range(0, len(curated_samples), batch_size):
            batch_end = min(i + batch_size, len(curated_samples))
            batch_data = []
            batch_labels = []
            
            for j in range(i, batch_end):
                data, label = curated_samples[j]  # ✅ 실제 라벨 사용 (Fine-tuning용)
                batch_data.append(data)
                batch_labels.append(label)
            
            if len(batch_data) == 0:
                continue
                
            batch_tensor = torch.stack(batch_data).to(device)
            batch_labels_tensor = torch.stack(batch_labels).to(device)
            
            # 🚨 클래스 수 안전 검증
            num_classes = model.fc.out_features if hasattr(model, 'fc') else model.classifier[-1].out_features
            batch_labels_tensor = torch.clamp(batch_labels_tensor, 0, num_classes - 1)
            
            total_samples += len(batch_data)
            
            # 실제 라벨로 Fine-tuning
            optimizer.zero_grad()
            output = model(batch_tensor)
            loss = criterion(output, batch_labels_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 정확도 계산
            _, predicted = torch.max(output.data, 1)
            correct_predictions += (predicted == batch_labels_tensor).sum().item()
        
        avg_loss = total_loss / max(1, total_samples // batch_size)
        accuracy = correct_predictions / max(1, total_samples) * 100
        
        if (epoch + 1) % 2 == 0:
            print(f"  에포크 {epoch+1}/{adaptation_epochs}: "
                  f"손실={avg_loss:.4f}, "
                  f"정확도={accuracy:.1f}% ({correct_predictions}/{total_samples})")
    
    print(f"✅ 타겟 도메인 Fine-tuning 완료 (최종 정확도: {accuracy:.1f}%)")

# 12. 향상된 평가 함수
def evaluate_model(model, data_loader, max_batches=20, domain_name=""):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"📊 {domain_name} 정확도: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

# 12. 결과 저장 및 분석
def save_comprehensive_results(results):
    """포괄적인 결과를 저장합니다."""
    
    # JSON 결과 저장
    with open('results/sda_u_comprehensive_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 성능 히스토리 저장
    with open('results/performance_history.json', 'w', encoding='utf-8') as f:
        json.dump(performance_history, f, indent=2, ensure_ascii=False)
    
    print("💾 포괄적인 결과 저장 완료")

# 13. 통합된 데이터 로딩 함수
def load_data(args):
    """통합된 데이터 로딩 함수 (Office-Home, Office-31, CIFAR 등 지원)"""
    
    print(f"📦 데이터 로딩 중...")
    print(f"🎯 데이터셋: {args.dataset}")
    
    if args.dataset == 'Office31':
        from office31_loader import Office31Loader
        loader = Office31Loader()
        
        print(f"📤 소스 도메인: {args.source_domain}")
        print(f"📥 타겟 도메인: {args.target_domain}")
        
        # 도메인별 데이터 로드
        source_train_dataset, source_test_dataset = loader.load_domain_data(args.source_domain)
        target_train_dataset, target_test_dataset = loader.load_domain_data(args.target_domain)
        
        # 데이터 로더 생성
        source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        num_classes = 31  # Office-31 클래스 수
        
    elif args.dataset == 'OfficeHome':
        from officehome_loader import OfficeHomeLoader
        loader = OfficeHomeLoader()
        
        print(f"📤 소스 도메인: {args.source_domain}")
        print(f"📥 타겟 도메인: {args.target_domain}")
        
        # 도메인별 데이터 로드
        source_train_dataset, source_test_dataset = loader.load_domain_data(args.source_domain)
        target_train_dataset, target_test_dataset = loader.load_domain_data(args.target_domain)
        
        # 데이터 로더 생성
        source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        num_classes = 65  # Office-Home 클래스 수
        
    else:
        # 기타 데이터셋 (SVHN, MNIST, CIFAR 등)
        from dataset_manager import DatasetManager
        manager = DatasetManager()
        
        # source_domain과 target_domain을 데이터셋 이름으로 사용
        source_dataset_name = args.source_domain
        target_dataset_name = args.target_domain
        
        print(f"📤 소스 데이터셋: {source_dataset_name}")
        print(f"📥 타겟 데이터셋: {target_dataset_name}")
        
        # 데이터셋 정보 가져오기
        source_info = manager.get_dataset_info(source_dataset_name)
        target_info = manager.get_dataset_info(target_dataset_name)
        
        print(f"📊 소스 클래스: {source_info['num_classes']}개")
        print(f"📊 타겟 클래스: {target_info['num_classes']}개")
        
        # 🚨 CIFAR 특별 처리: 클래스 수 불일치 해결
        if (source_dataset_name == 'CIFAR10' and target_dataset_name == 'CIFAR100') or \
           (source_dataset_name == 'CIFAR100' and target_dataset_name == 'CIFAR10'):
            
            print("🔧 CIFAR10↔CIFAR100 클래스 수 불일치 감지 - 특별 처리 모드")
            num_classes = 10  # CIFAR10 기준으로 통일
            print(f"✅ 클래스 수를 CIFAR10 기준(10개)으로 통일")
            print(f"⚠️ CIFAR100 라벨은 0-9 범위로 자동 클리핑됩니다")
            
        else:
            # 일반적인 경우: 더 작은 클래스 수 사용
            num_classes = min(source_info['num_classes'], target_info['num_classes'])
            print(f"🔧 통일된 클래스 수: {num_classes}개")
        
        # 데이터셋 로드
        source_train_loader, source_test_loader = manager.load_dataset(source_dataset_name, args.batch_size)
        target_train_loader, target_test_loader = manager.load_dataset(target_dataset_name, args.batch_size)
    
    print(f"✅ 데이터 로딩 완료! (클래스 수: {num_classes})")
    return source_train_loader, source_test_loader, target_train_loader, target_test_loader, num_classes

# 14. 메인 실행 함수
def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description='SDA-U (Selective Domain Adaptation with Unlearning)')
    
    # 데이터셋 설정
    parser.add_argument('--dataset', type=str, default='Office31', 
                       choices=['Office31', 'OfficeHome', 'SVHN', 'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10'],
                       help='데이터셋 선택')
    parser.add_argument('--source_domain', type=str, required=True,
                       help='소스 도메인 (Office31: amazon/webcam/dslr, OfficeHome: art/clipart/product/real_world)')
    parser.add_argument('--target_domain', type=str, required=True, 
                       help='타겟 도메인 (Office31: amazon/webcam/dslr, OfficeHome: art/clipart/product/real_world)')
    
    # 훈련 파라미터
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=20, help='훈련 에포크')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='학습률')
    parser.add_argument('--architecture', type=str, default='resnet50', 
                       choices=['resnet18', 'resnet50', 'simple_cnn'], help='모델 아키텍처')
    
    # SDA-U 파라미터
    parser.add_argument('--influence_samples', type=int, default=500, help='영향도 계산 샘플 수')
    parser.add_argument('--target_samples', type=int, default=800, help='타겟 샘플 수')
    parser.add_argument('--unlearn_steps', type=int, default=8, help='언러닝 스텝 수')
    parser.add_argument('--lambda_u', type=float, default=0.6, help='하이브리드 스코어링 λ_u')
    parser.add_argument('--beta', type=float, default=0.1, help='하이브리드 스코어링 β')
    parser.add_argument('--adaptation_epochs', type=int, default=10, help='적응 훈련 에포크')
    
    # 저장 경로 파라미터
    parser.add_argument('--model_save_dir', type=str, default='models', help='모델 저장 디렉토리')
    parser.add_argument('--results_file', type=str, default='results/sda_u_comprehensive_results.json', help='결과 파일 경로')
    
    # 🚀 사전 훈련 모델 관련 옵션
    parser.add_argument('--pretrained_model', type=str, default=None, 
                       help='사전 훈련된 모델 경로 (있으면 소스 도메인 훈련 스킵)')
    parser.add_argument('--skip_source_training', action='store_true', 
                       help='소스 도메인 훈련 스킵 (사전 훈련된 가중치 사용)')
    parser.add_argument('--use_pretrained_weights', action='store_true', default=True,
                       help='ImageNet 사전 훈련 가중치 사용')
    
    args = parser.parse_args()

    # GPU 설정 및 최적화
    try:
        from gpu_config import setup_gpu_optimizations, get_gpu_info
        print("🚀 A100 최적화 활성화!")
        setup_gpu_optimizations()
        gpu_info = get_gpu_info()
        print(f"🚀 설정 로드 완료: {gpu_info['name']}")
    except ImportError:
        print("⚠️ GPU 최적화 설정 파일이 없습니다. 기본 설정 사용")
        
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 사용 디바이스: {device}")

    # 데이터 로딩
    source_train_loader, source_test_loader, target_train_loader, target_test_loader, num_classes = load_data(args)
    
    # 모델 생성
    model = create_model(args.architecture, num_classes).to(device)
    
    print(f"🤖 모델 생성 완료: {args.architecture} (클래스: {num_classes})")
    print(f"📊 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 전체 SDA-U 실행
    print("\n" + "="*80)
    print("🚀 SDA-U 프레임워크 실행!")
    print("="*80)
    
    try:
        # 🚀 사전 훈련된 모델 로드 또는 소스 도메인 훈련
        if args.pretrained_model and os.path.exists(args.pretrained_model):
            print(f"📦 사전 훈련된 모델 로드: {args.pretrained_model}")
            checkpoint = torch.load(args.pretrained_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 사전 훈련된 모델 로드 완료!")
            
        elif args.skip_source_training:
            print("⏭️ 소스 도메인 훈련 스킵 - ImageNet 사전 훈련 가중치만 사용")
            print("💡 이 모드에서는 ResNet 등의 ImageNet 사전 훈련 가중치를 그대로 사용합니다")
            
        else:
            # 1. 소스 도메인 사전 훈련
            print("📚 1단계: 소스 도메인 사전 훈련")
            train_model_with_evaluation(model, source_train_loader, source_test_loader, num_epochs=args.num_epochs, 
                                      source_domain=args.source_domain, target_domain=args.target_domain)
        
        # 2. 타겟 도메인 평가 (사전 훈련 후)
        print("\n📊 2단계: 타겟 도메인 초기 평가")
        initial_target_acc = evaluate_model(model, target_test_loader, domain_name="타겟(초기)")
        
        # 소스 도메인 초기 성능도 측정 (참고용)
        initial_source_acc = evaluate_model(model, source_test_loader, domain_name="소스(초기)")
        print(f"📊 소스 도메인 초기 성능: {initial_source_acc:.2f}%")
        
        # 3. 올바른 SDA-U 알고리즘 수행
        print("\n🎯 SDA-U 알고리즘 시작 (올바른 순서)")
        
        # 3-1. 타겟 샘플 선별 (라벨 없이)
        curated_samples, uncertainty_scores = unsupervised_target_sample_selection(
            model, target_train_loader, num_samples=args.target_samples)
        
        # 3-2. 유해한 소스 샘플 선별
        harmful_samples, harmful_scores = compute_influence_and_select_harmful(
            model, source_train_loader, curated_samples, num_harmful=50)
        
        # 3-3. 머신 언러닝 수행
        print("\n🔄 3단계: 머신 언러닝 수행")
        unlearn_losses = perform_dos_unlearning(
            model, harmful_samples, harmful_scores, 
            num_steps=args.unlearn_steps)
        
        # 3-4. 선별된 타겟 샘플로 Fine-tuning
        retrain_with_curated_target_samples(model, curated_samples, args.adaptation_epochs)
        
        # 4. 최종 평가
        print("\n📈 4단계: 최종 성능 평가")
        final_source_acc = evaluate_model(model, source_test_loader, domain_name="소스(최종)")
        final_target_acc = evaluate_model(model, target_test_loader, domain_name="타겟(최종)")
        
        # 5. 결과 저장
        print("\n💾 5단계: 결과 저장")
        results = {
            'experiment_info': {
                'dataset': args.dataset,
                'source_domain': args.source_domain,
                'target_domain': args.target_domain,
                'architecture': args.architecture,
                'num_classes': num_classes
            },
            'performance': {
                'initial_source_accuracy': initial_source_acc,
                'initial_target_accuracy': initial_target_acc,
                'final_source_accuracy': final_source_acc,
                'final_target_accuracy': final_target_acc,
                'target_improvement': final_target_acc - initial_target_acc,
                'source_change': final_source_acc - initial_source_acc
            },
            'sda_u_metrics': {
                'harmful_samples': len(harmful_samples),
                'curated_samples': len(curated_samples),
                'unlearn_steps': args.unlearn_steps,
                'unlearn_losses': unlearn_losses
            },
            'hyperparameters': {
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'lambda_u': args.lambda_u,
                'beta': args.beta
            }
        }
        
        # 결과 저장 (지정된 경로 사용)
        results_dir = os.path.dirname(args.results_file)
        os.makedirs(results_dir, exist_ok=True)
        
        # 결과를 지정된 파일에 저장
        with open(args.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 결과 저장 완료: {args.results_file}")
        
        # 모델 저장 (지정된 경로 사용)
        model_dir = os.path.dirname(args.model_save_dir)
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"{args.source_domain}2{args.target_domain}_sda_u_model.pt"
        model_path = os.path.join(args.model_save_dir, model_filename)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'results': results,
            'args': vars(args)
        }
        torch.save(save_dict, model_path)
        print(f"🤖 모델 저장 완료: {model_path}")
        
        print("\n🎉 SDA-U 실험 완료!")
        print(f"📊 타겟 정확도 향상: {initial_target_acc:.2f}% → {final_target_acc:.2f}% (+{final_target_acc-initial_target_acc:.2f}%)")
        print(f"📊 소스 정확도 변화: {initial_source_acc:.2f}% → {final_source_acc:.2f}% ({final_source_acc-initial_source_acc:+.2f}%)")
        
        # 성능 요약
        if final_target_acc > initial_target_acc:
            print("✅ 타겟 도메인 성능 향상 성공!")
        else:
            print("❌ 타겟 도메인 성능 향상 실패")
            
        if abs(final_source_acc - initial_source_acc) < 2.0:  # 2% 이내 변화
            print("✅ 소스 도메인 성능 유지 성공!")
        else:
            print("⚠️ 소스 도메인 성능 크게 변화됨")
        
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main() 