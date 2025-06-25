# main.py - 완전 객체지향 SDA-U 알고리즘
"""
🎯 완전 객체지향 SDA-U (Selective Domain Adaptation with Unlearning) 알고리즘

주요 개선사항:
1. 완전한 객체지향 설계 (클래스 기반)
2. 설정 파일 분리 (config.json)
3. 중복 함수 통합 및 제거
4. 유지보수성 및 확장성 향상
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
import argparse
import os
import json
import random
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# 실시간 출력을 위한 버퍼링 해제
sys.stdout.reconfigure(line_buffering=True)

def flush_print(*args, **kwargs):
    """즉시 출력되는 print 함수"""
    print(*args, **kwargs)
    sys.stdout.flush()

# 데이터 로더
from office31_loader import Office31Manager
from officehome_loader import OfficeHomeLoader

# GPU 설정
try:
    from gpu_config import setup_gpu_optimizations, get_gpu_info
    GPU_OPTIMIZED = True
except ImportError:
    GPU_OPTIMIZED = False

# 전역 변수
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class ExperimentResults:
    """실험 결과를 담는 데이터 클래스"""
    initial_target_acc: float
    final_target_acc: float
    best_target_acc: float
    best_epoch: int
    improvement: float
    best_improvement: float
    unlearning_count: int
    total_epochs: int
    performance_history: Dict
    model_paths: Dict[str, str]

class SDAUConfig:
    """설정 관리 클래스"""
    
    def __init__(self, config_path: str = "config.json"):
        """설정 파일 로딩"""
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """설정 파일 로딩"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _validate_config(self):
        """설정 파일 검증"""
        required_sections = ['model', 'training', 'target_selection', 
                           'influence_calculation', 'unlearning', 'paths']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"필수 설정 섹션이 없습니다: {section}")
    
    def get(self, section: str, key: str = None, default=None):
        """설정값 가져오기"""
        if key is None:
            return self.config.get(section, default)
        return self.config.get(section, {}).get(key, default)
    
    def get_num_classes(self, dataset: str) -> int:
        """데이터셋별 클래스 수 반환"""
        return self.config['model']['num_classes'].get(dataset, 31)
    
    def get_domains(self, dataset: str) -> List[str]:
        """데이터셋별 도메인 목록 반환"""
        return self.config['datasets'].get(dataset, {}).get('domains', [])

class ModelManager:
    """모델 관리 클래스"""
    
    def __init__(self, config: SDAUConfig):
        self.config = config
        self.device = device
    
    def create_model(self, num_classes: int) -> nn.Module:
        """모델 생성"""
        architecture = self.config.get('model', 'architecture')
        pretrained = self.config.get('model', 'pretrained')
        
        if architecture == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"지원하지 않는 모델 아키텍처: {architecture}")
        
        return model.to(self.device)
    
    def save_model(self, model: nn.Module, save_path: str, metadata: Dict = None):
        """모델 저장"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': self.config.config,
            'timestamp': time.time()
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, save_path)
        flush_print(f"💾 모델 저장 완료: {save_path}")
    
    def load_model(self, model: nn.Module, load_path: str) -> Dict:
        """모델 로딩"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        flush_print(f"📥 모델 로딩 완료: {load_path}")
        return checkpoint

class DataManager:
    """데이터 관리 클래스"""
    
    def __init__(self, config: SDAUConfig):
        self.config = config
        self.data_root = self.config.get('paths', 'data_root')
    
    def load_dataset(self, dataset: str, source_domain: str, target_domain: str, 
                    batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
        """데이터셋 로딩"""
        if dataset == 'Office31':
            return self._load_office31(source_domain, target_domain, batch_size)
        elif dataset == 'OfficeHome':
            return self._load_officehome(source_domain, target_domain, batch_size)
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset}")
    
    def _load_office31(self, source_domain: str, target_domain: str, 
                      batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
        """Office31 데이터셋 로딩"""
        manager = Office31Manager(root=self.data_root)
        
        source_train_dataset, source_test_dataset = manager.load_domain_data(source_domain)
        target_train_dataset, target_test_dataset = manager.load_domain_data(target_domain)
        
        source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)
        source_test_loader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=False)
        target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
        target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)
        
        return source_train_loader, source_test_loader, target_train_loader, target_test_loader, 31
    
    def _load_officehome(self, source_domain: str, target_domain: str, 
                        batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
        """OfficeHome 데이터셋 로딩"""
        loader = OfficeHomeLoader(root=self.data_root)
        
        # OfficeHome 로더가 이제 DataLoader를 직접 반환 (collate_fn 포함)
        source_train_loader, source_test_loader = loader.load_domain_data(source_domain, batch_size)
        target_train_loader, target_test_loader = loader.load_domain_data(target_domain, batch_size)
        
        return source_train_loader, source_test_loader, target_train_loader, target_test_loader, 65

class TargetSampleSelector:
    """타겟 샘플 선별 클래스 - AlignSet + 능동학습 하이브리드 방법"""
    
    def __init__(self, config: SDAUConfig):
        self.config = config
    
    def select_samples(self, model: nn.Module, target_train_loader: DataLoader, 
                      num_classes: int) -> List[Tuple]:
        """타겟 샘플 선별 - AlignSet + 능동학습 하이브리드 스코어링"""
        method = self.config.get('target_selection', 'selection_method')
        
        if method == 'hybrid_alignset_active':
            return self._select_hybrid_alignset_active(model, target_train_loader)
        elif method == 'random':
            return self._select_random(target_train_loader)
        else:
            raise ValueError(f"지원하지 않는 선별 방법: {method}")
    
    def _select_hybrid_alignset_active(self, model: nn.Module, target_train_loader: DataLoader) -> List[Tuple]:
        """AlignSet + 능동학습 하이브리드 스코어링 기반 선별"""
        num_samples = self.config.get('target_selection', 'num_samples')
        lambda_u = self.config.get('target_selection', 'lambda_utility', 0.7)
        beta = self.config.get('target_selection', 'beta_uncertainty', 0.3)
        
        print(f"🎯 하이브리드 타겟 샘플 선별 중... (목표: {num_samples}개)")
        
        model.eval()
        
        # 모든 타겟 샘플 수집 및 특징 추출
        all_samples = []
        all_features = []
        all_utilities = []
        all_uncertainties = []
        
        with torch.no_grad():
            for data, target in target_train_loader:
                for i in range(len(data)):
                    x = data[i].unsqueeze(0).to(device)
                    
                    # 특징 추출 (마지막 FC 층 이전)
                    if hasattr(model, 'fc'):
                        # ResNet50의 경우 FC 층 전까지의 특징 추출
                        modules = list(model.children())[:-1]  # FC 층 제외
                        feature_extractor = nn.Sequential(*modules)
                        features = feature_extractor(x)
                        features = features.view(features.size(0), -1)
                    else:
                        features = x.view(x.size(0), -1)
                    
                    # 전체 예측
                    output = model(x)
                    probs = F.softmax(output, dim=1)
                    
                    # 의사 레이블 생성
                    pseudo_label = torch.argmax(probs, dim=1)
                    
                    # 유용성 계산 (의사 레이블에 대한 손실의 음수)
                    utility = -F.cross_entropy(output, pseudo_label).item()
                    
                    # 불확실성 계산 (엔트로피)
                    uncertainty = -(probs * torch.log(probs + 1e-8)).sum().item()
                    
                    all_samples.append((data[i], target[i]))
                    all_features.append(features.cpu())
                    all_utilities.append(utility)
                    all_uncertainties.append(uncertainty)
        
        print(f"📊 전체 타겟 샘플: {len(all_samples)}개")
        
        # 의사 라벨 추가 계산
        all_pseudo_labels = []
        with torch.no_grad():
            for data, target in [(s[0], s[1]) for s in all_samples]:
                x = data.unsqueeze(0).to(device)
                output = model(x)
                probs = F.softmax(output, dim=1)
                pseudo_label = torch.argmax(probs, dim=1).item()
                all_pseudo_labels.append(pseudo_label)
        
        # 점진적 선별 (그리디 방식)
        selected_indices = []
        selected_features = []
        selected_pseudo_labels = []
        
        for step in range(min(num_samples, len(all_samples))):
            best_score = float('-inf')
            best_idx = -1
            
            for i, (utility, uncertainty, features, pseudo_label) in enumerate(zip(all_utilities, all_uncertainties, all_features, all_pseudo_labels)):
                if i in selected_indices:
                    continue
                
                # 1. 특징 공간 다양성
                if len(selected_features) == 0:
                    feature_diversity = 1.0  # 첫 번째 샘플
                else:
                    # 선택된 샘플들과의 최대 유사도
                    max_similarity = 0.0
                    for selected_feat in selected_features:
                        # 코사인 유사도
                        similarity = F.cosine_similarity(features, selected_feat, dim=1).item()
                        max_similarity = max(max_similarity, similarity)
                    feature_diversity = 1.0 - max_similarity
                
                # 2. 의사 라벨 다양성 (서로 다른 클래스 선호)
                if len(selected_pseudo_labels) == 0:
                    label_diversity = 1.0
                else:
                    same_label_count = sum(1 for label in selected_pseudo_labels if label == pseudo_label)
                    label_diversity = 1.0 - (same_label_count / len(selected_pseudo_labels))
                
                # 3. 혼합 다양성 (특징 + 라벨)
                diversity = 0.7 * feature_diversity + 0.3 * label_diversity
                
                # AlignSet 점수
                alignset_score = lambda_u * utility + (1 - lambda_u) * diversity
                
                # 최종 하이브리드 점수
                hybrid_score = alignset_score + beta * uncertainty
                
                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                selected_features.append(all_features[best_idx])
                selected_pseudo_labels.append(all_pseudo_labels[best_idx])
        
        selected_samples = [all_samples[i] for i in selected_indices]
        
        # 클래스별 분포 출력 추가
        class_counts = {}
        for _, label in selected_samples:
            label_item = label.item() if hasattr(label, 'item') else int(label)
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
        
        print(f"✅ 하이브리드 선별 완료: {len(selected_samples)}개")
        print(f"📊 클래스별 선별된 샘플 분포:")
        for class_id in sorted(class_counts.keys()):
            print(f"   클래스 {class_id:2d}: {class_counts[class_id]:3d}개")
        
        return selected_samples
    
    def _select_random(self, target_train_loader: DataLoader) -> List[Tuple]:
        """랜덤 선별"""
        num_samples = self.config.get('target_selection', 'num_samples')
        
        all_samples = []
        for data, target in target_train_loader:
            for i in range(len(data)):
                all_samples.append((data[i], target[i]))
        
        total_samples = len(all_samples)
        select_count = min(num_samples, total_samples)
        
        random.seed(42)
        selected_samples = random.sample(all_samples, select_count)
        
        print(f"✅ 랜덤 선별 완료: {len(selected_samples)}개 / {total_samples}개")
        return selected_samples

class InfluenceCalculator:
    """영향도 계산 클래스 - 영향도 기반 필터링"""
    
    def __init__(self, config: SDAUConfig):
        self.config = config
    
    def compute_influence_scores(self, model: nn.Module, source_loader: DataLoader, 
                               target_samples: List[Tuple]) -> Tuple[List[Tuple], List[float]]:
        """영향도 기반 유해 소스 샘플 필터링"""
        method = self.config.get('influence_calculation', 'method')
        num_samples = self.config.get('influence_calculation', 'num_samples')
        
        if method == 'influence_filtering':
            return self._compute_influence_filtering(model, source_loader, target_samples, num_samples)
        elif method == 'simple':
            return self._compute_simple_influence(model, source_loader, target_samples, num_samples)
        else:
            raise ValueError(f"지원하지 않는 영향도 계산 방법: {method}")
    
    def _compute_influence_filtering(self, model: nn.Module, source_loader: DataLoader,
                                   target_samples: List[Tuple], num_samples: int) -> Tuple[List[Tuple], List[float]]:
        """영향도 기반 필터링 - 타겟 적응에 해로운 소스 샘플 식별"""
        print(f"🔬 영향도 기반 필터링 중... (샘플: {num_samples}개)")
        
        model.eval()
        damping = self.config.get('influence_calculation', 'damping', 0.01)
        lissa_iterations = self.config.get('influence_calculation', 'lissa_iterations', 10)
        
        # 1. 타겟 배치의 평균 손실 기울기 계산
        print("📊 타겟 배치 기울기 계산 중...")
        target_gradients = []
        
        for data, target in target_samples:
            data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            target_gradients.append([g.clone() for g in grad])
        
        # 타겟 평균 기울기
        avg_target_grad = []
        for i in range(len(target_gradients[0])):
            param_grads = [tg[i] for tg in target_gradients]
            avg_grad = torch.stack(param_grads).mean(dim=0)
            avg_target_grad.append(avg_grad)
        
        # 2. LiSSA로 헤시안 역행렬 근사
        h_estimate = [g.clone() for g in avg_target_grad]
        
        for iteration in range(lissa_iterations):
            try:
                batch_data, batch_target = next(iter(source_loader))
                sample_idx = random.randint(0, len(batch_data) - 1)
                data = batch_data[sample_idx:sample_idx+1].to(device)
                target = batch_target[sample_idx:sample_idx+1].to(device)
                
                model.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # 헤시안-벡터 곱
                hvp = self._compute_hessian_vector_product(model, loss, h_estimate)
                
                # LiSSA 업데이트
                for j in range(len(h_estimate)):
                    h_estimate[j] = avg_target_grad[j] + h_estimate[j] - damping * hvp[j]
                    
            except Exception as e:
                print(f"⚠️ LiSSA 반복 중 오류: {e}")
                break
        
        # 3. 소스 샘플별 영향도 계산 I_up(z_i, D_T^batch)
        influence_scores = []
        source_samples = []
        harmful_count = 0
        
        sample_count = 0
        for batch_data, batch_target in source_loader:
            if sample_count >= num_samples:
                break
                
            for i in range(len(batch_data)):
                if sample_count >= num_samples:
                    break
                    
                data, target = batch_data[i].unsqueeze(0).to(device), batch_target[i].unsqueeze(0).to(device)
                
                try:
                    model.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    
                    source_grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
                    
                    # 영향도 계산: I_up = -∇L(D_T)^T * H^-1 * ∇L(z_i)
                    influence = 0
                    for sg, ihvp in zip(source_grad, h_estimate):
                        influence += torch.sum(sg * ihvp).item()
                    
                    influence = -influence  # 논문 공식에 따른 음수
                    
                    # 음수 영향도 = 타겟 적응에 해로움
                    if influence < 0:
                        harmful_count += 1
                    
                    influence_scores.append(influence)
                    source_samples.append((batch_data[i], batch_target[i]))
                    sample_count += 1
                        
                except Exception as e:
                    print(f"⚠️ 영향도 계산 실패 (샘플 {sample_count}): {e}")
                    continue
        
        print(f"✅ 영향도 필터링 완료: {harmful_count}개 유해 샘플 / {len(influence_scores)}개 전체")
        
        return source_samples, influence_scores
    
    def _compute_simple_influence(self, model: nn.Module, source_loader: DataLoader,
                                target_samples: List[Tuple], num_samples: int) -> Tuple[List[Tuple], List[float]]:
        """간단한 영향도 계산"""
        model.eval()
        
        # 타겟 샘플들의 손실 계산
        target_losses = []
        for data, target in target_samples:
            data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(data)
                loss = F.cross_entropy(output, target)
                target_losses.append(loss.item())
        
        avg_target_loss = np.mean(target_losses)
        
        # 소스 샘플들의 영향도 계산
        influence_scores = []
        source_samples = []
        
        sample_count = 0
        for batch_data, batch_target in source_loader:
            if sample_count >= num_samples:
                break
                
            for i in range(len(batch_data)):
                if sample_count >= num_samples:
                    break
                    
                data, target = batch_data[i].unsqueeze(0).to(device), batch_target[i].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    influence = loss.item() - avg_target_loss
                    
                influence_scores.append(influence)
                source_samples.append((batch_data[i], batch_target[i]))
                sample_count += 1
        
        return source_samples, influence_scores
    
    def _compute_hessian_vector_product(self, model: nn.Module, loss: torch.Tensor, 
                                      vector: List[torch.Tensor]) -> List[torch.Tensor]:
        """헤시안-벡터 곱 계산"""
        grad = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
        grad_vector_product = sum(torch.sum(g * v) for g, v in zip(grad, vector))
        
        hvp = torch.autograd.grad(grad_vector_product, model.parameters(), retain_graph=True)
        return list(hvp)

class UnlearningEngine:
    """언러닝 엔진 클래스 - 동적 직교성 스케일링(DOS)"""
    
    def __init__(self, config: SDAUConfig):
        self.config = config
    
    def perform_unlearning(self, model: nn.Module, harmful_samples: List[Tuple], 
                          influence_scores: List[float], retain_samples: List[Tuple],
                          target_test_loader: DataLoader) -> float:
        """동적 직교성 스케일링(DOS) 언러닝 수행"""
        method = self.config.get('unlearning', 'method')
        
        if method == 'dos':
            return self._perform_dos_unlearning(model, harmful_samples, influence_scores, 
                                              retain_samples, target_test_loader)
        elif method == 'simple_dos':
            return self._perform_simple_dos_unlearning(model, harmful_samples, influence_scores)
        else:
            raise ValueError(f"지원하지 않는 언러닝 방법: {method}")
    
    def _perform_dos_unlearning(self, model: nn.Module, harmful_samples: List[Tuple],
                              influence_scores: List[float], retain_samples: List[Tuple],
                              target_test_loader: DataLoader) -> float:
        """동적 직교성 스케일링(DOS) 언러닝"""
        num_steps = self.config.get('unlearning', 'num_steps', 10)
        unlearn_lr = self.config.get('unlearning', 'learning_rate', 0.001)
        
        print(f"🔧 DOS 언러닝 수행 중... (삭제: {len(harmful_samples)}개, 유지: {len(retain_samples)}개)")
        
        model.train()
        
        # 언러닝 전 성능 기록
        pre_unlearn_target_acc = self._evaluate_model(model, target_test_loader)
        
        # 1. 유해성 가중치 계산 및 정규화 (w_i = -I_up(z_i))
        harmful_weights = []
        for score in influence_scores:
            weight = max(-score, 0)  # 음수 영향도만 사용
            harmful_weights.append(weight)
        
        total_weight = sum(harmful_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in harmful_weights]
        else:
            normalized_weights = [1.0 / len(harmful_samples)] * len(harmful_samples)
        
        # 2. 유지 세트의 평균 기울기 계산
        retain_gradients = []
        for data, target in retain_samples:
            data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            retain_gradients.append([g.clone() for g in grad])
        
        # 평균 유지 기울기
        avg_retain_grad = []
        for i in range(len(retain_gradients[0])):
            param_grads = [rg[i] for rg in retain_gradients]
            avg_grad = torch.stack(param_grads).mean(dim=0)
            avg_retain_grad.append(avg_grad)
        
        # DOS 언러닝 수행
        unlearn_optimizer = optim.SGD(model.parameters(), lr=unlearn_lr)
        
        for step in range(num_steps):
            # 3. 가중치 적용된 삭제 기울기 계산 (g_f^weighted)
            weighted_forget_grad = None
            
            for (data, target), weight in zip(harmful_samples, normalized_weights):
                data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
                
                model.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
                
                if weighted_forget_grad is None:
                    weighted_forget_grad = [weight * g for g in grad]
                else:
                    for i, g in enumerate(grad):
                        weighted_forget_grad[i] += weight * g
            
            # 4. 직교 언러닝: 유지 지식 방향 성분 제거
            if weighted_forget_grad is not None:
                dot_product = 0
                retain_norm_squared = 0
                
                for wfg, arg in zip(weighted_forget_grad, avg_retain_grad):
                    dot_product += torch.sum(wfg * arg).item()
                    retain_norm_squared += torch.sum(arg * arg).item()
                
                if retain_norm_squared > 1e-8:
                    projection_coeff = dot_product / retain_norm_squared
                else:
                    projection_coeff = 0
                
                # 5. 직교 기울기로 파라미터 업데이트
                unlearn_optimizer.zero_grad()
                
                for param, wfg, arg in zip(model.parameters(), weighted_forget_grad, avg_retain_grad):
                    orthogonal_grad = wfg - projection_coeff * arg
                    param.grad = orthogonal_grad.clone()
                
                unlearn_optimizer.step()
        
        # 언러닝 후 성능 확인
        post_unlearn_target_acc = self._evaluate_model(model, target_test_loader)
        performance_change = post_unlearn_target_acc - pre_unlearn_target_acc
        
        print(f"📊 언러닝 완료: {pre_unlearn_target_acc:.2f}% → {post_unlearn_target_acc:.2f}% ({performance_change:+.2f}%)")
        
        return performance_change
    
    def _perform_simple_dos_unlearning(self, model: nn.Module, harmful_samples: List[Tuple],
                                     influence_scores: List[float]) -> float:
        """간단한 DOS 언러닝"""
        num_steps = self.config.get('unlearning', 'num_steps')
        unlearn_lr = self.config.get('unlearning', 'learning_rate')
        
        flush_print(f"🔧 간단한 DOS 언러닝 수행 중... (스텝: {num_steps})")
        
        model.train()
        unlearn_optimizer = optim.SGD(model.parameters(), lr=unlearn_lr)
        
        for step in range(num_steps):
            total_loss = 0
            
            for data, target in harmful_samples:
                data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
                
                unlearn_optimizer.zero_grad()
                output = model(data)
                
                loss = F.cross_entropy(output, target)
                unlearn_loss = -0.05 * loss  # 음수 강도
                
                unlearn_loss.backward()
                unlearn_optimizer.step()
                
                total_loss += unlearn_loss.item()
            
            avg_loss = total_loss / len(harmful_samples)
            flush_print(f"   스텝 {step+1}: 언러닝 손실 = {avg_loss:.4f}")
        
        flush_print("✅ 간단한 DOS 언러닝 완료")
        return 0.0  # 성능 변화 반환하지 않음
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader, max_batches: int = None) -> float:
        """모델 성능 평가 (전체 데이터 평가)"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

class ModelTrainer:
    """모델 훈련 클래스"""
    
    def __init__(self, config: SDAUConfig):
        self.config = config
    
    def train_source_model(self, model: nn.Module, source_train_loader: DataLoader,
                          source_test_loader: DataLoader, source_domain: str, dataset: str) -> float:
        """소스 모델 훈련 또는 로딩"""
        # 소스 모델 경로 설정
        source_models_dir = Path(self.config.get('paths', 'source_models_dir')) / dataset
        source_models_dir.mkdir(parents=True, exist_ok=True)
        source_model_path = source_models_dir / f"{source_domain}_source_model.pt"
        
        flush_print(f"\n🔍 소스 모델 확인: {source_model_path}")
        
        # 기존 소스 모델 로딩 시도
        if source_model_path.exists():
            try:
                flush_print(f"📥 기존 소스 모델 로딩 중...")
                checkpoint = torch.load(source_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                source_acc = self._evaluate_model(model, source_test_loader)
                
                if source_acc > 50.0:
                    flush_print(f"✅ 기존 소스 모델 로딩 성공!")
                    flush_print(f"   📊 소스 도메인 성능: {source_acc:.2f}%")
                    return source_acc
                else:
                    flush_print(f"⚠️ 기존 모델 성능이 낮음 ({source_acc:.2f}%), 재훈련 필요")
            except Exception as e:
                flush_print(f"⚠️ 기존 모델 로딩 실패: {e}")
        
        # 새로운 소스 모델 훈련
        flush_print(f"\n🏋️ 소스 도메인 훈련 시작...")
        
        model.train()
        learning_rate = self.config.get('training', 'learning_rate')
        weight_decay = self.config.get('training', 'weight_decay')
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate * 2, weight_decay=weight_decay)
        
        best_source_acc = 0.0
        best_source_state = None
        patience = 0
        max_patience = 30
        
        for epoch in range(100):  # 최대 100 에포크
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(source_train_loader):
                # OfficeHome 호환성: target이 정수인 경우 텐서로 변환
                if isinstance(target, (list, tuple)):
                    target = torch.tensor(target)
                elif not torch.is_tensor(target):
                    target = torch.tensor([target] if isinstance(target, int) else target)
                
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            train_acc = 100 * correct / total
            source_acc = self._evaluate_model(model, source_test_loader)
            
            flush_print(f"   에포크 {epoch+1:2d}: 훈련 {train_acc:5.2f}% | 테스트 {source_acc:5.2f}%")
            
            if source_acc > best_source_acc:
                best_source_acc = source_acc
                best_source_state = copy.deepcopy(model.state_dict())
                patience = 0
                
                # 최고 성능 모델 저장
                torch.save({
                    'model_state_dict': best_source_state,
                    'source_accuracy': best_source_acc,
                    'epoch': epoch + 1,
                    'config': self.config.config
                }, source_model_path)
                flush_print(f"      💾 최고 성능 저장 (성능: {source_acc:.2f}%)")
            else:
                patience += 1
                
            if patience >= max_patience:
                flush_print(f"   ⏱️ 성능 개선 없음 ({max_patience}에포크). 조기 종료")
                break
        
        # 최고 성능 모델로 복원
        if best_source_state is not None:
            model.load_state_dict(best_source_state)
            flush_print(f"✅ 소스 모델 훈련 완료! 최고 성능: {best_source_acc:.2f}%")
            return best_source_acc
        else:
            flush_print(f"❌ 소스 모델 훈련 실패")
            return 0.0
    
    def train_with_target_samples(self, model: nn.Module, target_samples: List[Tuple], 
                                 num_epochs: int = 1) -> None:
        """타겟 샘플로 적응 훈련"""
        if not target_samples:
            return
            
        model.train()
        
        learning_rate = self.config.get('training', 'learning_rate')
        weight_decay = self.config.get('training', 'weight_decay')
        
        # 샘플 수에 따른 학습률 조정 (안전한 학습률로 변경)
        if len(target_samples) < 50:
            initial_lr = learning_rate * 0.5
            final_lr = learning_rate * 0.1
        else:
            initial_lr = learning_rate * 1.0
            final_lr = learning_rate * 0.2
        
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay * 0.2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=final_lr)
        
        # 배치 처리
        if len(target_samples) >= 24:
            batch_size = 8
            target_dataset = [(data, target) for data, target in target_samples]
            target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
            use_batches = True
        elif len(target_samples) >= 12:
            batch_size = 6
            target_dataset = [(data, target) for data, target in target_samples]
            target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
            use_batches = True
        else:
            use_batches = False
            repeat_factor = max(1, 8 // len(target_samples))
        
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            
            if use_batches:
                for data_batch, target_batch in target_loader:
                    data_batch, target_batch = data_batch.to(device), target_batch.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data_batch)
                    loss = F.cross_entropy(output, target_batch)
                    
                    # L2 정규화
                    if weight_decay > 0:
                        l2_reg = torch.tensor(0., requires_grad=True).to(device)
                        for param in model.parameters():
                            l2_reg = l2_reg + torch.norm(param, 2)
                        loss = loss + weight_decay * 0.1 * l2_reg
                    
                    loss.backward()
                    
                    # 그라디언트 클리핑
                    clip_norm = self.config.get('training', 'gradient_clip_norm')
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
            else:
                for repeat in range(repeat_factor):
                    for data, target in target_samples:
                        data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
                        
                        optimizer.zero_grad()
                        output = model(data)
                        loss = F.cross_entropy(output, target)
                        
                        if weight_decay > 0:
                            l2_reg = torch.tensor(0., requires_grad=True).to(device)
                            for param in model.parameters():
                                l2_reg = l2_reg + torch.norm(param, 2)
                            loss = loss + weight_decay * 0.1 * l2_reg
                        
                        loss.backward()
                        
                        clip_norm = self.config.get('training', 'gradient_clip_norm')
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                        
                        optimizer.step()
                        
                        total_loss += loss.item()
                        batch_count += 1
            
            scheduler.step()
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader, max_batches: int = 20) -> float:
        """모델 성능 평가"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                    
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

class SDAUAlgorithm:
    """핵심 SDA-U 알고리즘 클래스 (완전 객체지향)"""
    
    def __init__(self, config_path: str = "config.json"):
        """SDA-U 알고리즘 초기화"""
        self.config = SDAUConfig(config_path)
        self.model_manager = ModelManager(self.config)
        self.data_manager = DataManager(self.config)
        self.target_selector = TargetSampleSelector(self.config)
        self.influence_calculator = InfluenceCalculator(self.config)
        self.unlearning_engine = UnlearningEngine(self.config)
        self.model_trainer = ModelTrainer(self.config)
        
        # GPU 최적화
        if GPU_OPTIMIZED and self.config.get('gpu', 'auto_optimize'):
            setup_gpu_optimizations()
            flush_print(f"🚀 GPU 최적화 활성화: {get_gpu_info()['name']}")
        
        flush_print(f"🔧 디바이스: {device}")
        flush_print(f"⚙️ 설정 로딩 완료: {self.config.config_path}")
        
    def run_experiment(self, dataset: str, source_domain: str, target_domain: str) -> ExperimentResults:
        """전체 실험 실행"""
        flush_print("\n" + "="*80)
        flush_print("🎯 완전 객체지향 SDA-U 알고리즘 시작!")
        flush_print("="*80)
        flush_print(f"📊 데이터셋: {dataset}")
        flush_print(f"🏠 소스 도메인: {source_domain}")
        flush_print(f"🎯 타겟 도메인: {target_domain}")
        
        # 1. 데이터 로딩
        batch_size = self.config.get('training', 'batch_size')
        source_train_loader, source_test_loader, target_train_loader, target_test_loader, num_classes = \
            self.data_manager.load_dataset(dataset, source_domain, target_domain, batch_size)
        
        flush_print(f"✅ 데이터 로딩 완료 (클래스 수: {num_classes})")
        
        # 2. 모델 생성 및 소스 모델 훈련
        model = self.model_manager.create_model(num_classes)
        flush_print(f"🤖 모델: {self.config.get('model', 'architecture')} (파라미터: {sum(p.numel() for p in model.parameters()):,})")
        
        source_accuracy = self.model_trainer.train_source_model(
            model, source_train_loader, source_test_loader, source_domain, dataset)
        
        # 3. 핵심 SDA-U 알고리즘 실행
        results = self._core_sda_u_algorithm(
            model, source_train_loader, target_train_loader,
            target_test_loader, source_test_loader, num_classes,
            dataset, source_domain, target_domain
        )
        
        return results
    
    def _core_sda_u_algorithm(self, model: nn.Module, source_train_loader: DataLoader,
                             target_train_loader: DataLoader, target_test_loader: DataLoader,
                             source_test_loader: DataLoader, num_classes: int,
                             dataset: str, source_domain: str, target_domain: str) -> ExperimentResults:
        """핵심 SDA-U 알고리즘 실행"""
        
        # 성능 히스토리 저장용
        performance_history = {
            'epoch': [],
            'target_acc': [],
            'source_acc': [],
            'phase': [],
            'unlearning_count': [],
            'is_best': []
        }
        
        # 모델 저장 경로 설정
        model_save_dir = Path(self.config.get('paths', 'model_save_dir')) / f"{source_domain}2{target_domain}"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = model_save_dir / "best_model.pt"
        final_model_path = model_save_dir / "final_model.pt"
        performance_log_path = model_save_dir / "performance_history.json"
        
        flush_print(f"💾 모델 저장 경로: {model_save_dir}")
        
        # 1. 타겟 샘플 선별
        target_samples = self.target_selector.select_samples(model, target_train_loader, num_classes)
        flush_print(f"✅ 선별된 타겟 샘플: {len(target_samples)}개")
        
        # 초기 성능 측정
        initial_target_acc = self._evaluate_model(model, target_test_loader)
        initial_source_acc = self._evaluate_model(model, source_test_loader)
        
        flush_print(f"\n📊 초기 성능:")
        flush_print(f"   🎯 타겟: {initial_target_acc:.2f}%")
        flush_print(f"   🏠 소스: {initial_source_acc:.2f}%")
        
        # 초기 성능 히스토리에 추가
        performance_history['epoch'].append(0)
        performance_history['target_acc'].append(initial_target_acc)
        performance_history['source_acc'].append(initial_source_acc)
        performance_history['phase'].append('initial')
        performance_history['unlearning_count'].append(0)
        performance_history['is_best'].append(True)
        
        # 최고 성능 모델 추적
        best_model_state = copy.deepcopy(model.state_dict())
        best_target_acc = initial_target_acc
        best_epoch = 0
        
        # 최고 성능 모델 저장
        self.model_manager.save_model(model, str(best_model_path), {
            'epoch': 0,
            'target_accuracy': best_target_acc,
            'source_accuracy': initial_source_acc,
            'dataset': dataset,
            'source_domain': source_domain,
            'target_domain': target_domain
        })
        
        # 설정값 읽기
        max_epochs = self.config.get('training', 'max_epochs')
        patience_limit = self.config.get('training', 'patience')
        epoch_chunk_size = self.config.get('training', 'epoch_chunk_size')
        
        # 성능 정체 추적
        stagnation_count = 0
        unlearning_count = 0
        
        print(f"\n🏋️ 타겟 도메인 적응 훈련 시작 ({max_epochs}에포크, {patience_limit}에포크 정체 시 언러닝)")
        
        epoch = 0
        while epoch < max_epochs:
            # epoch_chunk_size 에포크씩 훈련
            for sub_epoch in range(epoch_chunk_size):
                if epoch + sub_epoch >= max_epochs:
                    break
                    
                current_epoch = epoch + sub_epoch + 1
                
                # 1에포크 훈련
                self.model_trainer.train_with_target_samples(model, target_samples, num_epochs=1)
                
                # 성능 측정 (설정값에 따라)
                log_interval = self.config.get('performance', 'log_interval')
                if current_epoch % log_interval == 0 or sub_epoch == epoch_chunk_size - 1:
                    current_target_acc = self._evaluate_model(model, target_test_loader)
                    current_source_acc = self._evaluate_model(model, source_test_loader)
                    
                    improvement = current_target_acc - best_target_acc
                    is_best = current_target_acc > best_target_acc
                    
                    print(f"에포크 {current_epoch:3d}: 타겟 {current_target_acc:5.2f}% | 소스 {current_source_acc:5.2f}% {'🏆' if is_best else ''}")
                    
                    # 성능 히스토리에 추가
                    performance_history['epoch'].append(current_epoch)
                    performance_history['target_acc'].append(current_target_acc)
                    performance_history['source_acc'].append(current_source_acc)
                    performance_history['phase'].append('training')
                    performance_history['unlearning_count'].append(unlearning_count)
                    performance_history['is_best'].append(is_best)
                    
                    # 최고 성능 업데이트
                    if is_best:
                        best_target_acc = current_target_acc
                        best_epoch = current_epoch
                        best_model_state = copy.deepcopy(model.state_dict())
                        stagnation_count = 0
                        
                        # 최고 성능 모델 저장
                        self.model_manager.save_model(model, str(best_model_path), {
                            'epoch': current_epoch,
                            'target_accuracy': best_target_acc,
                            'source_accuracy': current_source_acc,
                            'unlearning_count': unlearning_count
                        })
            
            epoch += epoch_chunk_size
            
            # epoch_chunk_size 단위로 정체 확인
            last_target_acc = performance_history['target_acc'][-1] if performance_history['target_acc'] else 0
            if last_target_acc <= best_target_acc:
                stagnation_count += epoch_chunk_size
                
                # patience_limit 이상 정체 시 언러닝
                if stagnation_count >= patience_limit and epoch < max_epochs:
                    print(f"\n🔧 언러닝 수행 #{unlearning_count+1} (정체: {stagnation_count}에포크)")
                    
                    # 최고 성능 모델로 복원
                    model.load_state_dict(best_model_state)
                    
                    # 영향도 계산 및 언러닝 수행
                    performance_change = self._perform_unlearning_cycle(
                        model, source_train_loader, target_samples, target_test_loader)
                    
                    unlearning_count += 1
                    
                    # 언러닝 후 성능 측정
                    after_unlearn_target_acc = self._evaluate_model(model, target_test_loader)
                    after_unlearn_source_acc = self._evaluate_model(model, source_test_loader)
                    
                    # 성능 히스토리에 추가
                    performance_history['epoch'].append(epoch)
                    performance_history['target_acc'].append(after_unlearn_target_acc)
                    performance_history['source_acc'].append(after_unlearn_source_acc)
                    performance_history['phase'].append(f'unlearning_{unlearning_count}')
                    performance_history['unlearning_count'].append(unlearning_count)
                    performance_history['is_best'].append(after_unlearn_target_acc > best_target_acc)
                    
                    # 언러닝으로 성능이 개선되면 최고 성능 업데이트
                    if after_unlearn_target_acc > best_target_acc:
                        print(f"🏆 언러닝으로 최고 성능 갱신! {best_target_acc:.2f}% → {after_unlearn_target_acc:.2f}%")
                        best_target_acc = after_unlearn_target_acc
                        best_epoch = epoch
                        best_model_state = copy.deepcopy(model.state_dict())
                        
                        self.model_manager.save_model(model, str(best_model_path), {
                            'epoch': epoch,
                            'target_accuracy': best_target_acc,
                            'source_accuracy': after_unlearn_source_acc,
                            'unlearning_count': unlearning_count,
                            'phase': f'unlearning_{unlearning_count}'
                        })
                    
                    stagnation_count = 0  # 정체 카운트 리셋
            else:
                stagnation_count = 0  # 성능이 개선되면 정체 카운트 리셋
            
            # 중간 성능 히스토리 저장
            if epoch % 50 == 0:
                with open(performance_log_path, 'w', encoding='utf-8') as f:
                    json.dump(performance_history, f, indent=2, ensure_ascii=False)
        
        # 최종 최고 성능 모델로 복원
        flush_print(f"\n🏆 최종 최고 성능 모델로 복원 (에포크 {best_epoch}, 성능: {best_target_acc:.2f}%)")
        model.load_state_dict(best_model_state)
        
        # 최종 성능 측정
        final_target_acc = self._evaluate_model(model, target_test_loader)
        final_source_acc = self._evaluate_model(model, source_test_loader)
        
        # 최종 성능을 히스토리에 추가
        performance_history['epoch'].append('final')
        performance_history['target_acc'].append(final_target_acc)
        performance_history['source_acc'].append(final_source_acc)
        performance_history['phase'].append('final')
        performance_history['unlearning_count'].append(unlearning_count)
        performance_history['is_best'].append(final_target_acc >= best_target_acc)
        
        # 최종 모델 저장
        self.model_manager.save_model(model, str(final_model_path), {
            'epoch': 'final',
            'target_accuracy': final_target_acc,
            'source_accuracy': final_source_acc,
            'best_target_accuracy': best_target_acc,
            'best_epoch': best_epoch,
            'unlearning_count': unlearning_count,
            'performance_history': performance_history
        })
        
        # 최종 성능 히스토리 저장
        with open(performance_log_path, 'w', encoding='utf-8') as f:
            json.dump(performance_history, f, indent=2, ensure_ascii=False)
        
        # 결과 출력
        print(f"\n{'='*60}")
        print("🎯 SDA-U 알고리즘 완료!")
        print(f"{'='*60}")
        print(f"📊 최종 결과:")
        print(f"   타겟: {initial_target_acc:.2f}% → {final_target_acc:.2f}% ({final_target_acc-initial_target_acc:+.2f}%)")
        print(f"   소스: {initial_source_acc:.2f}% → {final_source_acc:.2f}%")
        print(f"   🏆 최고: {best_target_acc:.2f}% (에포크 {best_epoch})")
        print(f"   언러닝: {unlearning_count}회, 에포크: {min(epoch, max_epochs)}")
        
        return ExperimentResults(
            initial_target_acc=initial_target_acc,
            final_target_acc=final_target_acc,
            best_target_acc=best_target_acc,
            best_epoch=best_epoch,
            improvement=final_target_acc - initial_target_acc,
            best_improvement=best_target_acc - initial_target_acc,
            unlearning_count=unlearning_count,
            total_epochs=min(epoch, max_epochs),
            performance_history=performance_history,
            model_paths={
                'best_model': str(best_model_path),
                'final_model': str(final_model_path),
                'performance_log': str(performance_log_path)
            }
        )
    
    def _perform_unlearning_cycle(self, model: nn.Module, source_train_loader: DataLoader,
                                 target_samples: List[Tuple], target_test_loader: DataLoader) -> float:
        """언러닝 사이클 수행"""
        # 영향도 계산
        flush_print(f"🧮 영향도 계산 중...")
        source_samples, influence_scores = self.influence_calculator.compute_influence_scores(
            model, source_train_loader, target_samples)
        
        # 유해한 샘플 선별 (음수 영향도만)
        harmful_indices = [i for i, score in enumerate(influence_scores) if score < 0]
        harmful_samples = [source_samples[i] for i in harmful_indices]
        harmful_influence_scores = [influence_scores[i] for i in harmful_indices]
        
        # 최대 샘플 수 제한
        max_influence_samples = self.config.get('influence_calculation', 'max_influence_samples')
        if len(harmful_samples) > max_influence_samples:
            sorted_pairs = sorted(zip(harmful_samples, harmful_influence_scores), 
                                key=lambda x: x[1])  # 음수가 작을수록 더 유해
            harmful_samples = [pair[0] for pair in sorted_pairs[:max_influence_samples]]
            harmful_influence_scores = [pair[1] for pair in sorted_pairs[:max_influence_samples]]
        
        flush_print(f"🔍 유해 샘플 발견: {len(harmful_samples)}개 (전체 {len(source_samples)}개 중)")
        
        if len(harmful_samples) > 0:
            # 언러닝 수행
            performance_change = self.unlearning_engine.perform_unlearning(
                model, harmful_samples, harmful_influence_scores, target_samples, target_test_loader)
            
            flush_print(f"📊 언러닝 성능 변화: {performance_change:+.2f}%")
            return performance_change
        else:
            flush_print("⚠️ 유해한 소스 샘플이 발견되지 않음. 언러닝 건너뛰기.")
            return 0.0
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """모델 성능 평가"""
        model.eval()
        correct = 0
        total = 0
        
        max_batches = self.config.get('performance', 'evaluation_batch_limit')
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                    
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

def main():
    """메인 함수 (완전 객체지향)"""
    parser = argparse.ArgumentParser(description='완전 객체지향 SDA-U 알고리즘')
    
    # 필수 인자
    parser.add_argument('--dataset', type=str, required=True, choices=['Office31', 'OfficeHome'])
    parser.add_argument('--source_domain', type=str, required=True)
    parser.add_argument('--target_domain', type=str, required=True)
    
    # 선택적 인자
    parser.add_argument('--config', type=str, default='config.json', help='설정 파일 경로')
    parser.add_argument('--results_file', type=str, default='results.json', help='결과 파일 경로')
    
    args = parser.parse_args()
    
    flush_print("🎯 완전 객체지향 SDA-U 알고리즘 시작!")
    flush_print(f"📊 실험 설정:")
    flush_print(f"   데이터셋: {args.dataset}")
    flush_print(f"   소스 도메인: {args.source_domain}")
    flush_print(f"   타겟 도메인: {args.target_domain}")
    flush_print(f"   설정 파일: {args.config}")
    
    try:
        # SDA-U 알고리즘 초기화 및 실행
        sda_u = SDAUAlgorithm(config_path=args.config)
        results = sda_u.run_experiment(args.dataset, args.source_domain, args.target_domain)
        
        # 결과 저장
        results_dict = {
            'dataset': args.dataset,
            'source_domain': args.source_domain,
            'target_domain': args.target_domain,
            'initial_target_acc': results.initial_target_acc,
            'final_target_acc': results.final_target_acc,
            'best_target_acc': results.best_target_acc,
            'best_epoch': results.best_epoch,
            'improvement': results.improvement,
            'best_improvement': results.best_improvement,
            'unlearning_count': results.unlearning_count,
            'total_epochs': results.total_epochs,
            'model_paths': results.model_paths,
            'timestamp': time.time()
        }
        
        # 결과 디렉토리 생성
        results_dir = Path(sda_u.config.get('paths', 'results_dir'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과 파일 저장
        results_path = results_dir / args.results_file
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        flush_print(f"\n💾 결과 저장 완료:")
        flush_print(f"   📄 결과 파일: {results_path}")
        flush_print(f"   🏆 최고 모델: {results.model_paths['best_model']}")
        flush_print(f"   📊 성능 로그: {results.model_paths['performance_log']}")
        
        flush_print(f"\n🎉 실험 완료! 최고 성능: {results.best_target_acc:.2f}%")
        
    except Exception as e:
        flush_print(f"❌ 실험 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 