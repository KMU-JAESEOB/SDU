# improved_target_sampling.py
"""
🎯 타겟 도메인 샘플링 개선 방안
- 계층적 샘플링 (Stratified Sampling)
- 클래스 균형 샘플링 (Class-balanced Sampling)  
- 다중 배치 샘플링 (Multi-batch Sampling)
- 적응적 샘플링 (Adaptive Sampling)
"""

import torch
import numpy as np
from collections import defaultdict, Counter
import random
from torch.utils.data import DataLoader, Subset

class ImprovedTargetSampler:
    """향상된 타겟 도메인 샘플링 클래스"""
    
    def __init__(self, target_dataset, num_classes=65):
        self.target_dataset = target_dataset
        self.num_classes = num_classes
        self.class_indices = self._build_class_indices()
        
    def _build_class_indices(self):
        """클래스별 샘플 인덱스 구축"""
        class_indices = defaultdict(list)
        
        for idx in range(len(self.target_dataset)):
            _, label = self.target_dataset[idx]
            class_indices[label].append(idx)
        
        return class_indices
    
    def stratified_sampling(self, total_samples=320, min_per_class=2):
        """계층적 샘플링: 각 클래스에서 균등하게 샘플링"""
        
        print(f"🎯 계층적 샘플링 시작 (총 {total_samples}개, 클래스당 최소 {min_per_class}개)")
        
        # 클래스별 샘플 수 계산
        available_classes = len(self.class_indices)
        base_per_class = max(min_per_class, total_samples // available_classes)
        remaining_samples = total_samples - (base_per_class * available_classes)
        
        selected_indices = []
        class_sample_counts = {}
        
        # 각 클래스에서 기본 샘플 수만큼 선택
        for class_label, indices in self.class_indices.items():
            if len(indices) >= base_per_class:
                sampled = random.sample(indices, base_per_class)
                selected_indices.extend(sampled)
                class_sample_counts[class_label] = base_per_class
            else:
                # 클래스에 충분한 샘플이 없으면 모든 샘플 사용
                selected_indices.extend(indices)
                class_sample_counts[class_label] = len(indices)
        
        # 남은 샘플을 큰 클래스에서 추가 선택
        if remaining_samples > 0:
            large_classes = [(label, indices) for label, indices in self.class_indices.items() 
                           if len(indices) > base_per_class]
            
            for i in range(remaining_samples):
                if large_classes:
                    class_label, indices = large_classes[i % len(large_classes)]
                    available_indices = [idx for idx in indices if idx not in selected_indices]
                    if available_indices:
                        selected_indices.append(random.choice(available_indices))
                        class_sample_counts[class_label] += 1
        
        print(f"✅ 계층적 샘플링 완료: {len(selected_indices)}개 선택")
        print(f"📊 클래스별 분포: {dict(class_sample_counts)}")
        
        return selected_indices
    
    def uncertainty_based_sampling(self, model, total_samples=320, device='cuda'):
        """불확실성 기반 샘플링: 모델이 가장 확신하지 못하는 샘플 선택"""
        
        print(f"🤔 불확실성 기반 샘플링 시작 (총 {total_samples}개)")
        
        model.eval()
        uncertainties = []
        
        # 전체 데이터셋에 대해 불확실성 계산
        temp_loader = DataLoader(self.target_dataset, batch_size=64, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(temp_loader):
                data = data.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                
                # 엔트로피 기반 불확실성 계산
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                uncertainties.extend(entropy.cpu().numpy())
        
        # 불확실성이 높은 순서로 정렬
        sorted_indices = sorted(range(len(uncertainties)), 
                              key=lambda i: uncertainties[i], reverse=True)
        
        selected_indices = sorted_indices[:total_samples]
        
        print(f"✅ 불확실성 기반 샘플링 완료: {len(selected_indices)}개 선택")
        print(f"📊 평균 불확실성: {np.mean([uncertainties[i] for i in selected_indices]):.4f}")
        
        return selected_indices
    
    def diverse_sampling(self, total_samples=320, diversity_weight=0.5):
        """다양성 기반 샘플링: 클래스 균형 + 무작위성"""
        
        print(f"🌈 다양성 기반 샘플링 시작 (총 {total_samples}개)")
        
        # 클래스별 최대 샘플 수 제한
        max_per_class = max(1, total_samples // self.num_classes)
        selected_indices = []
        
        # 각 클래스에서 제한된 수만큼 선택
        for class_label, indices in self.class_indices.items():
            sample_count = min(len(indices), max_per_class)
            if sample_count > 0:
                sampled = random.sample(indices, sample_count)
                selected_indices.extend(sampled)
        
        # 부족한 샘플을 무작위로 추가
        if len(selected_indices) < total_samples:
            all_indices = list(range(len(self.target_dataset)))
            remaining_indices = [idx for idx in all_indices if idx not in selected_indices]
            additional_needed = total_samples - len(selected_indices)
            
            if remaining_indices:
                additional = random.sample(remaining_indices, 
                                         min(additional_needed, len(remaining_indices)))
                selected_indices.extend(additional)
        
        print(f"✅ 다양성 기반 샘플링 완료: {len(selected_indices)}개 선택")
        
        return selected_indices
    
    def multi_batch_sampling(self, num_batches=10, batch_size=32):
        """다중 배치 샘플링: 여러 배치를 사용하여 더 많은 샘플 커버"""
        
        print(f"📦 다중 배치 샘플링 시작 ({num_batches}개 배치 × {batch_size}개)")
        
        # 셔플된 DataLoader 생성
        loader = DataLoader(self.target_dataset, batch_size=batch_size, 
                          shuffle=True, drop_last=False)
        
        selected_batches = []
        batch_count = 0
        
        for batch in loader:
            if batch_count >= num_batches:
                break
            selected_batches.append(batch)
            batch_count += 1
        
        total_samples = sum(len(batch[0]) for batch in selected_batches)
        print(f"✅ 다중 배치 샘플링 완료: {total_samples}개 선택")
        
        return selected_batches
    
    def adaptive_sampling(self, model, initial_samples=64, max_samples=512, 
                         threshold=0.1, device='cuda'):
        """적응적 샘플링: 성능에 따라 샘플 수를 동적 조정"""
        
        print(f"🔄 적응적 샘플링 시작 (초기: {initial_samples}개, 최대: {max_samples}개)")
        
        current_samples = initial_samples
        selected_indices = []
        
        while current_samples <= max_samples:
            # 현재 샘플 수로 계층적 샘플링 수행
            indices = self.stratified_sampling(current_samples)
            
            # 성능 평가 (간단한 불확실성 측정)
            subset = Subset(self.target_dataset, indices)
            temp_loader = DataLoader(subset, batch_size=32, shuffle=False)
            
            uncertainties = []
            model.eval()
            with torch.no_grad():
                for data, _ in temp_loader:
                    data = data.to(device)
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    uncertainties.extend(entropy.cpu().numpy())
            
            avg_uncertainty = np.mean(uncertainties)
            print(f"📊 {current_samples}개 샘플 평균 불확실성: {avg_uncertainty:.4f}")
            
            # 불확실성이 충분히 높으면 중단
            if avg_uncertainty > threshold:
                selected_indices = indices
                break
            
            # 샘플 수 증가
            current_samples = min(current_samples * 2, max_samples)
        
        print(f"✅ 적응적 샘플링 완료: {len(selected_indices)}개 선택")
        return selected_indices

def compare_sampling_methods(target_dataset, model, num_classes=65):
    """다양한 샘플링 방법 비교"""
    
    print("🔍 타겟 샘플링 방법 비교 분석")
    print("="*60)
    
    sampler = ImprovedTargetSampler(target_dataset, num_classes)
    
    methods = {
        'stratified': lambda: sampler.stratified_sampling(320),
        'uncertainty': lambda: sampler.uncertainty_based_sampling(model, 320),
        'diverse': lambda: sampler.diverse_sampling(320),
        'adaptive': lambda: sampler.adaptive_sampling(model, 64, 320)
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n🎯 {method_name.upper()} 샘플링 테스트")
        print("-" * 40)
        
        try:
            indices = method_func()
            
            # 클래스 분포 분석
            labels = [target_dataset[i][1] for i in indices]
            class_counts = Counter(labels)
            
            results[method_name] = {
                'total_samples': len(indices),
                'unique_classes': len(class_counts),
                'class_distribution': dict(class_counts),
                'balance_score': len(class_counts) / num_classes  # 클래스 커버리지
            }
            
            print(f"✅ 총 샘플: {len(indices)}개")
            print(f"📊 커버된 클래스: {len(class_counts)}/{num_classes}개")
            print(f"🎯 균형 점수: {results[method_name]['balance_score']:.3f}")
            
        except Exception as e:
            print(f"❌ {method_name} 실패: {e}")
            results[method_name] = {'error': str(e)}
    
    # 결과 요약
    print(f"\n📋 샘플링 방법 비교 요약")
    print("="*60)
    
    for method, result in results.items():
        if 'error' not in result:
            print(f"{method:12} | {result['total_samples']:3d}개 | "
                  f"{result['unique_classes']:2d}클래스 | "
                  f"균형: {result['balance_score']:.3f}")
    
    return results

def main():
    """테스트 실행"""
    print("🎯 타겟 도메인 샘플링 개선 방안 테스트")
    
    # 실제 사용 예시는 officehome_loader와 연동 필요
    print("💡 실제 사용을 위해서는 다음과 같이 통합:")
    print("1. officehome_loader.py와 연동")
    print("2. main.py의 타겟 샘플링 부분 교체")
    print("3. 실험 파라미터 조정")

if __name__ == "__main__":
    main() 