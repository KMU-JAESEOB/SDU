# download_officehome.py
"""
🏠 Office-Home 공식 데이터셋 자동 다운로드 및 설정
- 공식 데이터셋 자동 다운로드
- 압축 해제 및 디렉토리 구조 설정
- 데이터 검증 및 통계 정보 출력
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
import json
import time
from tqdm import tqdm
import hashlib

class OfficeHomeDownloader:
    """Office-Home 데이터셋 다운로드 관리자"""
    
    def __init__(self, root_dir='./data'):
        self.root_dir = Path(root_dir)
        self.officehome_dir = self.root_dir / 'OfficeHome'
        
        # Office-Home 공식 다운로드 정보
        self.download_info = {
            'dataset_name': 'Office-Home',
            'total_size': '2.1GB',
            'num_classes': 65,
            'num_images': 15588,
            'domains': {
                'Art': 2427,
                'Clipart': 4365,
                'Product': 4439,
                'Real World': 4357
            },
            # 공식 다운로드 URL들 (실제 URL로 업데이트 필요)
            'urls': {
                'official_site': 'https://www.hemanthdv.org/officeHomeDataset.html',
                'google_drive': 'https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg',
                'backup_url': 'https://dataset-mirrors.example.com/officehome.zip'  # 백업 URL
            },
            'file_info': {
                'filename': 'OfficeHome.zip',
                'md5_hash': 'placeholder_hash',  # 실제 해시값으로 업데이트 필요
                'extracted_size': '3.2GB'
            }
        }
        
        print("🏠 Office-Home 다운로더 초기화 완료!")
        print(f"📁 데이터 디렉토리: {self.root_dir}")
        print(f"🎯 타겟 디렉토리: {self.officehome_dir}")
    
    def check_existing_data(self):
        """기존 데이터셋 존재 여부 확인"""
        
        print("\n🔍 기존 데이터셋 확인 중...")
        
        if not self.officehome_dir.exists():
            print("❌ Office-Home 디렉토리가 존재하지 않습니다.")
            return False
        
        # 필수 도메인 디렉토리 확인
        required_domains = ['Art', 'Clipart', 'Product', 'Real World']
        missing_domains = []
        
        for domain in required_domains:
            domain_path = self.officehome_dir / domain
            if not domain_path.exists():
                missing_domains.append(domain)
            else:
                # 도메인별 샘플 수 확인
                sample_count = len(list(domain_path.rglob('*.jpg')))
                expected_count = self.download_info['domains'][domain]
                print(f"📦 {domain}: {sample_count:,}개 (예상: {expected_count:,}개)")
        
        if missing_domains:
            print(f"❌ 누락된 도메인: {', '.join(missing_domains)}")
            return False
        
        print("✅ 모든 도메인 디렉토리 존재")
        return True
    
    def download_progress_hook(self, block_num, block_size, total_size):
        """다운로드 진행률 표시"""
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        
        if hasattr(self, '_last_percent'):
            if percent - self._last_percent < 1:  # 1% 단위로만 업데이트
                return
        
        self._last_percent = percent
        bar_length = 50
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        
        print(f'\r📥 다운로드 중: |{bar}| {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)', end='')
    
    def download_from_google_drive(self, file_id, destination):
        """Google Drive에서 Office-Home 다운로드"""
        
        print(f"\n📥 Google Drive에서 다운로드 중...")
        
        # Google Drive 다운로드 URL 구성
        base_url = "https://drive.google.com/uc"
        params = f"?export=download&id={file_id}"
        
        try:
            # 파일 크기 확인을 위한 HEAD 요청
            import urllib.request
            req = urllib.request.Request(base_url + params, method='HEAD')
            with urllib.request.urlopen(req) as response:
                file_size = int(response.headers.get('Content-Length', 0))
            
            print(f"📊 파일 크기: {file_size / (1024*1024*1024):.2f} GB")
            
            # 실제 다운로드
            urllib.request.urlretrieve(
                base_url + params,
                destination,
                reporthook=self.download_progress_hook
            )
            
            print(f"\n✅ 다운로드 완료: {destination}")
            return True
            
        except Exception as e:
            print(f"\n❌ Google Drive 다운로드 실패: {e}")
            return False
    
    def download_from_url(self, url, destination):
        """일반 URL에서 다운로드"""
        
        print(f"\n📥 다운로드 중: {url}")
        
        try:
            urllib.request.urlretrieve(
                url, destination, reporthook=self.download_progress_hook
            )
            print(f"\n✅ 다운로드 완료: {destination}")
            return True
        except Exception as e:
            print(f"\n❌ 다운로드 실패: {e}")
            return False
    
    def extract_archive(self, archive_path):
        """압축 파일 해제"""
        
        print(f"\n📦 압축 해제 중: {archive_path}")
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # 진행률 표시와 함께 압축 해제
                    file_list = zip_ref.namelist()
                    
                    with tqdm(total=len(file_list), desc="압축 해제") as pbar:
                        for file in file_list:
                            zip_ref.extract(file, self.root_dir)
                            pbar.update(1)
            
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(self.root_dir)
            
            else:
                raise ValueError(f"지원하지 않는 압축 형식: {archive_path.suffix}")
            
            print("✅ 압축 해제 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 압축 해제 실패: {e}")
            return False
    
    def verify_dataset(self):
        """데이터셋 무결성 검증"""
        
        print("\n🔍 데이터셋 검증 중...")
        
        verification_results = {
            'total_images': 0,
            'total_classes': set(),
            'domain_stats': {},
            'issues': []
        }
        
        for domain_name in self.download_info['domains'].keys():
            domain_path = self.officehome_dir / domain_name
            
            if not domain_path.exists():
                verification_results['issues'].append(f"도메인 디렉토리 누락: {domain_name}")
                continue
            
            domain_images = 0
            domain_classes = set()
            
            for class_dir in domain_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    domain_classes.add(class_name)
                    verification_results['total_classes'].add(class_name)
                    
                    # 이미지 파일 카운트
                    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                    domain_images += len(images)
            
            verification_results['domain_stats'][domain_name] = {
                'images': domain_images,
                'classes': len(domain_classes),
                'expected_images': self.download_info['domains'][domain_name]
            }
            
            verification_results['total_images'] += domain_images
            
            # 예상 이미지 수와 비교
            expected = self.download_info['domains'][domain_name]
            if abs(domain_images - expected) > expected * 0.1:  # 10% 오차 허용
                verification_results['issues'].append(
                    f"{domain_name}: 이미지 수 불일치 ({domain_images} vs {expected})"
                )
        
        # 결과 출력
        print(f"\n📊 검증 결과:")
        print(f"   🖼️ 총 이미지: {verification_results['total_images']:,}개")
        print(f"   🎯 총 클래스: {len(verification_results['total_classes'])}개")
        
        for domain, stats in verification_results['domain_stats'].items():
            status = "✅" if stats['images'] == stats['expected_images'] else "⚠️"
            print(f"   {status} {domain}: {stats['images']:,}개 이미지, {stats['classes']}개 클래스")
        
        if verification_results['issues']:
            print(f"\n⚠️ 발견된 문제:")
            for issue in verification_results['issues']:
                print(f"   - {issue}")
            return False
        else:
            print(f"\n✅ 데이터셋 검증 완료! 모든 데이터가 정상입니다.")
            return True
    
    def create_dataset_info(self):
        """데이터셋 정보 파일 생성"""
        
        print("\n📝 데이터셋 정보 파일 생성 중...")
        
        info = {
            'dataset_name': 'Office-Home',
            'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Official Office-Home Dataset',
            'url': self.download_info['urls']['official_site'],
            'total_images': 0,
            'total_classes': 65,
            'domains': {}
        }
        
        # 실제 통계 수집
        for domain_name in self.download_info['domains'].keys():
            domain_path = self.officehome_dir / domain_name
            
            if domain_path.exists():
                images = list(domain_path.rglob('*.jpg')) + list(domain_path.rglob('*.png'))
                classes = set([img.parent.name for img in images])
                
                info['domains'][domain_name] = {
                    'images': len(images),
                    'classes': len(classes),
                    'class_list': sorted(list(classes))
                }
                
                info['total_images'] += len(images)
        
        # JSON 파일로 저장
        info_file = self.officehome_dir / 'dataset_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 데이터셋 정보 저장: {info_file}")
        return info
    
    def download_official_dataset(self):
        """공식 Office-Home 데이터셋 다운로드"""
        
        print("\n🚀 Office-Home 공식 데이터셋 다운로드 시작!")
        print("="*60)
        
        # 기존 데이터 확인
        if self.check_existing_data():
            response = input("\n✅ 기존 데이터가 존재합니다. 다시 다운로드하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                print("⏭️ 다운로드를 건너뛰고 기존 데이터를 사용합니다.")
                return self.verify_dataset()
        
        # 디렉토리 생성
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드 파일 경로
        download_file = self.root_dir / self.download_info['file_info']['filename']
        
        # 다운로드 시도 (여러 소스)
        download_success = False
        
        print(f"\n📋 다운로드 안내:")
        print(f"1. 공식 웹사이트: {self.download_info['urls']['official_site']}")
        print(f"2. 예상 파일 크기: {self.download_info['total_size']}")
        print(f"3. 압축 해제 후 크기: {self.download_info['file_info']['extracted_size']}")
        
        # 수동 다운로드 안내
        print(f"\n📥 수동 다운로드 방법:")
        print(f"1. {self.download_info['urls']['official_site']} 방문")
        print(f"2. OfficeHome 데이터셋 다운로드")
        print(f"3. 다운로드한 파일을 {download_file} 경로에 저장")
        print(f"4. 이 스크립트를 다시 실행")
        
        # 파일 존재 확인
        if download_file.exists():
            print(f"\n✅ 다운로드 파일 발견: {download_file}")
        else:
            print(f"\n⚠️ 수동 다운로드가 필요합니다.")
            print(f"파일을 다운로드한 후 다음 경로에 저장하세요: {download_file}")
            
            response = input("다운로드를 완료하셨습니까? (y/N): ")
            if response.lower() != 'y':
                print("❌ 다운로드를 취소합니다.")
                return False
        
        # 압축 해제
        if download_file.exists():
            print(f"\n📦 압축 파일 처리 중...")
            
            if self.extract_archive(download_file):
                # 압축 파일 삭제 (선택사항)
                response = input(f"\n🗑️ 압축 파일을 삭제하시겠습니까? ({download_file.name}) (y/N): ")
                if response.lower() == 'y':
                    download_file.unlink()
                    print("✅ 압축 파일 삭제 완료")
                
                # 데이터셋 검증
                if self.verify_dataset():
                    # 정보 파일 생성
                    self.create_dataset_info()
                    
                    print(f"\n🎉 Office-Home 데이터셋 설정 완료!")
                    print(f"📁 데이터 위치: {self.officehome_dir}")
                    print(f"🎯 4개 도메인, 65개 클래스 준비 완료")
                    return True
        
        print(f"\n❌ 데이터셋 설정 실패")
        return False

def main():
    """메인 실행 함수"""
    
    print("🏠 Office-Home 공식 데이터셋 다운로더")
    print("="*60)
    
    # 다운로더 생성
    downloader = OfficeHomeDownloader()
    
    # 다운로드 실행
    success = downloader.download_official_dataset()
    
    if success:
        print(f"\n🎉 설정 완료! 이제 Office-Home 실험을 실행할 수 있습니다.")
        print(f"실행 명령어: python officehome_full_experiments.py")
    else:
        print(f"\n❌ 설정 실패. 수동으로 데이터셋을 다운로드해주세요.")
    
    return success

if __name__ == "__main__":
    main() 