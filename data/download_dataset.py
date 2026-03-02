"""
数据集下载脚本
下载raw数据集到data/raw目录下
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, login
import requests
from tqdm import tqdm


DATASET_REPO = "CAS-SIAT-XinHai/ReactiGraph"
DATASET_FILE = "AutoRxn/exp_train_data_step_thinking_20k.jsonl"


def download_with_hf_hub(token: str, output_dir: str) -> str:
    """
    使用huggingface_hub下载
    """
    print(f"Method 1: Using huggingface_hub to download...")
    
    try:
        # 登录
        if token:
            login(token=token)
        
        # 下载文件
        file_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=DATASET_FILE,
            repo_type="dataset",
            token=token,
            cache_dir=None,  # 使用默认缓存
            force_download=False
        )
        
        print(f"Downloaded to: {file_path}")
        
        # 复制到输出目录
        output_path = Path(output_dir) / "exp_train_data_step_thinking_20k.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy(file_path, output_path)
        print(f"Copied to: {output_path}")
        
        return str(output_path)
    
    except Exception as e:
        print(f"Method 1 failed: {e}")
        raise


def download_with_wget(token: str, output_dir: str) -> str:
    """
    使用wget直接下载
    """
    print(f"Method 2: Using wget to download...")
    
    output_path = Path(output_dir) / "exp_train_data_step_thinking_20k.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # HuggingFace的直接下载URL
    url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{DATASET_FILE}"
    
    # 构建wget命令
    cmd = f'wget --header="Authorization: Bearer {token}" -O {output_path} {url}'
    
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Downloaded to: {output_path}")
        return str(output_path)
    else:
        raise Exception(f"wget failed: {result.stderr}")


def download_with_requests(token: str, output_dir: str) -> str:
    """
    使用requests下载(带进度条)
    """
    print(f"Method 3: Using requests to download...")
    
    output_path = Path(output_dir) / "exp_train_data_step_thinking_20k.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # HuggingFace的直接下载URL
    url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{DATASET_FILE}"
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # 发送请求
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    # 获取文件大小
    total_size = int(response.headers.get('content-length', 0))
    
    # 下载文件
    with open(output_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Download ReactiGraph dataset")
    parser.add_argument('--token', type=str, required=True, help='HuggingFace token')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--method', type=str, choices=['hf_hub', 'wget', 'requests', 'auto'],
                        default='auto', help='Download method')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    
    print(f"Downloading dataset to: {output_dir}")
    print(f"Dataset: {DATASET_REPO}/{DATASET_FILE}")
    print(f"Method: {args.method}")
    print("-" * 50)
    
    # 尝试不同的下载方法
    methods = {
        'hf_hub': download_with_hf_hub,
        'wget': download_with_wget,
        'requests': download_with_requests
    }
    
    if args.method == 'auto':
        # 自动尝试所有方法
        for method_name, method_func in methods.items():
            try:
                print(f"\nTrying method: {method_name}")
                file_path = method_func(args.token, str(output_dir))
                print(f"\n✓ Success! Dataset downloaded to: {file_path}")
                
                # 验证文件
                file_size = Path(file_path).stat().st_size
                print(f"File size: {file_size / 1024 / 1024:.2f} MB")
                
                return
            except Exception as e:
                print(f"✗ Method {method_name} failed: {e}")
                continue
        
        print("\n✗ All download methods failed!")
        sys.exit(1)
    else:
        # 使用指定的方法
        try:
            method_func = methods[args.method]
            file_path = method_func(args.token, str(output_dir))
            print(f"\n✓ Success! Dataset downloaded to: {file_path}")
            
            # 验证文件
            file_size = Path(file_path).stat().st_size
            print(f"File size: {file_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
