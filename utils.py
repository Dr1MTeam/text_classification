import os
from pathlib import Path
import requests

def download_file(url, dest_path):
    """Вспомогательная функция, скачивающая файл по ссылке в указанную директорию."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise

def check_data_path(path: str = './data', links: dict = None):
    """
    Проверяет существование датасета. Скачивает его, если указаны ссылки.

    Args:
        path (str): Путь к директории с данными. По умолчанию './data'.
        links (dict): Опциональный словарь с именами файлов (keys) и ссылками (values).
                      Пример: {"train.csv": "https://example.com/train.csv"}
    
    Returns:
        str: Результаты проверки.
    """
    path = Path(path)

    # Существует ли path
    if not path.exists():
        os.makedirs(path)
        print(f"Directory {path} created.")

    # Check files and optionally download them
    missing_files = []
    for file_name in (links.keys() if links else ['train.csv', 'test.csv', 'classes.txt']):
        file_path = path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
            if links and file_name in links:
                print(f"{file_name} not found. Downloading...")
                download_file(links[file_name], file_path)

    # Summarize results
    if links:
        if missing_files:
            return f"Missing files were downloaded: {', '.join(missing_files)}"
        else:
            return "Ok. All files exist."
    else:
        if missing_files:
            return f"Links not provided. Missing files: {', '.join(missing_files)}"
        else:
            return "Ok. All files exist."