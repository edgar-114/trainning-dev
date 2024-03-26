import os
import shutil
import gdown
import zipfile
import json
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, required=True)
    return parser.parse_args()

def download_and_unzip(urls, temp_data_dir):
    """
    Downloads files from a list of URLs and zips them into a single zip file.
    
    Args:
        urls (list): List of URLs of files to download.
        zip_filename (str): Name of the zip file to create.
        
    Returns:
        str: Path to the created zip file.
    """
    tmp = os.path.join('./', "tmp")
    temp_dir = os.path.join('./', "temp_dir")
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    # Download files
    downloaded_files = []
    for url in urls:
        filename = os.path.basename(url)
        file_path = os.path.join(tmp, filename)
        gdown.download(url, file_path, quiet=False)    
        downloaded_files.append(file_path)
    # extract all zip from downloaded_files to tempdir
    for file in downloaded_files:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(temp_data_dir)
    shutil.rmtree(os.path.join(temp_data_dir, "__MACOSX"), ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return temp_data_dir

def create_data_info(user_info, temp_data_dir):
    dataset_urls = user_info["datasets"]
    data_dir = download_and_unzip(dataset_urls, temp_data_dir)
    output_path = os.path.join("./", user_info["model_id"] +  ".json")
    return data_dir, output_path

def run_melody(user_info, temp_data_dir):
    data_dir, output_path = create_data_info(user_info, temp_data_dir)
    script = os.path.join("melody", "train.py")
    config_path = os.path.join("shakespeare", "config.json")
    command = (
        f"python {script} "
        f"--data-dir {data_dir} "
        f"--output-path {output_path} "
        f"--config-path {config_path}"

    )
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert result.returncode == 0, f"Command failed. Error: {result.stderr}"

def run_shakespeare(user_info, temp_data_dir):
    script = os.path.join("shakespeare", "rnn_training.py")
    data_dir, output_path = create_data_info(user_info, temp_data_dir)
    config_path = os.path.join("shakespeare", "config.json")
    command = (
        f"python {script} "
        f"--data-dir {data_dir} "
        f"--output-path {output_path} "
        f"--config-path {config_path}"
    )   
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert result.returncode == 0, f"Command failed. Error: {result.stderr}"

def run_perceptron(user_info, temp_data_dir):
    script = os.path.join("perceptron", "training_user.py")
    config_path = os.path.join("perceptron", "config.json")
    data_dir, output_path = create_data_info(user_info, temp_data_dir)
    command = (
        f"python {script} "
        f"-d {data_dir} "
        f"-c {config_path}"
        f"-o {output_path} "
    )   
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert result.returncode == 0, f"Command failed. Error: {result.stderr}"

if __name__ == "__main__":
    args = parse_args()
    json_path = args.json_path
    with open(json_path, "r") as f:
        user_infos = json.load(f)
    
    for task, u_i in user_infos:
        temp_data_dir = os.path.join('./', "temp_data_dir")
        if task == "melody":
            run_melody(u_i, temp_data_dir)
        elif task == "shakespeare":
            run_shakespeare(u_i, temp_data_dir)
        elif task == "perceptron":
            run_perceptron(u_i, temp_data_dir)
        else:
            raise ValueError(f"Task {task} not supported")
        shutil.rmtree(temp_data_dir, ignore_errors=True)
