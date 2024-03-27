import os
import shutil
import gdown
import zipfile
import json
import requests
import argparse
import subprocess

UPLOAD_MODEL_API = "https://api-dojo.eternalai.org/api/admin/dojo/upload-output?admin_key=eai2024"

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

def create_data_info(task, user_info, temp_data_dir):
    dataset_urls = []
    for info in user_info["datasets"]:
        dataset_urls.append(info["url"])

    data_dir = download_and_unzip(dataset_urls, temp_data_dir)
    output_dir = "./outputs_" + task
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, user_info["model_id"] +  ".json")
    return data_dir, output_path

def run_melody(user_info, temp_data_dir):
    data_dir, output_path = create_data_info("melody", user_info, temp_data_dir)
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
    return output_path

def run_shakespeare(user_info, temp_data_dir):
    script = os.path.join("shakespeare", "rnn_training.py")
    data_dir, output_path = create_data_info("shakespeare", user_info, temp_data_dir)
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
    return output_path

def run_perceptron(user_info, temp_data_dir):
    script = os.path.join("perceptron", "training_user.py")
    config_path = os.path.join("perceptron", "config.json")
    data_dir, output_path = create_data_info("perceptron", user_info, temp_data_dir)

    dataset_infos = user_info["datasets"]
    class_names = {}
    for info in dataset_infos:
        dataset_folder_name = os.path.splitext(os.path.basename(info['url']))[0]
        class_names[dataset_folder_name] = info['name']

    with open(config_path, 'r') as f:
        config = json.load(f)
    config['class_names'] = class_names    
    with open(config_path, 'w') as f:
        json.dump(config, f)

    command = (
        f"python {script} "
        f"-d {data_dir} "
        f"-c {config_path} "
        f"-o {output_path} "
    )   
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert result.returncode == 0, f"Command failed. Error: {result.stderr}"
    return output_path

def upload_output(output_path):
    model_id = os.path.basename(output_path).split('.')[0]
    payload = {'model_id': model_id}
    files=[
    ('output',(output_path.split('/')[-1], open(output_path,'rb'), 'application/json'))
    ]
    headers = {}
    response = requests.request("POST", UPLOAD_MODEL_API, headers=headers, data=payload, files=files)
    status = {
        "response": response.text,
        "model_id": model_id
    }
    if response.status_code == 200:
        status["status"] = "success"
    else:
        status["status"] = "failed"
    return status

if __name__ == "__main__":
    args = parse_args()
    json_path = args.json_path
    with open(json_path, "r") as f:
        user_infos = json.load(f)

    uploading_status = []
    
    for task, u_i in user_infos:
        temp_data_dir = os.path.join('./', "temp_data_dir")
        if task == "melody":
            output_path = run_melody(u_i, temp_data_dir)
        elif task == "shakespeare":
            output_path = run_shakespeare(u_i, temp_data_dir)
        elif task == "perceptron":
            output_path = run_perceptron(u_i, temp_data_dir)
        else:
            raise ValueError(f"Task {task} not supported")
        status = upload_output(output_path)
        uploading_status.append(status)
        shutil.rmtree(temp_data_dir, ignore_errors=True)
    
    with open("uploading_status.json", "w") as f:
        json.dump(uploading_status, f)
