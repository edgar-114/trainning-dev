import os
import json
import argparse
import subprocess
import tempfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    return parser.parse_args()

def download_data_from_url(urls):
    import urllib.request
    import os
    import zipfile

    temp_dir = tempfile.mkdtemp()
    for url in urls:
        zip_path = os.path.join(temp_dir, 'temp.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    return temp_dir

def create_data_info(user_info):
    dataset_urls = user_info["datasets"]
    data_dir = download_data_from_url(dataset_urls)
    name = user_info["model_id"]
    if not os.path.exists(name):
        os.makedirs(name)
    output_path = os.path.join(name, "model.json")
    return data_dir, output_path

def run_melody(user_info):
    data_dir, output_path = create_data_info(user_info)
    script = os.path.join("melody", "train.py")
    command = (
        f"python {script} "
        f"--data-dir {data_dir} "
        f"--output-path {output_path} "
    )
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert result.returncode == 0, f"Command failed. Error: {result.stderr}"

def run_shakespeare(user_info):
    script = os.path.join("shakespeare", "rnn_training.py")
    data_dir, output_path = create_data_info(user_info)
    command = (
        f"python {script} "
        f"--data-dir {data_dir} "
        f"--output-path {output_path} "
    )   
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert result.returncode == 0, f"Command failed. Error: {result.stderr}"

def run_perceptron(user_info):
    script = os.path.join("perceptron", "training_user.py")
    config_path = os.path.join("perceptron", "config.json")
    data_dir, output_path = create_data_info(user_info)
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

if __name__ == "main":
    args = parse_args()
    json_path = args.json_path
    task = args.task
    with open(json_path, "r") as f:
        user_infos = json.load(f)

    for u_i in user_infos:
        if task == "melody":
            run_melody(u_i)
        elif task == "shakespeare":
            run_shakespeare(u_i)
        elif task == "perceptron":
            run_perceptron(u_i)
        else:
            raise ValueError(f"Task {task} not supported")
