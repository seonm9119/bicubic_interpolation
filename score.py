import argparse
import base64
import csv
import json
import math
import shutil
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATASET_DIRECTORY = PROJECT_DIRECTORY / "datasets" / "feature_score"
ARCHIVE_DIRECTORY = DATASET_DIRECTORY / "archives"
RAW_DATASET_DIRECTORY = DATASET_DIRECTORY / "raw"
PREPARED_DATASET_DIRECTORY = DATASET_DIRECTORY / "prepared"
CACHE_DIRECTORY = DATASET_DIRECTORY / "cache"
RESULT_DIRECTORY = DATASET_DIRECTORY / "results"
SCALE_FACTOR = 4
DEFAULT_SAMPLE_COUNT = 20
DEFAULT_HIGH_RESOLUTION_SIZE = 512
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

DATASET_SOURCES = {
    "pcb": {
        "label": "PCB",
        "source_name": "DeepPCB",
        "source_url": "https://github.com/tangsanli5201/DeepPCB",
        "archive_name": "deeppcb-master.zip",
        "download_url": "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip",
        "raw_directory_name": "deeppcb",
        "path_keyword": "PCBData",
    },
    "fingerprint": {
        "label": "Fingerprint",
        "source_name": "Neurotechnology CrossMatch Sample DB",
        "source_url": "https://www.neurotechnology.com/download.html",
        "archive_name": "crossmatch-sample-db.zip",
        "download_url": "https://www.neurotechnology.com/download/CrossMatch_Sample_DB.zip",
        "raw_directory_name": "fingerprint_crossmatch",
        "path_keyword": "",
    },
    "iris": {
        "label": "Iris",
        "source_name": "UPOL Iris Database",
        "source_url": "https://phoenix.inf.upol.cz/iris/",
        "archive_name": "upol-iris1-8.zip",
        "download_url": "https://phoenix.inf.upol.cz/iris/download/iris1_8.zip",
        "raw_directory_name": "upol_iris",
        "path_keyword": "",
    },
}

METHODS = [
    {
        "id": "classic-bicubic",
        "label": "Bicubic",
        "api": "bicubic",
        "mode": "classic-bicubic",
    },
    {
        "id": "srgan",
        "label": "SRGAN",
        "api": "srgan",
        "endpoint": "/api/srgan/upscale",
    },
    {
        "id": "real-esrgan",
        "label": "Real-ESRGAN",
        "api": "srgan",
        "endpoint": "/api/realesrgan/upscale",
    },
    {
        "id": "feature-weighted",
        "label": "Feature-weighted Bicubic",
        "api": "bicubic",
        "mode": "feature-weighted",
    },
]


def main():
    arguments = parse_arguments()
    ensure_directories()

    should_run_all_steps = not arguments.download and not arguments.prepare and not arguments.score

    if arguments.download or should_run_all_steps:
        download_datasets()

    if arguments.prepare or should_run_all_steps:
        prepare_datasets(arguments.sample_count, arguments.hr_size)

    if arguments.score or should_run_all_steps:
        score_prepared_datasets(arguments)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Feature-critical SR benchmark scorer")
    parser.add_argument("--download", action="store_true", help="download public source datasets")
    parser.add_argument("--prepare", action="store_true", help="prepare fixed HR/LR image subset")
    parser.add_argument("--score", action="store_true", help="run API inference and write metric tables")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--hr-size", type=int, default=DEFAULT_HIGH_RESOLUTION_SIZE)
    parser.add_argument("--bicubic-url", default="http://sr-bicubic:8080")
    parser.add_argument("--sr-url", default="http://sr-benchmark:8080")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="recreate prepared images and cached outputs")

    return parser.parse_args()


def ensure_directories():
    ARCHIVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RAW_DATASET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PREPARED_DATASET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def download_datasets():
    for dataset_id, dataset_source in DATASET_SOURCES.items():
        archive_path = ARCHIVE_DIRECTORY / dataset_source["archive_name"]
        raw_dataset_directory = RAW_DATASET_DIRECTORY / dataset_source["raw_directory_name"]

        if not archive_path.exists():
            download_file(dataset_source["download_url"], archive_path)

        if contains_image_files(raw_dataset_directory):
            continue

        if raw_dataset_directory.exists():
            shutil.rmtree(raw_dataset_directory)

        raw_dataset_directory.mkdir(parents=True, exist_ok=True)
        extract_zip_file(archive_path, raw_dataset_directory)
        print(f"prepared raw dataset: {dataset_id}")


def download_file(download_url, archive_path):
    temporary_path = archive_path.with_suffix(archive_path.suffix + ".part")
    print(f"downloading {archive_path.name}")

    with urllib.request.urlopen(download_url, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS) as response:
        with temporary_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)

    temporary_path.replace(archive_path)


def extract_zip_file(archive_path, output_directory):
    with zipfile.ZipFile(archive_path) as zip_file:
        zip_file.extractall(output_directory)


def contains_image_files(directory):
    if not directory.exists():
        return False

    for image_path in directory.rglob("*"):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            return True

    return False


def prepare_datasets(sample_count, high_resolution_size):
    if PREPARED_DATASET_DIRECTORY.exists():
        shutil.rmtree(PREPARED_DATASET_DIRECTORY)

    PREPARED_DATASET_DIRECTORY.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for dataset_id, dataset_source in DATASET_SOURCES.items():
        raw_dataset_directory = RAW_DATASET_DIRECTORY / dataset_source["raw_directory_name"]
        source_image_paths = collect_source_image_paths(raw_dataset_directory, dataset_source["path_keyword"])
        selected_image_paths = select_evenly_spaced_paths(source_image_paths, sample_count)
        prepared_dataset_directory = PREPARED_DATASET_DIRECTORY / dataset_id
        prepared_dataset_directory.mkdir(parents=True, exist_ok=True)

        for image_index, source_image_path in enumerate(selected_image_paths, start=1):
            prepared_image_path = prepared_dataset_directory / f"{image_index:03d}-{source_image_path.stem}.png"

            try:
                high_resolution_image = create_high_resolution_image(source_image_path, high_resolution_size)
            except Exception as exception:
                print(f"skip unreadable image: {source_image_path} ({exception})")
                continue

            high_resolution_image.save(prepared_image_path)
            manifest_rows.append({
                "dataset": dataset_id,
                "source": dataset_source["source_name"],
                "sourceUrl": dataset_source["source_url"],
                "preparedImage": str(prepared_image_path.relative_to(DATASET_DIRECTORY)),
                "sourceImage": str(source_image_path.relative_to(DATASET_DIRECTORY)),
                "hrWidth": high_resolution_size,
                "hrHeight": high_resolution_size,
                "lrWidth": high_resolution_size // SCALE_FACTOR,
                "lrHeight": high_resolution_size // SCALE_FACTOR,
            })

        print(f"prepared {dataset_id}: {len(list(prepared_dataset_directory.glob('*.png')))} images")

    write_json(RESULT_DIRECTORY / "prepared_manifest.json", manifest_rows)


def collect_source_image_paths(raw_dataset_directory, path_keyword):
    source_image_paths = []

    for image_path in raw_dataset_directory.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path_keyword and path_keyword not in str(image_path):
            continue

        source_image_paths.append(image_path)

    return sorted(source_image_paths)


def select_evenly_spaced_paths(source_image_paths, sample_count):
    if not source_image_paths:
        return []

    if len(source_image_paths) <= sample_count:
        return source_image_paths

    selected_image_paths = []
    last_source_index = len(source_image_paths) - 1

    for sample_index in range(sample_count):
        source_index = round(sample_index * last_source_index / (sample_count - 1))
        selected_image_paths.append(source_image_paths[source_index])

    return selected_image_paths


def create_high_resolution_image(source_image_path, high_resolution_size):
    with Image.open(source_image_path) as source_image:
        rgb_image = source_image.convert("RGB")
        cropped_image = crop_center_square(rgb_image)

        return cropped_image.resize(
            (high_resolution_size, high_resolution_size),
            Image.Resampling.BICUBIC)


def crop_center_square(image):
    crop_size = min(image.width, image.height)
    left = (image.width - crop_size) // 2
    top = (image.height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    return image.crop((left, top, right, bottom))


def score_prepared_datasets(arguments):
    if arguments.force and CACHE_DIRECTORY.exists():
        shutil.rmtree(CACHE_DIRECTORY)
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    metric_rows = []

    for dataset_id, dataset_source in DATASET_SOURCES.items():
        prepared_dataset_directory = PREPARED_DATASET_DIRECTORY / dataset_id
        high_resolution_image_paths = sorted(prepared_dataset_directory.glob("*.png"))

        if not high_resolution_image_paths:
            print(f"skip empty prepared dataset: {dataset_id}")
            continue

        for image_index, high_resolution_image_path in enumerate(high_resolution_image_paths, start=1):
            print(f"scoring {dataset_id} {image_index}/{len(high_resolution_image_paths)}: {high_resolution_image_path.name}")
            low_resolution_image_path = create_low_resolution_image(high_resolution_image_path)

            for method in METHODS:
                output_image_path = CACHE_DIRECTORY / dataset_id / high_resolution_image_path.stem / f"{method['id']}.png"
                output_image_path.parent.mkdir(parents=True, exist_ok=True)

                metric_row = score_method_output(
                    arguments,
                    dataset_id,
                    dataset_source,
                    high_resolution_image_path,
                    low_resolution_image_path,
                    method,
                    output_image_path)
                metric_rows.append(metric_row)
                write_metric_outputs(metric_rows, arguments)

                if metric_row["status"] == "complete":
                    time.sleep(arguments.pause_seconds)

    write_metric_outputs(metric_rows, arguments)


def create_low_resolution_image(high_resolution_image_path):
    low_resolution_directory = CACHE_DIRECTORY / "low_resolution"
    low_resolution_directory.mkdir(parents=True, exist_ok=True)
    low_resolution_image_path = low_resolution_directory / high_resolution_image_path.name

    if low_resolution_image_path.exists():
        return low_resolution_image_path

    with Image.open(high_resolution_image_path) as high_resolution_image:
        low_resolution_size = (
            high_resolution_image.width // SCALE_FACTOR,
            high_resolution_image.height // SCALE_FACTOR)
        low_resolution_image = high_resolution_image.resize(low_resolution_size, Image.Resampling.BICUBIC)
        low_resolution_image.save(low_resolution_image_path)

    return low_resolution_image_path


def score_method_output(
    arguments,
    dataset_id,
    dataset_source,
    high_resolution_image_path,
    low_resolution_image_path,
    method,
    output_image_path):
    metric_row = {
        "dataset": dataset_id,
        "datasetLabel": dataset_source["label"],
        "source": dataset_source["source_name"],
        "sourceUrl": dataset_source["source_url"],
        "image": high_resolution_image_path.name,
        "method": method["id"],
        "methodLabel": method["label"],
        "status": "complete",
        "psnrDb": None,
        "ssim": None,
        "mse": None,
        "rmse": None,
        "error": None,
    }

    try:
        if not output_image_path.exists():
            request_method_output(arguments, method, low_resolution_image_path, output_image_path)

        metrics = calculate_metrics(high_resolution_image_path, output_image_path)
        metric_row.update(metrics)
    except Exception as exception:
        metric_row["status"] = "failed"
        metric_row["error"] = str(exception)
        print(f"failed {dataset_id} {method['id']}: {exception}")

    return metric_row


def request_method_output(arguments, method, low_resolution_image_path, output_image_path):
    if method["api"] == "bicubic":
        response_json = request_bicubic_output(
            arguments.bicubic_url,
            low_resolution_image_path,
            method["mode"],
            arguments.timeout_seconds)
        result_image_data_url = response_json["output"]["resultImage"]
    else:
        response_json = request_sr_output(
            arguments.sr_url,
            method["endpoint"],
            low_resolution_image_path,
            arguments.timeout_seconds)
        result_image_data_url = response_json["output"]["resultImage"]

    save_data_url_image(result_image_data_url, output_image_path)


def request_bicubic_output(bicubic_url, low_resolution_image_path, mode, timeout_seconds):
    endpoint_url = f"{bicubic_url.rstrip('/')}/api/bicubic/interpolate"
    form_fields = {
        "scaleFactor": str(SCALE_FACTOR),
        "featureWeightPercent": "100",
        "mode": mode,
    }

    return post_multipart_image(endpoint_url, low_resolution_image_path, form_fields, timeout_seconds)


def request_sr_output(sr_url, endpoint, low_resolution_image_path, timeout_seconds):
    endpoint_url = f"{sr_url.rstrip('/')}{endpoint}"

    return post_multipart_image(endpoint_url, low_resolution_image_path, {}, timeout_seconds)


def post_multipart_image(endpoint_url, image_path, form_fields, timeout_seconds):
    boundary = f"----score-boundary-{uuid.uuid4().hex}"
    request_body = create_multipart_body(boundary, image_path, form_fields)
    request = urllib.request.Request(endpoint_url, data=request_body)
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exception:
        error_body = exception.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{endpoint_url} returned {exception.code}: {error_body}") from exception

    return json.loads(response_body)


def create_multipart_body(boundary, image_path, form_fields):
    body_parts = []

    for field_name, field_value in form_fields.items():
        body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
        body_parts.append(f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode("utf-8"))
        body_parts.append(f"{field_value}\r\n".encode("utf-8"))

    body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
    body_parts.append(
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode("utf-8"))
    body_parts.append(b"Content-Type: image/png\r\n\r\n")
    body_parts.append(image_path.read_bytes())
    body_parts.append(b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    return b"".join(body_parts)


def save_data_url_image(data_url, output_image_path):
    if "," not in data_url:
        raise ValueError("API result image is not a data URL")

    image_base64 = data_url.split(",", 1)[1]
    output_image_path.write_bytes(base64.b64decode(image_base64))


def calculate_metrics(reference_image_path, output_image_path):
    with Image.open(reference_image_path) as reference_image:
        reference_array = np.asarray(reference_image.convert("RGB"), dtype=np.float64)

    with Image.open(output_image_path) as output_image:
        output_rgb_image = output_image.convert("RGB")

        if output_rgb_image.size != (reference_array.shape[1], reference_array.shape[0]):
            raise ValueError(
                f"output size {output_rgb_image.size} does not match reference "
                f"{reference_array.shape[1]}x{reference_array.shape[0]}")

        output_array = np.asarray(output_rgb_image, dtype=np.float64)

    difference_array = reference_array - output_array
    mse = float(np.mean(difference_array * difference_array))
    rmse = math.sqrt(mse)
    psnr_db = None if mse <= 0 else 10 * math.log10(255 * 255 / mse)
    ssim = calculate_global_luminance_ssim(reference_array, output_array)

    return {
        "psnrDb": round(psnr_db, 4) if psnr_db is not None else None,
        "ssim": round(ssim, 6),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
    }


def calculate_global_luminance_ssim(reference_array, output_array):
    reference_luminance = (
        0.2126 * reference_array[:, :, 0] +
        0.7152 * reference_array[:, :, 1] +
        0.0722 * reference_array[:, :, 2])
    output_luminance = (
        0.2126 * output_array[:, :, 0] +
        0.7152 * output_array[:, :, 1] +
        0.0722 * output_array[:, :, 2])

    reference_mean = float(np.mean(reference_luminance))
    output_mean = float(np.mean(output_luminance))
    reference_variance = float(np.mean(reference_luminance * reference_luminance) - reference_mean * reference_mean)
    output_variance = float(np.mean(output_luminance * output_luminance) - output_mean * output_mean)
    covariance = float(np.mean(reference_luminance * output_luminance) - reference_mean * output_mean)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * reference_mean * output_mean + c1) * (2 * covariance + c2)
    denominator = (reference_mean * reference_mean + output_mean * output_mean + c1) * (
        reference_variance + output_variance + c2)

    if denominator == 0:
        return 1.0

    return numerator / denominator


def write_metric_outputs(metric_rows, arguments):
    complete_metric_rows = [
        metric_row
        for metric_row in metric_rows
        if metric_row["status"] == "complete"
    ]
    summary_rows = create_summary_rows(complete_metric_rows)
    output_payload = {
        "config": {
            "scaleFactor": SCALE_FACTOR,
            "highResolutionSize": arguments.hr_size,
            "lowResolutionSize": arguments.hr_size // SCALE_FACTOR,
            "sampleCountPerDataset": arguments.sample_count,
            "metrics": ["PSNR", "SSIM", "MSE", "RMSE"],
        },
        "sources": create_source_rows(),
        "summary": summary_rows,
        "perImage": metric_rows,
    }

    write_json(RESULT_DIRECTORY / "feature_score_summary.json", output_payload)
    write_csv(RESULT_DIRECTORY / "feature_score_summary.csv", summary_rows)
    write_markdown_table(RESULT_DIRECTORY / "feature_score_summary.md", output_payload)


def create_summary_rows(metric_rows):
    summary_rows = []

    for dataset_id, dataset_source in DATASET_SOURCES.items():
        for method in METHODS:
            matching_rows = [
                metric_row
                for metric_row in metric_rows
                if metric_row["dataset"] == dataset_id and metric_row["method"] == method["id"]
            ]

            if not matching_rows:
                continue

            summary_rows.append({
                "dataset": dataset_source["label"],
                "method": method["label"],
                "samples": len(matching_rows),
                "psnrDb": average_metric(matching_rows, "psnrDb"),
                "ssim": average_metric(matching_rows, "ssim"),
                "mse": average_metric(matching_rows, "mse"),
                "rmse": average_metric(matching_rows, "rmse"),
            })

    return summary_rows


def average_metric(metric_rows, metric_name):
    metric_values = [
        metric_row[metric_name]
        for metric_row in metric_rows
        if metric_row[metric_name] is not None
    ]

    if not metric_values:
        return None

    precision = 6 if metric_name == "ssim" else 4

    return round(sum(metric_values) / len(metric_values), precision)


def create_source_rows():
    source_rows = []

    for dataset_id, dataset_source in DATASET_SOURCES.items():
        source_rows.append({
            "dataset": dataset_id,
            "label": dataset_source["label"],
            "source": dataset_source["source_name"],
            "url": dataset_source["source_url"],
        })

    return source_rows


def write_json(output_path, payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(output_path, summary_rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        field_names = ["dataset", "method", "samples", "psnrDb", "ssim", "mse", "rmse"]
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_markdown_table(output_path, output_payload):
    lines = [
        "# Feature-Critical Super Resolution Score",
        "",
        f"- scale factor: x{output_payload['config']['scaleFactor']}",
        f"- HR size: {output_payload['config']['highResolutionSize']} x {output_payload['config']['highResolutionSize']}",
        f"- LR size: {output_payload['config']['lowResolutionSize']} x {output_payload['config']['lowResolutionSize']}",
        "- metrics: PSNR higher, SSIM higher, MSE lower, RMSE lower",
        "",
        "| Dataset | Method | Samples | PSNR(dB) | SSIM | MSE | RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for summary_row in output_payload["summary"]:
        lines.append(
            f"| {summary_row['dataset']} | {summary_row['method']} | {summary_row['samples']} | "
            f"{summary_row['psnrDb']} | {summary_row['ssim']} | {summary_row['mse']} | {summary_row['rmse']} |")

    lines.extend([
        "",
        "## Sources",
        "",
    ])

    for source_row in output_payload["sources"]:
        lines.append(f"- {source_row['label']}: {source_row['source']} ({source_row['url']})")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
