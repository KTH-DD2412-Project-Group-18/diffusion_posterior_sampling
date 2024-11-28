from datasets import load_dataset, DownloadConfig

# Create a download configuration that only downloads validation
download_config = DownloadConfig(force_download=True)

# Load only validation split
ds = load_dataset(
    "benjamin-paine/imagenet-1k-256x256",
    split="validation",
    download_config=download_config
)

ds.save_to_disk("./datasets/256")