import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "aiswaryaramachandran/hindienglish-corpora",
    path="DeepLearning-S7-AI-ML-KTU-Lab/code/exp11"
)

print("Path to dataset files:", path)