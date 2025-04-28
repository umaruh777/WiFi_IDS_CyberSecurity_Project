
# ML-based Wi-Fi IDS Artifact (MSc Cyber Security)

## Description
Practical CNN-based Intrusion Detection System to detect and mitigate Wi-Fi deauthentication attacks.

## Setup Instructions
Install dependencies:
```
pip install -r requirements.txt
```

## Workflow
1. Capture traffic and extract features:
```
python preprocessing/real_time_feature_extraction.py
```
2. Train CNN model:
```
python model/cnn_training_script.py
```
3. Run real-time CNN-based detection:
```
python detection_mitigation/real_time_cnn_detection.py
```
