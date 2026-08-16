# Self-Supervised Video Forensics by Audio-Visual Anomaly Detection



---

This is the code for audio-visual forensics.

Steps to run the python code directly:

`pip install -r requirements.txt`

```python
# 1. test a sample fake video (path of video should be full path)
CUDA_VISIBLE_DEVICES=8 python detect.py --test_video_path /home/xxxx/test_video.mp4 --device cuda:0 --max-len 50 --n_workers 4  --bs 1 --lam 0 --output_dir /home/xxx/save

# 2. test a list of fake videos (path of .txt file should be full path, and list should contain full paths of testing videos)
CUDA_VISIBLE_DEVICES=8 python detect.py --test_video_path /home/xxxx/fake_videos.txt --device cuda:0 --max-len 50 --n_workers 4 --bs 1 --lam 0 --output_dir /home/xxx/save
```

(lam is a hyperparameter you can tune to combine scores from distributions over delays and audio-visual network activations mentioned in [paper](https://arxiv.org/pdf/2301.01767.pdf) method section. Default lam=0 is distributions over delays only.)

Audio-visual synchronization model checkpoint `sync_model.pth` can be donwloaded by this [link](https://drive.google.com/file/d/1BxaPiZmpiOJDsbbq8ZIDHJU7--RJE7Br/view?usp=sharing). Noted that AV synchronization model consists of video branch, audio branch, and audio-visual feature fusion transformer.

In the end, there would be a `output.log` file and a `testing_score.npy` file under output_dir generated to record scores for all the testing videos.

---

Audio-visual synchronization model code is based on [vit-pytorch](https://github.com/lucidrains/vit-pytorch)

Decoder only autoregressive model is partially based on [memory-compressed-attention](https://github.com/lucidrains/memory-compressed-attention)

Visual encoder is heavily borrowed from [action classifiction](https://github.com/TengdaHan/ActionClassification)

---



```text
@inproceedings{feng2023self,
  title={Self-supervised video forensics by audio-visual anomaly detection},
  author={Feng, Chao and Chen, Ziyang and Owens, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10491--10503},
  year={2023}
}

# Deepfake Detection System with Video Deblurring

A complete end-to-end deepfake detection system that combines video deblurring and audio-visual synchrony analysis to identify manipulated videos.

## 🌟 Features

- **Video Deblurring**: Automatically enhance blurry videos using UNet deep learning
- **Deepfake Detection**: Advanced AV-synchrony analysis using transformer models
- **REST API**: FastAPI-based backend for easy integration
- **Web Interface**: Modern React-based UI for video upload and analysis
- **Comprehensive Visualization**: 4-panel analysis with heatmaps and distributions
- **Export Capabilities**: CSV files with raw data and detection summaries

## 🎯 How It Works

The system uses a two-stage pipeline:

### Stage 1: Video Deblurring (Optional)
- UNet-based architecture processes each frame
- Improves video quality for better detection
- Can be skipped for clear videos to save time

### Stage 2: Deepfake Detection
1. **Frame & Audio Extraction**: Extract visual and audio streams
2. **Visual Encoding**: Process frames using ViT/ResNet backbone
3. **Audio Encoding**: Extract audio features using CNN encoder
4. **AV Fusion**: Transformer-based fusion of audio-visual features
5. **Delay Analysis**: Detect temporal inconsistencies in AV sync
6. **Feature Regression**: Analyze AV feature correlations
7. **Score Calculation**: Combine scores to determine authenticity

### Detection Principle

Real videos maintain natural audio-visual synchronization with delay distribution peaking at zero. Deepfakes, due to generation artifacts, show:
- Noisy delay distributions
- Anomalous AV feature correlations
- Inconsistent temporal patterns

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- 8GB+ RAM
- 5GB+ disk space for models

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd deepfake-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision torchaudio
pip install fastapi uvicorn python-multipart
pip install opencv-python numpy pandas matplotlib
pip install scikit-learn h5py loguru tqdm pillow
```

### Step 3: Download Model Files

Place the following model files in the project root:

```
sync_model.pth                    (AV-sync model - Required)
dist_regressive_model.pth         (Distribution regressor - Required)
avfeature_regressive_model.pth    (Feature regressor - Required)
pca.pkl                           (PCA model - Required)
checkpoints/deblur_model.pth      (Deblurring model - Optional)
```

### Step 4: Verify Setup

```bash
python test_system.py
```

This will check:
- ✓ Python version
- ✓ Dependencies installed
- ✓ Model files present
- ✓ Directory structure
- ✓ GPU availability

## 🚀 Quick Start

### Method 1: Web Interface (Easiest)

1. Start the server:
```bash
python integrated_server.py
```

2. Open `frontend.html` in your browser

3. Upload a video and click "Analyze Video"

### Method 2: Command Line

Process a single video:

```bash
# With deblurring
python workflow.py --video test.mp4

# Skip deblurring (faster)
python workflow.py --video test.mp4 --skip-deblur

# Use CPU
python workflow.py --video test.mp4 --device cpu
```

### Method 3: API Integration

Start the server:
```bash
python integrated_server.py
```

Make API calls:
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "x-api-key: my_secret_key" \
  -F "file=@video.mp4"
```

## 📖 Usage Examples

### CLI Workflow

```bash
# Basic detection
python workflow.py --video myvideo.mp4

# Custom output directory
python workflow.py --video myvideo.mp4 --output-dir results/

# GPU selection
python workflow.py --video myvideo.mp4 --device cuda:0

# Skip deblurring
python workflow.py --video myvideo.mp4 --skip-deblur
```

### API Usage

**Upload and Analyze:**
```python
import requests

url = "http://localhost:8000/detect"
headers = {"x-api-key": "my_secret_key"}
files = {"file": open("video.mp4", "rb")}

response = requests.post(url, headers=headers, files=files)
result = response.json()

print(f"Score: {result['score']}")
print(f"Label: {result['label']}")
```

**Download Results:**
```python
# Download visualization
viz_url = f"http://localhost:8000/download/visualization/video_analysis.png?x-api-key=my_secret_key"
response = requests.get(viz_url)

with open("visualization.png", "wb") as f:
    f.write(response.content)
```

### Batch Processing

```bash
# Process multiple videos
for video in videos/*.mp4; do
    python workflow.py --video "$video" --output-dir batch_results/
done
```

## 📊 Understanding Results

### Score Interpretation

| Score Range | Verdict | Meaning |
|-------------|---------|---------|
| 0.0 - 0.3   | REAL    | High confidence authentic |
| 0.3 - 0.5   | REAL    | Likely authentic |
| 0.5 - 0.7   | FAKE    | Likely manipulated |
| 0.7 - 1.0   | FAKE    | High confidence fake |
| > 1.0       | FAKE    | Very high confidence fake |

### Visualization Output

The 4-panel visualization includes:

1. **Delay Distribution Heatmap**: Shows AV synchronization delays over time
   - Real videos: Peak at 0 delay
   - Fake videos: Noisy, scattered distribution

2. **Average Delay Distribution**: Bar chart of average delays
   - Real videos: Clear peak at zero
   - Fake videos: Flat or multi-peaked

3. **AV Feature Correlation**: Heatmap of feature similarities
   - Measures consistency of audio-visual features

4. **Summary Statistics**: Scores and verdict
   - Distribution score
   - AV-feature score
   - Final combined score

### CSV Exports

**detection_summary.csv:**
```csv
video_path,final_score,verdict
test.mp4,0.7234,FAKE
```

**{video}_delays.csv:**
- Frame-by-frame delay distributions
- 31 columns (delays from -15 to +15 frames)
- Each row is a video frame

## 🏗️ Project Structure

```
deepfake-detection/
├── integrated_server.py           # Main API server
├── frontend.html                  # Web interface
├── workflow.py                    # CLI workflow runner
├── test_system.py                 # System verification tests
├── detect_with_visualization.py   # Detection with visualization
│
├── deblurring/
│   ├── model.py                   # UNet deblurring model
│   ├── infer.py                   # Deblurring inference
│   └── train.py                   # Training script
│
├── deepfake_detection/
│   ├── model.py                   # Deepfake detector
│   ├── infer.py                   # Detection inference
│   └── train.py                   # Training script
│
├── utils/
│   ├── dataset_loader.py          # Dataset utilities
│   └── preprocessing.py           # Video preprocessing
│
├── api_outputs/                   # API results (auto-created)
│   └── visualizations/
│
├── checkpoints/                   # Model checkpoints
│   └── deblur_model.pth
│
├── sync_model.pth                 # Main models (required)
├── dist_regressive_model.pth
├── avfeature_regressive_model.pth
├── pca.pkl
│
├── requirements.txt               # Python dependencies
├── SETUP_GUIDE.md                 # Detailed setup instructions
├── QUICK_REFERENCE.md             # Quick command reference
└── README.md                      # This file
```

## 🔧 Configuration

### API Server

Edit `integrated_server.py`:

```python
# Change API key
API_KEY = "your_secure_api_key"

# Change port
uvicorn.run(app, host="0.0.0.0", port=8000)

# File paths
OUTPUT_DIR = "api_outputs"
DEBLUR_CHECKPOINT = "checkpoints/deblur_model.pth"
```

### Frontend

Edit `frontend.html`:

```javascript
// Change API URL
const API_URL = "http://your-server:8000";

// Change API key
const API_KEY = "your_secure_api_key";
```

### Detection Parameters

Edit `config_deepfake.py` or pass command-line arguments:

```bash
--device cuda           # Use GPU
--max-len 50           # Maximum sequence length
--bs 1                 # Batch size
--lam 0.5              # Lambda for score blending
```

## 🐛 Troubleshooting

### Common Issues

**"Model files not found"**
- Ensure all .pth and .pkl files are in the project root
- Check file paths in scripts

**"CUDA out of memory"**
```bash
# Use CPU
export USE_CUDA="0"
# or
--device cpu
```

**"Port 8000 already in use"**
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>

# Or use different port
uvicorn integrated_server:app --port 8001
```

**"Import errors"**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

**"Deblurring fails"**
- System automatically uses original video as fallback
- Check if `checkpoints/deblur_model.pth` exists
- Or skip with `--skip-deblur`

### Debug Mode

Run with verbose output:

```bash
# Check system
python test_system.py

# Test workflow
python workflow.py --video test.mp4 --device cpu

# Server logs
tail -f api_outputs/output.log
```

## 🔒 Security

### Production Deployment

1. **Change API Key:**
```bash
export DEEPFAKE_API_KEY="$(openssl rand -hex 32)"
```

2. **Enable HTTPS:**
```bash
uvicorn integrated_server:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile key.pem \
  --ssl-certfile cert.pem
```

3. **Add Rate Limiting:**
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

4. **File Size Limits:**
Edit `integrated_server.py` to add:
```python
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_request_size=100_000_000  # 100MB
)
```

## 📈 Performance

### Typical Processing Times

| Video Length | Resolution | GPU (RTX 3090) | CPU (16-core) |
|--------------|------------|----------------|---------------|
| 10 seconds   | 720p       | ~15 seconds    | ~60 seconds   |
| 30 seconds   | 720p       | ~30 seconds    | ~120 seconds  |
| 1 minute     | 1080p      | ~50 seconds    | ~200 seconds  |

*With deblurring. Subtract ~30% if skipping deblurring.*

### Optimization Tips

1. **Use GPU** - 5-10x faster than CPU
2. **Skip deblurring** - Save 30-50% time for clear videos
3. **Reduce max_len** - For shorter videos
4. **Batch processing** - Process multiple videos in parallel

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed installation and setup
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command quick reference
- **API Docs** - Visit http://localhost:8000/docs when server is running

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request



## 🙏 Acknowledgments

This system uses:
- PyTorch for deep learning
- FastAPI for API framework
- OpenCV for video processing
- Vision Transformer (ViT) for visual encoding
- Transformer models for sequence analysis

## 📞 Support

For issues and questions:
- Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common problems
- Run `python test_system.py` to verify setup
- Check server logs in `api_outputs/output.log`

## 🔄 Updates

### Version 1.0.0
- Initial release
- Video deblurring integration
- AV-synchrony detection
- REST API with FastAPI
- Web interface
- Visualization and CSV export

---

