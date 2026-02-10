# 🔍 AI Image Detector with Grad-CAM

Detect whether an image is real or AI-generated using deep learning, with visual explanations powered by Grad-CAM.

## ✨ Features

- 📤 Upload images or provide URLs
- 🎯 97.78% test accuracy
- 🔥 Grad-CAM heatmaps showing AI attention
- 🎨 Interactive visualization
- 💡 Explainable AI predictions

## 🧠 Model Details

- **Architecture:** EfficientNet-B4
- **Training Data:** 33,333 images (real vs AI-generated)
- **Test Accuracy:** 97.78%
- **Explainability:** Grad-CAM attention maps

## 🚀 Live Demo

[Click here to try the live app!](https://your-app-url.streamlit.app)

## 💻 Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-detector-app.git
cd deepfake-detector-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
