import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2

# ==============================================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==============================================================================

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================


class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(DeepfakeDetector, self).__init__()

        self.backbone = models.efficientnet_b4(weights=None)
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ==============================================================================
# GRAD-CAM CLASS
# ==============================================================================


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        # Ensure correct shape
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Calculate weights
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def visualize(self, image, heatmap, alpha=0.5):
        """Create overlay of heatmap on image."""
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convert to uint8
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay, heatmap_colored

    def __del__(self):
        # Remove hooks
        try:
            self.forward_hook.remove()
            self.backward_hook.remove()
        except:
            pass

# ==============================================================================
# LOAD MODEL (Cached for performance)
# ==============================================================================


@st.cache_resource
def load_model():
    """Load model once and cache it."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        model = DeepfakeDetector(num_classes=2, dropout_rate=0.4).to(device)

        # Load checkpoint
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Initialize Grad-CAM with last feature layer
        target_layer = model.backbone.features[-1]
        gradcam = GradCAM(model, target_layer)

        return model, gradcam, device

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


# Load model
model, gradcam, device = load_model()

# ==============================================================================
# PREPROCESSING
# ==============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================


def predict_image(image, generate_gradcam=True):
    """Predict if image is real or AI-generated."""
    try:
        # Resize image
        image_resized = image.resize((224, 224))
        image_np = np.array(image_resized)

        # Transform
        img_tensor = transform(image_resized).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = output.argmax(1).item()
            confidence = probabilities[predicted_class].item() * 100

        label = "REAL" if predicted_class == 0 else "AI-GENERATED"

        result = {
            'label': label,
            'confidence': confidence,
            'prob_real': probabilities[0].item() * 100,
            'prob_fake': probabilities[1].item() * 100,
            'gradcam_heatmap': None,
            'gradcam_overlay': None
        }

        # Generate Grad-CAM
        if generate_gradcam:
            try:
                heatmap = gradcam.generate_cam(
                    img_tensor, target_class=predicted_class)
                overlay, heatmap_colored = gradcam.visualize(
                    image_np, heatmap, alpha=0.5)
                result['gradcam_heatmap'] = heatmap_colored
                result['gradcam_overlay'] = overlay
            except Exception as e:
                st.warning(f"⚠️ Grad-CAM generation skipped: {str(e)}")

        return result

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# ==============================================================================
# STREAMLIT UI
# ==============================================================================


# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 AI Image Detector with Explainability")
st.markdown(
    "### Detect if an image is real or AI-generated + See what the AI looks at")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI model detects whether an image is:
    - ✅ **REAL** (photographed)
    - ❌ **AI-GENERATED** (created by AI)
    
    **Grad-CAM Visualization:**
    Shows which parts of the image the AI focused on to make its decision.
    
    **Model Performance:**
    - Test Accuracy: 97.78%
    - Architecture: EfficientNet-B4
    - Training: 33k images
    
    **Best for:**
    - Social media photos
    - AI-generated landscapes/art
    - Educational demos
    
    **Limitations:**
    - May struggle with professional photography
    - Limited on AI faces (StyleGAN)
    """)

    st.markdown("---")

    # Options
    st.subheader("⚙️ Options")
    show_gradcam = st.checkbox("Show Grad-CAM", value=True,
                               help="Display AI attention heatmap")

    st.markdown("---")
    st.markdown(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Main content
tab1, tab2 = st.tabs(["📤 Upload Image", "🌐 URL Input"])

# Tab 1: File Upload
with tab1:
    st.markdown("### Upload an image from your computer")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Supported formats: JPG, PNG, BMP, WEBP"
    )

    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')

            # Predict
            with st.spinner("🔍 Analyzing image..."):
                result = predict_image(image, generate_gradcam=show_gradcam)

            if result is not None:
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Results")

                # Create columns based on Grad-CAM option
                if show_gradcam and result['gradcam_overlay'] is not None:
                    col1, col2, col3, col4 = st.columns(4)
                else:
                    col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📷 Original Image")
                    st.image(image, use_container_width=True)
                    st.caption(f"Size: {image.size[0]} x {image.size[1]} px")

                with col2:
                    st.markdown("#### 🎯 Prediction")
                    if result['label'] == "REAL":
                        st.success(f"### ✅ {result['label']}")
                    else:
                        st.error(f"### ❌ {result['label']}")

                    st.metric("Confidence", f"{result['confidence']:.2f}%")

                    # Progress bars
                    st.markdown("**Probabilities:**")
                    st.progress(result['prob_real'] / 100,
                                text=f"Real: {result['prob_real']:.1f}%")
                    st.progress(result['prob_fake'] / 100,
                                text=f"AI: {result['prob_fake']:.1f}%")

                if show_gradcam and result['gradcam_overlay'] is not None:
                    with col3:
                        st.markdown("#### 🔥 Grad-CAM Heatmap")
                        st.image(result['gradcam_heatmap'],
                                 use_container_width=True)
                        st.caption("Red = High attention")

                    with col4:
                        st.markdown("#### 🎨 Overlay")
                        st.image(result['gradcam_overlay'],
                                 use_container_width=True)
                        st.caption("Shows AI focus areas")

                # Interpretation
                st.markdown("---")
                st.markdown("### 💡 Interpretation")

                col_interp1, col_interp2 = st.columns(2)

                with col_interp1:
                    st.markdown("**Confidence Level:**")
                    if result['confidence'] >= 95:
                        st.info(
                            f"🟢 **VERY CONFIDENT** ({result['confidence']:.1f}%)\n\nThe model is extremely certain about this prediction.")
                    elif result['confidence'] >= 85:
                        st.info(
                            f"🟢 **CONFIDENT** ({result['confidence']:.1f}%)\n\nThe model has high confidence in this prediction.")
                    elif result['confidence'] >= 70:
                        st.warning(
                            f"🟡 **MODERATE** ({result['confidence']:.1f}%)\n\nThe model is somewhat confident but not certain.")
                    else:
                        st.warning(
                            f"🟡 **UNCERTAIN** ({result['confidence']:.1f}%)\n\nThis is a borderline case. The model is unsure.")

                with col_interp2:
                    if show_gradcam and result['gradcam_overlay'] is not None:
                        st.markdown("**Grad-CAM Explanation:**")
                        st.info("""
                        The **heatmap** shows where the AI looked:
                        - 🔴 **Red**: High attention (key areas)
                        - 🟡 **Yellow**: Moderate attention
                        - 🔵 **Blue**: Low attention (ignored)
                        
                        These areas influenced the AI's decision.
                        """)

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

# Tab 2: URL Input
with tab2:
    st.markdown("### Load an image from a URL")

    # Example URLs
    with st.expander("📋 Example URLs to try"):
        st.code("https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0")
        st.code("https://images.unsplash.com/photo-1494790108377-be9c29b29330")
        st.code("https://thispersondoesnotexist.com/")

    image_url = st.text_input(
        "Image URL",
        placeholder="https://example.com/image.jpg",
        help="Paste a direct link to an image"
    )

    if st.button("🔍 Analyze URL", type="primary"):
        if image_url:
            try:
                with st.spinner("📥 Downloading image..."):
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(
                        BytesIO(response.content)).convert('RGB')

                # Predict
                with st.spinner("🔍 Analyzing image..."):
                    result = predict_image(
                        image, generate_gradcam=show_gradcam)

                if result is not None:
                    # Display (same layout as Tab 1)
                    st.markdown("---")
                    st.markdown("## 📊 Results")

                    if show_gradcam and result['gradcam_overlay'] is not None:
                        col1, col2, col3, col4 = st.columns(4)
                    else:
                        col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 📷 Original Image")
                        st.image(image, use_container_width=True)

                    with col2:
                        st.markdown("#### 🎯 Prediction")
                        if result['label'] == "REAL":
                            st.success(f"### ✅ {result['label']}")
                        else:
                            st.error(f"### ❌ {result['label']}")

                        st.metric("Confidence", f"{result['confidence']:.2f}%")
                        st.progress(result['prob_real'] / 100,
                                    text=f"Real: {result['prob_real']:.1f}%")
                        st.progress(result['prob_fake'] / 100,
                                    text=f"AI: {result['prob_fake']:.1f}%")

                    if show_gradcam and result['gradcam_overlay'] is not None:
                        with col3:
                            st.markdown("#### 🔥 Grad-CAM")
                            st.image(result['gradcam_heatmap'],
                                     use_container_width=True)

                        with col4:
                            st.markdown("#### 🎨 Overlay")
                            st.image(result['gradcam_overlay'],
                                     use_container_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error downloading image: {str(e)}")
                st.info(
                    "Make sure the URL is valid and points directly to an image file.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter an image URL")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit 🎈 | Powered by EfficientNet-B4 🤖 | Explainable with Grad-CAM 🔍</p>
    <p><small>⚠️ Optimized for social media photos and AI-generated landscapes. 
    May not work well with professional photography or AI faces.</small></p>
</div>
""", unsafe_allow_html=True)
