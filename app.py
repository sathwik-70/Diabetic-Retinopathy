import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
import sys

# Add src to path if needed (though usually not needed if running from root)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src import config
from src.inference import RetinaAnalyzer

# Page config (Aesthetics)
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Custom Button Style */
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize the unified analyzer
    # The analyzer handles loading Grading, Segmentation, and Localization models internally
    analyzer = RetinaAnalyzer(device=device)
    return analyzer

def main():
    st.title("👁️ Diabetic Retinopathy Analysis System")
    st.markdown("### Unified Diagnostic Pipeline")
    st.markdown("Upload a retinal fundus image to detect Severity, Lesions, and Key Features.")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This system integrates three Deep Learning models:
    1. **Disease Grading**: Classifies DR severity (0-4).
    2. **Lesion Segmentation**: Detects Microaneurysms (MA), Hemorrhages (HE), Exudates (EX), and Soft Exudates (SE).
    3. **Localization**: Locates the Optic Disc (OD) and Fovea.
    """)
    
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Main Tabs
    tab_inference, tab_performance = st.tabs(["🔍 Analysis", "📈 Model Metrics"])
    
    with tab_inference:
        if uploaded_file is not None:
            # Layout
            col_input, col_results = st.columns([1, 1.5])
            
            with col_input:
                st.subheader("Input Image")
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_column_width=True)
                
                # --- Validation Logic ---
                def is_valid_fundus(img):
                    # Simple heuristic: Retinal images are overwhelmingly Red/Orange
                    # Logic: Mean Red > Mean Green and Mean Red > Mean Blue
                    # Also Blue channel is usually very low in Retinal images
                    img_array = np.array(img)
                    mean_r = np.mean(img_array[:, :, 0])
                    mean_g = np.mean(img_array[:, :, 1])
                    mean_b = np.mean(img_array[:, :, 2])
                    
                    # Heuristics
                    is_red_dominant = (mean_r > mean_g) and (mean_r > mean_b)
                    is_dark_blue = mean_b < (mean_r * 0.7) # Blue is typically much lower
                    
                    # Also check against purely white/black images
                    is_not_blank = mean_r > 20 and mean_r < 240
                    
                    return is_red_dominant and is_dark_blue and is_not_blank

                if not is_valid_fundus(image):
                    st.error("⚠️ Invalid Image Detected")
                    st.warning("Please upload a valid retinal fundus image.")
                    st.stop()
                
                analyze_btn = st.button("Run Full Analysis", use_container_width=True)

            with col_results:
                if analyze_btn:
                    with st.spinner('Running unified inference pipeline...'):
                        analyzer = load_analyzer()
                        # Save temp file for the analyzer if needed, but analyzer handles paths. 
                        # Actually analyzer.predict takes a path. We should save uploaded file briefly.
                        temp_dir = "temp"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        image.save(temp_path)
                        
                        try:
                            _, results = analyzer.predict(temp_path)
                            
                            # --- 1. Grading Results ---
                            st.subheader("1. Disease Severity")
                            grade = results.get('grade', 0)
                            probs = results.get('grade_probs', [])
                            labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
                            
                            grade_color = "green" if grade == 0 else "orange" if grade <= 2 else "red"
                            st.markdown(f":{grade_color}[**Prediction: {labels[grade]} (Grade {grade})**]")
                            
                            # Confidence Bar
                            if len(probs) > 0:
                                st.progress(float(probs[grade]), text=f"Confidence: {probs[grade]*100:.1f}%")
                            
                            with st.expander("Detailed Probabilities"):
                                for i, p in enumerate(probs):
                                    st.write(f"{labels[i]}: {p*100:.2f}%")
                            
                            st.divider()

                            # --- 2 & 3. Segmentation & Localization Visuals ---
                            st.subheader("2. & 3. Lesion & Feature Maps")
                            
                            # Create visualization
                            # We will use the analyzer's raw results to build a custom overlay for streamlit
                            
                            # Prepare masks overlay
                            w, h = image.size
                            combined_mask = np.zeros((h, w, 4), dtype=np.float32) # RGBA
                            
                            colors = [
                                (1, 0, 0, 0.4),   # MA - Red
                                (0, 1, 0, 0.4),   # HE - Green
                                (0, 0, 1, 0.4),   # EX - Blue
                                (1, 1, 0, 0.4),   # SE - Yellow
                                (0, 1, 1, 0.4)    # OD - Cyan (from segmentation model if present)
                            ]
                            seg_labels = ['Microaneurysms', 'Hemorrhages', 'Hard Exudates', 'Soft Exudates', 'Optic Disc (Seg)']
                            
                            if 'masks' in results:
                                masks = results['masks']
                                for i in range(min(len(masks), len(colors))):
                                    m = masks[i]
                                    # Resize mask to image size
                                    m_img = Image.fromarray((m * 255).astype(np.uint8))
                                    m_img = m_img.resize((w, h), resample=Image.NEAREST)
                                    m_np = np.array(m_img) / 255.0 # 0-1
                                    
                                    # Add to combined mask
                                    color = np.array(colors[i])
                                    # Simple weighted add (not perfect alpha blending but okay for viz)
                                    # Only add color where mask is active
                                    mask_indices = m_np > 0.5
                                    combined_mask[mask_indices] = color 
                            
                            # Convert combined_mask to Image
                            overlay_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
                            
                            # Localization markers
                            # We'll draw these on a separate layer or the same final image
                            final_viz = image.copy().convert("RGBA")
                            final_viz.alpha_composite(overlay_img)
                            
                            # Draw Localization Markers
                            from PIL import ImageDraw
                            draw = ImageDraw.Draw(final_viz)
                            
                            if 'localization' in results:
                                loc = results['localization']
                                od = loc.get('OD')
                                fov = loc.get('Fovea')
                                
                                radius = 15
                                width = 5
                                
                                if od:
                                    x, y = od
                                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="cyan", width=width)
                                    draw.text((x+radius, y), "OD", fill="cyan")
                                
                                if fov:
                                    x, y = fov
                                    draw.line((x-radius, y-radius, x+radius, y+radius), fill="magenta", width=width)
                                    draw.line((x-radius, y+radius, x+radius, y-radius), fill="magenta", width=width)
                                    draw.text((x+radius, y), "Fovea", fill="magenta")

                            st.image(final_viz, caption="Lesions (Masks) + Features (Markers)", use_column_width=True)
                            
                            # Legend
                            st.caption("Legend:")
                            cols = st.columns(5)
                            legend_data = zip(cols, seg_labels, ["Red", "Green", "Blue", "Yellow", "Cyan"])
                            for col, label, color_name in legend_data:
                                col.markdown(f"**{label}**: {color_name}")
                                
                            st.caption("*Markers: Cyan Circle (Optic Disc), Magenta Cross (Fovea)*")

                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            import traceback
                            st.write(traceback.format_exc())

        else:
            st.info("Please upload an image to start.")

    with tab_performance:
        st.header("Training Diagnostics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Loss")
            loss_curve_path = os.path.join(config.BASE_DIR, 'loss_curve.png')
            if os.path.exists(loss_curve_path):
                st.image(loss_curve_path, use_column_width=True)
            else:
                st.write("No loss curve found.")
                
        with col2:
            st.subheader("Confusion Matrix")
            cm_path = os.path.join(config.BASE_DIR, 'confusion_matrix.png')
            if os.path.exists(cm_path):
                st.image(cm_path, use_column_width=True)
            else:
                st.write("No confusion matrix found.")

if __name__ == "__main__":
    main()

