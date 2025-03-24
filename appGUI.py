import streamlit as st
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import io

# Function Definitions
def logistic_map(r, x, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)
    return np.array(sequence)

def fitness_function(encrypted_image):
    hist = np.histogram(encrypted_image, bins=256)[0]
    entropy = -np.sum((hist / np.sum(hist)) * np.log2(hist + 1e-9))
    correlation = np.corrcoef(encrypted_image.flatten(), np.roll(encrypted_image.flatten(), 1))[0, 1]
    return entropy - abs(correlation)

def pso_optimize(image, num_particles=5, iterations=10, r=3.99):
    M, N = image.shape
    num_pixels = M * N
    particles = []
    fitness_scores = []

    for _ in range(num_particles):
        x0 = random.uniform(0, 1)
        chaotic_seq = logistic_map(r, x0, num_pixels)
        chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
        encrypted_image = cv2.bitwise_xor(image, chaotic_seq)

        particles.append(encrypted_image)
        fitness_scores.append(fitness_function(encrypted_image))

    for _ in range(iterations):
        for i in range(num_particles):
            new_x0 = random.uniform(0, 1)
            chaotic_seq = logistic_map(r, new_x0, num_pixels)
            chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
            new_encrypted_image = cv2.bitwise_xor(image, chaotic_seq)

            new_fitness = fitness_function(new_encrypted_image)

            if new_fitness > fitness_scores[i]:
                particles[i] = new_encrypted_image
                fitness_scores[i] = new_fitness

    best_index = np.argmax(fitness_scores)
    return particles[best_index]

# Streamlit UI Setup
st.set_page_config(page_title="🔐 PSO Image Encryption", layout="wide")

# Sidebar Configuration
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Open_Lock.svg/1024px-Open_Lock.svg.png", width=100)
st.sidebar.title("⚙️ Encryption Settings")
st.sidebar.write("Adjust parameters before encryption.")

num_particles = st.sidebar.slider("🧩 Number of Particles", 2, 10, 5)
iterations = st.sidebar.slider("🔄 Iterations", 1, 20, 10)
r_value = st.sidebar.slider("⚡ Logistic Map r-value", 3.5, 4.0, 3.99)

# Main UI Layout
st.markdown("<h1 style='text-align: center;'>🔐 Image Encryption using PSO</h1>", unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("📤 Upload an Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = np.array(image)
    image = cv2.resize(image, (256, 256))

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼️ Original Image", use_column_width=True)

    if st.button("🚀 Encrypt Image", use_container_width=True):
        progress = st.progress(0)

        with st.spinner("🔒 Encrypting... Please wait ⏳"):
            for i in range(100):
                progress.progress(i + 1)

            best_encrypted_image = pso_optimize(image, num_particles, iterations, r_value)

        with col2:
            st.image(best_encrypted_image, caption="🔒 Encrypted Image", use_column_width=True)

        # Convert image to downloadable format
        img_pil = Image.fromarray(best_encrypted_image)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(label="📥 Download Encrypted Image",
                           data=byte_im,
                           file_name="encrypted_image.png",
                           mime="image/png",
                           use_container_width=True)
