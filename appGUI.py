import streamlit as st
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import io

# Generate a chaotic sequence using the logistic map
def logistic_map(r, x, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)
    return np.array(sequence)

# Compute fitness based on entropy and correlation
def fitness_function(encrypted_image):
    hist = np.histogram(encrypted_image, bins=256)[0]
    entropy = -np.sum((hist / np.sum(hist)) * np.log2(hist + 1e-9))
    correlation = np.corrcoef(encrypted_image.flatten(), np.roll(encrypted_image.flatten(), 1))[0, 1]
    return entropy - abs(correlation)

# Particle Swarm Optimization (PSO) based encryption process
def pso_optimize(image, num_particles=5, iterations=10, r=3.99):
    M, N = image.shape
    num_pixels = M * N
    particles = []
    fitness_scores = []

    # Generate initial encrypted images using chaotic sequences
    for _ in range(num_particles):
        x0 = random.uniform(0, 1)
        chaotic_seq = logistic_map(r, x0, num_pixels)
        chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
        encrypted_image = cv2.bitwise_xor(image, chaotic_seq)

        particles.append((encrypted_image, x0))  # Store x0 for decryption
        fitness_scores.append(fitness_function(encrypted_image))

    # Iterate to improve encryption using fitness evaluation
    for _ in range(iterations):
        for i in range(num_particles):
            new_x0 = random.uniform(0, 1)
            chaotic_seq = logistic_map(r, new_x0, num_pixels)
            chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
            new_encrypted_image = cv2.bitwise_xor(image, chaotic_seq)

            new_fitness = fitness_function(new_encrypted_image)

            if new_fitness > fitness_scores[i]:
                particles[i] = (new_encrypted_image, new_x0)
                fitness_scores[i] = new_fitness

    best_index = np.argmax(fitness_scores)
    return particles[best_index]

# Decryption process (XOR again with the same chaotic sequence)
def decrypt_image(encrypted_image, x0, r):
    M, N = encrypted_image.shape
    num_pixels = M * N
    chaotic_seq = logistic_map(r, x0, num_pixels)
    chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
    decrypted_image = cv2.bitwise_xor(encrypted_image, chaotic_seq)
    return decrypted_image

# Streamlit UI configuration
st.set_page_config(page_title="🔐 PSO Image Encryption & Decryption", layout="wide")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Open_Lock.svg/1024px-Open_Lock.svg.png", width=100)
st.sidebar.title("⚙️ Encryption Settings")
st.sidebar.write("Adjust parameters before encryption.")

# User-defined parameters for PSO
tnum_particles = st.sidebar.slider("🧩 Number of Particles", 2, 10, 5)
iterations = st.sidebar.slider("🔄 Iterations", 1, 20, 10)
r_value = st.sidebar.slider("⚡ Logistic Map r-value", 3.5, 4.0, 3.99)

st.markdown("<h1 style='text-align: center;'>🔐 Image Encryption & Decryption using PSO</h1>", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("📤 Upload an Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded image to grayscale
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)
    image = cv2.resize(image, (256, 256))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="🖼️ Original Image", use_column_width=True)

    # Encryption button
    if st.button("🚀 Encrypt Image", use_container_width=True):
        progress = st.progress(0)

        with st.spinner("🔒 Encrypting... Please wait ⏳"):
            for i in range(100):
                progress.progress(i + 1)

            best_encrypted_image, best_x0 = pso_optimize(image, tnum_particles, iterations, r_value)

        with col2:
            st.image(best_encrypted_image, caption="🔒 Encrypted Image", use_column_width=True)

        # Convert encrypted image for download
        img_pil = Image.fromarray(best_encrypted_image)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(label="📥 Download Encrypted Image",
                           data=byte_im,
                           file_name="encrypted_image.png",
                           mime="image/png",
                           use_container_width=True)

        # Decryption button
        if st.button("🔓 Decrypt Image", use_container_width=True):
            decrypted_image = decrypt_image(best_encrypted_image, best_x0, r_value)
            with col3:
                st.image(decrypted_image, caption="🔓 Decrypted Image", use_column_width=True)
