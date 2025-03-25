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
def pso_optimize(image, num_particles=5, iterations=10, r=3.99, w=0.7, c1=1.5, c2=1.5):
    M, N = image.shape
    num_pixels = M * N
    particles = [random.uniform(0, 1) for _ in range(num_particles)]
    velocities = [random.uniform(-0.1, 0.1) for _ in range(num_particles)]
    fitness_scores = []
    personal_best = particles[:]
    personal_best_scores = []
    global_best = None
    global_best_score = -np.inf

    for i in range(num_particles):
        chaotic_seq = logistic_map(r, particles[i], num_pixels)
        chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
        encrypted_image = cv2.bitwise_xor(image, chaotic_seq)
        fitness = fitness_function(encrypted_image)
        fitness_scores.append(fitness)
        personal_best_scores.append(fitness)

        if fitness > global_best_score:
            global_best = particles[i]
            global_best_score = fitness

    for _ in range(iterations):
        for i in range(num_particles):
            velocities[i] = (w * velocities[i] +
                             c1 * random.uniform(0, 1) * (personal_best[i] - particles[i]) +
                             c2 * random.uniform(0, 1) * (global_best - particles[i]))
            particles[i] += velocities[i]
            particles[i] = max(0, min(1, particles[i]))

            chaotic_seq = logistic_map(r, particles[i], num_pixels)
            chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
            encrypted_image = cv2.bitwise_xor(image, chaotic_seq)
            fitness = fitness_function(encrypted_image)

            if fitness > personal_best_scores[i]:
                personal_best[i] = particles[i]
                personal_best_scores[i] = fitness

            if fitness > global_best_score:
                global_best = particles[i]
                global_best_score = fitness

    chaotic_seq = logistic_map(r, global_best, num_pixels)
    chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
    best_encrypted_image = cv2.bitwise_xor(image, chaotic_seq)
    return best_encrypted_image, global_best

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

# User-defined parameters for PSO
num_particles = st.sidebar.slider("🧩 Number of Particles", 2, 10, 5)
iterations = st.sidebar.slider("🔄 Iterations", 1, 20, 10)
r_value = st.sidebar.slider("⚡ Logistic Map r-value", 3.5, 4.0, 3.99)

st.markdown("""
    <h1 style='text-align: center;'>🔐 Image Encryption & Decryption using PSO</h1>
""", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("📤 Upload an Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)
    image = cv2.resize(image, (256, 256))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(image, caption="🖼️ Original Image", use_container_width=True)

    best_encrypted_image, best_x0 = None, None
    with st.expander("🔒 Encrypt Image", expanded=True):
        if st.button("🚀 Encrypt", use_container_width=True):
            progress = st.progress(0)
            
            with st.spinner("🔒 Encrypting... Please wait ⏳"):
                for i in range(100):
                    progress.progress(i + 1)
                best_encrypted_image, best_x0 = pso_optimize(image, num_particles, iterations, r_value)
                progress.progress(100)

            with col2:
                st.image(best_encrypted_image, caption="🔒 Encrypted Image", use_container_width=True)

            buf = io.BytesIO()
            Image.fromarray(best_encrypted_image).save(buf, format="PNG")
            st.download_button("📥 Download Encrypted Image", buf.getvalue(), "encrypted_image.png", "image/png", use_container_width=True)
    
    if best_encrypted_image is not None and best_x0 is not None:
        with st.expander("🔓 Decrypt Image", expanded=True):
            if st.button("🔓 Decrypt", use_container_width=True):
                decrypted_image = decrypt_image(best_encrypted_image, best_x0, r_value)
                with col3:
                    st.image(decrypted_image, caption="🔓 Decrypted Image", use_container_width=True)
