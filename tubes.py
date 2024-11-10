import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menghitung derajat keanggotaan segitiga
def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

# Fungsi untuk membuat grafik fungsi keanggotaan
def plot_membership_functions():
    x_request = np.linspace(0, 1000, 1000)
    x_security = np.linspace(0, 10, 1000)
    x_anomalous = np.linspace(0, 500, 1000)

    # Fungsi keanggotaan untuk setiap variabel
    request_low = [triangular_membership(x, 0, 0, 500) for x in x_request]
    request_med = [triangular_membership(x, 0, 500, 1000) for x in x_request]
    request_high = [triangular_membership(x, 500, 1000, 1000) for x in x_request]

    security_low = [triangular_membership(x, 0, 0, 5) for x in x_security]
    security_med = [triangular_membership(x, 0, 5, 10) for x in x_security]
    security_high = [triangular_membership(x, 5, 10, 10) for x in x_security]

    anomalous_low = [triangular_membership(x, 0, 0, 250) for x in x_anomalous]
    anomalous_med = [triangular_membership(x, 0, 250, 500) for x in x_anomalous]
    anomalous_high = [triangular_membership(x, 250, 500, 500) for x in x_anomalous]

    # Plot setiap variabel dengan Matplotlib
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot Request Count
    axs[0].plot(x_request, request_low, label="Low")
    axs[0].plot(x_request, request_med, label="Medium")
    axs[0].plot(x_request, request_high, label="High")
    axs[0].set_title("Request Count Membership Functions")
    axs[0].legend()

    # Plot System Security Level
    axs[1].plot(x_security, security_low, label="Low")
    axs[1].plot(x_security, security_med, label="Medium")
    axs[1].plot(x_security, security_high, label="High")
    axs[1].set_title("System Security Level Membership Functions")
    axs[1].legend()

    # Plot Anomalous Data Volume
    axs[2].plot(x_anomalous, anomalous_low, label="Low")
    axs[2].plot(x_anomalous, anomalous_med, label="Medium")
    axs[2].plot(x_anomalous, anomalous_high, label="High")
    axs[2].set_title("Anomalous Data Volume Membership Functions")
    axs[2].legend()

    plt.tight_layout()
    return fig

# Fungsi Fuzzifikasi
def fuzzification(request_count, system_security_level, anomalous_data_volume):
    # Request Count Membership
    req_low = triangular_membership(request_count, 0, 0, 500)
    req_med = triangular_membership(request_count, 0, 500, 1000)
    req_high = triangular_membership(request_count, 500, 1000, 1000)

    # System Security Level Membership
    sec_low = triangular_membership(system_security_level, 0, 0, 5)
    sec_med = triangular_membership(system_security_level, 0, 5, 10)
    sec_high = triangular_membership(system_security_level, 5, 10, 10)

    # Anomalous Data Volume Membership
    anom_low = triangular_membership(anomalous_data_volume, 0, 0, 250)
    anom_med = triangular_membership(anomalous_data_volume, 0, 250, 500)
    anom_high = triangular_membership(anomalous_data_volume, 250, 500, 500)
    
    return {
        'req_low': req_low, 'req_med': req_med, 'req_high': req_high,
        'sec_low': sec_low, 'sec_med': sec_med, 'sec_high': sec_high,
        'anom_low': anom_low, 'anom_med': anom_med, 'anom_high': anom_high
    }

# Fungsi Evaluasi Aturan Fuzzy dan Agregasi Output
def rule_evaluation(fuzzy_values):
    # Definisikan aturan dengan output tetap (Sugeno)
    rules = [
        (fuzzy_values['req_low'], fuzzy_values['sec_med'], fuzzy_values['anom_low'], 25),  # Risiko rendah
        (fuzzy_values['req_med'], fuzzy_values['sec_low'], fuzzy_values['anom_med'], 50),  # Risiko sedang
        (fuzzy_values['req_high'], fuzzy_values['sec_low'], fuzzy_values['anom_high'], 75) # Risiko tinggi
    ]
    
    # Evaluasi setiap aturan menggunakan min (AND)
    weighted_outputs = []
    weights = []
    
    for rule in rules:
        firing_strength = min(rule[0], rule[1], rule[2])  # Ambil minimum dari tiap kondisi aturan
        output_value = rule[3]  # Output tetap untuk aturan ini
        weighted_outputs.append(firing_strength * output_value)
        weights.append(firing_strength)
    
    return weighted_outputs, weights

# Fungsi Defuzzifikasi dengan Metode Sugeno
def defuzzification(weighted_outputs, weights):
    if sum(weights) == 0:
        return 0
    return sum(weighted_outputs) / sum(weights)

# Fungsi utama untuk mengimplementasikan FIS
def fuzzy_inference_system(request_count, system_security_level, anomalous_data_volume):
    fuzzy_values = fuzzification(request_count, system_security_level, anomalous_data_volume)
    weighted_outputs, weights = rule_evaluation(fuzzy_values)
    result = defuzzification(weighted_outputs, weights)
    return result

# Dataset untuk uji coba
dataset = [
    (200, 3, 50, 30),  # (request_count, system_security_level, anomalous_data_volume, actual_risk_level)
    (400, 2, 150, 60),
    (150, 8, 30, 20),
    (900, 1, 400, 95),
    (250, 5, 80, 40),
    (700, 3, 300, 85),
    (100, 9, 20, 10),
    (500, 7, 100, 50),
    (800, 4, 350, 90),
    (300, 6, 60, 35),
]

# Menghitung MAE untuk mengevaluasi kinerja sistem
def calculate_mae(dataset):
    total_error = 0
    for data in dataset:
        request_count, system_security_level, anomalous_data_volume, actual_risk = data
        predicted_risk = fuzzy_inference_system(request_count, system_security_level, anomalous_data_volume)
        total_error += abs(predicted_risk - actual_risk)
    mae = total_error / len(dataset)
    return mae

# Antarmuka Streamlit
st.title("Cyber Attack Risk Level Prediction")
st.write("This app predicts the cyber attack risk level based on input values using Fuzzy Inference System (FIS).")

# Tampilkan fungsi keanggotaan
st.subheader("Membership Functions")
fig = plot_membership_functions()
st.pyplot(fig)

# Input dari pengguna
st.subheader("Input Values")
request_count = st.slider("Request Count", 0, 1000, 200)
system_security_level = st.slider("System Security Level", 0, 10, 3)
anomalous_data_volume = st.slider("Anomalous Data Volume", 0, 500, 50)

# Prediksi berdasarkan input pengguna
predicted_risk = fuzzy_inference_system(request_count, system_security_level, anomalous_data_volume)
st.write(f"Predicted Cyber Attack Risk Level: {predicted_risk:.2f}")

# Menghitung dan menampilkan MAE
mae = calculate_mae(dataset)
st.write(f"Mean Absolute Error (MAE) on dataset: {mae:.2f}")

# Tampilkan hasil prediksi untuk dataset uji coba
st.subheader("Dataset Predictions")
for data in dataset:
    req_count, sec_level, anom_volume, actual_risk = data
    pred_risk = fuzzy_inference_system(req_count, sec_level, anom_volume)
    st.write(f"Input: ({req_count}, {sec_level}, {anom_volume}) -> Predicted: {pred_risk:.2f}, Actual: {actual_risk}")
