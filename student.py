import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model dan scaler
model = tf.lite.Interpreter(model_path="student-performance.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

le = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Konfigurasi tampilan halaman
st.set_page_config(page_title="🎯 Prediksi Performa Belajar", layout="centered")
st.title("🎓✨ Prediksi Performa Belajar Siswa ✨🎓")
st.markdown("Selamat datang di aplikasi **prediksi performa belajar siswa**! 🎒📚 Yuk, isi data di bawah ini untuk melihat prediksi performa siswa kamu! 🔍📈")

# Input data
st.subheader("📝 Input Data Siswa")

gender = st.selectbox("🧍‍♂️🧍‍♀️ Jenis Kelamin", [
    (5, "👧 Perempuan (5)"),
    (10, "👦 Laki-laki (10)")
], format_func=lambda x: x[1])[0]

race = st.selectbox("🌐 Etnis/Asal", [
    (0, "🅰️ Group A (0)"),
    (1, "🅱️ Group B (1)"),
    (2, "🆑 Group C (2)"),
    (3, "🔷 Group D (3)"),
    (4, "🆘Group E (4)")
], format_func=lambda x: x[1])[0]

edulevel = st.selectbox("🎓 Tingkat Pendidikan Orang Tua", [
    (0, "🏫 Some High School (0)"),
    (1, "🎒 High School (1)"),
    (2, "📘 Some College (2)"),
    (3, "👨‍🎓 Associate Degree (3)"),
    (4, "👩‍🎓 Bachelor (4)"),
    (5, "🎓 Master (5)")
], format_func=lambda x: x[1])[0]

lunch = st.selectbox("🍱 Tipe Makan Siang", [
    (0, "💲 Free/Reduced (0)"),
    (1, "🍽️ Standard (1)")
], format_func=lambda x: x[1])[0]

prep_course = st.selectbox("📚 Kursus Persiapan Ujian", [
    (0, "❌ Tidak Mengikuti (0)"),
    (1, "✅ Mengikuti (1)")
], format_func=lambda x: x[1])[0]

math_scores = sorted([59, 96, 57, 70, 83, 68, 82, 46, 80])
reading_scores = sorted([70, 93, 76, 85, 57, 83, 61, 75])
writing_scores = sorted([78, 87, 77, 63, 86, 54, 80, 58, 73])

math_score = st.selectbox("🧮 Nilai Matematika", math_scores, index=math_scores.index(70))
reading_score = st.selectbox("📖 Nilai Membaca", reading_scores, index=reading_scores.index(70))
writing_score = st.selectbox("✍️ Nilai Menulis", writing_scores, index=writing_scores.index(78))

# Tombol prediksi
st.markdown("---")
if st.button("🔮 Prediksi Sekarang"):
    input_data = np.array([[gender, race, edulevel, lunch, prep_course, math_score, reading_score, writing_score]])
    scaled_input = scaler.transform(input_data)

    scaled_input = scaled_input.astype(np.float32)
    model.set_tensor(input_details[0]['index'], scaled_input)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = le.inverse_transform(predicted_class)

    st.success(f"📊 Performa Belajar yang Diprediksi: **✨ `{predicted_label[0]}` ✨** 🎉")
