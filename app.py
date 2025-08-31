import statistics
import av
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Cấu hình trang
st.set_page_config(
    page_title="Nhận dạng cảm xúc khuôn mặt",
    page_icon="😊",
    layout="wide"
)

st_autorefresh(interval=2000, key="emotion_refresh")

# Các lớp cảm xúc
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry': '#FF4444',
    'Disgust': '#8B4513',
    'Fear': '#800080',
    'Happy': '#32CD32',
    'Neutral': '#808080',
    'Sad': '#4169E1',
    'Surprise': '#FF8C00'
}


MODEL_CONFIG = {
    "CNN": {"input_size": (48,38), "to_rgb": False, "normalize": True},
    "VGG19": {"input_size": (48,48), "to_rgb": True, "normalize": True}
}

@st.cache_resource
def load_model_by_name(model_name):
    model_files = {
        "CNN": "fer_cnn.h5",
        "VGG19": "fer_vgg19.h5"
    }
    file_path = model_files.get(model_name)
    if not file_path:
        st.error(f"Model '{model_name}' không tồn tại trong danh sách.")
        return None
    try:
        model = tf.keras.models.load_model(file_path)
        st.success(f"Model '{model_name}' đã load thành công.")
        return model
    except Exception as e:
        st.error(f"Không thể load model '{model_name}': {e}")
        st.info("Vui lòng đảm bảo file model tồn tại và đường dẫn chính xác")
        return None


def preprocess_image(img, model_name):
    config = MODEL_CONFIG[model_name]

    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        if config["to_rgb"]:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3 and not config["to_rgb"]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, config["input_size"])

    if config["normalize"]:
        img = img.astype('float32') / 255.0

    if model_name == "CNN":
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = np.expand_dims(img, axis=0)

    return img


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


# def predict_emotion(model, face_image):
#     """Dự đoán cảm xúc từ ảnh khuôn mặt"""
#     processed_image = preprocess_image(face_image)
#     prediction = model.predict(processed_image, verbose=0)[0]
#     emotion_index = np.argmax(prediction)
#     confidence = np.max(prediction) * 100
#     return EMOTION_CLASSES[emotion_index], confidence

def predict_top_emotions(model, model_name,face_image, top_k=3):
    processed_image = preprocess_image(face_image, model_name)
    prediction = model.predict(processed_image, verbose=0)[0]
    top_indices = prediction.argsort()[-top_k:][::-1]
    top_emotions = [(EMOTION_CLASSES[i], float(prediction[i]) * 100) for i in top_indices]
    return top_emotions


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model_by_name()
        self.latencies = deque(maxlen=200)
        self.fps_values = deque(maxlen=200)
        self.emotions_log = []
        self.last_log_time = time.time()
        self.history = []
        self.device = "cpu"


    def recv(self, frame):
        start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame_emotions = []

        if self.model is not None:
            faces = detect_faces(img_rgb)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                face = img_rgb[y:y + h, x:x + w]

                if face.size > 0:
                    try:
                        top_emotions = predict_top_emotions(self.model, face, top_k=3)
                        frame_emotions.extend([emo for emo, _ in top_emotions])

                        for i, (emo, conf) in enumerate(top_emotions):
                            cv2.putText(img, f"{emo}: {conf:.1f}%",
                                        (x, y - 10 - i * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 1)

                    except Exception as e:
                        cv2.putText(img, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2)

        end_time = time.time()
        latency = end_time - start_time
        fps = 1.0 / latency if latency > 0 else 0

        self.latencies.append(latency)
        self.fps_values.append(fps)
        self.emotions_log.extend(frame_emotions)

        bench_text = f"Latency: {latency * 1000:.1f} ms | FPS: {fps:.1f}"
        cv2.putText(img, bench_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        now = time.time()
        if now - self.last_log_time >= 10.0:
            if self.latencies and self.fps_values:
                stats = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "latency_mean": statistics.mean(self.latencies),
                    "latency_median": statistics.median(self.latencies),
                    "latency_std": statistics.pstdev(self.latencies),
                    "fps_mean": statistics.mean(self.fps_values),
                    "fps_median": statistics.median(self.fps_values),
                    "fps_std": statistics.pstdev(self.fps_values),
                    "top_emotions": pd.Series(self.emotions_log).value_counts().head(3).to_dict()
                }

                self.history.append(stats)

                self.latencies.clear()
                self.fps_values.clear()
                self.emotions_log.clear()
                self.last_log_time = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_benchmark(self):
        if len(self.latencies) == 0:
            return None
        arr = np.array(self.latencies)
        fps_arr = np.array(self.fps_values)
        return {
            "Device": self.device.upper(),
            "Latency (ms)": {
                "mean": arr.mean() * 1000,
                "median": np.median(arr) * 1000,
                "std": arr.std() * 1000
            },
            "FPS": {
                "mean": fps_arr.mean(),
                "median": np.median(fps_arr),
                "std": fps_arr.std()
            }
        }

    def get_history_df(self):
        if self.history:
            return pd.DataFrame(self.history)
        else:
            return pd.DataFrame()


def main():
    st.title("🎭 Ứng dụng nhận dạng cảm xúc khuôn mặt")
    st.markdown("---")


    # Sidebar cho các tùy chọn
    st.sidebar.title("Tùy chọn")
    option = st.sidebar.selectbox(
        "Chọn phương thức nhận dạng:",
        ["📤 Upload ảnh", "📷 Sử dụng Camera"]
    )

    if option == "📤 Upload ảnh":
        st.header("Upload ảnh để nhận dạng cảm xúc")

        model_option = st.selectbox(
            "Chọn mô hình để nhận dạng:",
            ["CNN", "VGG19"]
        )

        model = load_model_by_name(model_option)

        uploaded_file = st.file_uploader(
            "Chọn ảnh...",
            type=['png', 'jpg', 'jpeg'],
            help="Hỗ trợ định dạng: PNG, JPG, JPEG"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ảnh gốc")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Kết quả nhận dạng")
                faces = detect_faces(image_array)
                if len(faces) > 0:
                    result_image = image_array.copy()
                    emotions_detected = []
                    for i, (x, y, w, h) in enumerate(faces):
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        face = image_array[y:y + h, x:x + w]
                        top_emotions = predict_top_emotions(model, model_option,face, top_k=3)
                        emotions_detected.append(top_emotions)  # lưu cả top 3

                        best_emotion, best_conf = top_emotions[0]
                        cv2.putText(result_image, f"{best_emotion}: {best_conf:.1f}%",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

                    st.image(result_image, use_container_width=True)

                    st.subheader("Chi tiết kết quả:")
                    for i, top_emotions in enumerate(emotions_detected):
                        st.markdown(f"<strong>Khuôn mặt {i + 1}:</strong>", unsafe_allow_html=True)
                        for emo, conf in top_emotions:
                            color = EMOTION_COLORS.get(emo, '#000000')
                            st.markdown(f"""
                            <div style='padding: 5px; border-left: 4px solid {color}; margin: 3px 0;'>
                                {emo}: {conf:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("Không phát hiện được khuôn mặt trong ảnh!")
                    st.info("💡 Mẹo: Hãy thử với ảnh có khuôn mặt rõ ràng và không bị che khuất")

    elif option == "📷 Sử dụng Camera":
        st.header("Sử dụng Camera để nhận dạng cảm xúc real-time")

        st.info("📌 Cho phép truy cập camera khi được yêu cầu")
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=VideoTransformer,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        stats_placeholder = st.empty()
        table_placeholder = st.empty()

        if webrtc_ctx.video_processor:
            df = webrtc_ctx.video_processor.get_history_df()
            bench = webrtc_ctx.video_processor.get_benchmark()

            if not df.empty:
                st.subheader("📊 Thống kê Benchmark & Cảm xúc")
                st.dataframe(df)

            if bench:
                st.markdown("### ⚡ Benchmark (Realtime)")
                st.write(f"**Device**: {bench['Device']}")
                st.write(f"**Latency (ms)** → Mean: {bench['Latency (ms)']['mean']:.1f}, "
                         f"Median: {bench['Latency (ms)']['median']:.1f}, "
                         f"Std: {bench['Latency (ms)']['std']:.1f}")
                st.write(f"**FPS** → Mean: {bench['FPS']['mean']:.1f}, "
                         f"Median: {bench['FPS']['median']:.1f}, "
                         f"Std: {bench['FPS']['std']:.1f}")

    with st.expander("ℹ️ Thông tin về các loại cảm xúc"):
        cols = st.columns(len(EMOTION_CLASSES))
        for i, emotion in enumerate(EMOTION_CLASSES):
            with cols[i]:
                color = EMOTION_COLORS[emotion]
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; border: 2px solid {color}; border-radius: 10px;'>
                    <strong style='color: {color};'>{emotion}</strong>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

if __name__ == "__main__":
    main()