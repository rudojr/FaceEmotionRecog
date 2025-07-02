import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Cấu hình trang
st.set_page_config(
    page_title="Nhận dạng cảm xúc khuôn mặt",
    page_icon="😊",
    layout="wide"
)

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


# Cache model để tránh load lại nhiều lần
@st.cache_resource
def load_model():
    """Load model đã train sẵn"""
    try:
        model = tf.keras.models.load_model('fer_cnn.h5')
        return model
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        st.info("Vui lòng đảm bảo file model tồn tại và đường dẫn chính xác")
        return None


def preprocess_image(image):
    """Tiền xử lý ảnh cho model"""
    # Chuyển sang grayscale nếu cần
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize về kích thước model expect (thường là 48x48 cho emotion recognition)
    image = cv2.resize(image, (48, 48))

    # Normalize pixel values
    image = image.astype('float32') / 255.0

    # Expand dimensions để phù hợp với input model
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    return image


def detect_faces(image):
    """Phát hiện khuôn mặt trong ảnh"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def predict_emotion(model, face_image):
    """Dự đoán cảm xúc từ ảnh khuôn mặt"""
    processed_image = preprocess_image(face_image)
    prediction = model.predict(processed_image, verbose=0)
    emotion_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return EMOTION_CLASSES[emotion_index], confidence


class VideoTransformer(VideoTransformerBase):
    """Class xử lý video stream từ camera"""

    def __init__(self):
        self.model = load_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.model is not None:
            faces = detect_faces(img_rgb)

            for (x, y, w, h) in faces:
                # Vẽ khung quanh khuôn mặt
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Cắt vùng khuôn mặt
                face = img_rgb[y:y + h, x:x + w]

                if face.size > 0:
                    try:
                        # Dự đoán cảm xúc
                        emotion, confidence = predict_emotion(self.model, face)

                        # Hiển thị kết quả
                        text = f"{emotion}: {confidence:.1f}%"
                        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                    except Exception as e:
                        cv2.putText(img, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2)

        return img


def main():
    st.title("🎭 Ứng dụng nhận dạng cảm xúc khuôn mặt")
    st.markdown("---")

    # Load model
    model = load_model()

    if model is None:
        st.stop()

    # Sidebar cho các tùy chọn
    st.sidebar.title("Tùy chọn")
    option = st.sidebar.selectbox(
        "Chọn phương thức nhận dạng:",
        ["📤 Upload ảnh", "📷 Sử dụng Camera"]
    )

    if option == "📤 Upload ảnh":
        st.header("Upload ảnh để nhận dạng cảm xúc")

        uploaded_file = st.file_uploader(
            "Chọn ảnh...",
            type=['png', 'jpg', 'jpeg'],
            help="Hỗ trợ định dạng: PNG, JPG, JPEG"
        )

        if uploaded_file is not None:
            # Hiển thị ảnh đã upload
            image = Image.open(uploaded_file)
            image_array = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ảnh gốc")
                st.image(image, use_column_width=True)

            with col2:
                st.subheader("Kết quả nhận dạng")

                # Phát hiện khuôn mặt
                faces = detect_faces(image_array)

                if len(faces) > 0:
                    # Tạo ảnh kết quả
                    result_image = image_array.copy()
                    emotions_detected = []

                    for i, (x, y, w, h) in enumerate(faces):
                        # Vẽ khung quanh khuôn mặt
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Cắt vùng khuôn mặt
                        face = image_array[y:y + h, x:x + w]

                        # Dự đoán cảm xúc
                        emotion, confidence = predict_emotion(model, face)
                        emotions_detected.append((emotion, confidence))

                        # Thêm text lên ảnh
                        cv2.putText(result_image, f"{emotion}: {confidence:.1f}%",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Hiển thị ảnh kết quả
                    st.image(result_image, use_column_width=True)

                    # Hiển thị chi tiết kết quả
                    st.subheader("Chi tiết kết quả:")
                    for i, (emotion, confidence) in enumerate(emotions_detected):
                        color = EMOTION_COLORS.get(emotion, '#000000')
                        st.markdown(f"""
                        <div style='padding: 10px; border-left: 4px solid {color}; margin: 5px 0;'>
                            <strong>Khuôn mặt {i + 1}:</strong> {emotion} ({confidence:.1f}% tin cậy)
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Không phát hiện được khuôn mặt trong ảnh!")
                    st.info("💡 Mẹo: Hãy thử với ảnh có khuôn mặt rõ ràng và không bị che khuất")

    elif option == "📷 Sử dụng Camera":
        st.header("Sử dụng Camera để nhận dạng cảm xúc real-time")

        st.info("📌 Cho phép truy cập camera khi được yêu cầu")

        # Cấu hình WebRTC
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("### 📊 Thống kê cảm xúc")
        if webrtc_ctx.video_transformer:
            st.info("Camera đang hoạt động. Hướng khuôn mặt vào camera để nhận dạng cảm xúc!")
        else:
            st.warning("Camera chưa được kích hoạt")

    # Thông tin về các loại cảm xúc
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

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🤖 Được phát triển bằng Streamlit và TensorFlow"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()