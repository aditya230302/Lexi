import cv2
import easyocr
import numpy as np
import streamlit as st
from textblob import TextBlob
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import tempfile
import os
from playsound import playsound
from PyPDF2 import PdfReader
import speech_recognition as sr
from PIL import Image
import io

# Initialize EasyOCR reader and Google Translator
reader = easyocr.Reader(['en'])  # Set the languages for OCR
translator = Translator()

languages = [
    "Afrikaans",
    "Akan",
    "Albanian",
    "Amharic",
    "Arabic",
    "Armenian",
    "Assamese",
    "Aymara",
    "Azerbaijani",
    "Bambara",
    "Bangla",
    "Basque",
    "Belarusian",
    "Bhojpuri",
    "Bosnian",
    "Bulgarian",
    "Burmese",
    "Catalan",
    "Cebuano",
    "Central Kurdish",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Corsican",
    "Croatian",
    "Czech",
    "Danish",
    "Divehi",
    "Dogri",
    "Dutch",
    "English",
    "Esperanto",
    "Estonian",
    "Ewe",
    "Filipino",
    "Finnish",
    "French",
    "Galician",
    "Ganda",
    "Georgian",
    "German",
    "Goan Konkani",
    "Greek",
    "Guarani",
    "Gujarati",
    "Haitian Creole",
    "Hausa",
    "Hawaiian",
    "Hebrew",
    "Hindi",
    "Hmong",
    "Hungarian",
    "Icelandic",
    "Igbo",
    "Iloko",
    "Indonesian",
    "Irish",
    "Italian",
    "Japanese",
    "Javanese",
    "Kannada",
    "Kazakh",
    "Khmer",
    "Kinyarwanda",
    "Korean",
    "Krio",
    "Kurdish",
    "Kyrgyz",
    "Lao",
    "Latin",
    "Latvian",
    "Lingala",
    "Lithuanian",
    "Luxembourgish",
    "Macedonian",
    "Maithili",
    "Malagasy",
    "Malay",
    "Malayalam",
    "Maltese",
    "Manipuri (Meitei Mayek)",
    "Māori",
    "Marathi",
    "Mizo",
    "Mongolian",
    "Nepali",
    "Northern Sotho",
    "Norwegian",
    "Nyanja",
    "Odia",
    "Oromo",
    "Pashto",
    "Persian",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Quechua",
    "Romanian",
    "Russian",
    "Samoan",
    "Sanskrit",
    "Scottish Gaelic",
    "Serbian",
    "Shona",
    "Sindhi",
    "Sinhala",
    "Slovak",
    "Slovenian",
    "Somali",
    "Southern Sotho",
    "Spanish",
    "Sundanese",
    "Swahili",
    "Swedish",
    "Tajik",
    "Tamil",
    "Tatar",
    "Telugu",
    "Thai",
    "Tigrinya",
    "Tsonga",
    "Turkish",
    "Turkmen",
    "Ukrainian",
    "Urdu",
    "Uyghur",
    "Uzbek",
    "Vietnamese",
    "Welsh",
    "Western Frisian",
    "Xhosa",
    "Yiddish",
    "Yoruba",
    "Zulu"
]

dic = {
    'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Azerbaijani': 'az', 'Basque': 'eu',
    'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Cebuano': 'ceb', 'Chichewa': 'ny',
    'Chinese (Simplified)': 'zh-cn', 'Chinese (Traditional)': 'zh-tw', 'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da',
    'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Filipino': 'tl', 'Finnish': 'fi', 'French': 'fr',
    'Frisian': 'fy', 'Galician': 'gl', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht',
    'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'he', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig',
    'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km',
    'Korean': 'ko', 'Kurdish (Kurmanji)': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv', 'Lithuanian': 'lt',
    'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi',
    'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar (Burmese)': 'my', 'Nepali': 'ne', 'Norwegian': 'no', 'Odia': 'or', 'Pashto': 'ps',
    'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm',
    'Scots Gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st', 'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala': 'si', 'Slovak': 'sk',
    'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tajik': 'tg', 'Tamil': 'ta',
    'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uyghur': 'ug', 'Uzbek': 'uz', 'Vietnamese': 'vi',
    'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'
}

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        st.write("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        return query
    except Exception as e:
        st.write("Could not understand, please try again.")
        return None

def translate_text(text, target_lang):
    if isinstance(text, str):  # If input is a single string
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    elif isinstance(text, list):  # If input is a list of strings
        translated_texts = []
        for sentence in text:
            translation = translator.translate(sentence, dest=target_lang)
            translated_texts.append(translation.text)
        return translated_texts
    else:
        raise ValueError("Invalid input type. Expected string or list of strings.")


def add_transparent_text(image, text, position, detected_lang, font_scale=0.8, thickness=2):
    # Create a copy of the image
    overlay = image.copy()

    # Get text size and position
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_width, text_height = text_size

    # Define background rectangle position
    background_position = (
        position[0],
        position[1] - text_height - 10
    )

    # Draw semi-transparent grey rectangle as background
    cv2.rectangle(
        overlay,
        background_position,
        (background_position[0] + text_width, background_position[1] + text_height),
        (128, 128, 128),
        -1
    )

    # Add text to the overlay image (100% white color)
    cv2.putText(
        overlay,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White color
        thickness
    )

    # Display detected language on top left corner
    cv2.putText(
        overlay,
        f"Translated Language: {detected_lang.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White color
        thickness
    )

    return overlay

def translate_pdf(pdf_reader, target_language):
    translator = Translator()
    translated_pages = []

    # Iterate over each page and translate the text
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()

        if text:
            # Translate the text to the target language
            translated = translator.translate(text, dest=target_language)
            translated_pages.append(translated.text)
        else:
            translated_pages.append("No text found on this page.")

    return translated_pages
def perform_ocr_on_image(image_bytes, target_lang):
    # Convert bytes to image
    img = Image.open(io.BytesIO(image_bytes))

    # Convert image to numpy array
    img_np = np.array(img)

    result = reader.readtext(img_np)
    ocr_result = ""
    for detection in result:
        text = detection[1]
        try:
            # Detect the language of the extracted text using TextBlob
            detected_lang = TextBlob(text).detect_language()
        except:
            detected_lang = 'en'  # Default to 'en' if language detection fails

        # Translate the text to the target language if it's different from the source language
        if detected_lang != target_lang:
            translated_text = translate_text(text, target_lang)
        else:
            translated_text = text

        ocr_result += translated_text + "\n"
    return ocr_result


def perform_ocr_on_frame(frame, target_lang):
    result = reader.readtext(frame)
    for detection in result:
        text = detection[1]
        bbox = detection[0]  # Bounding box of the detected text
        try:
            # Detect the language of the extracted text using TextBlob
            detected_lang = TextBlob(text).detect_language()
        except:
            detected_lang = 'en'  # Default to 'en' if language detection fails

        # Translate the text to the target language if it's different from the source language
        if detected_lang != target_lang:
            translated_text = translate_text(text, target_lang)
        else:
            translated_text = text

        # Display the translated text on the input image
        text_position = (int(bbox[0][0]), int(bbox[0][1]))  # Position of the detected text
        frame = add_transparent_text(frame, translated_text, text_position, detected_lang)

    return frame


def capture_and_process(target_lang='en'):
    st.title("Live OCR Translation")

    cap = cv2.VideoCapture(0)  # Open the default camera (0)

    while True:
        ret, frame = cap.read()  # Capture frame from the camera
        if not ret:
            break

        # Process the detected text
        processed_frame = perform_ocr_on_frame(frame, target_lang)

        # Display the processed frame using Streamlit
        st.image(processed_frame, channels="BGR", use_column_width=True)

        # Check for key press (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# Streamlit App
st.title("Language Nexus - LEXI")

# Functionality for Text-to-Text Translation
st.sidebar.subheader("Text-to-Text Translation")
source_text = st.sidebar.text_area("Enter text to translate:")
target_language = st.sidebar.selectbox("Choose target language for text translation:", languages,
                                       key="text_translation")
translate_button = st.sidebar.button('Translate Text', key="translate_text")

if translate_button:
    translated_text = translate_text(source_text, target_language)
    st.write("Translated Text:")
    st.write(translated_text)


# Functionality for Audio-to-Text Translation
st.sidebar.subheader("Audio-to-Text Translation")
uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
if uploaded_audio:
    selected_language_audio = st.sidebar.selectbox("Choose target language for audio translation:", languages,
                                                   key="audio_translation1")  # Unique key
    translate_audio_button = st.sidebar.button("Translate Audio", key="translate_audio1")  # Unique key

    if translate_audio_button:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_audio.read())
            temp_audio_path = temp_file.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                original_text = recognizer.recognize_google(audio_data)
                st.write(f"Original text: {original_text}")
                translated_text = translate_text(original_text, selected_language_audio)
                st.write(f"Translated text: {translated_text}")
            except sr.UnknownValueError:
                st.write("Could not understand the audio")
            except sr.RequestError:
                st.write("Could not request results; check your network connection")


# Functionality for Real-time Audio-to-Text Translation
st.sidebar.subheader("Real-time Audio-to-Text Translation")

# Language selection for audio translation
selected_language_audio = st.sidebar.selectbox("Choose target language for audio translation:", languages,
                                               key="audio_translation")

# Button to start real-time audio translation
start_realtime_audio_button = st.sidebar.button("Start Listening", key="start_realtime_audio_button")

# Initialize original_text variable
original_text = ""

if start_realtime_audio_button:
    st.write("Listening...")

    # Record audio
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
        audio_data = recognizer.listen(source)

    # Recognize speech
    try:
        original_text = recognizer.recognize_google(audio_data)
        st.write(f"Original text: {original_text}")
    except sr.UnknownValueError:
        st.write("Could not understand the audio")
    except sr.RequestError:
        st.write("Could not request results; check your network connection")

    # Translate text if original text is identified
    if original_text:
        translated_text = translate_text(original_text, selected_language_audio)
        st.write(f"Translated text: {translated_text}")



# Functionality for Document Translation
st.sidebar.subheader("Document Translation")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"], key="document_upload_pdf")  # Unique key
if uploaded_pdf:
    language_options = {
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh-cn",
    }
    selected_language_pdf = st.sidebar.selectbox("Select target language for document translation:",
                                                 language_options.keys(), key="document_translation_selectbox")  # Unique key

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_pdf.read())
        temp_file_path = temp_file.name

    pdf_reader = PdfReader(temp_file_path)
    translated_content = translate_pdf(pdf_reader, language_options[selected_language_pdf])

    st.write(f"Translated content into {selected_language_pdf}:")
    for page_num, content in enumerate(translated_content):
        st.markdown(f"### Page {page_num + 1}")
        st.text(content)


# Functionality for Live Translation
st.sidebar.subheader("Live Voice-to-Voice Translation")
lang_options = list(dic.keys())
chosen_lang = st.sidebar.selectbox("Choose target language for live translation:", lang_options,
                                   key="live_translation_selectbox")
if st.sidebar.button("Record and Translate", key="record_translate_button"):
    query = take_command()
    if query:
        st.write(f"You said: {query}")
        target_lang_code = dic[chosen_lang]
        translated_text = translate_text(query, target_lang_code)
        tts = gTTS(text=translated_text, lang=target_lang_code, slow=False)
        tts.save("translated_audio.mp3")
        playsound("translated_audio.mp3")
        os.remove("translated_audio.mp3")
        st.write(f"Translated text: {translated_text}")

# Functionality for OCR Translation
st.sidebar.subheader("OCR Translation & Live Video OCR")
supported_languages = list(LANGUAGES.values())  # Get the supported languages for OCR translation
selected_ocr_language = st.sidebar.selectbox("Select OCR Target Language", supported_languages, index=0)
start_ocr_button = st.sidebar.button("Start OCR & Live Video OCR", key="start_ocr_button")

if start_ocr_button:
    st.title("OCR & Live Video OCR")
    st.write("Press 'q' to stop the video.")

    cap = cv2.VideoCapture(0)

    # Create two columns for video feed and translated text
    col_video, col_text = st.columns(2)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Display the video feed
            col_video.image(frame, channels="BGR", use_column_width=True)

            # Process the detected text
            processed_frame = perform_ocr_on_frame(frame, selected_ocr_language)

            # Perform OCR on the processed frame
            detected_text = reader.readtext(processed_frame)

            # Display the translated text
            translated_text = translate_text(detected_text, selected_ocr_language)
            col_text.write(translated_text)

            if cv2.waitKeyEx(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# Functionality for Image OCR
st.sidebar.subheader("Image OCR")
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],
                                          key="image_uploader")  # Unique key
if uploaded_image:
    selected_ocr_language_image = st.sidebar.selectbox("Select OCR Target Language for Image", supported_languages,
                                                       index=0, key="ocr_language_selectbox")  # Unique key
    ocr_image_button = st.sidebar.button("Perform OCR on Image", key="perform_ocr_image_button")  # Unique key

    if ocr_image_button:
        image_bytes = uploaded_image.read()
        ocr_result_image = perform_ocr_on_image(image_bytes, selected_ocr_language_image)
        st.write("OCR Result:")
        st.write(ocr_result_image)
