import pyttsx3
import speech_recognition as sr
import webbrowser
from datetime import datetime
from googletrans import Translator
import pyautogui
import cv2
import os
import mediapipe as mp
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_disable_xla_backend"


# Initialize the speech recognition engine
r = sr.Recognizer()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        print("Recognizing...")
        text = r.recognize_google(audio)
        print("You said:", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
    except sr.RequestError:
        print("Sorry, there was an issue with the speech recognition service.")
    return ""

def translate_text(text, target_lang):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text

# Path to the pre-trained face detection model
face_cascade_path = 'haarcascade_frontalface_default.xml'

# Load the face detection model
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to draw bounding boxes around the detected faces
def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to process the captured video frames
def process_video_frames():
    video_capture = cv2.VideoCapture(0)  # Change the index to use a different camera (e.g., 1, 2, etc.)

    while True:
        ret, frame = video_capture.read()

        # Detect faces in the frame
        faces = detect_faces(frame)

        # Draw bounding boxes around the faces
        draw_faces(frame, faces)

        # Display the frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to process the images in a directory
def process_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)

            # Detect faces in the image
            faces = detect_faces(image)

            # Draw bounding boxes around the faces
            draw_faces(image, faces)

            # Display the image
            cv2.imshow('Image', image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    






def detect_faces():
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read each frame
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the OpenCV window
    video_capture.release()
    cv2.destroyAllWindows()

def detect_fingers(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    return image
def detect_fingers(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get the coordinates of the index finger
            index_finger_coords = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            height, width, _ = image.shape
            finger_x = int(index_finger_coords.x * width)
            finger_y = int(index_finger_coords.y * height)

            # Move the mouse cursor to the finger coordinates
            pyautogui.moveTo(finger_x, finger_y)

    return image
while True:
    user_input = listen()

    if "hello" in user_input:
        speak("Hello, How can I assist you")
    elif "goodbye" in user_input:
        speak("Goodbye!")
        break
    elif "what is your name" in user_input:
        speak("My name is Bond..... James Bond.")
        break
    elif "who are you" in user_input:
        speak("DON.")
        break   
    elif "who is your master" in user_input:
        speak("Kanishk... The Great.")
        break    
    elif "thanks" in user_input:
        speak("You're welcome! Is there anything else I can help in?")
        break    
    elif "shut up" in user_input:
        speak("Don't angry me.")
        break    
    elif "how are you" in user_input:
        speak("I am fine. How are you?")
        break    
    elif "hey bro" in user_input or "hi" in user_input:
        speak("Hey bro! How can I assist you?")
        break    
    elif "open youtube" in user_input:
        speak("Sure, opening YouTube.")
        webbrowser.open("https://www.youtube.com/")
        break    
    elif "open chat" in user_input:
        speak("Sure, opening the chat.")
        webbrowser.open("https://chat.openai.com/")
        break    
    elif "what is the time" in user_input:
        current_time = datetime.now().strftime("%I:%M %p")  
        speak(f"The current time is {current_time}.")
        break    
    elif "what is today's date" in user_input:
        current_date = datetime.now().strftime("%A, %B %d, %Y")  
        speak(f"Today's date is {current_date}.")
        break
    
    elif "translate" in user_input:
       speak("What would you like me to translate?")
       text_to_translate = listen()
       if text_to_translate:
           speak("Which language should I translate it to?")
           target_language = listen()
           if target_language:
              translation = translate_text(text_to_translate, target_language)
              speak(f"The translation is: {translation}")
           else:
              speak("I'm sorry, I didn't catch the target language.")
       else:
              speak("I'm sorry, I didn't catch the text to translate.")
      

    elif "recognise faces" in user_input:
        speak("Sure, please choose an option: video or directory.")
        option = listen()

        if "video" in option:
            speak("Opening video feed for face recognition.")
            process_video_frames()
            break
        elif "directory" in option:
            speak("Please provide the directory path.")
            directory_path = listen()
            if directory_path:
                speak("Processing images in the provided directory.")
                process_images_in_directory(directory_path)
                break
            else:
                speak("I'm sorry, I didn't catch the directory path.")
        else:
            speak("I'm sorry, I didn't catch the option.")
        
        break
    elif "detect faces" in user_input:
        speak("Sure, starting face detection.")
        detect_faces()
        break
    elif "detect hand" in user_input:
        speak("Sure, initiating hand detection.")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = detect_fingers(frame)

            cv2.imshow("Hand Detection", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        break
    else:
        speak("Sorry, I can't help with that.")




