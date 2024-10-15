import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import mysql.connector
from datetime import datetime, timedelta
import json
import bcrypt
import pandas as pd
import plotly.express as px
import logging
import cv2
import numpy as np
import time
from PIL import Image
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import numpy as np
import time
import random
import smtplib
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from textblob import TextBlob
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from moviepy.editor import VideoFileClip
from speech_recognition import Recognizer, AudioFile
import streamlit as st
import cv2
import tempfile
import os
import pyaudio
import wave
import threading
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import requests
from github import Github
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import streamlit as st
import streamlit_extras as stxa
import extra_streamlit_components as stx
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import json
from datetime import datetime, timedelta
import json
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from icalendar import Calendar, Event
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain import PromptTemplate, LLMChain
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import logging
import string
import cv2
from deepface import DeepFace
import numpy as np
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from pydub import AudioSegment
import io
import os
import tempfile
import logging
from typing import Optional, Dict, Any
import functools
from mysql.connector import IntegrityError
import base64
import pikepdf

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyAjxGxSRwazR6jQJrctjeTgMwbiWhkFT7Y"
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Chaithu@9515",
    database="glory"
)
cursor = db.cursor(dictionary=True)


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

def generate_speechx_questions(job_id: int) -> Optional[int]:
    logging.info(f"Generating SpeechX questions for job ID: {job_id}")
    
    # Fetch job details
    cursor.execute("SELECT title, department FROM job_postings WHERE id = %s", (job_id,))
    job_details = cursor.fetchone()
    
    if not job_details:
        st.error(f"No job found with ID: {job_id}")
        return None
    
    job_title, department = job_details
    
    prompt = PromptTemplate(
        input_variables=["job_title", "department"],
        template="""
        Generate a SpeechX assessment for the position of {job_title} in the {department} department. Include:
        1. 3 sentences for listening and repeating
        2. 3 sentences for reading aloud
        3. 3 topics for discussion

        Each item should be related to the job role or industry.
        Format the output as a JSON string with the following structure:
        [
            {{"type": "listen_repeat", "content": "Sentence 1"}},
            {{"type": "listen_repeat", "content": "Sentence 2"}},
            {{"type": "listen_repeat", "content": "Sentence 3"}},
            {{"type": "read_aloud", "content": "Sentence 4"}},
            {{"type": "read_aloud", "content": "Sentence 5"}},
            {{"type": "read_aloud", "content": "Sentence 6"}},
            {{"type": "topic_discussion", "content": "Topic 1"}},
            {{"type": "topic_discussion", "content": "Topic 2"}},
            {{"type": "topic_discussion", "content": "Topic 3"}}
        ]
        Ensure that the output is a valid JSON string.
        """
    )
    
    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(job_title=job_title, department=department)
        
        # Clean up the response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        
        st.write("Raw response from language model:", response)  # Debug output
        
        questions = json.loads(response)
        
        if not isinstance(questions, list) or len(questions) != 9:
            raise ValueError("Generated questions do not match the expected format")
        
        for q in questions:
            if not isinstance(q, dict) or 'type' not in q or 'content' not in q:
                raise ValueError("Question format is incorrect")
            if q['type'] not in ['listen_repeat', 'read_aloud', 'topic_discussion']:
                raise ValueError(f"Invalid question type: {q['type']}")
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse generated questions as JSON. Error: {str(e)}")
        st.error(f"Raw response: {response}")
        return None
    except ValueError as e:
        st.error(f"Generated questions are not in the correct format. Error: {str(e)}")
        st.error(f"Parsed questions: {questions}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while generating questions: {str(e)}")
        logging.exception("Unexpected error in generate_speechx_questions")
        return None
    
    try:
        cursor.execute("INSERT INTO speechx_assessments (job_id, created_by) VALUES (%s, %s)", (job_id, st.session_state.user['id']))
        assessment_id = cursor.lastrowid
        
        for question in questions:
            cursor.execute("INSERT INTO speechx_questions (assessment_id, question_type, content) VALUES (%s, %s, %s)",
                           (assessment_id, question['type'], question['content']))
        
        db.commit()
        st.success(f"Successfully created SpeechX assessment with ID: {assessment_id}")
        return assessment_id
    except Exception as e:
        db.rollback()
        st.error(f"Database error while creating SpeechX assessment: {str(e)}")
        logging.exception("Database error in generate_speechx_questions")
        return None
    

def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcript = "Speech not recognized"
        except sr.RequestError:
            transcript = "Could not request results from the speech recognition service"

    os.unlink(temp_audio_path)
    return transcript

def analyze_response(question, transcript):
    prompt = PromptTemplate(
        input_variables=["question", "transcript"],
        template="""
        Analyze the following response to the question/prompt: '{question}'.
        Response: '{transcript}'
        Provide a score out of 10 and detailed feedback.
        Format the output as a JSON string with the following structure:
        {{"score": 0, "feedback": "Detailed feedback here"}}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=question, transcript=transcript)
    
    try:
        analysis = json.loads(response)
        return analysis['score'], analysis['feedback']
    except json.JSONDecodeError:
        st.error("Failed to analyze response. Please try again.")
        return 0, "Analysis failed"

def save_audio_file1(frames, filename):
    try:
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"Audio file saved successfully: {filename}")  # Debug statement
    except Exception as e:
        print(f"Error saving audio file: {str(e)}")  # Debug statement
        raise

CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 44100

# Debugging helper function
def print_session_state():
    logging.debug("Current session state:")
    for key, value in st.session_state.items():
        if key.startswith('speechx_responses'):
            logging.debug(f"{key}:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key == 'audio_bytes' and sub_value:
                        logging.debug(f"  {sub_key}: {type(sub_value).__name__}, {len(sub_value)} bytes")
                    else:
                        logging.debug(f"  {sub_key}: {type(sub_value).__name__}")
            else:
                logging.debug(f"  Unexpected type: {type(value).__name__}")
        else:
            logging.debug(f"{key}: {type(value).__name__}")

def save_audio1(audio_bytes, application_id, question_id):
    audio_dir = "speechx_audio"
    os.makedirs(audio_dir, exist_ok=True)
    audio_filename = f"speechx_{application_id}_{question_id}.wav"
    audio_path = os.path.join(audio_dir, audio_filename)
    
    logging.debug(f"Attempting to save audio to: {audio_path}")
    
    try:
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_bytes)
        logging.debug(f"Audio successfully saved to: {audio_path}")
        return audio_path
    except Exception as e:
        error_msg = f"Error saving audio: {str(e)}"
        logging.exception(error_msg)
        raise


def speechx_assessment(application_id):
    st.subheader("SpeechX Assessment")
    logging.debug(f"Starting SpeechX assessment for application ID: {application_id}")
    
    # Fetch application details
    application = execute_query("SELECT job_id, speechx_status FROM applications WHERE id = %s", (application_id,))
    if not application:
        st.error(f"No application found with ID: {application_id}")
        return
    
    application = application[0]
    job_id = application['job_id']
    speechx_status = application['speechx_status']
    
    if speechx_status == 'completed':
        st.info("You have already completed the SpeechX Assessment.")
        return
    
    # Fetch or generate assessment
    result = execute_query("SELECT id FROM speechx_assessments WHERE job_id = %s", (job_id,))
    if result:
        assessment_id = result[0]['id']
    else:
        assessment_id = generate_speechx_questions(job_id)
    
    if not assessment_id:
        st.error("Failed to retrieve or generate SpeechX assessment.")
        return
    
    # Fetch questions
    questions = execute_query("SELECT * FROM speechx_questions WHERE assessment_id = %s ORDER BY question_type, id", (assessment_id,))
    
    # Initialize session state for responses if not already present
    if 'speechx_responses' not in st.session_state:
        st.session_state.speechx_responses = {}
    
    # Display progress
    total_questions = len(questions)
    answered_questions = len([r for r in st.session_state.speechx_responses.values() if r['audio_bytes']])
    st.progress(answered_questions / total_questions)
    st.write(f"Progress: {answered_questions}/{total_questions} questions answered")
    
    # Question loop
    for question in questions:
        st.write(f"### {question['question_type'].replace('_', ' ').title()}")
        st.write(question['content'])
        
        question_key = f"question_{question['id']}"
        if question_key not in st.session_state.speechx_responses:
            st.session_state.speechx_responses[question_key] = {'audio_bytes': None, 'transcript': None}
        
        audio_bytes = record_audio(question['id'])
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            transcript = transcribe_audio(audio_bytes)
            st.write(f"Transcript: {transcript}")
            
            st.session_state.speechx_responses[question_key] = {
                'audio_bytes': audio_bytes,
                'transcript': transcript
            }
        elif st.session_state.speechx_responses[question_key]['audio_bytes']:
            st.audio(st.session_state.speechx_responses[question_key]['audio_bytes'], format="audio/wav")
            st.write(f"Transcript: {st.session_state.speechx_responses[question_key]['transcript']}")
        else:
            st.warning("No audio recorded for this question.")
    
    print_session_state() 
    
    # Submit button
    # Submit button
    if st.button("Submit SpeechX Assessment"):
        logging.debug("Submit button clicked")
        if len(st.session_state.speechx_responses) != total_questions or \
        any(not r.get('audio_bytes') for r in st.session_state.speechx_responses.values()):
            st.error("Please answer all questions before submitting.")
            logging.error("Not all questions answered")
            return
        
        # Confirmation dialog
        # if not st.checkbox("I confirm that I want to submit my SpeechX Assessment"):
        #     st.warning("Please confirm your submission.")
        #     logging.warning("Submission not confirmed")
        #     return
        
        # Store all responses in a single transaction
        try:
            with db.cursor() as cursor:
                for question_key, response in st.session_state.speechx_responses.items():
                    question_id = int(question_key.split('_')[1])
                    logging.debug(f"Processing question ID: {question_id}")
                    
                    if not response['audio_bytes']:
                        logging.error(f"No audio bytes for question ID: {question_id}")
                        continue
                    
                    audio_url = save_audio1(response['audio_bytes'], application_id, question_id)
                    logging.debug(f"Audio saved at: {audio_url}")
                    
                    cursor.execute(
                        "INSERT INTO speechx_responses (application_id, question_id, audio_url, transcript) VALUES (%s, %s, %s, %s)",
                        (application_id, question_id, audio_url, response['transcript'])
                    )
                    logging.debug(f"Inserted response for question ID: {question_id}")
                
                cursor.execute("UPDATE applications SET speechx_status = 'completed' WHERE id = %s", (application_id,))
                logging.debug(f"Updated application status for ID: {application_id}")
            
            db.commit()
            logging.info("Transaction committed successfully")
            st.success("SpeechX Assessment completed! Your responses have been recorded.")
            # Clear session state after successful submission
            st.session_state.speechx_responses = {}
        except mysql.connector.Error as e:
            db.rollback()
            error_msg = f"Error storing responses: {str(e)}"
            st.error(error_msg)
            logging.exception(error_msg)
# Helper functions (unchanged)
def record_audio(question_id):
    audio_bytes = audio_recorder(key=f"audio_recorder_{question_id}")
    if audio_bytes:
        return audio_bytes
    return None

def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcript = "Speech not recognized"
        except sr.RequestError:
            transcript = "Could not request results from the speech recognition service"

    os.unlink(temp_audio_path)
    return transcript

def execute_query(query, params=None, fetch=True):
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        if fetch:
            result = cursor.fetchall()
            return result
        else:
            db.commit()
    except mysql.connector.Error as e:
        db.rollback()
        error_message = f"Database error: {e.errno} - {e.msg}"
        logging.error(error_message)
        st.error(error_message)
        return None
    finally:
        cursor.close()

def handle_database_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except mysql.connector.Error as e:
            error_message = f"Database error in {func.__name__}: {e.errno} - {e.msg}"
            logging.error(error_message)
            st.error(error_message)
            return None
    return wrapper

def analyze_speechx_responses(job_id):
    st.subheader("Analyze SpeechX Responses")
    
    # Fetch all responses for the job, including analyzed ones
    all_responses = execute_query("""
        SELECT sr.*, sq.content as question, sq.question_type, a.applicant_name, a.id as application_id,
               CASE WHEN sr.score IS NULL THEN 'Unanalyzed' ELSE 'Analyzed' END as analysis_status
        FROM speechx_responses sr
        JOIN speechx_questions sq ON sr.question_id = sq.id
        JOIN applications a ON sr.application_id = a.id
        WHERE a.job_id = %s
        ORDER BY a.applicant_name, sq.question_type
    """, (job_id,))
    
    if not all_responses:
        st.info("No SpeechX responses found for this job.")
        return
    
    # Group responses by applicant
    applicants = {}
    for response in all_responses:
        if response['applicant_name'] not in applicants:
            applicants[response['applicant_name']] = []
        applicants[response['applicant_name']].append(response)
    
    # Display responses grouped by applicant
    for applicant, responses in applicants.items():
        with st.expander(f"Responses for {applicant}"):
            unanalyzed_count = sum(1 for r in responses if r['analysis_status'] == 'Unanalyzed')
            st.write(f"Total responses: {len(responses)}, Unanalyzed: {unanalyzed_count}")
            
            for response in responses:
                st.write(f"### {response['question_type'].replace('_', ' ').title()}")
                st.write(f"Question: {response['question']}")
                st.write(f"Transcript: {response['transcript']}")
                st.audio(response['audio_url'], format="audio/wav")
                
                if response['analysis_status'] == 'Unanalyzed':
                    if st.button(f"Analyze Response (ID: {response['id']})"):
                        score, feedback = analyze_response(response['question'], response['transcript'])
                        execute_query("UPDATE speechx_responses SET score = %s, feedback = %s WHERE id = %s",
                                      (score, feedback, response['id']), fetch=False)
                        st.write(f"Score: {score}/10")
                        st.write(f"Feedback: {feedback}")
                        st.success("Response analyzed!")
                        st.rerun()  # Refresh the page to update the status
                else:
                    st.write(f"Score: {response['score']}/10")
                    st.write(f"Feedback: {response['feedback']}")
    
    if all(r['analysis_status'] == 'Analyzed' for r in all_responses):
        st.success("All responses for this job have been analyzed.")
    else:
        st.warning("There are still unanalyzed responses for this job.")

class VideoProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frames.append(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        self.frames = []  
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK)
        self.is_recording = True
        threading.Thread(target=self._record).start()

    def _record(self):
        while self.is_recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def reset(self):
        self.frames = []
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio = pyaudio.PyAudio() 

def initialize_session_state():
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'frames' not in st.session_state:
        st.session_state.frames = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'video_interview_id' not in st.session_state:
        st.session_state.video_interview_id = None
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None


initialize_session_state()

def toggle_recording():
    st.session_state.recording = not st.session_state.recording
    if st.session_state.recording:
        st.session_state.frames = []
        st.session_state.start_time = time.time()
        st.session_state.audio_recorder.start_recording()
    else:
        save_video()
        save_audio()

def save_video():
    if len(st.session_state.frames) > 0:
        video_dir = "video_interviews"
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"interview_{st.session_state.video_interview_id}.mp4")
        
        height, width, _ = st.session_state.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in st.session_state.frames:
            out.write(frame)
        out.release()
        
        cursor.execute("UPDATE video_interviews SET video_url = %s, status = 'completed' WHERE id = %s",
                       (video_path, st.session_state.video_interview_id))
        db.commit()
        
        st.success("Video interview completed and saved successfully!")
    else:
        st.error("No video data captured. Please try again.")

def save_audio_file(frames, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def save_audio():
    if st.session_state.audio_recorder.frames:
        audio_dir = "audio_interviews"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"interview_{st.session_state.video_interview_id}.wav")
        save_audio_file(st.session_state.audio_recorder.frames, audio_path)
        st.session_state.audio_file = audio_path
        cursor.execute("UPDATE video_interviews SET audio_url = %s, status = 'completed' WHERE id = %s",
                        (audio_path, st.session_state.video_interview_id))
        db.commit()
        st.success("Audio saved successfully!")
    else:
        st.warning("No audio data captured. Please try recording again.")

def record_video_interview(video_interview_id):
    initialize_session_state()
    st.session_state.video_interview_id = video_interview_id

    st.subheader("Video Interview")
    
    try:
        cursor.execute("SELECT question FROM video_interviews WHERE id = %s", (video_interview_id,))
        question = cursor.fetchone()['question']
        
        st.write(f"Question: {question}")
        st.write("You have 2 minutes to record your response.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start/Stop Recording", on_click=toggle_recording):
                pass  # The actual toggling is done in the callback

        with col2:
            if st.button("Reset Interview"):
                st.session_state.recording = False
                st.session_state.frames = []
                st.session_state.start_time = None
                st.session_state.audio_file = None
                st.session_state.audio_recorder.reset()  # Reset the audio recorder
                st.rerun()

        status_placeholder = st.empty()
        video_placeholder = st.empty()

        if st.session_state.recording:
            status_placeholder.write("Recording in progress...")
            camera = cv2.VideoCapture(0)
            
            while time.time() - st.session_state.start_time < 120 and st.session_state.recording:  # 2 minutes
                ret, frame = camera.read()
                if ret:
                    st.session_state.frames.append(frame)
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                remaining_time = int(120 - (time.time() - st.session_state.start_time))
                status_placeholder.write(f"Time remaining: {remaining_time} seconds")
                
               
                time.sleep(0.1)
            
            camera.release()
            
            if not st.session_state.recording: 
                save_video()
                save_audio()
            
            st.session_state.recording = False
        
        if st.session_state.audio_file:
            st.audio(st.session_state.audio_file)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Video interview error: {str(e)}")

def create_video_interview_request(application_id, question):
    cursor.execute("""
    INSERT INTO video_interviews (application_id, question)
    VALUES (%s, %s)
    """, (application_id, question))
    db.commit()
    video_interview_id = cursor.lastrowid

   
    cursor.execute("SELECT email FROM applications WHERE id = %s", (application_id,))
    candidate_email = cursor.fetchone()['email']

 
    subject = "Video Interview Request"
    body = f"""
    You have been invited to complete a video interview for your job application.
    Please log in to the recruitment platform to record your response to the following question:

    {question}

    You will have 2 minutes to record your response.
    """
    send_notification(candidate_email, subject, body)

    return video_interview_id

def clear_unread_results():
    while cursor.nextset():
        pass

logging.basicConfig(level=logging.ERROR)

def generate_job_description(title, department, required_qualifications, preferred_qualifications, responsibilities, experience):
    prompt = PromptTemplate(
        input_variables=["title", "department", "required_qualifications", "preferred_qualifications", "responsibilities", "experience"],
        template="""
        Create a detailed job description for the position of {title} in the {department} department and required qualifications for the postion is {required_qualifications}, and preferred qualifications are {preferred_qualifications} and skills required are {responsibilities}. Include
        1. Brief overview of the role
        2. Key responsibilities
        3. Required qualifications
        4. Preferred qualifications
        5. Required experience: {experience}
        6. Skills required
        7. Benefits and perks

        Required qualifications: {required_qualifications}
        Skills: {responsibilities}

        Format the output as a structured job description.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    job_description = chain.run(
        title=title,
        department=department,
        required_qualifications=required_qualifications,
        preferred_qualifications=preferred_qualifications,
        responsibilities=responsibilities,
        experience=experience
    )
    return job_description


def parse_resume(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_and_send_otp(email):
    otp = str(random.randint(100000, 999999))
    expiration_time = time.time() + 300 
    sender_email = "vtu19978.soc.cse@gmail.com"  
    sender_password = "Chaithu@9415" 

    message = MIMEMultipart("alternative")
    message["Subject"] = "Password Reset OTP"
    message["From"] = sender_email
    message["To"] = email

    text = f"Your OTP for password reset is: {otp}. This OTP is valid for 5 minutes."
    html = f"""\
    <html>
      <body>
        <p>Your OTP for password reset is: <strong>{otp}</strong></p>
        <p>This OTP is valid for 5 minutes.</p>
      </body>
    </html>
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    message.attach(part1)
    message.attach(part2)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        return otp, expiration_time
    except Exception as e:
        print(f"Error sending email: {e}")
        return None, None


def evaluate_resume(resume_text, job_description, required_qualifications, preferred_qualifications):
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    extract_prompt = PromptTemplate(
        input_variables=["resume_text"],
        template=(
            "Extract the following key information from the resume. If a piece of information is not present, write 'Not found'.\n"
            "1. Name\n2. Contact Information\n3. Education (list all degrees)\n4. Work Experience (list job titles and companies)\n"
            "5. Skills\n6. Certifications\n7. Projects\n8. Achievements\n\nResume:\n{resume_text}\n\n"
            "Provide the extracted information in a structured format."
        )
    )
    extract_chain = LLMChain(llm=model, prompt=extract_prompt)
    extracted_info = extract_chain.run(resume_text=resume_text)
    
    analysis_prompt = PromptTemplate(
        input_variables=["extracted_info", "job_description", "required_qualifications", "preferred_qualifications"],
        template=(
            "You are an expert resume evaluator. Analyze the following extracted resume information against the job requirements.\n\n"
            "Extracted Resume Information:\n{extracted_info}\n\n"
            "Job Description:\n{job_description}\n\n"
            "Required Qualifications:\n{required_qualifications}\n\n"
            "Preferred Qualifications:\n{preferred_qualifications}\n\n"
            "Provide a detailed analysis covering:\n"
            "1. Match with Job Description (score out of 100 and explanation)\n"
            "2. Required Qualifications Met (list each, whether met, and explanation)\n"
            "3. Preferred Qualifications Met (list each, whether met, and explanation)\n"
            "4. Key Strengths relevant to the position\n"
            "5. Areas for Improvement or Missing Qualifications\n"
            "6. Relevant Projects or Achievements\n"
            "7. Overall Fit (score out of 100 and explanation)\n\n"
            "Ensure your analysis is thorough, impartial, and based solely on the provided information."
        )
    )
    analysis_chain = LLMChain(llm=model, prompt=analysis_prompt)
    analysis = analysis_chain.run(
        extracted_info=extracted_info,
        job_description=job_description,
        required_qualifications=required_qualifications,
        preferred_qualifications=preferred_qualifications
    )
   
    scoring_prompt = PromptTemplate(
        input_variables=["analysis"],
        template=(
            "Based on the following detailed analysis, provide a final evaluation and score for the candidate.\n\n"
            "Analysis:\n{analysis}\n\n"
            "Final Evaluation:\n"
            "1. Provide an overall score from 0 to 100.\n"
            "2. Summarize the candidate's fit for the position in 2-3 sentences.\n"
            "3. List the top 3 reasons to consider this candidate.\n"
            "4. List the top 3 concerns or areas for improvement.\n\n"
            "Format your response as follows:\n"
            "Score: [0-100]\n"
            "Summary: [Your summary]\n"
            "Top Reasons to Consider:\n1. [Reason 1]\n2. [Reason 2]\n3. [Reason 3]\n"
            "Areas of Concern:\n1. [Concern 1]\n2. [Concern 2]\n3. [Concern 3]"
        )
    )
    scoring_chain = LLMChain(llm=model, prompt=scoring_prompt)
    final_evaluation = scoring_chain.run(analysis=analysis)
    return final_evaluation

def get_score(evaluation_text):
    try:
        score_line = [line for line in evaluation_text.split('\n') if "Overall match score" in line][0]
        score = int(score_line.split(':')[1].strip().split('/')[0])
    except:
        score = 0
    return score


def add_category(category_type, category_name):
    cursor.execute("""
    INSERT INTO job_categories (category_type, category_name)
    VALUES (%s, %s)
    """, (category_type, category_name))
    db.commit()

def get_categories(category_type):
    cursor.execute("SELECT category_name FROM job_categories WHERE category_type = %s", (category_type,))
    return [row['category_name'] for row in cursor.fetchall()]

# Job Posting Management
def create_job_posting(title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, deadline, campus_type, university):
    cursor.execute("""
    INSERT INTO job_postings (title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, deadline, campus_type, university)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, deadline, campus_type, university))
    db.commit()
    return cursor.lastrowid

# Application Form Management
def create_application_form(form_name, job_id, form_fields):
    cursor.execute("""
    INSERT INTO application_forms (form_name, job_id, form_fields)
    VALUES (%s, %s, %s)
    """, (form_name, job_id, json.dumps(form_fields)))
    db.commit()

def get_application_form(job_id):
    cursor.execute("SELECT * FROM application_forms WHERE job_id = %s", (job_id,))
    return cursor.fetchone()

# Application Management
def submit_application(job_id, applicant_name, email, phone, resume_file, form_responses):
    # Check if the user has already applied
    cursor.execute("SELECT COUNT(*) as count FROM applications WHERE job_id = %s AND email = %s", (job_id, email))
    result = cursor.fetchone()
    if result['count'] > 0:
        st.error("You have already applied for this job. You can only submit one application per job posting.")
        return

    # Check if the user is eligible for this job
    cursor.execute("SELECT campus_type, university FROM job_postings WHERE id = %s", (job_id,))
    job_details = cursor.fetchone()
    
    cursor.execute("SELECT university FROM users WHERE email = %s", (email,))
    user_details = cursor.fetchone()

    if job_details['campus_type'] == 'On Campus' and job_details['university'] != user_details['university']:
        st.error("You are not eligible to apply for this on-campus job.")
        return

    resume_text = parse_resume(resume_file)
    resume_binary = resume_file.read()

    cursor.execute("""
    INSERT INTO applications (job_id, applicant_name, email, phone, resume_file, resume_text, form_responses, submission_date, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (job_id, applicant_name, email, phone, resume_binary, resume_text, json.dumps(form_responses), datetime.now(), "Applied"))
    db.commit()
    st.success("Application submitted successfully!")


def update_application_status(application_id, new_status):
    cursor.execute("UPDATE applications SET status = %s WHERE id = %s", (new_status, application_id))
    db.commit()

# Interview Scheduling
def schedule_interview(application_id, interview_date, interview_time, interviewers, interview_mode, interview_location=None):
    cursor.execute("""
    INSERT INTO interviews (application_id, interview_date, interview_time, interviewers, interview_mode, interview_location)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (application_id, interview_date, interview_time, json.dumps(interviewers), interview_mode, interview_location))
    db.commit()

def create_interview_ics(interview_data):
    cal = Calendar()
    event = Event()
    event.add('summary', f"Interview for {interview_data['job_title']}")
    event.add('dtstart', interview_data['interview_datetime'])
    event.add('dtend', interview_data['interview_datetime'] + timedelta(hours=1))
    event.add('location', interview_data['interview_location'])
    description = f"Interview for {interview_data['applicant_name']} for the position of {interview_data['job_title']}"
    event.add('description', description)
    event['organizer'] = f"mailto:{interview_data['hr_email']}"
    event.add('attendee', f"mailto:{interview_data['applicant_email']}")
    for interviewer_email in interview_data['interviewer_emails']:
        event.add('attendee', f"mailto:{interviewer_email}")
    cal.add_component(event)
    return cal.to_ical()

def send_email(recipient, subject, body, ics_content):
    sender_email = "vtu19978.soc.cse@gmail.com"  # Replace with your email
    password = "aocu ljua lofj jrvk"  # Replace with your email password

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    ics_attachment = MIMEApplication(ics_content, Name="interview.ics")
    ics_attachment['Content-Disposition'] = 'attachment; filename="interview.ics"'
    message.attach(ics_attachment)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.send_message(message)

def add_to_google_calendar(event_data, credentials_path='credentials.json'):
    SCOPES = ['https://www.googleapis.com/auth/calendar.events']
    creds = None

    try:
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")

        with open(credentials_path, 'r') as file:
            try:
                json.load(file)  # Test if the file is valid JSON
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in credentials file: {credentials_path}")

        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)

        service = build('calendar', 'v3', credentials=creds)

        event = {
            'summary': f"Interview for {event_data['job_title']}",
            'location': event_data['interview_location'],
            'description': f"Interview for {event_data['applicant_name']} for the position of {event_data['job_title']}",
            'start': {
                'dateTime': event_data['interview_datetime'].isoformat(),
                'timeZone': 'IST',  # Replace with your timezone
            },
            'end': {
                'dateTime': (event_data['interview_datetime'] + timedelta(hours=1)).isoformat(),
                'timeZone': 'IST',  # Replace with your timezone
            },
            'attendees': [
                {'email': event_data['applicant_email']},
                *[{'email': email} for email in event_data['interviewer_emails']]
            ],
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        print(f"Event created: {event.get('htmlLink')}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the credentials file exists and the path is correct.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check the contents of your credentials file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your Google Cloud Console settings and ensure you have the necessary permissions.")

        
def schedule_interview_with_notifications(interview_data):
    ics_content = create_interview_ics(interview_data)

    # Send email to applicant
    applicant_subject = f"Interview Scheduled for {interview_data['job_title']}"
    applicant_body = f"Dear {interview_data['applicant_name']},\n\nYour interview for the {interview_data['job_title']} position has been scheduled for {interview_data['interview_datetime']}. Please find the calendar invite attached.\n\nBest regards,\nHR Team"
    send_email(interview_data['applicant_email'], applicant_subject, applicant_body, ics_content)

    # Send email to interviewers
    interviewer_subject = f"Interview Scheduled: {interview_data['applicant_name']} for {interview_data['job_title']}"
    interviewer_body = f"Dear Interviewer,\n\nAn interview has been scheduled with {interview_data['applicant_name']} for the {interview_data['job_title']} position on {interview_data['interview_datetime']}. Please find the calendar invite attached.\n\nBest regards,\nHR Team"
    for interviewer_email in interview_data['interviewer_emails']:
        send_email(interviewer_email, interviewer_subject, interviewer_body, ics_content)

    # Add to Google Calendar
    add_to_google_calendar(interview_data)

def get_interviews(application_id):
    cursor.execute("SELECT * FROM interviews WHERE applicant_id = %s", (application_id,))
    return cursor.fetchall()

# Notes Management
def add_note(application_id, note_text):
    cursor.execute("INSERT INTO notes (application_id, note_text) VALUES (%s, %s)", (application_id, note_text))
    db.commit()

def get_notes(application_id):
    cursor.execute("SELECT * FROM notes WHERE application_id = %s ORDER BY created_at DESC", (application_id,))
    return cursor.fetchall()

# Selected Candidates Management
def add_to_selected_list(application_id):
    cursor.execute("UPDATE applications SET selected = TRUE WHERE id = %s", (application_id,))
    db.commit()

def remove_from_selected_list(application_id):
    cursor.execute("UPDATE applications SET selected = FALSE WHERE id = %s", (application_id,))
    db.commit()

def get_selected_candidates():
    cursor.execute("SELECT * FROM applications WHERE selected = TRUE")
    return cursor.fetchall()

# Notification System
def send_notification(recipient_email, subject, body):
    sender_email = "vtu19978.soc.cse@gmail.com"
    password = "aocu ljua lofj jrvk"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.send_message(message)

# User Management

def create_user(username, email, password, full_name, role, university=None):
    try:
        # Check if username or email already exists
        cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            if existing_user['username'] == username:
                return "Error: Username already exists"
            elif existing_user['email'] == email:
                return "Error: Email already exists"
        
        # If no existing user, proceed with registration
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        if role == "Candidate":
            cursor.execute("""
            INSERT INTO users (username, email, password, full_name, role, university)
            VALUES (%s, %s, %s, %s, %s, %s)
            """, (username, email, hashed_password, full_name, role, university))
        else:
            cursor.execute("""
            INSERT INTO users (username, email, password, full_name, role)
            VALUES (%s, %s, %s, %s, %s)
            """, (username, email, hashed_password, full_name, role))
        
        db.commit()
        return "Success: User registered successfully"
    except IntegrityError as e:
        db.rollback()
        if "Duplicate entry" in str(e):
            if "username" in str(e):
                return "Error: Username already exists"
            elif "email" in str(e):
                return "Error: Email already exists"
        return f"Error: {str(e)}"
    except Exception as e:
        db.rollback()
        return f"Error: {str(e)}"

def authenticate_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return user
    return None

# Feedback Management

def add_feedback(interview_id, interviewer_id, feedback_text, rating):
    try:
        
        cursor.execute("SELECT applicant_id FROM interviews WHERE id = %s", (interview_id,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"No interview found with id {interview_id}")
        application_id = result['applicant_id']

        
        cursor.execute("""
        INSERT INTO feedback (application_id, interviewer_id, feedback_text, rating)
        VALUES (%s, %s, %s, %s)
        """, (application_id, interviewer_id, feedback_text, rating))
        db.commit()

        
        cursor.execute("UPDATE applications SET status = 'In Review' WHERE id = %s", (application_id,))
        db.commit()

        
        cursor.execute("UPDATE interviews SET status = 'Completed' WHERE id = %s", (interview_id,))
        db.commit()

    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        st.error(f"An error occurred: {err}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error(f"An unexpected error occurred: {e}")
    
def get_feedback(application_id):
    cursor.execute("""
    SELECT f.*, u.full_name as interviewer_name
    FROM feedback f
    JOIN users u ON f.interviewer_id = u.id
    WHERE f.application_id = %s
    ORDER BY f.created_at DESC
    """, (application_id,))
    return cursor.fetchall()


def generate_recruitment_metrics():
    cursor.execute("""
    SELECT 
        COUNT(*) as total_applications,
        SUM(CASE WHEN status = 'Offered' THEN 1 ELSE 0 END) as offers_made,
        AVG(DATEDIFF(CURDATE(), submission_date)) as avg_time_to_hire
    FROM applications
    """)
    metrics = cursor.fetchone()
    
    cursor.execute("""
    SELECT department, COUNT(*) as application_count
    FROM applications a
    JOIN job_postings j ON a.job_id = j.id
    GROUP BY department
    """)
    department_data = cursor.fetchall()
    
    return metrics, department_data


def ai_select_candidates(job_id, num_candidates):
    cursor.execute("""
    SELECT a.*, j.description, j.required_qualifications, j.preferred_qualifications
    FROM applications a
    JOIN job_postings j ON a.job_id = j.id
    WHERE a.job_id = %s AND a.status = 'Applied'
    """, (job_id,))
    applications = cursor.fetchall()
    
    selected_candidates = []
    for app in applications:
        evaluation = evaluate_resume(
            app['resume_text'], 
            app['description'], 
            app['required_qualifications'], 
            app['preferred_qualifications']
        )
        score = get_score(evaluation)
        selected_candidates.append((app['id'], score))
    
    selected_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = selected_candidates[:num_candidates]
    
    for candidate_id, score in top_candidates:
        update_application_status(candidate_id, "In Review")
        add_to_selected_list(candidate_id)
    
    return top_candidates

def save_job_draft(title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, created_by, campus_type, university):
    cursor.execute("""
    INSERT INTO job_drafts (title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, created_by, campus_type, university)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, created_by, campus_type, university))
    db.commit()
    return cursor.lastrowid

def get_job_draft(draft_id):
    cursor.execute("SELECT * FROM job_drafts WHERE id = %s", (draft_id,))
    return cursor.fetchone()

def update_job_draft(draft_id, title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, campus_type, university):
    cursor.execute("""
    UPDATE job_drafts
    SET title = %s, description = %s, department = %s, location = %s, employment_type = %s, salary_range = %s, experience = %s, required_qualifications = %s, preferred_qualifications = %s, responsibilities = %s, campus_type = %s, university = %s
    WHERE id = %s
    """, (title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, campus_type, university, draft_id))
    db.commit()

def schedule_interview(application_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location=None):
    cursor.execute("""
INSERT INTO interviews (applicant_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location)
VALUES (%s, %s, %s, %s, %s, %s, %s)
""", (application_id, job_id, interview_date, interview_time, json.dumps(interviewer_ids), interview_mode, interview_location))
    db.commit()

def get_interviews():
    cursor.execute("""
    SELECT i.*, a.applicant_name, j.title as job_title
    FROM interviews i
    JOIN applications a ON i.applicant_id = a.id
    JOIN job_postings j ON i.job_id = j.id
    WHERE i.interview_date >= CURDATE()
    ORDER BY i.interview_date, i.interview_time
    """)
    return cursor.fetchall()

def get_interviewers():
    cursor.execute("SELECT id, full_name FROM users WHERE role IN ('Interviewer', 'HR Manager')")
    return cursor.fetchall()

def cancel_interview(interview_id):
    cursor.execute("DELETE FROM interviews WHERE id = %s", (interview_id,))
    db.commit()

def get_binary_file_downloader_html(bin_file, file_label='File'):
    bin_str = base64.b64encode(bin_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">Download {file_label}</a>'
    return href

def analyze_pdf_structure(file_content):
    pdf_io = io.BytesIO(file_content)
    pdf_io.seek(0)
    
    # Check for PDF signature
    if not file_content.startswith(b'%PDF-'):
        return False, "File does not start with PDF signature"

    # Try to find the root object
    pdf_io.seek(-1024, 2)  # Start near the end of the file
    end_content = pdf_io.read().decode('latin-1')
    
    # Look for trailer and root object
    trailer_match = re.search(r'trailer\s*<<(.|\s)*>>', end_content)
    root_match = re.search(r'/Root\s+(\d+)\s+(\d+)\s+R', end_content)
    
    if not trailer_match:
        return False, "Could not find PDF trailer"
    if not root_match:
        return False, "Could not find root object reference in trailer"
    
    root_obj = root_match.group(1)
    logging.info(f"Found root object: {root_obj}")
    
    # Additional checks could be added here
    
    return True, f"PDF structure seems valid. Root object: {root_obj}"

def is_valid_pdf(file_content):
    try:
        is_valid, message = analyze_pdf_structure(file_content)
        if is_valid:
            logging.info(message)
        else:
            logging.error(message)
        return is_valid, message
    except Exception as e:
        error_message = f"Error analyzing PDF: {str(e)}"
        logging.error(error_message)
        return False, error_message


import logging

def verify_file_content(file_content, filename):
    logging.info(f"Verifying content for file: {filename}")
    logging.info(f"File content type: {type(file_content)}")
    logging.info(f"File content length: {len(file_content)} bytes")
    logging.info(f"First 100 bytes: {file_content[:100]}")
    
    # Check if content starts with PDF signature
    if file_content.startswith(b'%PDF-'):
        logging.info("File appears to be a valid PDF (starts with %PDF-)")
    else:
        logging.warning("File does not start with PDF signature")

import logging

logging.basicConfig(level=logging.DEBUG)

def log_pdf_details(file_content, filename):
    logging.info(f"Processing file: {filename}")
    logging.info(f"File content type: {type(file_content)}")
    logging.info(f"File content length: {len(file_content)} bytes")
    logging.info(f"First 20 bytes (hex): {file_content[:20].hex()}")
    
    # Check if content starts with PDF signature
    if file_content.startswith(b'%PDF-'):
        logging.info("File starts with PDF signature")
    else:
        logging.warning("File does not start with PDF signature")

# Use this function before attempting to validate or serve the PDF
def validate_pdf_with_pikepdf(file_content):
    try:
        pdf_io = io.BytesIO(file_content)
        with pikepdf.Pdf.open(pdf_io) as pdf:
            num_pages = len(pdf.pages)
        return True, f"Valid PDF with {num_pages} pages"
    except Exception as e:
        return False, f"Error validating PDF with pikepdf: {str(e)}"

from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf_from_text(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Split the text into paragraphs
    paragraphs = text.split('\n\n')
    
    # Create a list of Paragraph objects
    story = [Paragraph(p.replace('\n', '<br />'), styles['Normal']) for p in paragraphs]
    
    # Build the PDF
    doc.build(story)
    
    # Get the value of the BytesIO buffer
    pdf = buffer.getvalue()
    buffer.close()
    
    return pdf

def get_upcoming_interviews(interviewer_id):
    query = """
    SELECT i.*, a.applicant_name, j.title as job_title, a.resume_text
    FROM interviews i
    JOIN applications a ON i.applicant_id = a.id
    JOIN job_postings j ON a.job_id = j.id
    WHERE JSON_CONTAINS(i.interviewers, %s)
    AND i.interview_date >= CURDATE()
    ORDER BY i.interview_date, i.interview_time
    """
    cursor.execute(query, (json.dumps(interviewer_id),))
    return cursor.fetchall()



def clean_json_string(s):
    # Remove any leading/trailing non-JSON text
    json_pattern = r'\[.*\]'
    match = re.search(json_pattern, s, re.DOTALL)
    if match:
        return match.group()
    return s

def generate_ai_questions(job_id, mcq_count, coding_count, text_count, open_ended_count, difficulty):
    cursor.execute("SELECT title, description, required_qualifications FROM job_postings WHERE id = %s", (job_id,))
    job_info = cursor.fetchone()
    
    prompt = PromptTemplate(
        input_variables=["title", "description", "qualifications", "mcq_count", "coding_count", "text_count", "open_ended_count", "difficulty"],
        template="""
        Generate questions for a job application test based on the following job details:
        Job Title: {title}
        Job Description: {description}
        Required Qualifications: {qualifications}
        Difficulty Level: {difficulty}

        Generate:
        1. {mcq_count} multiple-choice questions related to the job skills (focus on fundamentals of the skills)
        2. {coding_count} coding questions (provide a simple function signature and expected output)
        3. {text_count} text-based fundamental questions on the given skills from the job description
        4. {open_ended_count} open-ended HR questions

        All questions should be at the {difficulty} difficulty level.

        Format the output as a JSON array with each question as an object containing 'type', 'question', 'options' (for MCQs), 'correct_answer' (except for the HR questions), and 'difficulty'.
        Use 'multiple_choice', 'coding', 'text', and 'open_ended' as the values for 'type'.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(
        title=job_info['title'], 
        description=job_info['description'], 
        qualifications=job_info['required_qualifications'],
        mcq_count=mcq_count,
        coding_count=coding_count,
        text_count=text_count,
        open_ended_count=open_ended_count,
        difficulty=difficulty
    )
    cleaned_result = clean_json_string(result)
    try:
        questions = json.loads(cleaned_result)
    except json.JSONDecodeError:
        st.error("Failed to generate questions in the correct format. Please try again.")
        st.text(result)
        return
    
    for question in questions:
        cursor.execute("""
        INSERT INTO test_questions (job_id, question_type, question_text, options, correct_answer, difficulty)
        VALUES (%s, %s, %s, %s, %s, %s)
        """, (job_id, question['type'], question['question'], json.dumps(question.get('options')), question.get('correct_answer'), question['difficulty']))
    db.commit()
    st.success("AI questions generated successfully!")
    return questions

def display_generated_questions(questions):
    st.subheader("Generated Questions")
    for i, question in enumerate(questions, 1):
        st.write(f"{i}. Type: {question['type']}, Difficulty: {question['difficulty']}")
        st.write(f"Question: {question['question']}")
        if question['type'] == 'multiple_choice':
            st.write(f"Options: {', '.join(question['options'])}")
            st.write(f"Correct Answer: {question['correct_answer']}")
        elif question['type'] == 'coding':
            st.write(f"Expected Output:")
        st.write("---")


def add_manual_question(job_id, question_text, question_type, options=None, correct_answer=None):
    cursor.execute("""
    INSERT INTO test_questions (job_id, question_type, question_text, options, correct_answer)
    VALUES (%s, %s, %s, %s, %s)
    """, (job_id, question_type, question_text, json.dumps(options), correct_answer))
    db.commit()

def get_test_questions(job_id):
    cursor.execute("SELECT * FROM test_questions WHERE job_id = %s", (job_id,))
    return cursor.fetchall()

def start_test_session(application_id):
    try:
        cursor.execute("""
        INSERT INTO test_sessions (application_id, start_time, status)
        VALUES (%s, %s, 'in_progress')
        """, (application_id, datetime.now()))
        db.commit()
        return cursor.lastrowid
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        db.rollback()
        return None
    finally:
        cursor.fetchall() 

def end_test_session(session_id):
    cursor.execute("""
    UPDATE test_sessions SET end_time = %s, status = 'completed'
    WHERE id = %s
    """, (datetime.now(), session_id))
    db.commit()

def save_candidate_response(application_id, question_id, answer):
    cursor.execute("""
    SELECT question_type FROM test_questions WHERE id = %s
    """, (question_id,))
    question_type = cursor.fetchone()['question_type']
    
    if question_type == 'Coding':
        
        cursor.execute("""
        INSERT INTO candidate_test_responses (application_id, question_id, candidate_answer, start_time)
        VALUES (%s, %s, %s, %s)
        """, (application_id, question_id, answer, datetime.now()))
    else:
        
        cursor.execute("""
        INSERT INTO candidate_test_responses (application_id, question_id, candidate_answer, start_time)
        VALUES (%s, %s, %s, %s)
        """, (application_id, question_id, answer, datetime.now()))
    db.commit()

def update_candidate_response(response_id, answer):
    cursor.execute("""
    UPDATE candidate_test_responses SET candidate_answer = %s, end_time = %s
    WHERE id = %s
    """, (answer, datetime.now(), response_id))
    db.commit()



def evaluate_candidate_answers(application_id, job_id):
    try:
        # Get the number of questions for this job
        cursor.execute("""
        SELECT COUNT(*) as question_count
        FROM test_questions
        WHERE job_id = %s
        """, (job_id,))
        question_count = cursor.fetchone()['question_count']

        if question_count == 0:
            return 0, ["No questions found for this test"]

        # Get the candidate's responses
        cursor.execute("""
        SELECT ctr.*, tq.question_text, tq.correct_answer, tq.question_type
        FROM candidate_test_responses ctr
        JOIN test_questions tq ON ctr.question_id = tq.id
        WHERE ctr.application_id = %s
        ORDER BY ctr.id DESC
        LIMIT %s
        """, (application_id,question_count,))
        responses = cursor.fetchall()

        if len(responses) == 0:
            return 0, ["No responses found for this test"]

        total_score = 0
        max_possible_score = len(responses) * 100
        explanations = []

        def extract_score_and_explanation(text):
            score_match = re.search(r'score"?\s*:\s*(\d+)', text, re.IGNORECASE)
            explanation_match = re.search(r'explanation"?\s*:\s*"([^"]+)"', text, re.IGNORECASE)
            
            score = int(score_match.group(1)) if score_match else 0
            explanation = explanation_match.group(1) if explanation_match else "No explanation provided."
            
            return score, explanation

        for response in responses:
            if response['question_type'] == 'Coding':
                evaluation_prompt = PromptTemplate(
                    input_variables=["question", "correct_answer", "candidate_answer"],
                    template="""
                    Coding Question: {question}
                    Expected Output or Functionality: {correct_answer}
                    Candidate's Code: {candidate_answer}
                    
                    Evaluate the candidate's code. Consider correctness, efficiency, and coding style.
                    Provide a score between 0 and 100, where 100 is a perfect answer and 0 is completely incorrect.
                    Also provide a brief explanation for the score.
                    Your response should be in the format: score: <score>, explanation: "<explanation>"
                    """
                )
            else:
                evaluation_prompt = PromptTemplate(
                    input_variables=["question", "correct_answer", "candidate_answer"],
                    template="""
                    Question: {question}
                    Correct Answer: {correct_answer}
                    Candidate Answer: {candidate_answer}
                    
                    Evaluate the candidate's answer. Provide a score between 0 and 100, where 100 is a perfect answer and 0 is completely incorrect.
                    Also provide a brief explanation for the score.
                    Your response should be in the format: score: <score>, explanation: "<explanation>"
                    """
                )
            
            evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)
            result = evaluation_chain.run(question=response['question_text'], correct_answer=response['correct_answer'], candidate_answer=response['candidate_answer'])
            
            score, explanation = extract_score_and_explanation(result)
            
            total_score += score
            explanations.append(f"Question {response['id']}: {explanation}")
            
            cursor.execute("UPDATE candidate_test_responses SET score = %s WHERE id = %s", (score, response['id']))

        if max_possible_score > 0:
            average_score = (total_score / max_possible_score) * 100
        else:
            average_score = 0

        cursor.execute("UPDATE applications SET test_score = %s, test_status = 'evaluated' WHERE id = %s", (average_score, application_id))
        db.commit()

        return average_score, explanations

    except Exception as e:
        error_message = f"An error occurred during evaluation: {str(e)}"
        print(error_message)
        return 0, [error_message]


def ai_proctoring(frame, tab_switched):
    violations = []
    
    
    if tab_switched:
        violations.append("Tab switching detected")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        violations.append("No face detected")
    elif len(faces) > 1:
        violations.append("Multiple faces detected")
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) < 2:
            violations.append("Eyes not clearly visible")
        
        if 'prev_face_center' not in ai_proctoring.cache:
            ai_proctoring.cache['prev_face_center'] = None
        
        face_center = (x + w//2, y + h//2)
        if ai_proctoring.cache['prev_face_center']:
            distance = np.sqrt((face_center[0] - ai_proctoring.cache['prev_face_center'][0])**2 + 
                               (face_center[1] - ai_proctoring.cache['prev_face_center'][1])**2)
            if distance > 50:  
                violations.append("Rapid head movement detected")
        ai_proctoring.cache['prev_face_center'] = face_center
    

    edges = cv2.Canny(frame, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000: 
            (x, y, w, h) = cv2.boundingRect(contour)
            if y < frame.shape[0] // 2: 
                violations.append("Suspicious object detected near face")
                break
    
    return len(violations) == 0, violations

ai_proctoring.cache = {}

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))

    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))

    ear = (A + B) / (2.0 * C)

    return ear

def analyze_video_interview(interview_id):
    try:
        cursor.execute("SELECT video_url, audio_url, question FROM video_interviews WHERE id = %s", (interview_id,))
        result = cursor.fetchone()
        if not result or not result['video_url'] or not result['audio_url']:
            return 0, "Video or audio not found"
        
        video_path = result['video_url']
        audio_path = result['audio_url']
        question = result['question']
        
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            return 0, "Video or audio file not found on the server"
        
        # Transcribe audio
        recognizer = Recognizer()
        with AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        
        try:
            transcript = recognizer.recognize_google(audio)
        except Exception as transcribe_error:
            logging.error(f"Error transcribing audio: {str(transcribe_error)}")
            return 0, "Failed to transcribe audio"
        
        # Analyze transcript with LLM
        text_analysis_result = analyze_transcript_with_llm(transcript, question)
        
        # Analyze facial emotions
        emotion_analysis_result = analyze_facial_emotions(video_path)
        
        # Analyze emotions with LLM
        emotion_llm_analysis = analyze_emotions_with_llm(emotion_analysis_result['emotion_analysis'], question)
        
        # Combine results
        combined_score = (text_analysis_result['overall_score'] + emotion_analysis_result['emotion_score'] + emotion_llm_analysis['emotion_llm_score']) / 3
        combined_analysis = f"""
        Text Analysis:
        {text_analysis_result['analysis_text']}
        
        Emotion Analysis:
        {emotion_analysis_result['emotion_analysis']}
        
        LLM Emotion Analysis:
        {emotion_llm_analysis['emotion_llm_analysis']}
        
        Overall Score: {combined_score:.2f}
        """
        
        cursor.execute("UPDATE video_interviews SET score = %s, analysis_result = %s, status = 'analyzed' WHERE id = %s",
                       (combined_score, combined_analysis, interview_id))
        db.commit()
        
        return combined_score, combined_analysis
    
    except Exception as e:
        logging.error(f"Error in analyze_video_interview: {str(e)}")
        return 0, f"An error occurred during analysis: {str(e)}"

def analyze_facial_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions.append(result[0]['emotion'])
        except:
            pass
    
    cap.release()
    
    # Calculate average emotions
    avg_emotions = {emotion: np.mean([e[emotion] for e in emotions]) for emotion in emotions[0].keys()}
    
    # Calculate emotion score (you may want to adjust this based on your preferences)
    emotion_score = (avg_emotions['happy'] + avg_emotions['surprise']) - (avg_emotions['angry'] + avg_emotions['sad'] + avg_emotions['fear'])
    emotion_score = max(0, min(10, emotion_score * 2))  # Scale to 0-10
    
    emotion_analysis = f"""
    Average Emotions:
    Happy: {avg_emotions['happy']:.2f}
    Sad: {avg_emotions['sad']:.2f}
    Angry: {avg_emotions['angry']:.2f}
    Surprise: {avg_emotions['surprise']:.2f}
    Fear: {avg_emotions['fear']:.2f}
    Disgust: {avg_emotions['disgust']:.2f}
    Neutral: {avg_emotions['neutral']:.2f}
    
    Emotion Score: {emotion_score:.2f}
    """
    
    return {'emotion_score': emotion_score, 'emotion_analysis': emotion_analysis}

def analyze_emotions_with_llm(emotion_analysis, question):
    prompt = PromptTemplate(
        input_variables=["emotion_analysis", "question"],
        template="""Analyze the following emotion data from a video interview. Consider how these emotions might relate to the interview question and the candidate's overall performance. Provide insights on what these emotions might indicate about the candidate's confidence, interest, and suitability for the role. Also, suggest any potential concerns or positive indicators based on the emotional data.

Emotion Data:
{emotion_analysis}

Interview Question: {question}

Please provide your analysis in the following format:
1. Overall Emotional State:
2. Confidence Assessment:
3. Interest and Engagement:
4. Potential Concerns:
5. Positive Indicators:
6. Suitability for the Role:
7. Recommendations for Follow-up:

Based on your analysis, provide an overall score from 0-10, where 10 indicates an ideal emotional state for the interview.

Overall Emotion-based Score: [Your score here]

Detailed Analysis:
[Your detailed analysis here]"""
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(emotion_analysis=emotion_analysis, question=question)
    
    # Extract the score using regex
    import re
    score_match = re.search(r'Overall Emotion-based Score: (\d+(?:\.\d+)?)', response)
    if score_match:
        emotion_llm_score = float(score_match.group(1))
    else:
        emotion_llm_score = 5.0  # Default score if not found
    
    return {
        'emotion_llm_score': emotion_llm_score,
        'emotion_llm_analysis': response
    }

import re
from typing import Dict, Any

def analyze_transcript_with_llm(transcript: str, question: str) -> Dict[str, Any]:
    prompt = PromptTemplate(
        input_variables=["question", "transcript"],
        template="""Analyze the following interview transcript based on the given question. Provide scores (0-10) and detailed feedback for each of these aspects:
1. Sentiment Analysis: Evaluate the overall tone and emotion of the response.
2. Grammar: Assess the grammatical correctness of the response.
3. Fluency: Evaluate the smoothness and coherence of the speech.
4. Topic Relevance: Determine how well the response addresses the given question.
5. Knowledge Demonstration: Assess the depth and accuracy of knowledge shown in the response.

Question: {question}

Transcript: {transcript}

Please provide your analysis in the following format:
Sentiment Analysis:
Score: [0-10]
Feedback: [Your detailed feedback]

Grammar:
Score: [0-10]
Feedback: [Your detailed feedback]

Fluency:
Score: [0-10]
Feedback: [Your detailed feedback]

Topic Relevance:
Score: [0-10]
Feedback: [Your detailed feedback]

Knowledge Demonstration:
Score: [0-10]
Feedback: [Your detailed feedback]

Overall Score: [Average of all scores, rounded to two decimal places]

Summary: [A brief summary of the overall performance]"""
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(question=question, transcript=transcript)
    
    # Extract scores using regex
    scores = re.findall(r'Score: (\d+(?:\.\d+)?)', response)
    
    # Convert scores to floats
    scores = [float(score) for score in scores]
    
    # Calculate overall score
    if scores:
        overall_score = round(sum(scores) / len(scores), 2)
    else:
        overall_score = 0.00
    
    # Update the response with the calculated overall score
    response = re.sub(r'Overall Score:.*', f'Overall Score: {overall_score}', response)
    
    return {
        'overall_score': overall_score,
        'analysis_text': response
    }

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_vector_db(data):
    texts = [str(item) for item in data]
    vector_db = FAISS.from_texts(texts, embeddings)
    return vector_db

def query_vector_db(vector_db, query):
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())
    return qa_chain.run(query)

def historical_recruitment_analysis():
    st.title("Historical Recruitment Analysis Engine")

    # Fetch historical data
    try:
        cursor.execute("""
            SELECT jp.title, jp.department, jp.location, jp.employment_type,
                   a.applicant_name, a.status, a.test_status,
                   ts.status as test_session_status,
                   vi.status as video_interview_status,
                   f.rating as interview_rating
            FROM job_postings jp
            JOIN applications a ON jp.id = a.job_id
            LEFT JOIN test_sessions ts ON a.id = ts.application_id
            LEFT JOIN video_interviews vi ON a.id = vi.application_id
            LEFT JOIN feedback f ON a.id = f.application_id
            ORDER BY jp.created_at DESC
            LIMIT 1000
        """)
        historical_data = cursor.fetchall()
    except mysql.connector.Error as err:
        st.error(f"Error fetching data: {err}")
        return

    if not historical_data:
        st.warning("No historical data found.")
        return

    vector_db = create_vector_db(historical_data)

    analysis_type = st.selectbox("Select Analysis Type", 
                                 ["Hiring Trends", 
                                  "Successful Candidate Profiles", 
                                  "Department Performance", 
                                  "Location-based Insights"])

    if analysis_type == "Hiring Trends":
        if st.button("Analyze Hiring Trends"):
            query = """Analyze the hiring trends in the recent job postings and applications. Consider:
            1. Most common job titles, departments, and employment types
            2. Application statuses distribution
            3. Test completion rates and performance
            4. Video interview completion rates
            5. Overall success rate (candidates with high interview ratings)
            Provide insights on popular roles and any noticeable patterns in the hiring process."""
            insights = query_vector_db(vector_db, query)
            st.write(insights)

    elif analysis_type == "Successful Candidate Profiles":
        if st.button("Analyze Successful Candidates"):
            query = """Examine the profiles of candidates who received high interview ratings. Consider:
            1. Their test performance (test_status and test_session_status)
            2. Their video interview performance (video_interview_status)
            3. Their overall application status
            4. The types of roles and departments they applied to
            Provide insights on common characteristics of successful candidates in our hiring process."""
            insights = query_vector_db(vector_db, query)
            st.write(insights)

    elif analysis_type == "Department Performance":
        departments = list(set(item['department'] for item in historical_data if item['department']))
        if departments:
            department = st.selectbox("Select Department", departments)
            if st.button("Analyze Department Performance"):
                query = f"""Analyze the hiring performance of the {department} department. Consider:
                1. Number and types of job postings in this department
                2. Application rates and statuses for jobs in this department
                3. Test completion and success rates for this department's applications
                4. Video interview completion rates for this department
                5. Overall success rate (candidates with high interview ratings) for this department
                Compare this department's performance to others and suggest potential improvements."""
                insights = query_vector_db(vector_db, query)
                st.write(insights)
        else:
            st.warning("No department data available.")

    elif analysis_type == "Location-based Insights":
        locations = list(set(item['location'] for item in historical_data if item['location']))
        if locations:
            location = st.selectbox("Select Location", locations)
            if st.button("Analyze Location-based Insights"):
                query = f"""Analyze the recruitment patterns for {location}. Consider:
                1. Types of roles and departments most common in this location
                2. Application rates and statuses for jobs in this location
                3. Test completion and success rates for applications in this location
                4. Video interview completion rates for this location
                5. Overall success rate (candidates with high interview ratings) for this location
                Identify any unique challenges or advantages for hiring in this location compared to others."""
                insights = query_vector_db(vector_db, query)
                st.write(insights)
        else:
            st.warning("No location data available.")

    if st.checkbox("Show Raw Historical Data"):
        st.write(pd.DataFrame(historical_data))

def github_profile_analysis():
    st.title("GitHub Profile Analysis")
    github_username = st.text_input("Enter GitHub Username")
    if st.button("Analyze GitHub Profile"):
        if github_username:
            try:
                user_url = f"https://api.github.com/users/{github_username}"
                user_response = requests.get(user_url)
                user_data = user_response.json()

                if user_response.status_code == 200:
                    repos_url = f"https://api.github.com/users/{github_username}/repos"
                    repos_response = requests.get(repos_url)
                    repos_data = repos_response.json()

                    if repos_response.status_code == 200:
                        profile_data = {
                            "User Info": user_data,
                            "Repositories": repos_data[:5] 
                        }
                        query = f"""Analyze the following GitHub profile and provide a detailed report:
                        1. Overview of the user's GitHub activity and profile
                        2. Detailed analysis of their top 5 projects, including:
                        - Project description
                        - Technologies used
                        - Project complexity
                        - Potential impact or usefulness
                        3. Overall skill assessment based on the projects
                        4. Areas of expertise
                        5. Suggestions for improvement
                        6. Rate the overall profile on a scale of 1-10, with justification

                        GitHub Profile Data:
                        {profile_data}
                        """

                        analysis = query_vector_db(create_vector_db([str(profile_data)]), query)

                        
                        st.subheader("GitHub Profile Analysis")
                        st.write(analysis)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a GitHub username.")


def send_offer_letter(recipient_email, candidate_name, job_title, offer_deadline):
    sender_email = "vtu19978.soc.cse@gmail.com"  # Replace with your email
    password = "aocu ljua lofj jrvk"  # Replace with your email password
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = f"Job Offer: {job_title}"
    
    body = f"""
    Dear {candidate_name},

    We are pleased to offer you the position of {job_title} at our company.
    
    Please review the attached offer letter for more details.
    
    This offer is valid until {offer_deadline.strftime('%Y-%m-%d')}.
    Please respond with your decision by this date.

    Congratulations!

    Best regards,
    HR Department
    """
    
    message.attach(MIMEText(body, "plain"))
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.send_message(message)

def offers_section():
    st.subheader("Offered Candidates")

    # Fetch all unique job titles with offered candidates
    cursor.execute("""
    SELECT DISTINCT j.title, j.id
    FROM applications a
    JOIN job_postings j ON a.job_id = j.id
    WHERE a.offer_status IN ('Not Offered', 'Offered')
    """)
    job_data = cursor.fetchall()

    # Create a dictionary of job titles and IDs
    job_dict = {f"{row['title']} (ID: {row['id']})": row['id'] for row in job_data}

    # Create a selectbox with job titles and IDs
    selected_job = st.selectbox("Select Job Title", list(job_dict.keys()))

    if selected_job:
        selected_job_id = job_dict[selected_job]

        cursor.execute("""
        SELECT a.*, j.title as job_title
        FROM applications a
        JOIN job_postings j ON a.job_id = j.id
        WHERE a.offer_status IN ('Not Offered', 'Offered') AND j.id = %s and a.status="Offered"
        """, (selected_job_id,))
        candidates = cursor.fetchall()

        st.write(f"Number of candidates: {len(candidates)}")

        for candidate in candidates:
            with st.expander(f"{candidate['applicant_name']} - {candidate['job_title']}"):
                st.write(f"Email: {candidate['email']}")
                st.write(f"Application Date: {candidate['submission_date']}")
                st.write(f"Offer Status: {candidate['offer_status']}")

                if candidate['offer_status'] == 'Not Offered':
                    if st.button(f"Send Offer to {candidate['applicant_name']}", key=f"send_offer_{candidate['id']}"):
                        offer_deadline = datetime.now() + timedelta(days=7)
                        try:
                            send_offer_letter(candidate['email'], candidate['applicant_name'], candidate['job_title'], offer_deadline)
                            cursor.execute("""
                            UPDATE applications
                            SET offer_status = 'Offered', offer_deadline = %s
                            WHERE id = %s
                            """, (offer_deadline, candidate['id']))
                            db.commit()
                            st.success(f"Offer letter sent to {candidate['applicant_name']} ({candidate['email']})")
                        except Exception as e:
                            st.error(f"Failed to send offer letter: {str(e)}")
                
                elif candidate['offer_status'] == 'Offered':
                    st.write(f"Offer Deadline: {candidate['offer_deadline']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Accept Offer", key=f"accept_{candidate['id']}"):
                            cursor.execute("""
                            UPDATE applications
                            SET offer_status = 'Accepted', offer_accepted_date = CURRENT_DATE
                            WHERE id = %s
                            """, (candidate['id'],))
                            db.commit()
                            st.success("Offer accepted!")
                    
                    with col2:
                        if st.button("Decline Offer", key=f"decline_{candidate['id']}"):
                            cursor.execute("""
                            UPDATE applications
                            SET offer_status = 'Declined'
                            WHERE id = %s
                            """, (candidate['id'],))
                            db.commit()
                            st.warning("Offer declined.")

    st.subheader("Accepted Offers")
    cursor.execute("""
    SELECT a.*, j.title as job_title
    FROM applications a
    JOIN job_postings j ON a.job_id = j.id
    WHERE a.offer_status = 'Accepted'
    """)
    accepted_candidates = cursor.fetchall()

    for candidate in accepted_candidates:
        with st.expander(f"{candidate['applicant_name']} - {candidate['job_title']}"):
            st.write(f"Email: {candidate['email']}")
            st.write(f"Offer Accepted Date: {candidate['offer_accepted_date']}")
            
            onboarding_date = candidate['offer_accepted_date'] + timedelta(days=14)
            st.write(f"Onboarding Date: {onboarding_date}")

            cursor.execute("""
            INSERT INTO onboarding (application_id, onboarding_date, status)
            VALUES (%s, %s, 'Pending')
            ON DUPLICATE KEY UPDATE onboarding_date = %s
            """, (candidate['id'], onboarding_date, onboarding_date))
            db.commit()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_onboarding_data():
    try:
        # Remove duplicates, keeping the entry with the most complete information
        cursor.execute("""
        DELETE o1 FROM onboarding o1
        INNER JOIN onboarding o2
        WHERE o1.application_id = o2.application_id
        AND o1.id > o2.id
        """)
        
        # Update the status of the remaining entry if there's a completed checklist
        cursor.execute("""
        UPDATE onboarding
        SET status = 'Completed'
        WHERE checklist IS NOT NULL AND JSON_LENGTH(checklist) > 0
        """)
        
        db.commit()
        logger.info("Onboarding data cleanup completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during onboarding data cleanup: {str(e)}")
        return False

def onboarding_section():
    st.subheader("Onboarding")

    # Cleanup onboarding data
    if cleanup_onboarding_data():
        st.success("Onboarding data has been cleaned up. Duplicates have been removed.")
    else:
        st.error("There was an error cleaning up the onboarding data. Please check the logs.")

    # Fetch all onboarding candidates
    cursor.execute("""
    SELECT o.*, a.applicant_name, j.title as job_title
    FROM onboarding o
    JOIN applications a ON o.application_id = a.id
    JOIN job_postings j ON a.job_id = j.id
    ORDER BY o.onboarding_date
    """)
    onboarding_candidates = cursor.fetchall()

    # Group candidates by onboarding status
    pending = [c for c in onboarding_candidates if c['status'] == 'Pending']
    in_progress = [c for c in onboarding_candidates if c['status'] == 'In Progress']
    completed = [c for c in onboarding_candidates if c['status'] == 'Completed']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Pending")
        for candidate in pending:
            with st.expander(f"{candidate['applicant_name']} - {candidate['job_title']}"):
                st.write(f"Onboarding Date: {candidate['onboarding_date']}")
                if st.button("Start Onboarding", key=f"start_{candidate['id']}"):
                    cursor.execute("""
                    UPDATE onboarding
                    SET status = 'In Progress'
                    WHERE id = %s
                    """, (candidate['id'],))
                    db.commit()
                    st.success("Onboarding started!")
                    st.rerun()

    with col2:
        st.write("### In Progress")
        for candidate in in_progress:
            with st.expander(f"{candidate['applicant_name']} - {candidate['job_title']}"):
                st.write(f"Onboarding Date: {candidate['onboarding_date']}")
                checklist = candidate['checklist'] or {}
                for item in ['Paperwork', 'IT Setup', 'Training', 'Introduction']:
                    checklist[item] = st.checkbox(item, value=checklist.get(item, False), key=f"{candidate['id']}_{item}")
                
                if st.button("Update Checklist", key=f"update_{candidate['id']}"):
                    cursor.execute("""
                    UPDATE onboarding
                    SET checklist = %s,
                        status = CASE WHEN %s THEN 'Completed' ELSE 'In Progress' END
                    WHERE id = %s
                    """, (json.dumps(checklist), all(checklist.values()), candidate['id']))
                    db.commit()
                    st.success("Checklist updated!")
                    st.rerun()

    with col3:
        st.write("### Completed")
        for candidate in completed:
            with st.expander(f"{candidate['applicant_name']} - {candidate['job_title']}"):
                st.write(f"Onboarding Date: {candidate['onboarding_date']}")
                st.write("All onboarding tasks completed.")

    # Provide option to manually refresh
    if st.button("Refresh Onboarding Data"):
        st.rerun()

def view_offers_section(user_id):
    # First, check the user's role
    cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
    user_role = cursor.fetchone()

    if not user_role or user_role['role'] != 'Candidate':
        st.error("Access denied. Only candidates can view offers.")
        return

    st.subheader("View Your Offers")

    # Fetch offers for the specific user
    cursor.execute("""
    SELECT a.*, j.title as job_title, o.offer_letter_content, o.salary
    FROM applications a
    JOIN job_postings j ON a.job_id = j.id
    LEFT JOIN offers o ON a.id = o.application_id
    JOIN users u ON a.email = u.email
    WHERE a.offer_status IN ('Offered', 'Accepted', 'Declined')
    AND u.id = %s
    ORDER BY a.offer_deadline DESC
    """, (user_id,))
    offers = cursor.fetchall()

    if not offers:
        st.info("You have no offers at the moment.")
        return

    # Filter options
    status_filter = st.multiselect("Filter by Status", ['Offered', 'Accepted', 'Declined'])
    job_filter = st.multiselect("Filter by Job", list(set(offer['job_title'] for offer in offers)))

    filtered_offers = [
        offer for offer in offers
        if (not status_filter or offer['offer_status'] in status_filter)
        and (not job_filter or offer['job_title'] in job_filter)
    ]

    for offer in filtered_offers:
        with st.expander(f"{offer['job_title']} ({offer['offer_status']})"):
            st.write(f"Offer Status: {offer['offer_status']}")
            st.write(f"Offer Deadline: {offer['offer_deadline']}")
            if offer['offer_status'] == 'Accepted':
                st.write(f"Accepted Date: {offer['offer_accepted_date']}")
            if offer['salary']:
                st.write(f"Offered Salary: ${offer['salary']:,.2f}")
            if offer['offer_letter_content']:
                st.text_area("Offer Letter", offer['offer_letter_content'], height=200, disabled=True)
            
            # Option to accept or decline offer
            if offer['offer_status'] == 'Offered':
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Accept Offer", key=f"accept_{offer['id']}"):
                        cursor.execute("""
                        UPDATE applications a
                        JOIN users u ON a.email = u.email
                        SET a.offer_status = 'Accepted', a.offer_accepted_date = CURRENT_DATE
                        WHERE a.id = %s AND u.id = %s
                        """, (offer['id'], user_id))
                        db.commit()
                        if cursor.rowcount > 0:
                            st.success("Offer accepted!")
                            st.rerun()
                        else:
                            st.error("Failed to accept offer. Please try again.")
                with col2:
                    if st.button("Decline Offer", key=f"decline_{offer['id']}"):
                        cursor.execute("""
                        UPDATE applications a
                        JOIN users u ON a.email = u.email
                        SET a.offer_status = 'Declined'
                        WHERE a.id = %s AND u.id = %s
                        """, (offer['id'], user_id))
                        db.commit()
                        if cursor.rowcount > 0:
                            st.warning("Offer declined.")
                            st.rerun()
                        else:
                            st.error("Failed to decline offer. Please try again.")

embeddings = HuggingFaceEmbeddings()

def create_vector_db(data):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text("\n".join(str(item) for item in data))
    vector_db = FAISS.from_texts(texts, embeddings)
    return vector_db

def query_vector_db(vector_db, query):
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())
    return qa_chain.run(query)

def ai_talent_marketplace():
    st.title("AI-Driven Talent Marketplace")

    if st.session_state.user['role'] in ['Admin', 'HR Manager', 'Recruiter']:
        hr_talent_marketplace()
    elif st.session_state.user['role'] == 'Candidate':
        candidate_talent_marketplace()
    else:
        st.error("Access denied. This feature is not available for your role.")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def hr_talent_marketplace():
    st.subheader("Talent Pool Analysis and Job Matching")

    try:
        # Fetch all job postings
        cursor.execute("SELECT id, title, department, required_qualifications, preferred_qualifications FROM job_postings")
        job_postings = cursor.fetchall()

        # Fetch all candidates
        cursor.execute("""
            SELECT a.id, a.applicant_name, a.resume_text, jp.title as applied_job
            FROM applications a
            JOIN job_postings jp ON a.job_id = jp.id
        """)
        candidates = cursor.fetchall()

        # Create embeddings for job postings and candidates
        job_embeddings = [embeddings.embed_query(f"{job['title']} {job['required_qualifications']} {job['preferred_qualifications']}") for job in job_postings]
        candidate_embeddings = [embeddings.embed_query(f"{candidate['resume_text']} {candidate['applied_job']}") for candidate in candidates]

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(candidate_embeddings, job_embeddings)

        # Create a DataFrame for easy analysis
        df = pd.DataFrame(similarity_matrix, 
                          columns=[f"{job['id']} - {job['title']}" for job in job_postings], 
                          index=[f"{candidate['id']} - {candidate['applicant_name']}" for candidate in candidates])

        # Talent Pool Analysis
        st.subheader("Talent Pool Analysis")
        if st.button("Generate Talent Pool Insights"):
            talent_pool_data = [f"{c['applicant_name']}: {c['resume_text']}" for c in candidates]
            vector_db = create_vector_db(talent_pool_data)
            query = """Analyze the talent pool and provide insights on:
                       1. Common skills and qualifications among candidates
                       2. Any skill gaps compared to current job openings
                       3. Recommendations for upskilling or reskilling initiatives
                       4. Suggestions for new roles that could leverage the available talent"""
            insights = query_vector_db(vector_db, query)
            st.write(insights)

        # Job Matching
        st.subheader("Job Matching")
        selected_job_for_matching = st.selectbox("Select a job to find top candidates", 
                                                 [f"{job['id']} - {job['title']}" for job in job_postings],
                                                 key="job_matching")
        if selected_job_for_matching:
            job_id = int(selected_job_for_matching.split('-')[0].strip())
            top_candidates = df[selected_job_for_matching].nlargest(5)
            st.write("Top 5 Candidates for this job:")
            st.write(top_candidates)

        # Job Description Optimization
        st.subheader("Job Description Optimization")
        selected_job_for_optimization = st.selectbox("Select a job to optimize", 
                                                     [f"{job['id']} - {job['title']}" for job in job_postings],
                                                     key="job_optimization")
        if selected_job_for_optimization:
            job_id = int(selected_job_for_optimization.split('-')[0].strip())
            job = next(job for job in job_postings if job['id'] == job_id)

            if st.button("Optimize Job Description"):
                vector_db = create_vector_db([job, *candidates])
                query = f"""Based on the current job description for {job['title']} and the available talent pool,
                            suggest optimizations to the job description that could attract more qualified candidates.
                            Consider:
                            1. Skills that are abundant in the talent pool but not mentioned in the job description
                            2. Alternative phrasings for required qualifications that might resonate better with candidates
                            3. Additional preferred qualifications that could widen the talent pool"""
                optimized_description = query_vector_db(vector_db, query)
                st.write(optimized_description)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please try again later or contact support.")

def candidate_talent_marketplace():
    st.subheader("Personalized Job Recommendations and Skill Development")

    # Fetch candidate's data
    cursor.execute("""
        SELECT a.id, a.resume_text, jp.title as applied_job, jp.department, ctr.score
        FROM applications a
        JOIN job_postings jp ON a.job_id = jp.id
        LEFT JOIN candidate_test_responses ctr ON a.id = ctr.application_id
        WHERE a.applicant_name = %s
    """, (st.session_state.user['full_name'],))
    candidate_data = cursor.fetchall()

    # Fetch all job postings (removed the 'status' condition)
    cursor.execute("SELECT id, title, department, required_qualifications, preferred_qualifications FROM job_postings")
    job_postings = cursor.fetchall()

    # Create embeddings
    candidate_embedding = embeddings.embed_query(f"{candidate_data[0]['resume_text']} {candidate_data[0]['applied_job']}")
    job_embeddings = [embeddings.embed_query(f"{job['title']} {job['required_qualifications']} {job['preferred_qualifications']}") for job in job_postings]

    # Calculate similarity
    similarities = cosine_similarity([candidate_embedding], job_embeddings)[0]

    # Create a DataFrame for job recommendations
    df = pd.DataFrame({
        'Job Title': [job['title'] for job in job_postings],
        'Department': [job['department'] for job in job_postings],
        'Match Score': similarities
    })
    df = df.sort_values('Match Score', ascending=False).head(5)

    st.subheader("Top Job Recommendations")
    st.table(df)

    # Skill gap analysis and development plan
    if st.button("Generate Skill Development Plan"):
        vector_db = create_vector_db([*candidate_data, *job_postings])
        query = f"""Based on the candidate's resume and the requirements of their top job matches,
                    provide a skill development plan. Include:
                    1. Key skills the candidate should focus on developing
                    2. Suggested learning resources or courses for each skill
                    3. Estimated time investment for skill acquisition
                    4. How these skills align with their career trajectory and job market trends"""
        development_plan = query_vector_db(vector_db, query)
        st.write(development_plan)

    # AI-powered application enhancement
    st.subheader("Application Enhancement")
    selected_job = st.selectbox("Select a job to apply for", df['Job Title'])
    if st.button("Enhance Application"):
        job = next(job for job in job_postings if job['title'] == selected_job)
        vector_db = create_vector_db([candidate_data[0], job])
        query = f"""Based on the candidate's resume and the requirements for the {selected_job} role,
                    suggest enhancements to the application. Include:
                    1. Key experiences or skills to highlight in the cover letter
                    2. Suggestions for tailoring the resume to this specific role
                    3. Potential talking points for an interview
                    4. Any additional qualifications or projects that could strengthen the application"""
        application_tips = query_vector_db(vector_db, query)
        st.write(application_tips)

def generate_otp(length=6):
    """Generate a random OTP of specified length."""
    return ''.join(random.choices(string.digits, k=length))

def send_otp_email(email, otp):
    """Send an email with the OTP to the specified email address."""
    # Email configuration
    sender_email = "vtu19978.soc.cse@gmail.com"  # Replace with your email
    sender_password = "aocu ljua lofj jrvk"  # Replace with your email password
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server
    smtp_port = 587  # Replace with your SMTP port

    # Create the email message
    subject = "Your Password Reset OTP"
    body = f"""
    Dear User,

    Your One-Time Password (OTP) for password reset is: {otp}

    This OTP is valid for 5 minutes. Please do not share it with anyone.

    If you didn't request this password reset, please ignore this email.

    Best regards,
    Your Application Team
    """

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        print(f"OTP sent successfully to {email}")
        return True
    except Exception as e:
        print(f"Failed to send OTP email: {str(e)}")
        return False

def get_user_details(user_id):
    cursor.execute("""
        SELECT username, email, full_name, role, created_at, university
        FROM users
        WHERE id = %s
    """, (user_id,))
    user = cursor.fetchone()
    return user

def update_user_details(user_id, full_name):
    cursor.execute("""
        UPDATE users
        SET full_name = %s
        WHERE id = %s
    """, (full_name, user_id))
    db.commit()

def get_job_postings():
    cursor.execute("SELECT id, title FROM job_postings WHERE deadline >= CURDATE() ORDER BY deadline")
    jobs = cursor.fetchall()
    
    return [{"id": job["id"], "title": job['title']} for job in jobs]

def generate_interview_questions(resume_text, num_questions):
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    question_prompt = PromptTemplate(
        input_variables=["resume_text", "num_questions"],
        template=(
            "Based on the following resume, generate {num_questions} interview questions. "
            "Include a mix of questions about the candidate's experience, skills, "
            "and some general HR questions. Adapt the questions based on the "
            "candidate's background.\n\nResume:\n{resume_text}\n\n"
            "Provide the questions in a numbered list format."
        )
    )
    question_chain = LLMChain(llm=model, prompt=question_prompt)
    questions = question_chain.run(resume_text=resume_text, num_questions=num_questions)
    return questions.split("\n")

def ai_mock_interview():
    st.title("AI Mock Interview")

    # Add a reset button
    if st.button("Reset Interview"):
        for key in list(st.session_state.keys()):
            if key.startswith('interview_'):
                del st.session_state[key]
        st.rerun()

    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")
    
    if uploaded_file is not None:
        resume_text = parse_resume(uploaded_file)
        
        # Add input for number of questions
        num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=12)
        
        if 'interview_started' not in st.session_state:
            st.session_state.interview_started = False

        if not st.session_state.interview_started:
            if st.button("Start Mock Interview"):
                st.session_state.interview_started = True
                st.session_state.interview_questions = generate_interview_questions(resume_text, num_questions)
                st.session_state.interview_question_index = 0
                st.session_state.interview_responses = []
                st.rerun()
        else:
            conduct_interview(resume_text, st.session_state.interview_questions)

def conduct_interview(resume_text, questions):
    st.subheader("AI Mock Interview")
    st.write("Welcome to your AI mock interview. Please answer each question and click 'Next' to proceed.")

    # Initialize session state variables if they don't exist
    if 'interview_question_index' not in st.session_state:
        st.session_state.interview_question_index = 0
    if 'interview_responses' not in st.session_state:
        st.session_state.interview_responses = []

    if st.session_state.interview_question_index < len(questions):
        current_question = questions[st.session_state.interview_question_index]
        st.write(f"Question {st.session_state.interview_question_index + 1}: {current_question}")
        
        response = st.text_area("Your answer", key=f"response_{st.session_state.interview_question_index}")
        
        if st.button("Next"):
            st.session_state.interview_responses.append(response)
            
            if st.session_state.interview_question_index < len(questions) - 1:
                follow_up_prompt = PromptTemplate(
                    input_variables=["resume_text", "question", "response", "next_question"],
                    template=(
                        "Based on the candidate's resume and their response to the previous question, "
                        "determine if the next question is still relevant or if it should be adjusted. "
                        "If an adjustment is needed, provide a new, more relevant question.\n\n"
                        "Resume: {resume_text}\n"
                        "Previous Question: {question}\n"
                        "Candidate's Response: {response}\n"
                        "Next Planned Question: {next_question}\n\n"
                        "If the next question should be changed, provide a new question. "
                        "Otherwise, respond with 'Keep the original question.'"
                    )
                )
                follow_up_chain = LLMChain(llm=model, prompt=follow_up_prompt)
                result = follow_up_chain.run(
                    resume_text=resume_text,
                    question=current_question,
                    response=response,
                    next_question=questions[st.session_state.interview_question_index + 1]
                )
                
                if result.strip() != "Keep the original question.":
                    questions[st.session_state.interview_question_index + 1] = result.strip()

            st.session_state.interview_question_index += 1
            st.rerun()

    else:
        st.write("Interview completed. Click 'Evaluate Interview' to see your results.")
        if st.button("Evaluate Interview"):
            evaluation = evaluate_interview(resume_text, questions, st.session_state.interview_responses)
            st.subheader("Interview Evaluation")
            st.write(evaluation)
            st.download_button(
                label="Download Evaluation",
                data=evaluation,
                file_name="mock_interview_evaluation.txt",
                mime="text/plain"
            )
        
        if st.button("Start New Interview"):
            for key in list(st.session_state.keys()):
                if key.startswith('interview_'):
                    del st.session_state[key]
            st.rerun()

    return st.session_state.interview_responses

def evaluate_interview(resume_text, questions, responses):
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    evaluation_prompt = PromptTemplate(
        input_variables=["resume_text", "interview_qa"],
        template=(
            "You are an expert interviewer. Evaluate the following mock interview "
            "based on the candidate's resume and their responses to the interview questions. "
            "Provide a detailed analysis covering:\n"
            "1. Overall Performance (score out of 100 and explanation)\n"
            "2. Strengths Demonstrated\n"
            "3. Areas for Improvement\n"
            "4. Alignment with Resume\n"
            "5. Communication Skills\n"
            "6. Specific Feedback on Each Answer\n"
            "7. Recommendations for the Candidate\n\n"
            "Resume:\n{resume_text}\n\n"
            "Interview Q&A:\n{interview_qa}\n\n"
            "Ensure your evaluation is thorough, impartial, and constructive. "
            "Pay close attention to the content and quality of each response. "
            "Consider how well the candidate answered each specific question, "
            "and whether their answers demonstrate the skills and experience "
            "required for the position. If a candidate's answer is insufficient "
            "or off-topic, mention this in your evaluation."
        )
    )
    
    interview_qa = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, responses)])
    
    evaluation_chain = LLMChain(llm=model, prompt=evaluation_prompt)
    evaluation = evaluation_chain.run(
        resume_text=resume_text,
        interview_qa=interview_qa
    )
    
    return evaluation(interview_qa=interview_qa)
    
    return evaluation

def evaluate_interview(resume_text, questions, responses):
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    evaluation_prompt = PromptTemplate(
        input_variables=["resume_text", "interview_qa"],
        template=(
            "You are an expert interviewer. Evaluate the following mock interview "
            "based on the candidate's resume and their responses to the interview questions. "
            "Provide a detailed analysis covering:\n"
            "1. Overall Performance (score out of 100 and explanation)\n"
            "2. Strengths Demonstrated\n"
            "3. Areas for Improvement\n"
            "4. Alignment with Resume\n"
            "5. Communication Skills\n"
            "6. Specific Feedback on Each Answer\n"
            "7. Recommendations for the Candidate\n\n"
            "Resume:\n{resume_text}\n\n"
            "Interview Q&A:\n{interview_qa}\n\n"
            "Ensure your evaluation is thorough, impartial, and constructive. "
            "Pay close attention to the content and quality of each response. "
            "Consider how well the candidate answered each specific question, "
            "and whether their answers demonstrate the skills and experience "
            "required for the position. If a candidate's answer is insufficient "
            "or off-topic, mention this in your evaluation."
        )
    )
    
    interview_qa = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, responses)])
    
    evaluation_chain = LLMChain(llm=model, prompt=evaluation_prompt)
    evaluation = evaluation_chain.run(
        resume_text=resume_text,
        interview_qa=interview_qa
    )
    
    return evaluation
    evaluation_chain = LLMChain(llm=model, prompt=evaluation_prompt)
    evaluation = evaluation_chain.run(
        resume_text=resume_text,
        interview_qa=interview_qa
    )
    
    return evaluation

def ai_mock_interview():
    st.title("AI Mock Interview")

    # Add a reset button
    if st.button("Reset Interview"):
        for key in list(st.session_state.keys()):
            if key.startswith('interview_'):
                del st.session_state[key]
        st.rerun()

    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")
    
    if uploaded_file is not None:
        resume_text = parse_resume(uploaded_file)
        
        # Add input for number of questions
        num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=12)
        
        if 'interview_started' not in st.session_state:
            st.session_state.interview_started = False

        if not st.session_state.interview_started:
            if st.button("Start Mock Interview"):
                st.session_state.interview_started = True
                st.session_state.interview_questions = generate_interview_questions(resume_text, num_questions)
                st.session_state.interview_question_index = 0
                st.session_state.interview_responses = []
                st.rerun()
        else:
            conduct_interview(resume_text, st.session_state.interview_questions)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def clean_json_string1(json_string):
    json_string = re.sub(r'```json\s*|\s*```', '', json_string)
    json_string = json_string.strip()
    return json_string

def generate_interview_questions1(resume_text, num_technical, num_hr, num_resume):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    
    prompt = PromptTemplate.from_template(
        """Based on the following resume, generate {num_technical} technical questions, 
        {num_hr} HR questions, and {num_resume} resume-based questions. 
        The technical questions should be challenging and related to the skills mentioned in the resume. 
        Include at least one coding question if applicable.
        
        Resume:
        {resume_text}
        
        Format the output as a JSON object with keys 'technical', 'hr', and 'resume', 
        each containing an array of question strings. Do not include any markdown formatting in your response."""
    )
    
    formatted_prompt = prompt.format(
        num_technical=num_technical,
        num_hr=num_hr,
        num_resume=num_resume,
        resume_text=resume_text
    )
    
    response = model.predict(formatted_prompt)
    cleaned_response = clean_json_string1(response)
    
    try:
        questions = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        questions = {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_response": response,
            "cleaned_response": cleaned_response
        }
    
    return questions

def display_questions(questions):
    for category, category_questions in questions.items():
        st.subheader(f"{category.capitalize()} Questions")
        for i, question in enumerate(category_questions, 1):
            st.write(f"{i}. {question}")
        st.write("---")
        
def main():
    st.set_page_config(page_title="TalentForge AI", page_icon="🎯", layout="wide")
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    .stApp {
        background-color: #f0f8ff;
        font-family: 'Poppins', sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 1.5rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(50,50,93,.11), 0 1px 3px rgba(0,0,0,.08);
    }
    .stButton > button:hover {
        background-color: #357ae8;
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50,50,93,.1), 0 3px 6px rgba(0,0,0,.08);
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.7rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74,144,226,0.2);
    }
    .stExpander > div > div > div > button {
        background-color: #f8fafc;
        border: none;
        border-radius: 10px;
        padding: 0.7rem;
        font-weight: 600;
        color: #2c5282;
        transition: all 0.3s ease;
    }
    .stExpander > div > div > div > button:hover {
        background-color: #edf2f7;
    }
    .nav-link {
        color: #4a5568 !important;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .nav-link:hover {
        color: #2c5282 !important;
        background-color: #edf2f7;
    }
    .nav-link.active {
        color: #2c5282 !important;
        font-weight: 700;
        background-color: #e2e8f0 !important;
    }
    .custom-card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    .job-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c5282;
        margin-bottom: 0.5rem;
    }
    .job-details {
        font-size: 0.9rem;
        color: #4a5568;
    }
    .custom-metric {
        background-color: #ebf4ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c5282;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4a5568;
        margin-top: 0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)

    def load_lottieurl(url: str):
        try:
            r = requests.get(url)
            r.raise_for_status() 
            return r.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading Lottie URL: {e}")
        except json.JSONDecodeError:
            st.error("Error decoding JSON from Lottie URL")
        return None
    
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
    lottie_json = load_lottieurl(lottie_url)

    with st.sidebar:
        
        st.image("ten.png", width=300)
        
        if 'user' in st.session_state and st.session_state.user:
            st.write(f"Welcome, {st.session_state.user['full_name']}!")
            
            if st.session_state.user['role'] in ['Admin', 'HR Manager', 'Recruiter']:
                menu_items = {
                    "Job Management": ["Post Job", "Job Drafts", "Manage Categories"],
                    "Applications": ["Review Applications", "Applied Candidates", "In Review", "Selected Candidates"],
                    "Interviews": ["Interview Scheduling", "Upcoming Interviews", "Video Interviews1", "View Candidate Feedback"],
                    "Assessments": ["Evaluate Tests", "SpeechX Assessment"],
                    "Offers": ["Offers", "Onboarding"],
                    "Analytics": ["Analytics", "Historical Recruitment Analysis"],
                    "Talent Marketplace": ["AI-Driven Talent Marketplace"],
                    "Profile": ["View Profile"]
                }
            elif st.session_state.user['role'] == 'Interviewer':
                menu_items = {
                    "Interviews": ["Upcoming Interviews", "Provide Feedback"],
                    "Analysis": ["GitHub Profile Analysis", "Generate Interview Questions"],
                    "Profile": ["View Profile"]
                }
            else:  # Candidate
                menu_items = {
                    "Jobs": ["View Job Postings"],
                    "Applications": ["Submit Application", "Application Status", "AI-Driven Talent Marketplace"],
                    "Assessments": ["Take Test", "SpeechX Assessment"],
                    "Interviews": ["Upcoming Interviews1","Video Interviews", "Interview Feedback", "AI-Mock Interview"],
                    "Offers": ["View Offers"],
                    "Profile": ["View Profile"]
                }

            selected_menu = option_menu(
                menu_title="Navigation",
                options=list(menu_items.keys()),
                icons=["briefcase", "file-earmark-text", "camera-video", "clipboard-check", "envelope", "graph-up"],
                menu_icon="list",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#ffffff"},
                    "icon": {"color": "#4a90e2", "font-size": "1rem"}, 
                    "nav-link": {"font-size": "0.9rem", "text-align": "left", "margin":"0px", "--hover-color": "#edf2f7"},
                    "nav-link-selected": {"background-color": "#e2e8f0"},
                }
            )

            selected_submenu = option_menu(
                menu_title=None,
                options=menu_items[selected_menu],
                icons=["chevron-right"] * len(menu_items[selected_menu]),
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#ffffff"},
                    "icon": {"color": "#718096", "font-size": "0.8rem"}, 
                    "nav-link": {"font-size": "0.8rem", "text-align": "left", "margin":"0px", "--hover-color": "#edf2f7"},
                    "nav-link-selected": {"background-color": "#e2e8f0"},
                }
            )

            if st.button("Logout", key="logout_button"):
                st.session_state.user = None
                st.rerun()
        else:
            st.info("Please login to access the system.")

    # Main content area
    if 'user' not in st.session_state or st.session_state.user is None:
        
        st.markdown("<h1 class='main-header'>Welcome to TalentForge AI</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            auth_option = option_menu(
                menu_title=None,
                options=["Login", "Register", "Forgot Password"],
                icons=["box-arrow-in-right", "person-plus", "key"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
                styles={
                    "container": {"padding": "0!important", "background-color": "#ffffff"},
                    "icon": {"color": "#4a90e2", "font-size": "1rem"}, 
                    "nav-link": {"font-size": "0.9rem", "text-align": "center", "margin":"0px", "--hover-color": "#edf2f7"},
                    "nav-link-selected": {"background-color": "#e2e8f0"},
                }
            )

            if auth_option == "Login":
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    submit_button = st.form_submit_button("Login")
                    if submit_button:
                        user = authenticate_user(username, password)
                        if user:
                            st.session_state.user = user
                            st.success(f"Welcome, {user['full_name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")

            elif auth_option == "Register":
                with st.form("register_form"):
                    username = st.text_input("Username")
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    full_name = st.text_input("Full Name")
                    role = st.selectbox("Role", ["Candidate", "Recruiter", "Interviewer", "HR Manager", "Admin"])
                    
                    # Only show university field if role is Candidate
                    university = st.text_input("University Name") if role == "Candidate" else None
                    
                    submit_button = st.form_submit_button("Register")
                    
                    if submit_button:
                        if not username or not email or not password or not full_name:
                            st.error("Please fill in all required fields.")
                        else:
                            result = create_user(username, email, password, full_name, role, university)
                            if result.startswith("Success"):
                                st.success(result)
                            else:
                                st.error(result)

            elif auth_option == "Forgot Password":
                email = st.text_input("Enter your email")
                if st.button("Send OTP"):
                    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                    user = cursor.fetchone()
                    print(f"User data: {user}")
                    if user:
                        user_id = user['id']  # Now we're sure user is not None
                        user_id = user[0] if isinstance(user, tuple) else user['id']
                        print(f"User ID type: {type(user_id)}, value: {user_id}")
                        otp = generate_otp()
                        otp_created_at = datetime.now()
                        otp_expires_at = otp_created_at + timedelta(minutes=5)
                        
                        # Update user record with OTP information
                        cursor.execute("""
                            UPDATE users 
                            SET otp = %s, otp_created_at = %s, otp_expires_at = %s 
                            WHERE id = %s
                        """, (otp, otp_created_at, otp_expires_at, user_id))
                        db.commit()
                        
                        # Send OTP to user's email
                        if send_otp_email(email, otp):
                            st.success("OTP sent to your email. Valid for 5 minutes.")
                            st.session_state.reset_email = email
                        else:
                            st.error("Failed to send OTP. Please try again.")
                    else:
                        st.error("No user found with this email address.")
                
                # OTP verification and password reset
                if 'reset_email' in st.session_state:
                    otp = st.text_input("Enter OTP")
                    new_password = st.text_input("Enter new password", type="password")
                    confirm_password = st.text_input("Confirm new password", type="password")
                    
                    if st.button("Reset Password"):
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        else:
                            cursor.execute("""
                                SELECT id FROM users
                                WHERE email = %s AND otp = %s AND otp_expires_at > NOW()
                            """, (st.session_state.reset_email, otp))
                            user = cursor.fetchone()
                            
                            if user:
                                user_id = user['id']
                                hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                                
                                cursor.execute("""
                                    UPDATE users 
                                    SET password = %s, otp = NULL, otp_created_at = NULL, otp_expires_at = NULL 
                                    WHERE id = %s
                                """, (hashed_password, user_id))
                                
                                db.commit()
                                st.success("Password reset successfully. Please login with your new password.")
                                del st.session_state.reset_email
                            else:
                                st.error("Invalid or expired OTP")
    else:
        st.markdown(f"<h1 class='main-header'>{selected_submenu}</h1>", unsafe_allow_html=True)

        if selected_submenu == "Post Job":
            st.markdown("<h2>Create Job Posting</h2>", unsafe_allow_html=True)
            
            if 'editing_draft' in st.session_state:
                draft = get_job_draft(st.session_state.editing_draft)
                del st.session_state.editing_draft
            else:
                draft = None

            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Job Title", value=draft['title'] if draft else "")
                department = st.selectbox("Department", get_categories("Department"), index=get_categories("Department").index(draft['department']) if draft else 0)
                location = st.selectbox("Location", get_categories("Location"), index=get_categories("Location").index(draft['location']) if draft else 0)
                campus_type = st.selectbox("Campus Type", ["On Campus", "Off Campus"], index=["On Campus", "Off Campus"].index(draft['campus_type']) if draft and 'campus_type' in draft else 1)
            
            with col2:
                employment_type = st.selectbox("Employment Type", get_categories("Employment Type"), index=get_categories("Employment Type").index(draft['employment_type']) if draft else 0)
                salary_range = st.text_input("Salary Range", value=draft['salary_range'] if draft else "")
                experience = st.text_input("Required Experience", value=draft['experience'] if draft and 'experience' in draft else "")
                deadline = st.date_input("Application Deadline")
                deadline_time = st.time_input("Deadline Time")

            if campus_type == "On Campus":
                university = st.text_input("University", value=draft['university'] if draft and 'university' in draft else "")
            else:
                university = None

            required_qualifications = st.text_area("Required Qualifications", value=draft['required_qualifications'] if draft else "")
            preferred_qualifications = st.text_area("Preferred Qualifications", value=draft['preferred_qualifications'] if draft else "")
            responsibilities = st.text_area("Responsibilities", value=draft['responsibilities'] if draft else "")

            if st.button("Generate Job Description", key="generate_description"):
                with st.spinner("Generating job description..."):
                    job_description = generate_job_description(title, department, required_qualifications, preferred_qualifications, responsibilities, experience)
                st.text_area("Generated Job Description", job_description, height=300)
            
            description = st.text_area("Job Description (Edit if needed)", value=draft['description'] if draft else "", height=300)

            deadline_datetime = datetime.combine(deadline, deadline_time)
            col1, col2, col3 = st.columns(3)
            
            if col1.button("Save as Draft", key="save_draft"):
                if draft:
                    update_job_draft(draft['id'], title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, campus_type, university)
                    st.success("Draft updated successfully!")
                else:
                    save_job_draft(title, description, department, location, employment_type, salary_range, experience, required_qualifications, preferred_qualifications, responsibilities, st.session_state.user['id'], campus_type, university)
                    st.success("Draft saved successfully!")
            
            if col2.button("Post Job"):
                job_id = create_job_posting(
                    title, description, department, location, employment_type, salary_range, 
                    experience, required_qualifications, preferred_qualifications, 
                    responsibilities, deadline_datetime, campus_type, university
                )
                st.success(f"Job posted successfully! Job ID: {job_id}")
                if draft:
                    cursor.execute("DELETE FROM job_drafts WHERE id = %s", (draft['id'],))
                    db.commit()
            
            if draft and col3.button("Discard Changes"):
                st.rerun()
        

        elif selected_submenu == "Upcoming Interviews1":
            st.subheader("📅 Your Upcoming Interviews")
            # Fetch upcoming interviews for the candidate
            cursor.execute("""
            SELECT 
                u.id AS user_id,
                u.email AS user_email,
                u.full_name AS candidate_name,
                a.id AS application_id,
                j.id AS job_id,
                j.title AS job_title,
                i.id AS interview_id,
                i.interview_date,
                i.interview_time,
                i.interview_mode,
                i.interview_location,
                i.interviewers
            FROM 
                users u
            JOIN 
                applications a ON u.email = a.email
            JOIN 
                job_postings j ON a.job_id = j.id
            JOIN 
                interviews i ON a.id = i.applicant_id
            WHERE 
                u.id = %s 
                AND i.interview_date >= CURDATE() and i.status!='Completed'
            ORDER BY 
                i.interview_date, i.interview_time
            """, (st.session_state.user['id'],))
            upcoming_interviews = cursor.fetchall()

            if upcoming_interviews:
                for interview in upcoming_interviews:
                    with st.expander(f"🗓️ {interview['interview_date']} - {interview['job_title']}"):
                        # Convert timedelta to a formatted string
                        interview_time = (datetime.min + interview['interview_time']).time()
                        st.write(f"🕒 Time: {interview_time.strftime('%I:%M %p')}")
                        st.write(f"🏢 Job: {interview['job_title']}")
                        st.write(f"🤝 Mode: {interview['interview_mode']}")
                        if interview['interview_location']:
                            st.write(f"📍 Location: {interview['interview_location']}")
                        
                        # Fetch interviewer names
                        interviewer_ids = json.loads(interview['interviewers'])
                        if interviewer_ids:
                            placeholders = ', '.join(['%s'] * len(interviewer_ids))
                            query = f"SELECT full_name FROM users WHERE id IN ({placeholders})"
                            cursor.execute(query, tuple(interviewer_ids))
                            interviewer_names = [row['full_name'] for row in cursor.fetchall()]
                            st.write(f"👥 Interviewers: {', '.join(interviewer_names)}")
                        else:
                            st.write("👥 Interviewers: Not assigned yet")
                        
                        # Add preparation tips
                        st.write("### 📚 Preparation Tips")
                        st.write("- Research the company and role")
                        st.write("- Review common interview questions")
                        st.write("- Prepare questions for the interviewers")
                        st.write("- Test your equipment if it's a video interview")
                        st.write("- Plan your outfit and route to the interview location")

                        # Add option to reschedule or cancel
                        col1, col2 = st.columns(2)
                        if col1.button("📅 Request Reschedule", key=f"reschedule_{interview['interview_id']}"):
                            st.session_state.reschedule_request = interview['interview_id']
                        if col2.button("❌ Cancel Interview", key=f"cancel_{interview['interview_id']}"):
                            st.session_state.cancel_request = interview['interview_id']

                # Handle reschedule request
                if 'reschedule_request' in st.session_state:
                    st.write("### 📅 Request Reschedule")
                    new_date = st.date_input("Preferred Date", min_value=datetime.now().date())
                    new_time = st.time_input("Preferred Time")
                    reason = st.text_area("Reason for Rescheduling")
                    if st.button("Submit Reschedule Request"):
                        # Here you would typically send this request to HR for approval
                        st.success("Your reschedule request has been submitted for approval.")
                        del st.session_state.reschedule_request

                # Handle cancel request
                if 'cancel_request' in st.session_state:
                    st.write("### ❌ Cancel Interview")
                    reason = st.text_area("Reason for Cancellation")
                    if st.button("Confirm Cancellation"):
                        # Here you would typically update the interview status and notify HR
                        st.success("Your interview has been cancelled. We've notified the HR team.")
                        del st.session_state.cancel_request

            else:
                st.info("📭 You have no upcoming interviews scheduled at this time.")

        elif selected_submenu == "SpeechX Assessment":
            if st.session_state.user['role'] == 'Candidate':
                application = execute_query("SELECT id, speechx_status FROM applications WHERE email = %s ORDER BY id DESC LIMIT 1", (st.session_state.user['email'],))
                if application:
                    application = application[0]  # Get the first (and only) result
                    if application['speechx_status'] != 'completed':
                        speechx_assessment(application['id'])
                    else:
                        st.info("You have already completed the SpeechX Assessment.")
                else:
                    st.warning("No active application found. Please apply for a job first.")
            
            elif st.session_state.user['role'] in ['Admin', 'HR Manager', 'Recruiter']:
                st.subheader("SpeechX Assessment Management")
                job_options = get_job_postings()
                if job_options:
                    selected_job = st.selectbox(
                        "Select Job",
                        options=job_options,
                        format_func=lambda x: x['title']
                    )
                    job_id = selected_job['id']
                    job_title = selected_job['title']
                    
                    if st.button("Generate SpeechX Assessment"):
                        assessment_id = generate_speechx_questions(job_id)
                        if assessment_id:
                            st.success(f"SpeechX Assessment generated for Job: {job_title}")
                        else:
                            st.error("Failed to generate SpeechX Assessment. Please try again.")
                    
                    # Add a summary of SpeechX responses
                    response_summary = execute_query("""
                        SELECT 
                            COUNT(*) as total_responses,
                            SUM(CASE WHEN score IS NULL THEN 1 ELSE 0 END) as unanalyzed_responses,
                            COUNT(DISTINCT application_id) as total_applicants
                        FROM speechx_responses sr
                        JOIN applications a ON sr.application_id = a.id
                        WHERE a.job_id = %s
                    """, (job_id,))
                    
                    if response_summary and response_summary[0]['total_responses'] > 0:
                        st.write("### SpeechX Response Summary")
                        st.write(f"Total Applicants: {response_summary[0]['total_applicants']}")
                        st.write(f"Total Responses: {response_summary[0]['total_responses']}")
                        st.write(f"Unanalyzed Responses: {response_summary[0]['unanalyzed_responses']}")
                    else:
                        st.info("No SpeechX responses found for this job yet.")
                    
                    analyze_speechx_responses(job_id)
                else:
                    st.warning("No active job postings found.")
            else:
                st.warning("You don't have permission to access this feature.")

        elif selected_submenu == "AI-Mock Interview":
            ai_mock_interview()


        elif selected_submenu == "View Candidate Feedback":
            st.subheader("👀 Candidate Interview Feedback")

            # Fetch all feedback
            cursor.execute("""
            SELECT cf.*, a.applicant_name, j.title as job_title
            FROM candidate_feedback cf
            JOIN interviews i ON cf.interview_id = i.id
            JOIN applications a ON i.applicant_id = a.id
            JOIN job_postings j ON i.job_id = j.id
            ORDER BY cf.submitted_at DESC
            """)
            all_feedback = cursor.fetchall()

            if all_feedback:
                for feedback in all_feedback:
                    with st.expander(f"💼 {feedback['applicant_name']} - {feedback['job_title']}"):
                        rating_emoji = {
                            "Terrible": "😞",
                            "Bad": "🙁",
                            "Okay": "😐",
                            "Good": "🙂",
                            "Amazing": "😃"
                        }
                        st.write(f"Rating: {rating_emoji[feedback['rating']]} {feedback['rating']}")
                        st.write(f"💭 Feedback: {feedback['feedback_text']}")
                        st.write(f"⏰ Submitted at: {feedback['submitted_at']}")
            else:
                st.info("📭 No candidate feedback available at this time.")




        elif selected_submenu == "Interview Feedback":
            st.subheader("📝 Provide Interview Feedback")
            
            # Fetch completed interviews for the candidate
            cursor.execute("""
            SELECT i.id, j.title, i.interview_date, i.interview_time, u.id,i.applicant_id
            FROM users u
            JOIN applications a ON u.email = a.email
            JOIN interviews i ON a.job_id = i.applicant_id
            JOIN job_postings j ON i.job_id = j.id
            WHERE u.id = %s AND i.status = 'Completed' AND i.id NOT IN (
                SELECT interview_id FROM candidate_feedback
            )
            ORDER BY i.interview_date DESC, i.interview_time DESC
            """, (st.session_state.user['id'],))
            completed_interviews = cursor.fetchall()
            
            if completed_interviews:
                selected_interview = st.selectbox(
                    "Select Interview",
                    options=[f"{i['applicant_id']}: {i['title']} on {i['interview_date']} at {i['interview_time']}" for i in completed_interviews],
                    format_func=lambda x: x.split(": ", 1)[1]
                )
                interview_id = int(selected_interview.split(":")[0])
                print("interview_id",interview_id)

                st.write("## What do you think of the interview experience?")
                rating = st.select_slider(
                    "Rate your experience",
                    options=[
                        "😞 Terrible",
                        "🙁 Bad",
                        "😐 Okay",
                        "🙂 Good",
                        "😃 Amazing"
                    ],
                    value="😐 Okay"
                )

                feedback_text = st.text_area("💭 What are the main reasons for your rating?")

                if st.button("Submit Feedback"):
                    # Extract rating without emoji
                    rating_without_emoji = rating.split(" ", 1)[1]
                    
                    # Fetch the interview ID to confirm existence
                    cursor.execute("SELECT id FROM interviews WHERE applicant_id = %s", (interview_id,))
                    hello = cursor.fetchone() 
                    print("interview_id",hello)

                    cursor.execute("""
                    INSERT INTO candidate_feedback (interview_id, rating, feedback_text)
                    VALUES (%s, %s, %s)
                    """, (hello['id'], rating_without_emoji, feedback_text))
                    db.commit()
                    st.success("🎉 Thank you for your feedback!")
            else:
                st.info("📅 You have no completed interviews that require feedback at this time.")


        elif selected_submenu == "View Profile":
            
            # st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("User Profile")
            
            user = get_user_details(st.session_state.user['id'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Username:** {user['username']}")
                st.write(f"**Email:** {user['email']}")
                st.write(f"**Role:** {user['role']}")
                st.write(f"**Joined:** {user['created_at'].strftime('%Y-%m-%d')}")
                if user['role'] == 'Candidate':
                    st.write(f"**University:** {user['university']}")
            
            with col2:
                new_full_name = st.text_input("Full Name", value=user['full_name'])
            
            if st.button("Update Profile"):
                update_user_details(st.session_state.user['id'], new_full_name)
                st.success("Profile updated successfully!")
                st.session_state.user['full_name'] = new_full_name  # Update session state
                st.rerun()  # Refresh the page to show updated information
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a section for changing password
            # st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Change Password")
            
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Change Password"):
                # Verify current password
                cursor.execute("SELECT password FROM users WHERE id = %s", (st.session_state.user['id'],))
                stored_password = cursor.fetchone()['password']
                
                if bcrypt.checkpw(current_password.encode('utf-8'), stored_password.encode('utf-8')):
                    if new_password == confirm_new_password:
                        hashed_new_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                        cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_new_password, st.session_state.user['id']))
                        db.commit()
                        st.success("Password changed successfully!")
                    else:
                        st.error("New passwords do not match.")
                else:
                    st.error("Current password is incorrect.")
            
            st.markdown("</div>", unsafe_allow_html=True)

        elif selected_submenu == "AI-Driven Talent Marketplace":
            ai_talent_marketplace()

        elif selected_submenu == "Evaluate Tests":
            def safe_format_score(score):
                if score is None:
                    return 'N/A'
                try:
                    return f"{float(score):.2f}"
                except (ValueError, TypeError):
                    return str(score)

            st.subheader("Evaluate Candidate Tests")

            if 'evaluated_tests' not in st.session_state:
                st.session_state.evaluated_tests = {}

            cursor.execute("""
            SELECT a.id, a.applicant_name, j.id as job_id, j.title as job_title, a.test_score, a.test_status
            FROM applications a
            JOIN job_postings j ON a.job_id = j.id
            WHERE a.test_status = 'completed' AND (a.test_score IS NULL OR a.status = 'Applied')
            """)
            completed_tests = cursor.fetchall()

            st.write(f"Found {len(completed_tests)} tests to evaluate")

            if st.button("Evaluate All Tests"):
                with st.spinner("Evaluating all tests..."):
                    for test in completed_tests:
                        if test['id'] not in st.session_state.evaluated_tests:
                            try:
                                average_score, explanations = evaluate_candidate_answers(test['id'], test['job_id'])

                                st.session_state.evaluated_tests[test['id']] = {
                                    'score': average_score,
                                    'explanations': explanations
                                }

                                cursor.execute("UPDATE applications SET test_score = %s WHERE id = %s", (average_score, test['id']))
                                db.commit()

                            except Exception as e:
                                st.error(f"An error occurred during evaluation of {test['applicant_name']}'s test: {str(e)}")

                    st.success("All tests evaluated successfully!")

            if st.session_state.evaluated_tests:
                cutoff_score = st.number_input("Set cutoff score for all candidates", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

                if st.button("Update Statuses"):
                    for test_id, evaluation in st.session_state.evaluated_tests.items():
                        score = evaluation['score']
                        new_status = "In Review" if score >= cutoff_score else "Rejected"

                        cursor.execute("""
                            UPDATE applications 
                            SET status = %s, test_status = 'Evaluated', test_score = %s
                            WHERE id = %s
                        """, (new_status, score, test_id))
                        db.commit()

                    st.success("All statuses updated successfully in the database!")
                    st.session_state.evaluated_tests = {}  # Clear evaluated tests

            # Display results
            st.write("### Evaluation Results")
            for test in completed_tests:
                cursor.execute("""
                    SELECT test_score, status 
                    FROM applications 
                    WHERE id = %s
                """, (test['id'],))
                updated_data = cursor.fetchone()

                score = updated_data['test_score'] if updated_data else None
                status = updated_data['status'] if updated_data else "Not Evaluated"

                st.write(f"Candidate: {test['applicant_name']}, Job: {test['job_title']}")
                st.write(f"Score: {safe_format_score(score)}, Status: {status}")
                st.write("---")


        elif selected_submenu == "Applied Candidates":
                st.subheader("Applied Candidates")
                
                # Select job
                cursor.execute("SELECT id, title FROM job_postings")
                jobs = cursor.fetchall()
                selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs])
                job_id = int(selected_job.split(":")[0])
                
                # Get applied candidates
                cursor.execute("""
                SELECT * FROM applications 
                WHERE job_id = %s AND status = 'Applied'
                """, (job_id,))
                candidates = cursor.fetchall()
                
                if candidates:
                    st.write(f"Number of applied candidates: {len(candidates)}")
                    
                    # Choose between manual and AI-generated questions
                    question_type = st.radio("Question Type", ["Manual", "AI-generated"])
                    
                    if question_type == "Manual":
                        st.subheader("Add Manual Questions")
                        question_text = st.text_area("Enter question")
                        question_type = st.selectbox("Question Type", ["Multiple Choice", "Coding", "Open-ended"])
                        options = st.text_input("Options (comma-separated, for MCQ only)")
                        correct_answer = st.text_input("Correct Answer (for MCQ and Coding)")
                        
                        if st.button("Add Question"):
                            add_manual_question(job_id, question_text, question_type, options.split(',') if options else None, correct_answer)
                            st.success("Question added successfully!")
                    
                    elif question_type == "AI-generated":
                        mcq_count = st.number_input("Number of Multiple Choice Questions", min_value=0, value=2)
                        coding_count = st.number_input("Number of Coding Questions", min_value=0, value=1)
                        text_count = st.number_input("Number of Fundamentals Questions on the skills", min_value=0, value=1)
                        open_ended_count = st.number_input("Number of Open-ended Questions", min_value=0, value=1)
                        difficulty = st.selectbox("Question Difficulty", ["Easy", "Medium", "Hard"])
                        
                        if st.button("Generate AI Questions"):
                            generated_questions = generate_ai_questions(job_id, mcq_count, coding_count, text_count, open_ended_count, difficulty)
                            if generated_questions:
                                display_generated_questions(generated_questions)
                            st.success("AI questions generated successfully!")
                    
                    # Set test deadline
                    test_deadline_date = st.date_input("Set Test Deadline Date")
                    test_deadline_time = st.time_input("Set Test Deadline Time")
                    test_deadline = datetime.combine(test_deadline_date, test_deadline_time)
                    if st.button("Set Deadline and Notify Candidates"):
                        for candidate in candidates:
                            # Update application status and send notification
                            cursor.execute("UPDATE applications SET test_status = 'not_started' WHERE id = %s", (candidate['id'],))
                            send_notification(candidate['email'], "Test Invitation", f"You have been invited to take a test for the {selected_job.split(':')[1].strip()} position. Please complete the test by {test_deadline}.")
                        db.commit()
                        st.success("Candidates notified and test deadline set!")
                
                else:
                    st.write("No applied candidates found for this job.")


        elif selected_submenu == "Take Test":
            if st.session_state.user['role'] == 'Candidate':
                st.subheader("Candidate Test")

                cursor.execute("""
                SELECT a.id, j.title
                FROM applications a
                JOIN job_postings j ON a.job_id = j.id
                WHERE a.email = %s AND a.test_status = 'not_started'
                """, (st.session_state.user['email'],))
                available_tests = cursor.fetchall()

                if available_tests:
                    selected_test = st.selectbox("Select Test to Take", [f"{test['id']}: {test['title']}" for test in available_tests])
                    application_id = int(selected_test.split(":")[0])

                    if st.button("Start Test"):
                        session_id = start_test_session(application_id)
                        if session_id:
                            st.session_state.current_test = session_id
                            st.session_state.test_start_time = datetime.now()
                            st.rerun()
                        else:
                            st.error("Failed to start test. Please try again later.")

                    if 'current_test' in st.session_state:
                        test_id = st.session_state.current_test
                        cursor.execute("""
                        SELECT tq.id, tq.question_text, tq.options, tq.question_type
                        FROM test_questions tq
                        JOIN applications a ON tq.job_id = a.job_id
                        WHERE a.id = %s
                        """, (application_id,))
                        questions = cursor.fetchall()
                        
                        for question in questions:
                            st.write(question['question_text'])
                            
                            if question['question_type'] == 'Coding':
                                answer = st.text_area("Your Code:", key=f"question_{question['id']}_code", height=300)
                                st.write("Example solution:")
                                st.code("""
                        def sum_even_numbers(arr):
                            return sum(num for num in arr if num % 2 == 0)
                                """, language="python")
                            elif question['options']:
                                try:
                                    options = json.loads(question['options'])
                                    if options:
                                        answer = st.radio("Select your answer:", options, key=f"question_{question['id']}_radio")
                                    else:
                                        # If no options are available, provide a text area instead
                                        answer = st.text_area("Your answer:", key=f"question_{question['id']}_text", height=150)
                                except json.JSONDecodeError:
                                    st.write("Error in question options format.")
                                    # Provide a text area in case of error
                                    answer = st.text_area("Your answer:", key=f"question_{question['id']}_text", height=150)
                            else:
                                # For any other type of question, always provide a text area
                                answer = st.text_area("Your answer:", key=f"question_{question['id']}_text", height=150)
                            
                            if answer is not None:
                                save_candidate_response(application_id, question['id'], answer)
                        
                        time_elapsed = datetime.now() - st.session_state.test_start_time
                        st.write(f"Time elapsed: {time_elapsed.total_seconds() // 60} minutes")
                        
                        if st.button("Submit Test") or time_elapsed > timedelta(minutes=30):
                            end_test_session(test_id)
                            cursor.execute("UPDATE applications SET test_status = 'completed' WHERE id = %s", (application_id,))
                            db.commit()
                            del st.session_state.current_test
                            del st.session_state.test_start_time
                            st.success("Test submitted successfully!")
                            st.rerun()
                else:
                    st.write("No tests available at the moment.")
            else:
                st.write("You don't have permission to access this section.")

        elif selected_submenu == "Interview Scheduling":
            st.subheader("Interview Scheduling Dashboard")

            if 'interview_action_performed' in st.session_state and st.session_state.interview_action_performed:
                del st.session_state.interview_action_performed
                st.rerun()

            def has_scheduled_interview(application_id):
                cursor.execute("""
                SELECT COUNT(*) as count
                FROM interviews
                WHERE applicant_id = %s AND interview_date >= CURDATE() AND status != 'Completed'
                """, (application_id,))
                result = cursor.fetchone()
                return result['count'] > 0

            def get_interviews():
                cursor.execute("""
                SELECT i.*, a.applicant_name, j.title as job_title
                FROM interviews i
                JOIN applications a ON i.applicant_id = a.id
                JOIN job_postings j ON i.job_id = j.id
                WHERE i.interview_date >= CURDATE() AND i.status != 'Completed'
                ORDER BY i.interview_date, i.interview_time
                """)
                return cursor.fetchall()

            # Upcoming Interviews Summary
            upcoming_interviews = get_interviews()
            st.metric("Upcoming Interviews", len(upcoming_interviews))

            # Interviews List Table
            st.subheader("Scheduled Interviews")
            for interview in upcoming_interviews:
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 3])
                col1.write(interview['applicant_name'])
                col2.write(interview['job_title'])
                col3.write(interview['interview_date'].strftime("%Y-%m-%d"))
                
                # Convert timedelta to string representation
                interview_time = (datetime.min + interview['interview_time']).time()
                col4.write(interview_time.strftime("%H:%M"))
                
                if col5.button("View", key=f"view_{interview['id']}"):
                    st.session_state.viewing_interview = interview['id']
                if col5.button("Reschedule", key=f"reschedule_{interview['id']}"):
                    st.session_state.rescheduling_interview = interview['id']
                if col5.button("Cancel", key=f"cancel_{interview['id']}"):
                    cancel_interview(interview['id'])
                    st.success("Interview cancelled successfully!")
                    st.session_state.interview_action_performed = True
                    st.rerun()

            if st.button("Schedule New Interview"):
                st.session_state.scheduling_new_interview = True

            # Interview Scheduling Screen
            if 'scheduling_new_interview' in st.session_state and st.session_state.scheduling_new_interview:
                st.subheader("Schedule New Interview")
                
                cursor.execute("""
                SELECT DISTINCT j.id, j.title
                FROM job_postings j
                JOIN applications a ON j.id = a.job_id
                WHERE a.status = 'Interview'
                """)
                jobs_with_interviews = cursor.fetchall()
                if jobs_with_interviews:
                    selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs_with_interviews])
                    job_id = int(selected_job.split(":")[0])

                    # Fetch candidates for the selected job who are in the 'Interview' stage
                    cursor.execute("""
                    SELECT id, applicant_name
                    FROM applications
                    WHERE job_id = %s AND status = 'Interview'
                    """, (job_id,))
                    interview_candidates = cursor.fetchall()

                    if interview_candidates:
                        selected_candidate = st.selectbox("Select Candidate", [f"{c['id']}: {c['applicant_name']}" for c in interview_candidates])
                        application_id = int(selected_candidate.split(":")[0])

                        interview_date = st.date_input("Interview Date")
                        interview_time = st.time_input("Interview Time")

                        # Fetch users with interviewer roles
                        cursor.execute("""
                        SELECT id, full_name 
                        FROM users 
                        WHERE role IN ('Interviewer', 'HR Manager')
                        """)
                        interviewers = cursor.fetchall()

                        selected_interviewers = st.multiselect(
                            "Select Interviewers", 
                            options=[f"{i['id']}: {i['full_name']}" for i in interviewers],
                            format_func=lambda x: x.split(": ")[1]
                        )

                        interview_mode = st.selectbox("Interview Mode", ["In-Person", "Video Call", "Phone Call"])
                        interview_location = st.text_input("Interview Location (if applicable)")

                        if st.button("Schedule Interview"):
                            if has_scheduled_interview(application_id):
                                st.error("An interview is already scheduled for this candidate.")
                            else:
                                # Extract interviewer IDs
                                interviewer_ids = [int(i.split(":")[0]) for i in selected_interviewers]

                                # Schedule the interview in the database
                                cursor.execute("""
                                INSERT INTO interviews (applicant_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location, status)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, 'Scheduled')
                                """, (application_id, job_id, interview_date, interview_time, json.dumps(interviewer_ids), interview_mode, interview_location))
                                db.commit()

                                # Fetch applicant email
                                cursor.execute("SELECT email FROM applications WHERE id = %s", (application_id,))
                                applicant_email = cursor.fetchone()['email']

                                # Fetch interviewer emails
                                placeholders = ', '.join(['%s'] * len(interviewer_ids))
                                cursor.execute(f"SELECT email FROM users WHERE id IN ({placeholders})", interviewer_ids)
                                interviewer_emails = [row['email'] for row in cursor.fetchall()]

                                # Prepare interview data
                                interview_data = {
                                    'job_title': selected_job.split(":")[1].strip(),
                                    'interview_datetime': datetime.combine(interview_date, interview_time),
                                    'interview_location': interview_location,
                                    'applicant_name': selected_candidate.split(":")[1].strip(),
                                    'applicant_email': applicant_email,
                                    'hr_email': 'hr@company.com',  # Replace with actual HR email
                                    'interviewer_emails': interviewer_emails
                                }

                                # Schedule interview (send emails and add to Google Calendar)
                                schedule_interview_with_notifications(interview_data)

                                st.success("Interview scheduled successfully! Emails sent and event added to Google Calendar.")
                                st.session_state.scheduling_new_interview = False
                                st.session_state.interview_action_performed = True
                                st.rerun()
                    else:
                        st.write("No candidates are currently in the 'Interview' stage for this job.")
                else:
                    st.write("No jobs currently have candidates in the 'Interview' stage.")

            # Interview Details Screen
            if 'viewing_interview' in st.session_state:
                interview_id = st.session_state.viewing_interview
                cursor.execute("""
                SELECT i.*, a.applicant_name, a.email, a.phone, j.title as job_title, j.department, j.location
                FROM interviews i
                JOIN applications a ON i.applicant_id = a.id
                JOIN job_postings j ON i.job_id = j.id
                WHERE i.id = %s
                """, (interview_id,))
                interview = cursor.fetchone()

                st.subheader("Interview Details")
                st.write(f"Applicant Name: {interview['applicant_name']}")
                st.write(f"Email: {interview['email']}")
                st.write(f"Phone: {interview['phone']}")
                st.write(f"Job Title: {interview['job_title']}")
                st.write(f"Department: {interview['department']}")
                st.write(f"Location: {interview['location']}")
                st.write(f"Interview Date: {interview['interview_date']}")
                
                # Convert timedelta to string representation
                interview_time = (datetime.min + interview['interview_time']).time()
                st.write(f"Interview Time: {interview_time.strftime('%H:%M')}")
                
                st.write(f"Interview Mode: {interview['interview_mode']}")
                if interview['interview_location']:
                    st.write(f"Interview Location: {interview['interview_location']}")
                
                # Fetch interviewer names
                interviewer_ids = json.loads(interview['interviewers'])
                cursor.execute("SELECT full_name FROM users WHERE id IN %s", (tuple(interviewer_ids),))
                interviewer_names = [row['full_name'] for row in cursor.fetchall()]
                st.write(f"Interviewers: {', '.join(interviewer_names)}")

                if st.button("Close"):
                    del st.session_state.viewing_interview

            # Reschedule Interview Screen
            if 'rescheduling_interview' in st.session_state:
                interview_id = st.session_state.rescheduling_interview
                st.subheader("Reschedule Interview")

                cursor.execute("SELECT applicant_id, job_id FROM interviews WHERE id = %s", (interview_id,))
                current_interview = cursor.fetchone()

                new_date = st.date_input("New Interview Date", min_value=datetime.now().date())
                new_time = st.time_input("New Interview Time")

                if st.button("Reschedule Interview"):
                    # Check for conflicts
                    cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM interviews
                    WHERE applicant_id = %s AND job_id = %s AND id != %s
                    AND interview_date = %s AND interview_time = %s
                    """, (current_interview['applicant_id'], current_interview['job_id'], interview_id, new_date, new_time))
                    conflict = cursor.fetchone()['count'] > 0

                    if conflict:
                        st.error("There is already an interview scheduled at this time for this candidate and job.")
                    else:
                        cursor.execute("UPDATE interviews SET interview_date = %s, interview_time = %s WHERE id = %s",
                                    (new_date, new_time, interview_id))
                        db.commit()
                        st.success("Interview rescheduled successfully!")
                        del st.session_state.rescheduling_interview
                        st.session_state.interview_action_performed = True
                        st.rerun()
                    
        elif selected_submenu == "Manage Categories":
            st.subheader("Manage Job Categories")
            
            category_type = st.selectbox("Category Type", ["Department", "Location", "Employment Type"])
            category_name = st.text_input("Category Name")
            
            if st.button("Add Category"):
                add_category(category_type, category_name)
                st.success(f"Added {category_name} to {category_type} categories")

        elif selected_submenu == "Review Applications":
            st.subheader("Review Applications")
            
            cursor.execute("SELECT id, title FROM job_postings")
            jobs = cursor.fetchall()
            selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs], key="review_applications_job_select")
            job_id = int(selected_job.split(":")[0])

            cursor.execute("""
    SELECT a.*, j.description, j.required_qualifications, j.preferred_qualifications
    FROM applications a
    JOIN job_postings j ON a.job_id = j.id
    WHERE a.job_id = %s AND a.status != 'Rejected'
    """, (job_id,))
            applications = cursor.fetchall()

            for app in applications:
                st.write(f"Applicant: {app['applicant_name']}")
                st.write(f"Email: {app['email']}")
                st.write(f"Status: {app['status']}")
                
                if st.button(f"Analyze Resume - {app['applicant_name']}"):
                    with st.spinner("Analyzing resume..."):
                        evaluation = evaluate_resume(
                            app['resume_text'],
                            app['description'],
                            app['required_qualifications'],
                            app['preferred_qualifications']
                        )
                        score = get_score(evaluation)
                        st.write(f"Match Score: {score}")
                        st.text_area("Resume Evaluation", evaluation, height=300)
                
                new_status = st.selectbox(f"Update Status for {app['applicant_name']}", 
                                        ["Applied", "In Review", "Interview", "Offered", "Rejected"],
                                        index=["Applied", "In Review", "Interview", "Offered", "Rejected"].index(app['status']))
                
                if new_status != app['status']:
                    if st.button(f"Update Status for {app['applicant_name']}"):
                        print("Hello")
                        update_application_status(app['id'], new_status)
                        st.success(f"Status updated for {app['applicant_name']}")
                        
                        if new_status == "Interview":
                            # Schedule interview
                            st.subheader(f"Schedule Interview for {app['applicant_name']}")
                            
                            interview_date = st.date_input("Interview Date")
                            interview_time = st.time_input("Interview Time")
                            
                            # Fetch users with interviewer roles for this specific job
                            cursor.execute("""
                            SELECT DISTINCT u.id, u.full_name 
                            FROM users u
                            WHERE u.role IN ('Interviewer', 'HR Manager')
                            """, (job_id,))
                            interviewers = cursor.fetchall()
                            
                            if interviewers:
                                selected_interviewers = st.multiselect(
                                    "Select Interviewers", 
                                    options=[f"{i['id']}: {i['full_name']}" for i in interviewers],
                                    format_func=lambda x: x.split(": ")[1]
                                )
                                
                                interview_mode = st.selectbox("Interview Mode", ["In-Person", "Video Call", "Phone Call"])
                                interview_location = st.text_input("Interview Location (if applicable)")

                                if st.button("Schedule Interview"):
                                    # Extract interviewer IDs
                                    interviewer_ids = [int(i.split(":")[0]) for i in selected_interviewers]
                                    
                                    # Schedule the interview
                                    cursor.execute("""
                                    INSERT INTO interviews (applicant_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    """, (app['id'], job_id, interview_date, interview_time, json.dumps(interviewer_ids), interview_mode, interview_location))
                                    db.commit()
                                    
                                    st.success("Interview scheduled successfully!")
                            else:
                                st.error("No interviewers assigned to this job. Please assign interviewers first.")


                if st.button(f"Select {app['applicant_name']}"):
                    add_to_selected_list(app['id'])
                    send_notification(app['email'], "Application Update", "Congratulations! You have been selected for further consideration.")
                    st.success(f"{app['applicant_name']} added to the selected list.")

                st.write("---")


    # Add this new elif block for the "Schedule Interview" option
        elif selected_submenu == "Schedule Interview":
            st.subheader("Schedule Interview")

            # Fetch jobs with candidates in the 'Interview' stage
            cursor.execute("""
            SELECT DISTINCT j.id, j.title
            FROM job_postings j
            JOIN applications a ON j.id = a.job_id
            WHERE a.status = 'Interview'
            """)
            jobs_with_interviews = cursor.fetchall()

            if jobs_with_interviews:
                selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs_with_interviews])
                job_id = int(selected_job.split(":")[0])

                # Fetch candidates for the selected job who are in the 'Interview' stage
                cursor.execute("""
                SELECT id, applicant_name
                FROM applications
                WHERE job_id = %s AND status = 'Interview'
                """, (job_id,))
                interview_candidates = cursor.fetchall()

                if interview_candidates:
                    selected_candidate = st.selectbox("Select Candidate", [f"{c['id']}: {c['applicant_name']}" for c in interview_candidates])
                    application_id = int(selected_candidate.split(":")[0])

                    interview_date = st.date_input("Interview Date")
                    interview_time = st.time_input("Interview Time")

                    # Fetch users with interviewer roles
                    cursor.execute("""
                    SELECT id, full_name 
                    FROM users 
                    WHERE role IN ('Interviewer', 'HR Manager')
                    """)
                    interviewers = cursor.fetchall()

                    selected_interviewers = st.multiselect(
                        "Select Interviewers", 
                        options=[f"{i['id']}: {i['full_name']}" for i in interviewers],
                        format_func=lambda x: x.split(": ")[1]
                    )

                    interview_mode = st.selectbox("Interview Mode", ["In-Person", "Video Call", "Phone Call"])
                    interview_location = st.text_input("Interview Location (if applicable)")

                    if st.button("Schedule Interview"):
                        # Extract interviewer IDs
                        interviewer_ids = [int(i.split(":")[0]) for i in selected_interviewers]

                        # Schedule the interview
                        cursor.execute("""
                        INSERT INTO interviews (applicant_id, job_id, interview_date, interview_time, interviewers, interview_mode, interview_location)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (application_id, job_id, interview_date, interview_time, json.dumps(interviewer_ids), interview_mode, interview_location))
                        db.commit()

                        st.success("Interview scheduled successfully!")
                else:
                    st.write("No candidates are currently in the 'Interview' stage for this job.")
            else:
                st.write("No jobs currently have candidates in the 'Interview' stage.")

        elif selected_submenu == "Selected Candidates":
            st.subheader("Selected Candidates")
            selected_candidates = get_selected_candidates()
            for candidate in selected_candidates:
                st.write(f"Name: {candidate['applicant_name']}")
                st.write(f"Email: {candidate['email']}")
                st.write(f"Status: {candidate['status']}")
                st.write("---")

        elif selected_submenu == "Analytics":
            st.subheader("Recruitment Analytics Dashboard")
            
            # Job selection
            cursor.execute("SELECT id, title FROM job_postings")
            jobs = cursor.fetchall()
            selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs], key="analytics_job_select")
            job_id = int(selected_job.split(":")[0])

            # Fetch job-specific metrics
            cursor.execute("""
            SELECT 
                COUNT(*) as total_applications,
                SUM(CASE WHEN status = 'Offered' THEN 1 ELSE 0 END) as offers_made,
                AVG(DATEDIFF(CURDATE(), submission_date)) as avg_time_to_hire,
                SUM(CASE WHEN status = 'Rejected' THEN 1 ELSE 0 END) as rejections,
                SUM(CASE WHEN status = 'Interview' THEN 1 ELSE 0 END) as interviews_scheduled
            FROM applications
            WHERE job_id = %s
            """, (job_id,))
            metrics = cursor.fetchone()

            # Display key metrics in a more visually appealing way
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Applications", int(metrics['total_applications']) if metrics['total_applications'] is not None else 0)
            col2.metric("Offers Made", int(metrics['offers_made']) if metrics['offers_made'] is not None else 0)
            col3.metric("Avg. Time to Hire (days)", round(float(metrics['avg_time_to_hire']), 1) if metrics['avg_time_to_hire'] is not None else 0)
            col4.metric("Rejections", int(metrics['rejections']) if metrics['rejections'] is not None else 0)
            col5.metric("Interviews Scheduled", int(metrics['interviews_scheduled']) if metrics['interviews_scheduled'] is not None else 0)

            # Application status breakdown
            st.subheader("Application Status Breakdown")
            cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM applications
            WHERE job_id = %s
            GROUP BY status
            """, (job_id,))
            status_data = cursor.fetchall()
            
            df = pd.DataFrame(status_data)
            fig = px.pie(df, values='count', names='status', title='', hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Application timeline
            st.subheader("Application Timeline")
            cursor.execute("""
            SELECT DATE(submission_date) as date, COUNT(*) as count
            FROM applications
            WHERE job_id = %s
            GROUP BY DATE(submission_date)
            ORDER BY date
            """, (job_id,))
            timeline_data = cursor.fetchall()
            
            df_timeline = pd.DataFrame(timeline_data)
            fig_timeline = px.line(df_timeline, x='date', y='count', title='Daily Application Count')
            fig_timeline.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_timeline, use_container_width=True)

            st.subheader("Candidate Selection")
            selection_method = st.radio("Selection Method", ["Manual", "AI-based"])

            cursor.execute("SELECT id, title FROM job_postings")
            jobs = cursor.fetchall()
            selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs])
            job_id = int(selected_job.split(":")[0])

            if selection_method == "Manual":
                cursor.execute("""
                SELECT a.*, j.description, j.required_qualifications, j.preferred_qualifications
                FROM applications a
                JOIN job_postings j ON a.job_id = j.id
                WHERE a.job_id = %s AND a.status = 'Applied'
                """, (job_id,))
                applications = cursor.fetchall()

                for app in applications:
                    st.write(f"Applicant: {app['applicant_name']}")
                    st.write(f"Email: {app['email']}")
                    
                    selected = st.checkbox(f"Select {app['applicant_name']}", value=app['selected'])
                    if selected != app['selected']:
                        if selected:
                            add_to_selected_list(app['id'])
                            update_application_status(app['id'], "In Review")
                            st.success(f"{app['applicant_name']} added to the selected list.")
                        else:
                            remove_from_selected_list(app['id'])
                            update_application_status(app['id'], "Applied")
                            st.success(f"{app['applicant_name']} removed from the selected list.")

                    # Add button for resume analysis
                    if st.button(f"Analyze Resume - {app['applicant_name']}"):
                        with st.spinner("Analyzing resume..."):
                            evaluation = evaluate_resume(
                                app['resume_text'],
                                app['description'],
                                app['required_qualifications'],
                                app['preferred_qualifications']
                            )
                            score = get_score(evaluation)
                            st.write(f"Match Score: {score}")
                            st.text_area("Resume Evaluation", evaluation, height=300)

                    st.write("---")

            elif selection_method == "AI-based":
                num_candidates = st.number_input("Number of candidates to select", min_value=1, max_value=20, value=5)
                if st.button("Run AI Selection"):
                    selected_candidates = ai_select_candidates(job_id, num_candidates)
                    st.success(f"Selected {len(selected_candidates)} candidates based on AI evaluation.")
                    for candidate_id, score in selected_candidates:
                        cursor.execute("SELECT applicant_name, email FROM applications WHERE id = %s", (candidate_id,))
                        candidate = cursor.fetchone()
                        st.write(f"Selected: {candidate['applicant_name']} (Score: {score})")
                        st.write(f"Email: {candidate['email']}")
                        st.write("---")

        elif selected_submenu == "Upcoming Interviews":
            st.subheader("Upcoming Interviews")
            interviewer_id = st.session_state.user['id']
            interviews = get_upcoming_interviews(interviewer_id)
            
            if interviews:
                for index, interview in enumerate(interviews):
                    with st.expander(f"{interview['applicant_name']} - {interview['job_title']} ({interview['interview_date']})"):
                        st.write(f"Date: {interview['interview_date']}")
                        st.write(f"Time: {interview['interview_time']}")
                        st.write(f"Mode: {interview['interview_mode']}")
                        if interview['interview_location']:
                            st.write(f"Location: {interview['interview_location']}")
                        
                        # Generate PDF from resume text
                        if interview['resume_text']:
                            resume_pdf = create_pdf_from_text(interview['resume_text'])
                            resume_filename = f"{interview['applicant_name']}_resume.pdf"
                            
                            st.download_button(
                                label="Download Resume",
                                data=resume_pdf,
                                file_name=resume_filename,
                                mime="application/pdf",
                                key=f"download_resume_{interview['id']}_{index}"  # Unique key for each download button
                            )
                        else:
                            st.info("No resume available for download.")
                        
                        # Add feedback form
                        feedback = st.text_area("Interview Feedback", key=f"feedback_{interview['id']}")
                        rating = st.slider("Rating", 1, 5, 3, key=f"rating_{interview['id']}")
                        if st.button("Submit Feedback", key=f"submit_{interview['id']}"):
                            add_feedback(interview['id'], interviewer_id, feedback, rating)
                            st.success("Feedback submitted successfully!")
                            st.rerun()
            else:
                st.write("No upcoming interviews found.")

        elif selected_submenu == "Job Drafts":
            st.subheader("Job Drafts")
            
            cursor.execute("""
            SELECT id, title, department, created_at
            FROM job_drafts
            WHERE created_by = %s
            ORDER BY created_at DESC
            """, (st.session_state.user['id'],))
            drafts = cursor.fetchall()

            if drafts:
                for draft in drafts:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    col1.write(f"{draft['title']} - {draft['department']}")
                    col2.write(draft['created_at'].strftime("%Y-%m-%d"))
                    
                    if col3.button("Edit", key=f"edit_{draft['id']}"):
                        st.session_state.editing_draft = draft['id']
                        st.rerun()
                    
                    if col3.button("Delete", key=f"delete_{draft['id']}"):
                        cursor.execute("DELETE FROM job_drafts WHERE id = %s", (draft['id'],))
                        db.commit()
                        st.success("Draft deleted successfully!")
                        st.rerun()
            else:
                st.write("No job drafts found.")

            if 'editing_draft' in st.session_state:
                draft_id = st.session_state.editing_draft
                cursor.execute("SELECT * FROM job_drafts WHERE id = %s", (draft_id,))
                draft = cursor.fetchone()
                
                # Pre-fill the form with draft data
                title = st.text_input("Job Title", value=draft['title'])
                description = st.text_area("Job Description", value=draft['description'])
                department = st.selectbox("Department", get_categories("Department"), index=get_categories("Department").index(draft['department']))
                location = st.selectbox("Location", get_categories("Location"), index=get_categories("Location").index(draft['location']))
                employment_type = st.selectbox("Employment Type", get_categories("Employment Type"), index=get_categories("Employment Type").index(draft['employment_type']))
                salary_range = st.text_input("Salary Range", value=draft['salary_range'])
                experience = st.text_input("Required Experience", value=draft['experience'] if 'experience' in draft else "")
                required_qualifications = st.text_area("Required Qualifications", value=draft['required_qualifications'])
                preferred_qualifications = st.text_area("Preferred Qualifications", value=draft['preferred_qualifications'])
                responsibilities = st.text_area("Responsibilities", value=draft['responsibilities'])
                campus_type = st.selectbox("Campus Type", ["On Campus", "Off Campus"], index=["On Campus", "Off Campus"].index(draft['campus_type']) if 'campus_type' in draft else 1)
                university = st.text_input("University", value=draft['university'] if 'university' in draft else "")

                if st.button("Update Draft"):
                    update_job_draft(draft_id, title, description, department, location, employment_type, salary_range, experience,
                                    required_qualifications, preferred_qualifications, responsibilities, campus_type, university)
                    st.success("Draft updated successfully!")
                    del st.session_state.editing_draft
                    st.rerun()

                if st.button("Post Job"):
                    deadline = st.date_input("Application Deadline")
                    deadline_time = st.time_input("Deadline Time")
                    deadline_datetime = datetime.combine(deadline, deadline_time)
                    job_id = create_job_posting(title, description, department, location, employment_type, salary_range, experience,
                                                required_qualifications, preferred_qualifications, responsibilities, deadline_datetime, campus_type, university)
                    st.success(f"Job posted successfully! Job ID: {job_id}")
                    cursor.execute("DELETE FROM job_drafts WHERE id = %s", (draft_id,))
                    db.commit()
                    del st.session_state.editing_draft
                    st.rerun()
        elif selected_submenu == "Offers":
            offers_section()

        elif selected_submenu == "Onboarding":
             onboarding_section()

        elif selected_submenu == "View Offers":
            view_offers_section(st.session_state.user['id'])


        elif selected_submenu == "Provide Feedback":
            st.subheader("Provide Interview Feedback")
            interviewer_id = st.session_state.user['id']
            cursor.execute("""
    SELECT i.id, a.applicant_name, j.title as job_title
    FROM interviews i
    JOIN applications a ON i.applicant_id = a.id
    JOIN job_postings j ON a.job_id = j.id
    WHERE JSON_CONTAINS(i.interviewers, %s)
    AND i.interview_date >= CURDATE()
    ORDER BY i.interview_date DESC
    """, (json.dumps(interviewer_id),))
            past_interviews = cursor.fetchall()

            if past_interviews:
                selected_interview = st.selectbox("Select Interview", [f"{i['applicant_name']} - {i['job_title']}" for i in past_interviews])
                interview_id = past_interviews[[f"{i['applicant_name']} - {i['job_title']}" for i in past_interviews].index(selected_interview)]['id']

                feedback_text = st.text_area("Feedback")
                rating = st.slider("Rating", 1, 5, 3)

                if st.button("Submit Feedback"):
                    add_feedback(interview_id, interviewer_id, feedback_text, rating)
                    st.success("Feedback submitted successfully!")
            else:
                st.write("No past interviews found.")

        elif selected_submenu == "View Job Postings":
            st.subheader("Available Job Postings")
            
            # Get the current user's university
            cursor.execute("SELECT university FROM users WHERE id = %s", (st.session_state.user['id'],))
            user_university = cursor.fetchone()['university']

            # Fetch job postings
            cursor.execute("""
            SELECT * FROM job_postings 
            WHERE deadline >= NOW() 
            AND (campus_type = 'Off Campus' OR (campus_type = 'On Campus' AND university = %s))
            ORDER BY created_at DESC
            """, (user_university,))
            jobs = cursor.fetchall()

            if 'viewing_job' in st.session_state:
                # Display job details
                job_id = st.session_state.viewing_job
                cursor.execute("SELECT * FROM job_postings WHERE id = %s", (job_id,))
                job = cursor.fetchone()

                if job:
                    st.subheader(f"Job Details: {job['title']}")
                    st.write(f"**Department:** {job['department']}")
                    st.write(f"**Location:** {job['location']}")
                    st.write(f"**Salary:** {job['salary_range']}")
                    st.write(f"**Deadline:** {job['deadline'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Description:** {job['description']}")
                    st.write(f"**Campus Type:** {job['campus_type']}")
                    if job['campus_type'] == 'On Campus':
                        st.write(f"**University:** {job['university']}")

                    if st.button("Back to Job Postings"):
                        del st.session_state.viewing_job
                        st.rerun()
                else:
                    st.write("Job not found.")

            else:
                # Display job postings list
                for job in jobs:
                    with st.expander(f"{job['title']} - {job['department']}"):
                        st.write(f"**Location:** {job['location']}")
                        st.write(f"**Salary:** {job['salary_range']}")
                        st.write(f"**Deadline:** {job['deadline'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Description:** {job['description']}")
                        st.write(f"**Campus Type:** {job['campus_type']}")
                        if job['campus_type'] == 'On Campus':
                            st.write(f"**University:** {job['university']}")
                        if st.button(f"View Details", key=f"view_details_{job['id']}"):
                            st.session_state.viewing_job = job['id']
                            st.rerun()


        elif selected_submenu == "Submit Application":
            st.subheader("Submit Job Application")
            
            # Get the current user's university
            cursor.execute("SELECT university FROM users WHERE id = %s", (st.session_state.user['id'],))
            user_university = cursor.fetchone()['university']

            # Fetch eligible job postings
            cursor.execute("""
            SELECT id, title, department, deadline, campus_type, university 
            FROM job_postings 
            WHERE deadline >= NOW() 
            AND (campus_type = 'Off Campus' OR (campus_type = 'On Campus' AND university = %s))
            """, (user_university,))
            jobs = cursor.fetchall()

            job_options = [f"{job['id']}: {job['title']} - {job['department']} (Deadline: {job['deadline'].strftime('%Y-%m-%d %H:%M')}) - {job['campus_type']}" for job in jobs]
            
            selected_job = st.selectbox("Select Job", job_options)
            job_id = int(selected_job.split(":")[0])

            # Check if the selected job is eligible for the current user
            cursor.execute("SELECT campus_type, university FROM job_postings WHERE id = %s", (job_id,))
            job_details = cursor.fetchone()
            
            if job_details['campus_type'] == 'On Campus' and job_details['university'] != user_university:
                st.error("You are not eligible to apply for this on-campus job.")
            else:
                applicant_name = st.text_input("Full Name")
                email = st.text_input("Email Address")
                phone = st.text_input("Phone Number")
                resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

                # Get custom application form
                application_form = get_application_form(job_id)
                form_responses = {}
                if application_form:
                    st.subheader("Additional Information")
                    form_fields = json.loads(application_form['form_fields'])
                    for field in form_fields:
                        if field['type'] == "Text":
                            form_responses[field['name']] = st.text_input(field['name'])
                        elif field['type'] == "Dropdown":
                            form_responses[field['name']] = st.selectbox(field['name'], field['options'])
                        elif field['type'] == "Checkbox":
                            form_responses[field['name']] = st.multiselect(field['name'], field['options'])

                if st.button("Submit Application"):
                    if resume_file:
                        submit_application(job_id, applicant_name, email, phone, resume_file, form_responses)
                        st.success("Application submitted successfully!")
                        
                        # Send confirmation email
                        send_notification(email, "Application Received", f"Thank you for applying to the position of {selected_job.split(':')[1].split('-')[0].strip()}. We have received your application and will review it shortly.")
                    else:
                        st.error("Please upload your resume.")

        elif selected_submenu == "In Review":
            if st.session_state.user['role'] in ['Admin', 'HR Manager', 'Recruiter']:
                st.subheader("Applications In Review")
                
                query = """
                SELECT a.id, a.applicant_name, j.title as job_title, f.feedback_text, f.rating, u.full_name as interviewer_name
                FROM applications a
                JOIN job_postings j ON a.job_id = j.id
                JOIN feedback f ON a.id = f.application_id
                JOIN users u ON f.interviewer_id = u.id
                WHERE a.status = 'In Review'
                ORDER BY a.submission_date DESC
                """
                cursor.execute(query)
                applications = cursor.fetchall()

                for app in applications:
                    with st.expander(f"{app['applicant_name']} - {app['job_title']}"):
                        st.write(f"Interviewer: {app['interviewer_name']}")
                        st.write(f"Feedback: {app['feedback_text']}")
                        st.write(f"Rating: {app['rating']}/5")

                        decision = st.selectbox("Decision", ["Select", "Offer", "Reject"], key=f"decision_{app['id']}")
                        if decision != "Select":
                            if st.button("Confirm Decision", key=f"confirm_{app['id']}"):
                                new_status = "Offered" if decision == "Offer" else "Rejected"
                                cursor.execute("UPDATE applications SET status = %s WHERE id = %s", (new_status, app['id']))
                                db.commit()
                                st.success(f"Application status updated to {new_status}")
                                st.rerun()
            else:
                st.write("You don't have permission to access this section.")

        elif selected_submenu == "Video Interviews1":
            st.subheader("Video Interviews")
        
            # Create new video interview request
            if st.checkbox("Create New Video Interview Request"):
                cursor.execute("SELECT id, applicant_name FROM applications WHERE status = 'In Review'")
                candidates = cursor.fetchall()
                selected_candidate = st.selectbox("Select Candidate", [f"{c['id']}: {c['applicant_name']}" for c in candidates])
                application_id = int(selected_candidate.split(":")[0])
            
                question = st.text_area("Enter the interview question")
                if st.button("Send Video Interview Request"):
                    create_video_interview_request(application_id, question)
                    st.success("Video interview request sent successfully!")

            # View and analyze completed video interviews
            st.subheader("Completed Video Interviews")
            
            # Get all job postings
            cursor.execute("SELECT id, title FROM job_postings")
            jobs = cursor.fetchall()
            selected_job = st.selectbox("Select Job", [f"{job['id']}: {job['title']}" for job in jobs])
            job_id = int(selected_job.split(":")[0])

            # Get all candidates with completed video interviews for the selected job
            cursor.execute("""
            SELECT vi.id, a.id as application_id, a.applicant_name, vi.status, vi.score
            FROM video_interviews vi
            JOIN applications a ON vi.application_id = a.id
            WHERE a.job_id = %s AND vi.status IN ('completed', 'analyzed')
            """, (job_id,))
            completed_interviews = cursor.fetchall()

            for interview in completed_interviews:
                with st.expander(f"{interview['applicant_name']} - Status: {interview['status']}"):
                    if interview['status'] == 'completed':
                        if st.button(f"Analyze Interview {interview['id']}"):
                            score, analysis = analyze_video_interview(interview['id'])
                            st.success(f"Analysis complete. Overall score: {score:.2f}")
                            st.text_area("Analysis Result", analysis, height=300)
                            
                            # Update the interview status
                            cursor.execute("UPDATE video_interviews SET status = 'analyzed', score = %s, analysis_result = %s WHERE id = %s",
                                        (score, analysis, interview['id']))
                            db.commit()
                    elif interview['status'] == 'analyzed':
                        st.write(f"Score: {interview['score']:.2f}")
                        cursor.execute("SELECT analysis_result FROM video_interviews WHERE id = %s", (interview['id'],))
                        analysis = cursor.fetchone()['analysis_result']
                        st.text_area("Analysis Result", analysis, height=300)
                    
                    # Manual selection
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Select for Interview", key=f"interview_{interview['application_id']}"):
                            cursor.execute("UPDATE applications SET status = 'Interview' WHERE id = %s", (interview['application_id'],))
                            db.commit()
                            st.success("Candidate selected for interview.")
                    with col2:
                        if st.button("Reject", key=f"reject_{interview['application_id']}"):
                            cursor.execute("UPDATE applications SET status = 'Rejected' WHERE id = %s", (interview['application_id'],))
                            db.commit()
                            st.success("Candidate rejected.")

            # AI-based selection
            st.subheader("AI-based Selection")
            cutoff_score = st.slider("Set cutoff score", 0.0, 10.0, 7.0, 0.1)
            if st.button("Apply AI Selection"):
                cursor.execute("""
                UPDATE applications a
                JOIN video_interviews vi ON a.id = vi.application_id
                SET a.status = CASE WHEN vi.score >= %s THEN 'Interview' ELSE 'Rejected' END
                WHERE a.job_id = %s AND vi.status = 'analyzed'
                """, (cutoff_score, job_id))
                db.commit()
                st.success("AI selection applied successfully.")

            # Display results after AI selection
            st.subheader("Selection Results")
            cursor.execute("""
            SELECT a.applicant_name, a.status, vi.score
            FROM applications a
            JOIN video_interviews vi ON a.id = vi.application_id
            WHERE a.job_id = %s AND vi.status = 'analyzed'
            ORDER BY vi.score DESC
            """, (job_id,))
            results = cursor.fetchall()
            
            for result in results:
                st.write(f"{result['applicant_name']} - Score: {result['score']:.2f} - Status: {result['status']}")

        

        elif selected_submenu == "Video Interviews":
            st.subheader("Video Interviews")
            
            cursor.execute("""
            SELECT vi.id, vi.question, vi.status
            FROM video_interviews vi
            JOIN applications a ON vi.application_id = a.id
            WHERE a.email = %s AND vi.status = 'pending'
            """, (st.session_state.user['email'],))
            pending_interviews = cursor.fetchall()
            
            if pending_interviews:
                st.write("You have pending video interviews. Please complete them:")
                for interview in pending_interviews:
                    st.write(f"Question: {interview['question']}")
                    if st.button(f"Take Interview {interview['id']}"):
                        record_video_interview(interview['id'])
            else:
                st.write("You have no pending video interviews at this time.")

        elif selected_submenu == "Historical Recruitment Analysis":
            historical_recruitment_analysis()

        elif selected_submenu == "Generate Interview Questions":
            st.subheader("Generate Interview Questions")
            
            uploaded_file = st.file_uploader("Upload Candidate Resume (PDF)", type="pdf")
            
            if uploaded_file is not None:
                resume_text = extract_text_from_pdf(uploaded_file)
                
                st.write("Resume successfully uploaded and processed.")
                
                num_technical = st.number_input("Number of Technical Questions", min_value=1, max_value=10, value=3)
                num_hr = st.number_input("Number of HR Questions", min_value=1, max_value=10, value=2)
                num_resume = st.number_input("Number of Resume-based Questions", min_value=1, max_value=10, value=2)
                
                if st.button("Generate Questions"):
                    with st.spinner("Generating questions..."):
                        questions = generate_interview_questions1(resume_text, num_technical, num_hr, num_resume)
                        
                    if "error" in questions:
                        st.error(f"Error generating questions: {questions['error']}")
                        st.subheader("Raw response:")
                        st.text(questions["raw_response"])
                        st.subheader("Cleaned response:")
                        st.text(questions["cleaned_response"])
                    else:
                        display_questions(questions)
                    
                    # Option to save questions
                    if st.button("Save Questions"):
                        # Here you would implement the logic to save the questions to your database
                        st.success("Questions saved successfully!")
            else:
                st.write("Please upload a resume to generate questions.")

        elif selected_submenu == "Application Status":
            st.subheader("Your Application Status")
            cursor.execute("""
            SELECT a.*, j.title as job_title
            FROM applications a
            JOIN job_postings j ON a.job_id = j.id
            WHERE a.email = %s
            ORDER BY a.submission_date DESC
            """, (st.session_state.user['email'],))
            applications = cursor.fetchall()

            for app in applications:
                st.write(f"Job: {app['job_title']}")
                st.write(f"Status: {app['status']}")
                st.write(f"Submitted: {app['submission_date']}")
                
                if app['status'] == 'Interview':
                    cursor.execute("SELECT * FROM interviews WHERE applicant_id = %s", (app['id'],))
                    interviews = cursor.fetchall()
                    for interview in interviews:
                        st.write(f"Interview scheduled for: {interview['interview_date']} at {interview['interview_time']}")
                        st.write(f"Mode: {interview['interview_mode']}")
                        if interview['interview_location']:
                            st.write(f"Location: {interview['interview_location']}")
                
                st.write("---")

        elif selected_submenu == "GitHub Profile Analysis":
            github_profile_analysis()
    

if __name__ == "__main__":
    main()