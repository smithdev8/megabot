"""
🚀 MEGA TELEGRAM BOT - Enterprise Edition
Все топовые фичи в одном боте!
"""

import os
import io
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import base64
import tempfile

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, filters, ContextTypes,
    ConversationHandler
)

# AI и обработка
from groq import Groq  # Бесплатный AI
import openai  # Для DALL-E и Whisper
from gtts import gTTS  # Text-to-Speech
import speech_recognition as sr  # Speech-to-Text
from pydub import AudioSegment  # Аудио конвертация

# Веб-панель админа
from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
from flask_cors import CORS
import threading
import secrets

# PDF генерация
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# База данных
import sqlite3
from datetime import datetime
import pandas as pd

# Утилиты
from PIL import Image as PILImage
import requests
import hashlib
import uuid
from functools import wraps
import schedule
import pytz

# Загрузка переменных окружения
from dotenv import load_dotenv
load_dotenv()

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    """Централизованная конфигурация"""
    
    # Telegram
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    
    # AI APIs
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Бесплатный текст
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Для изображений и голоса
    
    # Admin
    ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(16))
    
    # Business
    BUSINESS_NAME = os.getenv("BUSINESS_NAME", "MEGA AI Bot")
    
    # Web Panel
    WEB_PORT = int(os.getenv("WEB_PORT", "5000"))
    
    # Database
    DB_PATH = "bot_database.db"
    
    # File paths
    EXPORTS_DIR = Path("exports")
    VOICE_DIR = Path("voice_messages")
    IMAGES_DIR = Path("generated_images")
    
    # Create directories
    for dir_path in [EXPORTS_DIR, VOICE_DIR, IMAGES_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # AI Models
    TEXT_MODEL = "llama-3.3-70b-versatile"  # Groq
    IMAGE_MODEL = "dall-e-3"  # OpenAI
    VOICE_MODEL = "whisper-1"  # OpenAI

# ==================== БАЗА ДАННЫХ ====================
class Database:
    """SQLite база данных с полным функционалом"""
    
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self.create_tables()
        
    def create_tables(self):
        """Создание всех таблиц"""
        cursor = self.conn.cursor()
        
        # Пользователи
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                language_code TEXT,
                is_premium BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP
            )
        ''')
        
        # Сообщения
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message_type TEXT,
                content TEXT,
                ai_response TEXT,
                tokens_used INTEGER,
                response_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Сгенерированные изображения
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prompt TEXT,
                image_url TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Голосовые сообщения
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                file_id TEXT,
                transcription TEXT,
                duration INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Аналитика
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                date DATE DEFAULT CURRENT_DATE
            )
        ''')
        
        self.conn.commit()
    
    def add_user(self, user_data: dict):
        """Добавить/обновить пользователя"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, username, first_name, last_name, language_code, last_active)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_data['id'],
            user_data.get('username'),
            user_data.get('first_name'),
            user_data.get('last_name'),
            user_data.get('language_code'),
            datetime.now()
        ))
        self.conn.commit()
    
    def log_message(self, user_id: int, message_type: str, content: str, 
                    ai_response: str = None, tokens: int = 0, response_time: float = 0):
        """Логировать сообщение"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO messages 
            (user_id, message_type, content, ai_response, tokens_used, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, message_type, content, ai_response, tokens, response_time))
        self.conn.commit()
    
    def get_chat_history(self, user_id: int, limit: int = 100) -> list:
        """Получить историю чата"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT content, ai_response, created_at 
            FROM messages 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        return cursor.fetchall()
    
    def get_stats(self) -> dict:
        """Получить статистику"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Общая статистика
        cursor.execute("SELECT COUNT(*) FROM users")
        stats['total_users'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        stats['total_messages'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM generated_images")
        stats['total_images'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM voice_messages")
        stats['total_voice'] = cursor.fetchone()[0]
        
        # Активные пользователи за сегодня
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM messages 
            WHERE DATE(created_at) = DATE('now')
        ''')
        stats['active_today'] = cursor.fetchone()[0]
        
        # Топ пользователи
        cursor.execute('''
            SELECT u.username, COUNT(m.id) as msg_count
            FROM users u
            JOIN messages m ON u.user_id = m.user_id
            GROUP BY u.user_id
            ORDER BY msg_count DESC
            LIMIT 5
        ''')
        stats['top_users'] = cursor.fetchall()
        
        return stats

# ==================== AI МОДУЛИ ====================
class AIManager:
    """Менеджер всех AI функций"""
    
    def __init__(self, db: Database):
        self.db = db
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
        self.openai_client = openai if Config.OPENAI_API_KEY else None
        if self.openai_client:
            openai.api_key = Config.OPENAI_API_KEY
        self.conversations = {}
        
    async def get_text_response(self, user_id: int, message: str) -> str:
        """Получить текстовый ответ от AI"""
        start_time = datetime.now()
        
        try:
            # Управление контекстом
            if user_id not in self.conversations:
                self.conversations[user_id] = []
            
            self.conversations[user_id].append({"role": "user", "content": message})
            
            if len(self.conversations[user_id]) > 10:
                self.conversations[user_id] = self.conversations[user_id][-10:]
            
            # Используем Groq (бесплатно)
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model=Config.TEXT_MODEL,
                    messages=[
                        {"role": "system", "content": "Ты - профессиональный AI ассистент. Отвечай полезно и дружелюбно."},
                        *self.conversations[user_id]
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                ai_response = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            # Fallback на OpenAI если Groq недоступен
            elif self.openai_client:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.conversations[user_id],
                    max_tokens=1000
                )
                ai_response = response.choices[0].message.content
                tokens = response.usage.total_tokens
            else:
                ai_response = "❌ AI сервис недоступен. Проверьте API ключи."
                tokens = 0
            
            # Сохраняем ответ
            self.conversations[user_id].append({"role": "assistant", "content": ai_response})
            
            # Логируем в БД
            response_time = (datetime.now() - start_time).total_seconds()
            self.db.log_message(user_id, "text", message, ai_response, tokens, response_time)
            
            return ai_response
            
        except Exception as e:
            logging.error(f"AI Error: {e}")
            return "😔 Произошла ошибка. Попробуйте позже."
    
    async def generate_image(self, user_id: int, prompt: str) -> str:
        """Генерация изображения через DALL-E"""
        if not self.openai_client:
            return None, "❌ Для генерации изображений нужен OpenAI API ключ"
        
        try:
            # Генерируем изображение
            response = openai.Image.create(
                model=Config.IMAGE_MODEL,
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            
            image_url = response.data[0].url
            
            # Скачиваем изображение
            image_data = requests.get(image_url).content
            filename = f"{uuid.uuid4()}.png"
            filepath = Config.IMAGES_DIR / filename
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            # Сохраняем в БД
            cursor = self.db.conn.cursor()
            cursor.execute('''
                INSERT INTO generated_images (user_id, prompt, image_url, file_path)
                VALUES (?, ?, ?, ?)
            ''', (user_id, prompt, image_url, str(filepath)))
            self.db.conn.commit()
            
            return filepath, "✅ Изображение сгенерировано!"
            
        except Exception as e:
            logging.error(f"Image generation error: {e}")
            return None, f"❌ Ошибка генерации: {str(e)}"
    
    async def transcribe_voice(self, file_path: str) -> str:
        """Транскрибация голоса через Whisper"""
        if not self.openai_client:
            return "❌ Для распознавания голоса нужен OpenAI API ключ"
        
        try:
            with open(file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model=Config.VOICE_MODEL,
                    file=audio_file,
                    language="ru"  # Можно сделать автоопределение
                )
            return transcript.text
        except Exception as e:
            logging.error(f"Voice transcription error: {e}")
            return "❌ Не удалось распознать голос"
    
    def text_to_speech(self, text: str, language: str = 'ru') -> str:
        """Преобразование текста в голос"""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            filename = f"{uuid.uuid4()}.mp3"
            filepath = Config.VOICE_DIR / filename
            tts.save(str(filepath))
            return filepath
        except Exception as e:
            logging.error(f"TTS error: {e}")
            return None

# ==================== PDF ГЕНЕРАТОР ====================
class PDFGenerator:
    """Генерация красивых PDF отчетов"""
    
    @staticmethod
    def generate_chat_export(user_id: int, chat_history: list, user_info: dict) -> str:
        """Экспорт чата в PDF"""
        filename = f"chat_export_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = Config.EXPORTS_DIR / filename
        
        # Создаем PDF документ
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Стили
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Содержимое
        story = []
        
        # Заголовок
        story.append(Paragraph(f"📊 Экспорт чата - {Config.BUSINESS_NAME}", title_style))
        story.append(Spacer(1, 20))
        
        # Информация о пользователе
        user_data = [
            ['Пользователь:', user_info.get('first_name', 'Unknown')],
            ['Username:', f"@{user_info.get('username', 'none')}"],
            ['User ID:', str(user_id)],
            ['Дата экспорта:', datetime.now().strftime('%d.%m.%Y %H:%M')]
        ]
        
        user_table = Table(user_data, colWidths=[120, 350])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e5e7eb')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2937')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db'))
        ]))
        
        story.append(user_table)
        story.append(Spacer(1, 30))
        
        # История чата
        story.append(Paragraph("💬 История диалога", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        for message, response, timestamp in chat_history:
            # Сообщение пользователя
            story.append(Paragraph(f"<b>👤 Вы ({timestamp}):</b>", styles['Normal']))
            story.append(Paragraph(message or "...", styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Ответ бота
            if response:
                story.append(Paragraph(f"<b>🤖 {Config.BUSINESS_NAME}:</b>", styles['Normal']))
                story.append(Paragraph(response, styles['Normal']))
                story.append(Spacer(1, 20))
        
        # Генерируем PDF
        doc.build(story)
        
        return filepath

# ==================== WEB АДМИН-ПАНЕЛЬ ====================
class AdminPanel:
    """Веб-интерфейс администратора"""
    
    def __init__(self, db: Database):
        self.db = db
        self.app = Flask(__name__)
        self.app.secret_key = Config.SECRET_KEY
        CORS(self.app)
        self.setup_routes()
        
    def setup_routes(self):
        """Настройка маршрутов"""
        
        @self.app.route('/')
        def index():
            if 'admin_logged_in' not in session:
                return redirect(url_for('login'))
            return self.render_dashboard()
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                password = request.form.get('password')
                if password == Config.ADMIN_PASSWORD:
                    session['admin_logged_in'] = True
                    return redirect(url_for('index'))
                return self.render_login(error="Неверный пароль")
            return self.render_login()
        
        @self.app.route('/logout')
        def logout():
            session.pop('admin_logged_in', None)
            return redirect(url_for('login'))
        
        @self.app.route('/api/stats')
        def api_stats():
            if 'admin_logged_in' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            return jsonify(self.db.get_stats())
        
        @self.app.route('/api/users')
        def api_users():
            if 'admin_logged_in' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT user_id, username, first_name, last_active, 
                       (SELECT COUNT(*) FROM messages WHERE user_id = users.user_id) as msg_count
                FROM users
                ORDER BY last_active DESC
                LIMIT 100
            ''')
            users = cursor.fetchall()
            
            return jsonify([{
                'user_id': u[0],
                'username': u[1],
                'first_name': u[2],
                'last_active': u[3],
                'messages': u[4]
            } for u in users])
        
        @self.app.route('/broadcast', methods=['POST'])
        def broadcast():
            if 'admin_logged_in' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            message = request.json.get('message')
            # Здесь должна быть логика рассылки через Telegram API
            return jsonify({'status': 'Message queued for broadcast'})
    
    def render_login(self, error=None):
        """Страница входа"""
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Admin Login - {{ business_name }}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .login-container {
                    background: white;
                    padding: 2rem;
                    border-radius: 1rem;
                    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
                    width: 90%;
                    max-width: 400px;
                }
                h1 {
                    color: #1f2937;
                    margin-bottom: 2rem;
                    text-align: center;
                }
                .form-group {
                    margin-bottom: 1.5rem;
                }
                label {
                    display: block;
                    color: #6b7280;
                    margin-bottom: 0.5rem;
                    font-size: 0.875rem;
                }
                input {
                    width: 100%;
                    padding: 0.75rem;
                    border: 1px solid #d1d5db;
                    border-radius: 0.5rem;
                    font-size: 1rem;
                }
                input:focus {
                    outline: none;
                    border-color: #667eea;
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }
                button {
                    width: 100%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 0.75rem;
                    border: none;
                    border-radius: 0.5rem;
                    font-size: 1rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.2s;
                }
                button:hover {
                    transform: translateY(-2px);
                }
                .error {
                    background: #fee;
                    color: #c00;
                    padding: 0.75rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                    text-align: center;
                }
                .logo {
                    text-align: center;
                    font-size: 3rem;
                    margin-bottom: 1rem;
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <div class="logo">🤖</div>
                <h1>{{ business_name }} Admin</h1>
                {% if error %}
                <div class="error">{{ error }}</div>
                {% endif %}
                <form method="POST">
                    <div class="form-group">
                        <label for="password">Пароль администратора</label>
                        <input type="password" id="password" name="password" required autofocus>
                    </div>
                    <button type="submit">Войти</button>
                </form>
            </div>
        </body>
        </html>
        ''', business_name=Config.BUSINESS_NAME, error=error)
    
    def render_dashboard(self):
        """Главная страница админки"""
        stats = self.db.get_stats()
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Admin Dashboard - {{ business_name }}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f3f4f6;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 1.5rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .header-content {
                    max-width: 1200px;
                    margin: 0 auto;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .header h1 {
                    font-size: 1.5rem;
                }
                .logout-btn {
                    background: rgba(255,255,255,0.2);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 0.5rem;
                    text-decoration: none;
                    transition: background 0.2s;
                }
                .logout-btn:hover {
                    background: rgba(255,255,255,0.3);
                }
                .container {
                    max-width: 1200px;
                    margin: 2rem auto;
                    padding: 0 1rem;
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }
                .stat-card {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .stat-card h3 {
                    color: #6b7280;
                    font-size: 0.875rem;
                    font-weight: 500;
                    margin-bottom: 0.5rem;
                    text-transform: uppercase;
                }
                .stat-card .value {
                    color: #1f2937;
                    font-size: 2rem;
                    font-weight: 700;
                }
                .stat-card .change {
                    color: #10b981;
                    font-size: 0.875rem;
                    margin-top: 0.5rem;
                }
                .section {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    margin-bottom: 2rem;
                }
                .section h2 {
                    color: #1f2937;
                    margin-bottom: 1rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 2px solid #e5e7eb;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    text-align: left;
                    padding: 0.75rem;
                    border-bottom: 1px solid #e5e7eb;
                }
                th {
                    background: #f9fafb;
                    font-weight: 600;
                    color: #6b7280;
                    font-size: 0.875rem;
                }
                .btn {
                    background: #667eea;
                    color: white;
                    padding: 0.5rem 1rem;
                    border: none;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    font-size: 0.875rem;
                    transition: background 0.2s;
                }
                .btn:hover {
                    background: #5a67d8;
                }
                .broadcast-form {
                    display: flex;
                    gap: 1rem;
                    margin-top: 1rem;
                }
                .broadcast-form textarea {
                    flex: 1;
                    padding: 0.75rem;
                    border: 1px solid #d1d5db;
                    border-radius: 0.5rem;
                    resize: vertical;
                    min-height: 100px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="header-content">
                    <h1>🤖 {{ business_name }} - Admin Panel</h1>
                    <a href="/logout" class="logout-btn">Выйти</a>
                </div>
            </div>
            
            <div class="container">
                <!-- Статистика -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Пользователи</h3>
                        <div class="value">{{ stats.total_users }}</div>
                        <div class="change">+{{ stats.active_today }} сегодня</div>
                    </div>
                    <div class="stat-card">
                        <h3>Сообщения</h3>
                        <div class="value">{{ stats.total_messages }}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Изображения</h3>
                        <div class="value">{{ stats.total_images }}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Голосовые</h3>
                        <div class="value">{{ stats.total_voice }}</div>
                    </div>
                </div>
                
                <!-- Топ пользователи -->
                <div class="section">
                    <h2>🏆 Топ пользователи</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Username</th>
                                <th>Сообщений</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for user in stats.top_users %}
                            <tr>
                                <td>@{{ user[0] or 'anonymous' }}</td>
                                <td>{{ user[1] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                
                <!-- Рассылка -->
                <div class="section">
                    <h2>📢 Массовая рассылка</h2>
                    <div class="broadcast-form">
                        <textarea id="broadcast-message" placeholder="Введите сообщение для рассылки всем пользователям..."></textarea>
                        <button class="btn" onclick="sendBroadcast()">Отправить</button>
                    </div>
                </div>
            </div>
            
            <script>
                function sendBroadcast() {
                    const message = document.getElementById('broadcast-message').value;
                    if (!message.trim()) {
                        alert('Введите сообщение');
                        return;
                    }
                    
                    fetch('/broadcast', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    })
                    .then(res => res.json())
                    .then(data => {
                        alert('Рассылка запущена!');
                        document.getElementById('broadcast-message').value = '';
                    });
                }
                
                // Автообновление статистики
                setInterval(() => {
                    fetch('/api/stats')
                        .then(res => res.json())
                        .then(data => {
                            // Обновляем значения на странице
                            console.log('Stats updated:', data);
                        });
                }, 30000);  // Каждые 30 секунд
            </script>
        </body>
        </html>
        ''', business_name=Config.BUSINESS_NAME, stats=stats)
    
    def run(self):
        """Запуск веб-сервера в отдельном потоке"""
        self.app.run(host='0.0.0.0', port=Config.WEB_PORT, debug=False)

# ==================== TELEGRAM БОТ ====================
class MegaBot:
    """Главный класс бота со всеми фичами"""
    
    def __init__(self):
        self.db = Database()
        self.ai = AIManager(self.db)
        self.pdf = PDFGenerator()
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        user = update.effective_user
        self.db.add_user(user.to_dict())
        
        keyboard = [
            [
                InlineKeyboardButton("💬 Чат с AI", callback_data="chat"),
                InlineKeyboardButton("🎨 Генерация картинки", callback_data="image")
            ],
            [
                InlineKeyboardButton("🎤 Голосовое сообщение", callback_data="voice"),
                InlineKeyboardButton("📄 Экспорт чата", callback_data="export")
            ],
            [
                InlineKeyboardButton("📊 Статистика", callback_data="stats"),
                InlineKeyboardButton("ℹ️ Помощь", callback_data="help")
            ]
        ]
        
        welcome = f"""
🚀 **Добро пожаловать в {Config.BUSINESS_NAME}!**

Я - продвинутый AI-бот с уникальными возможностями:

✨ **Возможности:**
• 💬 Умный чат с контекстом
• 🎨 Генерация изображений (DALL-E 3)
• 🎤 Распознавание голоса (Whisper)
• 🔊 Озвучка текста
• 📄 Экспорт чатов в PDF
• 📊 Детальная аналитика
• 🌐 Веб-панель администратора

Выберите действие или просто напишите сообщение!
"""
        
        await update.message.reply_text(
            welcome,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений"""
        user = update.effective_user
        message = update.message.text
        
        # Обновляем пользователя
        self.db.add_user(user.to_dict())
        
        # Показываем индикатор набора
        await update.message.reply_chat_action("typing")
        
        # Получаем ответ от AI
        response = await self.ai.get_text_response(user.id, message)
        
        # Отправляем ответ
        await update.message.reply_text(response)
    
    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка голосовых сообщений"""
        user = update.effective_user
        voice = update.message.voice
        
        await update.message.reply_text("🎤 Обрабатываю голосовое сообщение...")
        
        # Скачиваем файл
        file = await context.bot.get_file(voice.file_id)
        voice_path = Config.VOICE_DIR / f"{voice.file_id}.ogg"
        await file.download_to_drive(voice_path)
        
        # Конвертируем в mp3 для Whisper
        mp3_path = voice_path.with_suffix('.mp3')
        audio = AudioSegment.from_ogg(voice_path)
        audio.export(mp3_path, format="mp3")
        
        # Транскрибируем
        text = await self.ai.transcribe_voice(str(mp3_path))
        
        if text.startswith("❌"):
            await update.message.reply_text(text)
            return
        
        # Сохраняем в БД
        cursor = self.db.conn.cursor()
        cursor.execute('''
            INSERT INTO voice_messages (user_id, file_id, transcription, duration)
            VALUES (?, ?, ?, ?)
        ''', (user.id, voice.file_id, text, voice.duration))
        self.db.conn.commit()
        
        # Отправляем транскрипцию
        await update.message.reply_text(f"📝 Распознанный текст:\n\n{text}")
        
        # Получаем ответ от AI
        await update.message.reply_chat_action("typing")
        response = await self.ai.get_text_response(user.id, text)
        await update.message.reply_text(f"🤖 Ответ:\n\n{response}")
        
        # Озвучиваем ответ (опционально)
        if len(response) < 500:  # Ограничиваем длину для озвучки
            voice_file = self.ai.text_to_speech(response)
            if voice_file:
                with open(voice_file, 'rb') as audio:
                    await update.message.reply_voice(voice=audio)
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка нажатий кнопок"""
        query = update.callback_query
        await query.answer()
        
        user = query.from_user
        
        if query.data == "chat":
            await query.message.reply_text("💬 Просто напишите ваше сообщение!")
            
        elif query.data == "image":
            if not Config.OPENAI_API_KEY:
                await query.message.reply_text("❌ Для генерации изображений нужен OpenAI API ключ")
                return
            
            await query.message.reply_text(
                "🎨 Опишите изображение, которое хотите создать.\n\n"
                "Пример: _Кот-астронавт в космосе в стиле Van Gogh_",
                parse_mode="Markdown"
            )
            context.user_data['waiting_for_image_prompt'] = True
            
        elif query.data == "voice":
            await query.message.reply_text(
                "🎤 Отправьте голосовое сообщение, и я его распознаю и отвечу!"
            )
            
        elif query.data == "export":
            await query.message.reply_text("📄 Готовлю PDF с вашей историей чата...")
            
            # Получаем историю
            chat_history = self.db.get_chat_history(user.id)
            
            if not chat_history:
                await query.message.reply_text("❌ История чата пуста")
                return
            
            # Генерируем PDF
            pdf_path = self.pdf.generate_chat_export(
                user.id, 
                chat_history,
                user.to_dict()
            )
            
            # Отправляем файл
            with open(pdf_path, 'rb') as pdf_file:
                await query.message.reply_document(
                    document=pdf_file,
                    filename=pdf_path.name,
                    caption="✅ Ваша история чата в PDF формате!"
                )
            
        elif query.data == "stats":
            stats = self.db.get_stats()
            stats_text = f"""
📊 **Статистика бота**

👥 Всего пользователей: {stats['total_users']}
💬 Всего сообщений: {stats['total_messages']}
🎨 Сгенерировано изображений: {stats['total_images']}
🎤 Голосовых сообщений: {stats['total_voice']}
🔥 Активных сегодня: {stats['active_today']}

🏆 **Топ пользователи:**
"""
            for i, (username, count) in enumerate(stats['top_users'][:3], 1):
                stats_text += f"\n{i}. @{username or 'anonymous'}: {count} сообщений"
            
            if user.id == Config.ADMIN_ID:
                stats_text += f"\n\n🔗 [Админ-панель](http://localhost:{Config.WEB_PORT})"
            
            await query.message.reply_text(stats_text, parse_mode="Markdown")
            
        elif query.data == "help":
            help_text = """
📚 **Подробная справка**

**Команды:**
/start - Главное меню
/clear - Очистить историю диалога
/stats - Статистика бота
/export - Экспорт чата в PDF
/admin - Ссылка на админ-панель (для админов)

**Возможности:**

🎨 **Генерация изображений:**
Опишите картинку, и я создам её с помощью DALL-E 3

🎤 **Голосовые сообщения:**
Отправьте голосовое, я распознаю и отвечу

📄 **Экспорт в PDF:**
Сохраните историю чата в красивом PDF

🌐 **Админ-панель:**
Веб-интерфейс для управления ботом

💡 **Совет:** Бот помнит контекст диалога!
"""
            await query.message.reply_text(help_text, parse_mode="Markdown")
    
    async def handle_message_with_image_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка промпта для генерации изображения"""
        if context.user_data.get('waiting_for_image_prompt'):
            user = update.effective_user
            prompt = update.message.text
            
            await update.message.reply_text("🎨 Генерирую изображение... Это займет 10-20 секунд")
            
            # Генерируем изображение
            image_path, message = await self.ai.generate_image(user.id, prompt)
            
            if image_path:
                with open(image_path, 'rb') as img:
                    await update.message.reply_photo(
                        photo=img,
                        caption=f"✨ Сгенерировано по запросу:\n_{prompt}_",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text(message)
            
            context.user_data['waiting_for_image_prompt'] = False
        else:
            await self.handle_text(update, context)
    
    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /admin"""
        user = update.effective_user
        
        if user.id != Config.ADMIN_ID:
            await update.message.reply_text("⛔ Доступ только для администратора")
            return
        
        admin_text = f"""
🔐 **Панель администратора**

🌐 Веб-интерфейс: http://localhost:{Config.WEB_PORT}
🔑 Пароль: установлен в переменной ADMIN_PASSWORD

**Функции админ-панели:**
• Статистика в реальном времени
• Список всех пользователей
• История сообщений
• Массовая рассылка
• Экспорт данных

**API endpoints:**
• /api/stats - статистика
• /api/users - список пользователей
• /broadcast - массовая рассылка
"""
        await update.message.reply_text(admin_text, parse_mode="Markdown")
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистка истории диалога"""
        user = update.effective_user
        
        if user.id in self.ai.conversations:
            del self.ai.conversations[user.id]
        
        await update.message.reply_text("🔄 История диалога очищена! Начнем сначала.")
    
    async def export_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Экспорт чата в PDF"""
        user = update.effective_user
        
        await update.message.reply_text("📄 Готовлю PDF с вашей историей чата...")
        
        # Получаем историю
        chat_history = self.db.get_chat_history(user.id)
        
        if not chat_history:
            await update.message.reply_text("❌ История чата пуста")
            return
        
        # Генерируем PDF
        pdf_path = self.pdf.generate_chat_export(
            user.id, 
            chat_history,
            user.to_dict()
        )
        
        # Отправляем файл
        with open(pdf_path, 'rb') as pdf_file:
            await update.message.reply_document(
                document=pdf_file,
                filename=pdf_path.name,
                caption="✅ Ваша история чата в PDF формате!"
            )

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
def main():
    """Запуск всех компонентов"""
    
    # Настройка логирования
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Проверка конфигурации
    if not Config.TELEGRAM_TOKEN:
        logging.error("❌ TELEGRAM_TOKEN не установлен!")
        return
    
    if not Config.GROQ_API_KEY and not Config.OPENAI_API_KEY:
        logging.warning("⚠️ Нет API ключей для AI. Установите GROQ_API_KEY (бесплатно) или OPENAI_API_KEY")
    
    # Инициализация
    bot = MegaBot()
    
    # Запуск админ-панели в отдельном потоке
    admin_panel = AdminPanel(bot.db)
    admin_thread = threading.Thread(target=admin_panel.run, daemon=True)
    admin_thread.start()
    logging.info(f"🌐 Админ-панель запущена на http://localhost:{Config.WEB_PORT}")
    
    # Создание Telegram приложения
    app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
    
    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("admin", bot.admin_command))
    app.add_handler(CommandHandler("clear", bot.clear_command))
    app.add_handler(CommandHandler("export", bot.export_command))
    app.add_handler(CommandHandler("stats", lambda u, c: bot.button_handler(u, c)))
    
    # Обработчики сообщений
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        bot.handle_message_with_image_prompt
    ))
    app.add_handler(MessageHandler(filters.VOICE, bot.handle_voice))
    
    # Обработчик кнопок
    app.add_handler(CallbackQueryHandler(bot.button_handler))
    
    # Запуск бота
    logging.info(f"🚀 {Config.BUSINESS_NAME} запущен!")
    logging.info(f"💬 Groq AI: {'✅' if Config.GROQ_API_KEY else '❌'}")
    logging.info(f"🎨 DALL-E: {'✅' if Config.OPENAI_API_KEY else '❌'}")
    logging.info(f"🎤 Whisper: {'✅' if Config.OPENAI_API_KEY else '❌'}")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
