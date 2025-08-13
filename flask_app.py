import os
import tempfile
import base64
import sqlite3
import io
import datetime
import json
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory, session, redirect, url_for, flash
from flask.json.provider import DefaultJSONProvider
import logging
from flask.logging import default_handler
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
import google.generativeai as genai
from google.auth.exceptions import RefreshError
import cv2  # Import OpenCV
import numpy as np # Import numpy

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('eNivaran')
logger.addHandler(default_handler)

def handle_json_error(e):
    logger.error(f"JSON Serialization Error: {str(e)}")
    return jsonify({'error': 'Internal server error occurred.', 'details': str(e) if app.debug else None}), 500

def handle_value_error(e):
    logger.error(f"Value Error: {str(e)}")
    return jsonify({'error': 'Invalid data format in request.', 'details': str(e) if app.debug else None}), 400

def handle_key_error(e):
    logger.error(f"Key Error: {str(e)}")
    return jsonify({'error': 'Required data missing from request.', 'details': str(e) if app.debug else None}), 400

def handle_sqlite_error(e):
    logger.error(f"Database Error: {str(e)}")
    return jsonify({'error': 'Database operation failed.', 'details': str(e) if app.debug else None}), 500


class CustomJSONEncoder(DefaultJSONProvider):
    def __init__(self, app):
        super().__init__(app)
        self.options = {'ensure_ascii': False, 'sort_keys': False, 'compact': True}
    def default(self, obj):
        try:
            if isinstance(obj, datetime.datetime): return obj.isoformat()
            if isinstance(obj, sqlite3.Row): return dict(obj)
            if isinstance(obj, bytes): return base64.b64encode(obj).decode('utf-8')
            return super().default(obj)
        except Exception as e:
            print(f"JSON encoding error: {e}")
            return None
    def dumps(self, obj, **kwargs):
        def convert(o):
            if isinstance(o, datetime.datetime): return o.isoformat()
            elif isinstance(o, sqlite3.Row): return dict(o)
            elif isinstance(o, bytes): return base64.b64encode(o).decode('utf-8')
            elif isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)): return [convert(v) for v in o]
            return o
        try: return json.dumps(convert(obj), ensure_ascii=False, **kwargs)
        except Exception as e:
            print(f"JSON dumps error: {e}")
            return json.dumps(None)
    def loads(self, s, **kwargs):
        try: return json.loads(s, **kwargs)
        except Exception as e:
            print(f"JSON loads error: {e}")
            return None

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError

from pothole_detection import run_pothole_detection, assess_road_video
from duplication_detection_code import get_duplicate_detector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-replace-later'

# --- Firebase Initialization ---
# Check for the existence of the service account file and provide a clear warning if it's missing.
SERVICE_ACCOUNT_FILE = 'firebase-service-account.json'
if not os.path.exists(SERVICE_ACCOUNT_FILE):
    app.logger.critical(
        "--- FIREBASE SERVICE ACCOUNT FILE NOT FOUND ---\n"
        f"The file '{SERVICE_ACCOUNT_FILE}' is required to connect to Firebase for real-time chat features.\n"
        "Please download your service account credentials from the Firebase Console and place the file in the root directory of this project.\n"
        "The application will not function correctly without it."
    )
    # Exit or raise an exception if Firebase is critical for startup
    # For now, we will let it proceed but log a critical error.
    
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://enivaran-1e89f-default-rtdb.firebaseio.com'})
    app.logger.info("Firebase Admin SDK initialized successfully.")
except Exception as e:
    app.logger.error(f"Failed to initialize Firebase Admin SDK: {e}", exc_info=True)
    # Depending on the app's requirements, you might want to re-raise the exception
    # to prevent the app from starting without a valid Firebase connection.
    # raise e

detector = get_duplicate_detector(location_threshold=0.1)

try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=GEMINI_API_KEY)
    chat_model = genai.GenerativeModel('gemini-2.5-flash')
    app.logger.info("Google Gemini AI configured successfully.")
except Exception as e:
    app.logger.error(f"Failed to configure Google Gemini AI: {e}")
    chat_model = None

def load_existing_complaints_into_detector():
    app.logger.info("Loading existing complaints into duplicate detector...")
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        complaints_to_load = conn.execute('SELECT id, text, location_lat, location_lon, issue_type, image FROM complaints WHERE is_duplicate = 0').fetchall()
        for complaint in complaints_to_load:
            if not all(k in complaint for k in ['id', 'text', 'location_lat', 'location_lon', 'issue_type', 'image']):
                app.logger.warning(f"Skipping incomplete complaint record: {complaint.get('id')}")
                continue
            report_dict = {
                'id': complaint['id'], 'text': complaint['text'],
                'location': (complaint['location_lat'], complaint['location_lon']),
                'issue_type': complaint['issue_type'], 'image_bytes': complaint['image']
            }
            detector.add_report(report_dict)
    app.logger.info(f"Loaded {len(complaints_to_load)} complaints into the detector.")
    if len(complaints_to_load) > 1:
        detector.build_clusters()
        app.logger.info("Detector clusters have been built.")

def b64encode_filter(data):
    if data is None: return None
    return base64.b64encode(data).decode('utf-8')
app.jinja_env.filters['b64encode'] = b64encode_filter
app.json = CustomJSONEncoder(app)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
VIDEO_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'video_uploads')
PROCESSED_VIDEO_FOLDER = os.path.join(BASE_DIR, 'processed_videos')
CHAT_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'chat_files')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_VIDEO_FOLDER, exist_ok=True)
os.makedirs(CHAT_UPLOAD_FOLDER, exist_ok=True)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    VIDEO_UPLOAD_FOLDER=VIDEO_UPLOAD_FOLDER,
    PROCESSED_VIDEO_FOLDER=PROCESSED_VIDEO_FOLDER,
    SECRET_KEY='dev-secret-key-replace-later',
    MAX_CONTENT_LENGTH=100 * 1024 * 1024
)

app.register_error_handler(json.JSONDecodeError, handle_json_error)
app.register_error_handler(ValueError, handle_value_error)
app.register_error_handler(KeyError, handle_key_error)
app.register_error_handler(sqlite3.Error, handle_sqlite_error)

@app.errorhandler(404)
def not_found_error(error):
    if request.is_json: return jsonify({'error': 'Resource not found'}), 404
    flash('The requested page was not found.', 'error')
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    if request.is_json: return jsonify({'error': 'An internal server error occurred'}), 500
    flash('An unexpected error has occurred.', 'error')
    return render_template('index.html'), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error(f'Unhandled Exception: {str(e)}')
    if request.is_json: return jsonify({'error': 'An unexpected error occurred', 'details': str(e) if app.debug else None}), 500
    flash('An unexpected error has occurred.', 'error')
    return render_template('index.html'), 500

if not app.debug:
    import logging.handlers
    file_handler = logging.handlers.RotatingFileHandler('enivaran.log', maxBytes=1024 * 1024, backupCount=10)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('eNivaran startup')

APP_DB = os.path.join(BASE_DIR, 'enivaran.db')

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        value = row[idx]
        column_name = col[0]
        if isinstance(value, str) and column_name in ['submitted_at', 'detected_at', 'created_at', 'last_updated']:
            try: value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                try: value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except ValueError: app.logger.warning(f"Invalid timestamp format in column {column_name}: {value!r}. Row data: {dict(zip([col[0] for col in cursor.description], row))}")
        d[column_name] = value
    return d

def get_coordinates_from_address(street, city, state, zipcode):
    geolocator = Nominatim(user_agent="eNivaran-app")
    address = f"{street}, {city}, {state}, {zipcode}, India"
    try:
        location = geolocator.geocode(address)
        return (location.latitude, location.longitude) if location else (None, None)
    except GeocoderServiceError: return None, None

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            app.logger.info('User not in session, redirecting to login')
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
            
        # Verify user exists and session is valid
        try:
            with sqlite3.connect(APP_DB) as conn:
                conn.row_factory = dict_factory
                user = conn.execute('SELECT id, username, role FROM users WHERE id = ?', 
                                 (session['user_id'],)).fetchone()
                
                if not user:
                    app.logger.warning(f'Invalid session for user_id {session["user_id"]}, clearing session')
                    session.clear()
                    flash('Your session has expired. Please log in again.', 'error')
                    return redirect(url_for('login'))
                    
                # Update session with latest user data
                session['username'] = user['username']
                session['role'] = user['role']
                session['is_admin'] = user['role'] in ['admin', 'higher_admin']
                session['is_higher_admin'] = user['role'] == 'higher_admin'
                
        except sqlite3.Error as e:
            app.logger.error(f'Database error in login_required: {e}')
            flash('A system error occurred. Please try again.', 'error')
            return redirect(url_for('login'))
            
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/tools')
@login_required
def tools():
    return render_template('tools.html')


@app.route('/unified_detector')
@login_required
def unified_detector():
    if not session.get('is_higher_admin'):
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('admin_dashboard'))
    return render_template('unified_detector.html')

@app.route('/api/detect_from_kartaview', methods=['POST'])
@login_required
def detect_from_kartaview():
    if not session.get('is_higher_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    if not data or 'address' not in data:
        return jsonify({'error': 'Address is required.'}), 400
    
    # This is a placeholder for where you would call the KartaView API
    # and the pothole detection model.
    # We will simulate a response with error handling.
    try:
        # Simulate an API call that might fail
        if "error" in data['address'].lower():
            raise ValueError("Simulated API error: Could not connect to KartaView.")
        
        # Simulate a successful response with no images found
        if "no images" in data['address'].lower():
            return jsonify({'frames': [], 'sequenceId': '12345'})

        # Simulate a successful response with detected potholes
        return jsonify({
            "sequenceId": "12345",
            "frames": [
                {
                    "raw_image_url": "https://via.placeholder.com/600x400.png?text=Original+Image+1",
                    "annotated_image_url": "https://via.placeholder.com/600x400.png?text=Annotated+Image+1",
                    "captured_at": "2023-10-27T10:00:00Z",
                    "detections": {
                        "potholes": [
                            {"confidence": 0.95, "box": [100, 150, 50, 50]}
                        ]
                    }
                }
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/detect_from_address', methods=['POST'])
@login_required
def detect_from_address():
    if not session.get('is_higher_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
        
    data = request.get_json()
    if not data or not all(k in data for k in ['street', 'city', 'pincode']):
        return jsonify({'error': 'Street, city, and pincode are required.'}), 400

    try:
        # Simulate an API call that might fail
        if "error" in data['city'].lower():
            raise ValueError("Simulated API error: Address not found.")

        # Simulate a successful response
        return jsonify({
            "raw_image_url": "https://via.placeholder.com/600x400.png?text=Original+Image",
            "annotated_image_url": "https://via.placeholder.com/600x400.png?text=Annotated+Image",
            "captured_at": "2023-10-27T10:00:00Z",
            "detections": {
                "potholes": [
                    {"confidence": 0.88, "box": [200, 250, 60, 40]}
                ]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def init_database():
    with sqlite3.connect(APP_DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        c = conn.cursor()
        # Existing tables...
        c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, full_name TEXT, password_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS complaints (id INTEGER PRIMARY KEY, text TEXT, location_lat REAL, location_lon REAL, city TEXT, issue_type TEXT, image BLOB, image_filename TEXT, submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, is_duplicate INTEGER DEFAULT 0, original_report_id INTEGER, user_id INTEGER, status TEXT DEFAULT 'Submitted', upvotes INTEGER DEFAULT 0, remarks TEXT DEFAULT 'Complaint sent for supervision.', FOREIGN KEY (user_id) REFERENCES users (id), FOREIGN KEY (original_report_id) REFERENCES complaints (id))''')
        
        # --- CORRECTED: Create feedback table with schema migration checks ---
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                complaint_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                feedback_image_path TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (complaint_id) REFERENCES complaints (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
        ''')

        # Existing tables...
        c.execute('''CREATE TABLE IF NOT EXISTS pothole_detections (id INTEGER PRIMARY KEY, input_image BLOB, input_filename TEXT, detection_result TEXT, annotated_image BLOB, detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, user_id INTEGER, FOREIGN KEY (user_id) REFERENCES users (id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS pothole_stats (id INTEGER PRIMARY KEY CHECK (id = 1), total_potholes INTEGER DEFAULT 0, high_priority_count INTEGER DEFAULT 0, medium_priority_count INTEGER DEFAULT 0, low_priority_count INTEGER DEFAULT 0, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        if c.execute('SELECT COUNT(*) FROM pothole_stats').fetchone()[0] == 0: c.execute('INSERT INTO pothole_stats (id) VALUES (1)')
        c.execute('''CREATE TABLE IF NOT EXISTS upvotes (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, complaint_id INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE, FOREIGN KEY (complaint_id) REFERENCES complaints (id) ON DELETE CASCADE, UNIQUE (user_id, complaint_id))''')

        # Add indexes...
        c.execute('CREATE INDEX IF NOT EXISTS idx_complaints_user_id ON complaints(user_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_complaints_submitted_at ON complaints(submitted_at)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_upvotes_user_complaint ON upvotes(user_id, complaint_id)')
        
        # --- NEW: Add columns for completion details to complaints table ---
        c.execute("PRAGMA table_info(complaints)")
        columns = [row[1] for row in c.fetchall()]
        if 'completion_details' not in columns:
            app.logger.info("Adding 'completion_details' column to 'complaints' table.")
            c.execute("ALTER TABLE complaints ADD COLUMN completion_details TEXT")
        if 'completion_image' not in columns:
            app.logger.info("Adding 'completion_image' column to 'complaints' table.")
            c.execute("ALTER TABLE complaints ADD COLUMN completion_image BLOB")
            
        # Add other columns if they don't exist...
        c.execute("PRAGMA table_info(complaints)")
        columns = [row[1] for row in c.fetchall()]
        if 'city' not in columns:
            app.logger.info("Adding 'city' column to 'complaints' table.")
            c.execute("ALTER TABLE complaints ADD COLUMN city TEXT")
        c.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in c.fetchall()]
        if 'role' not in columns:
            app.logger.info("Adding 'role' column to 'users' table.")
            c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        
        # --- THIS IS THE FIX ---
        # Check if the columns in the 'feedback' table exist and add them if they don't.
        c.execute("PRAGMA table_info(feedback)")
        columns = [row[1] for row in c.fetchall()]
        if 'feedback_image_path' not in columns:
            app.logger.info("Adding 'feedback_image_path' column to 'feedback' table.")
            c.execute("ALTER TABLE feedback ADD COLUMN feedback_image_path TEXT")
        if 'submitted_at' not in columns:
            app.logger.info("Adding 'submitted_at' column to 'feedback' table.")
            c.execute("ALTER TABLE feedback ADD COLUMN submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        
        # Ensure admin users exist...
        c.execute("SELECT id FROM users WHERE username = ?", ('admin001',))
        if c.fetchone() is None:
            c.execute("INSERT INTO users (username, full_name, password_hash, role) VALUES (?, ?, ?, ?)",
                      ('admin001', 'Admin User', generate_password_hash('admin$001'), 'admin'))
            app.logger.info("Created default admin user.")

        c.execute("SELECT id FROM users WHERE username = ?", ('higher001',))
        if c.fetchone() is None:
            c.execute("INSERT INTO users (username, full_name, password_hash, role) VALUES (?, ?, ?, ?)",
                      ('higher001', 'Higher Admin User', generate_password_hash('higher$001'), 'higher_admin'))
            app.logger.info("Created default higher_admin user.")


        conn.commit()

def init_app():
    app.logger.info("Starting application initialization...")
    try:
        init_database()
        app.logger.info("Database initialized successfully")
    except Exception as e:
        app.logger.error(f"Database initialization failed: {e}")
        raise
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        app.logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
    except Exception as e:
        app.logger.error(f"Failed to create upload folder: {e}")
        raise
    @app.before_request
    def enable_foreign_keys():
        if request.endpoint != 'static_files':
            conn = sqlite3.connect(APP_DB)
            conn.execute('PRAGMA foreign_keys = ON')
            conn.close()
    try:
        load_existing_complaints_into_detector()
        app.logger.info("Loaded existing complaints into duplicate detector")
    except Exception as e:
        app.logger.error(f"Failed to load complaints into detector: {e}")
    app.logger.info("Application initialization completed")

init_app()

@app.route('/detect_pothole', methods=['POST'])
@login_required
def detect_pothole():
    if 'image' not in request.files: return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    result_json, annotated_image_bytes = run_pothole_detection(file_path)
    os.remove(file_path)
    if result_json is None: return jsonify({'error': 'Detection failed'}), 500
    annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode('utf-8')
    return jsonify({'result': result_json, 'annotated_image_b64': annotated_image_b64})

@app.route('/detect_video', methods=['POST'])
@login_required
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(input_path)
        app.logger.info(f"Video saved to {input_path}")
        output_filename = f"processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(app.config['PROCESSED_VIDEO_FOLDER'], output_filename)
        processed_path, avg_damage = assess_road_video(input_path, output_path, model_path='Pothole-Detector.pt')
        os.remove(input_path)
        if processed_path is None:
            return jsonify({'error': 'Video processing failed on the server.'}), 500
        video_url = url_for('serve_processed_video', filename=output_filename)
        return jsonify({'success': True, 'video_url': video_url})
    except Exception as e:
        app.logger.error(f"Error during video detection: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/processed_videos/<path:filename>')
def serve_processed_video(filename):
    """Serve a processed video file from the designated folder."""
    return send_from_directory(app.config['PROCESSED_VIDEO_FOLDER'], filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(APP_DB) as conn:
            conn.row_factory = dict_factory
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['is_admin'] = user['role'] in ['admin', 'higher_admin']
            session['is_higher_admin'] = user['role'] == 'higher_admin'
            flash(f'Welcome back, {user["full_name"]}!', 'success')
            if session['is_admin']:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        full_name = request.form['full_name']
        password = request.form['password']
        if not all([username, full_name, password]):
            flash('All fields are required.', 'error')
            return redirect(url_for('signup'))
        try:
            with sqlite3.connect(APP_DB) as conn:
                if conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone():
                    flash('Username already exists.', 'error')
                    return render_template('signup.html')
                password_hash = generate_password_hash(password)
                conn.execute('INSERT INTO users (username, full_name, password_hash) VALUES (?, ?, ?)', (username, full_name, password_hash))
                conn.commit()
        except sqlite3.Error as e:
            flash(f"Database error: {e}", "error")
            return render_template('signup.html')
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = session.get('username', 'User')
    is_admin = session.get('is_admin', False)
    session.clear()
    if is_admin:
        flash('Administrator logged out successfully.', 'success')
    else:
        flash(f'Goodbye, {username}! You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/complaints')
@login_required
def view_complaints():
    sort_by = request.args.get('sort', 'time_desc')
    search_id = request.args.get('search_id', type=int)
    order_clause = "ORDER BY submitted_at DESC"
    if sort_by == 'upvotes_desc':
        order_clause = "ORDER BY upvotes DESC, submitted_at DESC"
    elif sort_by == 'time_asc':
        order_clause = "ORDER BY submitted_at ASC"
    where_conditions = ["(c.is_duplicate = 0 OR c.is_duplicate IS NULL)"]
    params = []
    if search_id:
        where_conditions.append("c.id = ?")
        params.append(search_id)
    where_clause = "WHERE " + " AND ".join(where_conditions)
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        query = f'''
            SELECT c.id, c.text, c.city, CAST(c.location_lat AS FLOAT) as location_lat, CAST(c.location_lon AS FLOAT) as location_lon,
                   c.issue_type, c.image, c.submitted_at, c.status, c.upvotes, c.remarks, c.is_duplicate, c.original_report_id, u.username
            FROM complaints c LEFT JOIN users u ON c.user_id = u.id {where_clause} {order_clause}
        '''
        complaints_raw = conn.execute(query, params).fetchall()
        processed_complaints = []
        for complaint in complaints_raw:
            try:
                comp_dict = {'id': int(complaint['id']), 'text': str(complaint['text'] or ''), 'city': str(complaint['city'] or ''),
                             'location_lat': float(complaint['location_lat'] or 0),
                             'location_lon': float(complaint['location_lon'] or 0), 'issue_type': str(complaint['issue_type'] or ''),
                             'status': str(complaint['status'] or 'Submitted'), 'upvotes': int(complaint['upvotes'] or 0),
                             'remarks': str(complaint['remarks'] or ''), 'username': str(complaint['username'] or ''),
                             'is_duplicate': bool(complaint.get('is_duplicate')),
                             'original_report_id': int(complaint['original_report_id']) if complaint.get('original_report_id') else None}
                if isinstance(complaint['submitted_at'], str):
                    try: submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try: submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S')
                        except ValueError: submitted_at = datetime.datetime.now()
                else: submitted_at = complaint['submitted_at'] or datetime.datetime.now()
                comp_dict['submitted_at'] = submitted_at
                if complaint.get('image'):
                    try: comp_dict['image'] = base64.b64encode(complaint['image']).decode('utf-8')
                    except: comp_dict['image'] = None
                else: comp_dict['image'] = None
                processed_complaints.append(comp_dict)
            except Exception as e:
                app.logger.error(f"Error processing complaint {complaint.get('id', 'unknown')}: {str(e)}")
                continue
    return render_template('complaints.html', complaints=processed_complaints, sort_by=sort_by, search_id=search_id)

@app.route('/upvote_complaint/<int:complaint_id>', methods=['POST'])
@login_required
def upvote_complaint(complaint_id):
    if session.get('is_admin'):
        return jsonify({'error': 'Admins cannot upvote.'}), 403
    user_id = session['user_id']
    with sqlite3.connect(APP_DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        existing_vote = cursor.execute('SELECT id FROM upvotes WHERE user_id = ? AND complaint_id = ?', (user_id, complaint_id)).fetchone()
        if existing_vote:
            app.logger.warning(f"User {user_id} attempted to upvote complaint {complaint_id} again.")
            return jsonify({'error': 'You have already upvoted this complaint.'}), 409
        try:
            cursor.execute('INSERT INTO upvotes (user_id, complaint_id) VALUES (?, ?)', (user_id, complaint_id))
            cursor.execute('UPDATE complaints SET upvotes = upvotes + 1 WHERE id = ?', (complaint_id,))
            
            
            new_count_result = cursor.execute('SELECT upvotes FROM complaints WHERE id = ?', (complaint_id,)).fetchone()
            if new_count_result:
                return jsonify({'success': True, 'new_count': new_count_result['upvotes']})
            else:
                raise Exception("Complaint not found after upvoting.")
        except sqlite3.Error as e:
            app.logger.error(f"Database error during upvote transaction for complaint {complaint_id} by user {user_id}: {e}")
            return jsonify({'error': 'A database error occurred during the upvote process.'}), 500

@app.route('/my_complaints')
@login_required
def my_complaints():
    if session.get('is_admin'):
        flash("Admin users can view all complaints via the admin dashboard.", "info")
        return redirect(url_for('admin_dashboard'))
    user_id = session['user_id']
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        complaints_raw = conn.execute('SELECT c.id, c.text, c.city, c.issue_type, c.image, c.submitted_at, c.status, c.upvotes, c.remarks, c.is_duplicate, c.original_report_id, u.username FROM complaints c LEFT JOIN users u ON c.user_id = u.id WHERE c.user_id = ? ORDER BY c.submitted_at DESC', (user_id,)).fetchall()
        processed_complaints = []
        for complaint in complaints_raw:
            try:
                comp_dict = {'id': int(complaint['id']), 'text': str(complaint['text'] or ''), 'city': str(complaint['city'] or ''),
                             'issue_type': str(complaint['issue_type'] or ''),
                             'status': str(complaint['status'] or 'Submitted'), 'upvotes': int(complaint['upvotes'] or 0),
                             'remarks': str(complaint['remarks'] or ''), 'username': str(complaint['username'] or ''),
                             'is_duplicate': bool(complaint['is_duplicate']),
                             'original_report_id': int(complaint['original_report_id']) if complaint['original_report_id'] else None}
                if complaint.get('image'):
                    try: comp_dict['image'] = base64.b64encode(complaint['image']).decode('utf-8')
                    except: comp_dict['image'] = None
                else: comp_dict['image'] = None
                if isinstance(complaint['submitted_at'], str):
                    try: submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try: submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S')
                        except ValueError: submitted_at = datetime.datetime.now()
                else: submitted_at = complaint['submitted_at'] or datetime.datetime.now()
                comp_dict['submitted_at'] = submitted_at
                processed_complaints.append(comp_dict)
            except Exception as e:
                app.logger.error(f"Error processing complaint {complaint.get('id', 'unknown')}: {str(e)}")
                continue
    return render_template('my_complaints.html', complaints=processed_complaints, now=datetime.datetime.now())

@app.route('/raise_complaint', methods=['POST'])
@login_required
def raise_complaint():
    user_id = session['user_id']
    if session.get('is_admin') or not isinstance(user_id, int):
        return jsonify({'error': 'Invalid user session for raising a complaint.'}), 403
    form = request.form
    # The evidence file is named 'image' in the form, even if it's a video
    if not all([form.get(k) for k in ['text', 'issue_type', 'street', 'city', 'state', 'zipcode']]) or 'image' not in request.files:
        return jsonify({'error': 'All fields and an evidence file are required.'}), 400

    evidence_file = request.files['image']
    image_bytes_for_db = None

    # --- NEW LOGIC: Handle video by extracting first frame for thumbnail ---
    if evidence_file.content_type.startswith('video'):
        app.logger.info("Video file detected for complaint. Extracting thumbnail.")
        # Save to a temporary file to be read by OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=secure_filename(evidence_file.filename)) as temp:
            evidence_file.save(temp.name)
            video_path = temp.name
        
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        os.remove(video_path) # Clean up the temporary file

        if success:
            # Encode the first frame as JPEG bytes to be stored in the DB
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                image_bytes_for_db = buffer.tobytes()
            else:
                app.logger.error("Failed to encode video frame to JPEG.")
                return jsonify({'error': 'Could not process video thumbnail.'}), 500
        else:
            app.logger.error("Failed to read first frame from uploaded video.")
            return jsonify({'error': 'Could not read first frame from video.'}), 500
    else:
        # --- Existing logic for images ---
        app.logger.info("Image file detected for complaint.")
        image_bytes_for_db = evidence_file.read()
    
    if not image_bytes_for_db:
        return jsonify({'error': 'Failed to process evidence file.'}), 500

    lat, lon = get_coordinates_from_address(form['street'], form['city'], form['state'], form['zipcode'])
    if not lat:
        return jsonify({'error': 'Could not find coordinates for the address.'}), 400

    new_report_data = {
        'text': form['text'],
        'location': (lat, lon),
        'issue_type': form['issue_type'],
        'image_bytes': image_bytes_for_db  # Use the processed bytes
    }
    
    is_duplicate, similar_reports, confidence = detector.find_duplicates(new_report_data)
    original_id = None
    
    if is_duplicate and similar_reports:
        original_id = similar_reports[0].get('id')
        app.logger.info(f"Duplicate detected with confidence {confidence:.2f}. Original report ID: {original_id}")
    else:
        app.logger.info("No significant duplicate found. Registering as a new complaint.")

    with sqlite3.connect(APP_DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO complaints (
                text, location_lat, location_lon, city, issue_type,
                image, image_filename, user_id,
                is_duplicate, original_report_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            form['text'], lat, lon, form['city'], form['issue_type'],
            image_bytes_for_db, secure_filename(evidence_file.filename),
            user_id, 1 if is_duplicate else 0, original_id
        ))
        
        new_complaint_id = cursor.lastrowid
        conn.commit()
        

    if not is_duplicate:
        new_report_data['id'] = new_complaint_id
        detector.add_report(new_report_data)
        app.logger.info(f"New complaint #{new_complaint_id} added to the live detector.")

    if is_duplicate:
        return jsonify({'message': f'Complaint registered, but it appears to be a duplicate of report #{original_id}. Your report has been linked.'}), 200
    else:
        return jsonify({'message': 'Complaint registered successfully.'}), 200

@app.route('/debug/reset_complaints', methods=['POST'])
def debug_reset_complaints():
    if not app.debug:
        return jsonify({'error': 'This route is only available in debug mode'}), 403
    try:
        with sqlite3.connect(APP_DB) as conn:
            conn.execute('PRAGMA foreign_keys = OFF')
            conn.execute('DELETE FROM complaints')
            conn.execute('DELETE FROM sqlite_sequence WHERE name="complaints"')
            conn.execute('UPDATE pothole_stats SET total_potholes = 0, high_priority_count = 0, medium_priority_count = 0, low_priority_count = 0')
            conn.commit()
            flash('All complaints have been cleared successfully.', 'success')
            return jsonify({'message': 'All complaints cleared successfully'}), 200
    except Exception as e:
        app.logger.error(f'Error resetting complaints: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/admin')
@login_required
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    is_higher_admin = session.get('is_higher_admin', False)
    search_id = request.args.get('search_id', type=int)
    search_city = request.args.get('city', '')

    where_conditions = []
    params = []
    if search_id:
        where_conditions.append("c.id = ?")
        params.append(search_id)
    if search_city:
        where_conditions.append("c.city LIKE ?")
        params.append(f"%{search_city}%")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        query = f'''
            SELECT c.id, c.text, c.city, CAST(c.location_lat AS FLOAT) as location_lat, CAST(c.location_lon AS FLOAT) as location_lon,
                   c.issue_type, c.image, c.completion_image, c.completion_details, c.submitted_at, c.status, c.upvotes, c.remarks, c.is_duplicate, c.original_report_id,
                   u.username, u.full_name as reporter_name, c.user_id
            FROM complaints c LEFT JOIN users u ON c.user_id = u.id {where_clause} ORDER BY c.submitted_at DESC
        '''
        complaints_raw = conn.execute(query, params).fetchall()
        
        if is_higher_admin:
            feedback_query = 'SELECT * FROM feedback ORDER BY submitted_at DESC'
            all_feedback = conn.execute(feedback_query).fetchall()
            feedback_by_complaint = {}
            for fb in all_feedback:
                feedback_by_complaint.setdefault(fb['complaint_id'], []).append(fb)

        processed_complaints = []
        for complaint in complaints_raw:
            try:
                comp_dict = {'id': int(complaint['id']), 'text': str(complaint['text'] or ''), 'city': str(complaint.get('city') or ''),
                             'location_lat': float(complaint['location_lat'] or 0),
                             'location_lon': float(complaint['location_lon'] or 0), 'issue_type': str(complaint['issue_type'] or ''),
                             'status': str(complaint['status'] or 'Submitted'), 'upvotes': int(complaint['upvotes'] or 0),
                             'remarks': str(complaint['remarks'] or ''), 'username': str(complaint['username'] or ''),
                             'reporter_name': str(complaint['reporter_name'] or ''), 'is_duplicate': bool(complaint.get('is_duplicate')),
                             'original_report_id': int(complaint['original_report_id']) if complaint.get('original_report_id') else None,
                             'user_id': int(complaint['user_id']),
                             'completion_details': complaint.get('completion_details')}
                if isinstance(complaint['submitted_at'], str):
                    try: submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try: submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S')
                        except ValueError: submitted_at = datetime.datetime.now()
                else: submitted_at = complaint['submitted_at'] or datetime.datetime.now()
                comp_dict['submitted_at'] = submitted_at
                
                # Handle images
                if complaint.get('image'):
                    try: comp_dict['image'] = base64.b64encode(complaint['image']).decode('utf-8')
                    except: comp_dict['image'] = None
                else: comp_dict['image'] = None
                
                if complaint.get('completion_image'):
                    try: comp_dict['completion_image'] = base64.b64encode(complaint['completion_image']).decode('utf-8')
                    except: comp_dict['completion_image'] = None
                else: comp_dict['completion_image'] = None

                if is_higher_admin:
                    comp_dict['feedback'] = feedback_by_complaint.get(complaint['id'], [])
                
                processed_complaints.append(comp_dict)
            except Exception as e:
                app.logger.error(f"Error processing complaint {complaint.get('id', 'unknown')}: {str(e)}")
                continue
    return render_template('admin_dashboard.html', complaints=processed_complaints, search_id=search_id, search_city=search_city)

@app.route('/api/admin/stats/complaints_by_status')
@login_required
def get_complaints_by_status():
    if not session.get('is_admin'): return jsonify({'error': 'Unauthorized'}), 403
    with sqlite3.connect(APP_DB) as conn:
        data = conn.execute("SELECT status, COUNT(*) as count FROM complaints GROUP BY status").fetchall()
        return jsonify({'labels': [d[0] for d in data], 'series': [d[1] for d in data]})

@app.route('/api/admin/stats/complaints_over_time')
@login_required
def get_complaints_over_time():
    if not session.get('is_admin'): return jsonify({'error': 'Unauthorized'}), 403
    with sqlite3.connect(APP_DB) as conn:
        # For SQLite, date functions are tricky. This groups by day.
        data = conn.execute("SELECT strftime('%Y-%m-%d', submitted_at), COUNT(*) FROM complaints GROUP BY 1 ORDER BY 1 LIMIT 30").fetchall()
        return jsonify({'labels': [d[0] for d in data], 'series': [d[1] for d in data]})

# --- START: NEW API ENDPOINTS ---
@app.route('/api/admin/stats/complaints_by_type')
@login_required
def get_complaints_by_type():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    with sqlite3.connect(APP_DB) as conn:
        # This query now normalizes the issue type and separates original vs. duplicate counts
        query = """
            SELECT
                LOWER(issue_type) as type,
                SUM(CASE WHEN is_duplicate = 0 OR is_duplicate IS NULL THEN 1 ELSE 0 END) as originals,
                SUM(CASE WHEN is_duplicate = 1 THEN 1 ELSE 0 END) as duplicates
            FROM complaints
            WHERE issue_type IS NOT NULL AND issue_type != ''
            GROUP BY LOWER(issue_type)
            ORDER BY (originals + duplicates) DESC
            LIMIT 10
        """
        data = conn.execute(query).fetchall()
        
        # Prepare data for a stacked bar chart in ApexCharts
        response_data = {
            'labels': [d[0] for d in data],
            'series': [
                {'name': 'Original', 'data': [d[1] for d in data]},
                {'name': 'Duplicate', 'data': [d[2] for d in data]}
            ]
        }
        return jsonify(response_data)

@app.route('/api/admin/stats/complaints_by_city')
@login_required
def get_complaints_by_city():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    with sqlite3.connect(APP_DB) as conn:
        data = conn.execute(
            "SELECT city, COUNT(*) as count FROM complaints WHERE city IS NOT NULL AND city != '' GROUP BY city ORDER BY count DESC LIMIT 10"
        ).fetchall()
        return jsonify({'labels': [d[0] for d in data], 'series': [d[1] for d in data]})
# --- END: NEW API ENDPOINTS ---

@app.route('/api/admin/stats/feedback_sentiment')
@login_required
def get_feedback_sentiment():
    if not session.get('is_higher_admin'): return jsonify({'error': 'Unauthorized'}), 403
    with sqlite3.connect(APP_DB) as conn:
        data = conn.execute("SELECT CASE WHEN rating >= 3 THEN 'Positive' ELSE 'Negative' END as sentiment, COUNT(*) as count FROM feedback GROUP BY 1").fetchall()
        return jsonify({'labels': [d[0] for d in data], 'series': [d[1] for d in data]})

@app.route('/submit_feedback/<int:complaint_id>', methods=['POST'])
@login_required
def submit_feedback(complaint_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User not logged in'}), 401
    
    rating = request.form.get('rating')
    comment = request.form.get('comment')
    feedback_image = request.files.get('feedback_image')
    
    if not rating:
        return jsonify({'error': 'Rating is required'}), 400
        
    feedback_image_filename = None
    if feedback_image:
        # Create a dedicated folder for feedback images
        feedback_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'feedback_files')
        os.makedirs(feedback_upload_folder, exist_ok=True)
        
        filename = secure_filename(feedback_image.filename)
        feedback_image_path = os.path.join(feedback_upload_folder, filename)
        feedback_image.save(feedback_image_path)
        feedback_image_filename = filename # Store just the filename

    with sqlite3.connect(APP_DB) as conn:
        conn.execute(
            'INSERT INTO feedback (complaint_id, user_id, rating, comment, feedback_image_path) VALUES (?, ?, ?, ?, ?)',
            (complaint_id, user_id, rating, comment, feedback_image_filename)
        )
        # Also update the complaint status to 'Resolved' or similar
        conn.execute(
            "UPDATE complaints SET status = 'Resolved & Reviewed' WHERE id = ?", (complaint_id,)
        )
        conn.commit()
        
    flash('Thank you! Your feedback has been submitted successfully.', 'success')
    return jsonify({'success': True, 'message': 'Feedback submitted successfully!'})

@app.route('/api/summarize_case/<int:complaint_id>')
@login_required
def summarize_case(complaint_id):
    if not session.get('is_higher_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        # Fetch complaint details including new completion fields
        complaint = conn.execute(
            'SELECT *, (completion_image IS NOT NULL) as has_completion_image FROM complaints WHERE id = ?', (complaint_id,)
        ).fetchone()
        
        # Fetch feedback
        feedback = conn.execute(
            'SELECT *, (feedback_image_path IS NOT NULL) as has_feedback_image FROM feedback WHERE complaint_id = ? ORDER BY submitted_at DESC', (complaint_id,)
        ).fetchall()

    if not complaint:
        return jsonify({'error': 'Complaint not found'}), 404

    # Summarize chat history
    chat_history = db.reference(f'chats/{complaint_id}/messages').get()
    chat_summary = ""
    if chat_history and isinstance(chat_history, dict):
        # Sort messages by timestamp before summarizing
        sorted_messages = sorted(chat_history.values(), key=lambda x: x.get('timestamp', ''))
        for message in sorted_messages:
            chat_summary += f"{message.get('sender_name', 'Unknown')}: {message.get('text', '')}\n"
    else:
        chat_summary = "No chat history available."
        
    # Summarize feedback
    feedback_summary = ""
    if feedback:
        for item in feedback:
            rating_text = 'Positive' if item['rating'] >= 3 else 'Negative' if item['rating'] > 0 else 'Not Rated'
            feedback_summary += f"Feedback (Rating: {rating_text}): {item['comment']}\n"
            if item.get('has_feedback_image'):
                feedback_summary += "(User provided a feedback image.)\n"
    else:
        feedback_summary = "No user feedback submitted yet."

    # Build the prompt
    prompt = f"""
    You are an AI assistant for a civic issues platform. Summarize the following case for a higher authority.
    *Case ID:* {complaint['id']}
    *Initial Complaint:* {complaint['text']}
    *Location:* {complaint['city']}
    *Submitted On:* {complaint['submitted_at']}
    
    *Admin/User Chat History:*
    {chat_summary}
    
    *Final Status:* {complaint['status']}
    *Admin's Completion Report:*
    {complaint['completion_details'] if complaint['completion_details'] else 'No completion details provided.'}
    {'*Admin provided a completion proof image.*' if complaint.get('has_completion_image') else ''}

    *User Feedback:*
    {feedback_summary}
    ---
    Provide a neutral, chronological summary of the case. Start with the initial report, summarize the communication, describe the resolution action taken by the admin, and conclude with the user's feedback. Highlight key events and any points of conflict.
    """
    
    try:
        response = chat_model.generate_content(prompt)
        return jsonify({'summary': response.text})
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return jsonify({'error': 'The AI service is currently unavailable or encountered an error.'}), 500

@app.route('/update_complaint_status/<int:complaint_id>', methods=['POST'])
@login_required
def update_complaint_status(complaint_id):
    if not session.get('is_admin'):
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))

    status = request.form.get('status')
    remarks = request.form.get('remarks')
    
    if not status:
        flash('Status is required.', 'error')
        return redirect(url_for('admin_dashboard'))

    with sqlite3.connect(APP_DB) as conn:
        cursor = conn.cursor()
        if status == 'Completed':
            completion_details = request.form.get('completion_details', '')
            completion_image_file = request.files.get('completion_image')
            
            if not completion_details:
                flash('Completion details are required when status is "Completed".', 'error')
                return redirect(url_for('admin_dashboard'))
            
            completion_image_blob = completion_image_file.read() if completion_image_file else None
            
            cursor.execute(
                'UPDATE complaints SET status = ?, remarks = ?, completion_details = ?, completion_image = ? WHERE id = ?',
                (status, remarks, completion_details, completion_image_blob, complaint_id)
            )
        else:
            cursor.execute(
                'UPDATE complaints SET status = ?, remarks = ? WHERE id = ?',
                (status, remarks, complaint_id)
            )
        conn.commit()
    
    # Push update to Firebase chat if it exists
    try:
        chat_ref = db.reference(f'chats/{complaint_id}/messages')
        chat_ref.push({
            'text': f"Status updated to: {status}\nAdmin Remarks: {remarks}",
            'sender_id': 'admin',
            'sender_name': 'Admin',
            'timestamp': datetime.datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        app.logger.error(f"Failed to push status update to Firebase for complaint #{complaint_id}: {e}")

    flash('Complaint status updated successfully.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/delete_complaint/<int:complaint_id>', methods=['POST'])
@login_required
def delete_complaint(complaint_id):
    if not session.get('is_admin'):
        app.logger.warning(f"Non-admin user {session.get('user_id')} attempted to delete complaint {complaint_id}.")
        return jsonify({'error': 'Unauthorized access.'}), 403
    try:
        with sqlite3.connect(APP_DB) as conn:
            conn.execute('PRAGMA foreign_keys = ON')
            cursor = conn.cursor()
            result = cursor.execute('SELECT id FROM complaints WHERE id = ?', (complaint_id,)).fetchone()
            if not result:
                return jsonify({'error': 'Complaint not found.'}), 404
            cursor.execute('DELETE FROM complaints WHERE id = ?', (complaint_id,))
            conn.commit()
            app.logger.info(f"Admin {session.get('username')} deleted complaint #{complaint_id}.")
            flash('Complaint deleted successfully.', 'success')
            return jsonify({'success': True, 'message': 'Complaint deleted successfully.'})
    except sqlite3.Error as e:
        app.logger.error(f"Database error while deleting complaint {complaint_id}: {e}")
        return jsonify({'error': 'Database operation failed.', 'details': str(e)}), 500

@app.route('/api/cities')
@login_required
def get_cities():
    """Returns a JSON list of unique city names from the complaints database."""
    try:
        with sqlite3.connect(APP_DB) as conn:
            # Use a factory to return a simple list of strings
            conn.row_factory = lambda cursor, row: row[0]
            cities = conn.execute(
                "SELECT DISTINCT city FROM complaints WHERE city IS NOT NULL AND city != '' ORDER BY city ASC"
            ).fetchall()
        return jsonify(cities)
    except sqlite3.Error as e:
        app.logger.error(f"Failed to fetch cities from database: {e}")
        return jsonify({'error': 'Could not fetch city list due to a database error.'}), 500

@app.route('/pothole_stats')
def pothole_stats():
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        result = conn.execute('SELECT CAST(total_potholes AS INTEGER) as total_potholes, CAST(high_priority_count AS INTEGER) as high_priority_count, CAST(medium_priority_count AS INTEGER) as medium_priority_count, CAST(low_priority_count AS INTEGER) as low_priority_count, last_updated FROM pothole_stats WHERE id = 1').fetchone()
        if result:
            try:
                processed = {'total_potholes': int(result['total_potholes'] or 0), 'high_priority_count': int(result['high_priority_count'] or 0),
                             'medium_priority_count': int(result['medium_priority_count'] or 0), 'low_priority_count': int(result['low_priority_count'] or 0),
                             'last_updated': result['last_updated'].isoformat() if result['last_updated'] else datetime.datetime.now().isoformat()}
                return jsonify(processed)
            except Exception as e:
                app.logger.error(f"Error processing pothole stats: {str(e)}")
    return jsonify({'total_potholes': 0, 'high_priority_count': 0, 'medium_priority_count': 0, 'low_priority_count': 0,
                    'last_updated': datetime.datetime.now().isoformat()})

@app.route('/chat/unread_counts', methods=['GET'])
@login_required
def get_unread_counts():
    try:
        user_id = session.get('user_id')
        is_admin = session.get('is_admin', False)
        all_chats = db.reference('chats').get()
        if not all_chats:
            return jsonify({})
        unread_counts = {}
        is_list = isinstance(all_chats, list)
        with sqlite3.connect(APP_DB) as conn:
            conn.row_factory = dict_factory
            if is_admin:
                complaints = conn.execute('SELECT id FROM complaints').fetchall()
                complaint_ids = {c['id'] for c in complaints}
                participant_id = 'admin'
                for comp_id in complaint_ids:
                    chat_data = all_chats.get(str(comp_id)) if not is_list else (all_chats[comp_id] if comp_id < len(all_chats) else None)
                    if not chat_data or 'messages' not in chat_data: continue
                    last_read = chat_data.get('metadata', {}).get(participant_id, {}).get('last_read', '1970-01-01T00:00:00Z')
                    count = sum(1 for msg in chat_data['messages'].values() if msg.get('sender_id') != participant_id and msg.get('timestamp') > last_read)
                    if count > 0: unread_counts[comp_id] = count
            else:
                complaints = conn.execute('SELECT id FROM complaints WHERE user_id = ?', (user_id,)).fetchall()
                complaint_ids = {c['id'] for c in complaints}
                participant_id = f"user_{user_id}"
                for comp_id in complaint_ids:
                    chat_data = all_chats.get(str(comp_id)) if not is_list else (all_chats[comp_id] if comp_id < len(all_chats) else None)
                    if not chat_data or 'messages' not in chat_data: continue
                    last_read = chat_data.get('metadata', {}).get(participant_id, {}).get('last_read', '1970-01-01T00:00:00Z')
                    count = sum(1 for msg in chat_data['messages'].values() if msg.get('sender_id') == 'admin' and msg.get('timestamp') > last_read)
                    if count > 0: unread_counts[comp_id] = count
        return jsonify(unread_counts)
    except RefreshError as e:
        app.logger.error(f"Firebase authentication error: {e}. Please check your service account credentials.", exc_info=True)
        return jsonify({'error': 'Firebase authentication failed. Please contact the administrator.'}), 500
    except Exception as e:
        app.logger.error(f"Failed to get unread counts: {e}", exc_info=True)
        return jsonify({'error': 'Failed to retrieve unread message counts.'}), 500

@app.route('/chat/<int:complaint_id>/mark_read', methods=['POST'])
@login_required
def mark_chat_as_read(complaint_id):
    try:
        participant_id = 'admin' if session.get('is_admin') else f"user_{session['user_id']}"
        with sqlite3.connect(APP_DB) as conn:
            complaint_owner_id = conn.execute('SELECT user_id FROM complaints WHERE id = ?', (complaint_id,)).fetchone()
        if not complaint_owner_id: return jsonify({'error': 'Complaint not found.'}), 404
        if not session.get('is_admin') and (complaint_owner_id[0] != session.get('user_id')):
            return jsonify({'error': 'Unauthorized.'}), 403
        now_iso = datetime.datetime.utcnow().isoformat() + "Z"
        metadata_ref = db.reference(f'chats/{complaint_id}/metadata/{participant_id}')
        metadata_ref.update({'last_read': now_iso})
        app.logger.info(f"Chat for complaint {complaint_id} marked as read for {participant_id}.")
        return jsonify({'success': True})
    except RefreshError as e:
        app.logger.error(f"Firebase authentication error: {e}. Please check your service account credentials.", exc_info=True)
        return jsonify({'error': 'Firebase authentication failed. Please contact the administrator.'}), 500
    except Exception as e:
        app.logger.error(f"Failed to mark chat as read for complaint #{complaint_id}: {e}")
        return jsonify({'error': 'Failed to update read status.'}), 500

@app.route('/chat/<int:complaint_id>/send', methods=['POST'])
@login_required
def send_chat_message(complaint_id):
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Message text is required.'}), 400
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        complaint = conn.execute('SELECT user_id FROM complaints WHERE id = ?', (complaint_id,)).fetchone()
    if not complaint: return jsonify({'error': 'Complaint not found.'}), 404
    if not session.get('is_admin') and (complaint['user_id'] != session['user_id']):
        return jsonify({'error': 'Unauthorized to send messages to this chat.'}), 403
    is_admin_session = session.get('is_admin', False)
    sender_id = 'admin' if is_admin_session else f"user_{session['user_id']}"
    sender_name = 'Admin' if is_admin_session else session.get('username', 'Unknown')
    message = {'text': data['text'], 'sender_id': sender_id, 'sender_name': sender_name, 'timestamp': datetime.datetime.utcnow().isoformat() + "Z"}
    try:
        chat_ref = db.reference(f'chats/{complaint_id}/messages')
        chat_ref.push(message)
        return jsonify({'success': True, 'message': 'Message sent.'})
    except RefreshError as e:
        app.logger.error(f"Firebase authentication error: {e}. Please check your service account credentials.", exc_info=True)
        return jsonify({'error': 'Firebase authentication failed. Please contact the administrator.'}), 500
    except Exception as e:
        app.logger.error(f"FIREBASE PUSH FAILED for complaint #{complaint_id}: {e}", exc_info=True)
        return jsonify({'error': 'Failed to send message to the database.'}), 500

@app.route('/chat/<int:complaint_id>/clear', methods=['POST'])
@login_required
def clear_chat_history(complaint_id):
    try:
        with sqlite3.connect(APP_DB) as conn:
            complaint_owner_id = conn.execute('SELECT user_id FROM complaints WHERE id = ?', (complaint_id,)).fetchone()
        if not complaint_owner_id: return jsonify({'error': 'Complaint not found.'}), 404
        if not session.get('is_admin') and (complaint_owner_id[0] != session.get('user_id')):
            return jsonify({'error': 'Unauthorized.'}), 403
        chat_ref = db.reference(f'chats/{complaint_id}/messages')
        chat_ref.delete()
        app.logger.info(f"Chat history for complaint #{complaint_id} cleared by user {session.get('user_id')}.")
        return jsonify({'success': True, 'message': 'Chat history cleared.'})
    except RefreshError as e:
        app.logger.error(f"Firebase authentication error: {e}. Please check your service account credentials.", exc_info=True)
        return jsonify({'error': 'Firebase authentication failed. Please contact the administrator.'}), 500
    except Exception as e:
        app.logger.error(f"Failed to clear chat history for complaint #{complaint_id}: {e}")
        return jsonify({'error': 'Failed to clear chat history.'}), 500

@app.route('/chat/<int:complaint_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(complaint_id):
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        complaint = conn.execute('SELECT user_id FROM complaints WHERE id = ?', (complaint_id,)).fetchone()
    if not complaint: return jsonify({'error': 'Complaint not found.'}), 404
    if not session.get('is_admin') and (complaint['user_id'] != session['user_id']):
        return jsonify({'error': 'Unauthorized to view this chat.'}), 403
    try:
        messages = db.reference(f'chats/{complaint_id}/messages').get()
        return jsonify(messages or {})
    except RefreshError as e:
        app.logger.error(f"Firebase authentication error: {e}. Please check your service account credentials.", exc_info=True)
        return jsonify({'error': 'Firebase authentication failed. Please contact the administrator.'}), 500
    except Exception as e:
        app.logger.error(f"Failed to retrieve Firebase messages for complaint #{complaint_id}: {e}")
        return jsonify({'error': 'Failed to retrieve messages.'}), 500

@app.route('/chat/ai', methods=['POST'])
@login_required
def ai_chat_handler():
    if not chat_model:
        return jsonify({'error': 'AI model is not configured on the server.'}), 503
    data = request.get_json()
    if not data or 'history' not in data or 'message' not in data:
        return jsonify({'error': 'Invalid request format.'}), 400
    history = data['history']
    user_message = data['message']
    gemini_history = []
    system_instruction = ("You are 'eNivaran', an AI assistant for India's civic issue reporting platform.\n\n"
                        "### RULES OF ENGAGEMENT\n"
                        "* Only respond to questions about civic complaints, reporting steps, or status checks.\n"
                        "* If the question is outside civic issues, say:\n"
                        "  - 'क्षमा करें, मैं केवल नागरिक शिकायतों से संबंधित प्रश्नों का उत्तर दे सकता हूँ।'\n"
                        "  - or the same in user's language (if Hindi/Bengali/etc).\n\n"
                        "### LANGUAGE HANDLING\n"
                        "* Detect the language of the user message.\n"
                        "* Reply in the same language for instructions.\n"
                        "* But the actual complaint form MUST be filled in *English only*.\n"
                        "* If the user enters data in Hindi/Bengali, translate that into English and guide accordingly.\n\n"
                        "### COMPLAINT REPORTING FLOW\n"
                        "When users ask how to report an issue like a pothole, give instructions in the following structure:\n"
                        "# How to Report a Pothole\n"
                        "* Issue Type: Select 'Pothole'\n"
                        "* Description: (standard format — see below)\n"
                        "* Road Name:\n"
                        "* City:\n"
                        "* State:\n"
                        "* Pincode:\n"
                        "* Upload Image: Attach a clear image of the pothole\n\n"
                        "### DESCRIPTION TEMPLATE (Important for Duplicate Detection)\n"
                        "Always generate a 1-2 sentence English description using this strict format:\n"
                        "'Pothole located at [Road Name], [City], [State] - [Pincode]. It is approximately [size/shape/severity if provided] and poses a risk to commuters.'\n"
                        "* NEVER add emojis, extra lines, or story-like text.\n"
                        "* Maintain this format to ensure the backend's duplication model works correctly.\n\n"
                        "### EXAMPLE RESPONSE\n"
                        "User asks in Hindi: 'मैं गड्ढे की शिकायत कैसे करूं?'\n"
                        "Bot responds in Hindi:\n"
                        "# गड्ढे की शिकायत कैसे दर्ज करें\n"
                        "* Issue Type: 'Pothole' चुनें\n"
                        "* विवरण: नीचे दिए गए फॉर्मेट में लिखें\n"
                        "* Road Name, City, State, और Pincode दर्ज करें\n"
                        "* pothole की तस्वीर अपलोड करें\n"
                        "* नीचे एक उदाहरण विवरण है:\n"
                        "'Pothole located at MG Road, Pune, Maharashtra - 411001. It is approximately 2 feet wide and poses a risk to commuters.'\n\n"
                        "### WARNING\n"
                        "* Never generate answers outside this format.\n"
                        "* Always enforce location + description + image requirement.\n")
    for i, msg in enumerate(history):
        role = 'user' if msg['sender'] == 'user' else 'model'
        text = msg['text']
        if i == 0 and role == 'user':
            text = f"{system_instruction}\n\nUSER: {text}"
        gemini_history.append({'role': role, 'parts': [{'text': text}]})
    try:
        chat_session = chat_model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        app.logger.error(f"Gemini API call failed: {e}")
        return jsonify({'error': 'The AI service is currently unavailable or encountered an error. Please try again later.'}), 500

@app.route('/upload_chat_file', methods=['POST'])
@login_required
def upload_chat_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(CHAT_UPLOAD_FOLDER, filename)
        file.save(save_path)
        app.logger.info(f"User {session['user_id']} uploaded chat file: {filename}")
        return jsonify({'success': True, 'message': f'File "{filename}" uploaded successfully.', 'filename': filename})
    return jsonify({'error': 'File upload failed.'}), 500

@app.route('/leaderboard')
@login_required
def leaderboard():
    app.logger.info(f"User {session.get('user_id')} accessing leaderboard")
    try:
        # Verify database connection first
        with sqlite3.connect(APP_DB) as conn:
            conn.execute('SELECT 1').fetchone()
        
        # Only proceed if database connection is successful
        return render_template('leaderboard.html')
        
    except sqlite3.Error as e:
        app.logger.error(f"Database error in leaderboard route: {e}")
        flash('A database error occurred. Please try again.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"Unexpected error in leaderboard route: {e}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/api/leaderboard')
@login_required
def get_leaderboard_data():
    app.logger.info(f"User {session.get('user_id')} requesting leaderboard data")
    
    try:
        user_id = session.get('user_id')
        if not user_id:
            app.logger.error("No user_id in session for leaderboard request")
            return jsonify({'error': 'User session invalid'}), 401

        with sqlite3.connect(APP_DB) as conn:
            conn.row_factory = dict_factory
            cursor = conn.cursor()

            # First verify user exists
            user_check = cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if not user_check:
                app.logger.error(f"User {user_id} not found in database")
                return jsonify({'error': 'User not found'}), 404

            # Use a Common Table Expression (CTE) to calculate points for all users first.
            # This is more efficient and avoids repeating the calculation logic.
            query_cte = """
                WITH UserStats AS (
                    SELECT
                        u.id,
                        u.full_name,
                        COALESCE((
                            SELECT 
                                (CAST(COALESCE(SUM(CASE WHEN c.is_duplicate = 0 THEN 1 ELSE 0 END), 0) AS REAL) * 15) + 
                                (CAST(COALESCE(SUM(c.upvotes), 0) AS REAL) * 2)
                            FROM complaints c 
                            WHERE c.user_id = u.id
                        ), 0) as points,
                        (SELECT COUNT(*) FROM complaints WHERE user_id = u.id AND (is_duplicate = 0 OR is_duplicate IS NULL)) as total_complaints,
                        (SELECT COALESCE(SUM(upvotes), 0) FROM complaints WHERE user_id = u.id) as total_votes
                    FROM users u
                )
            """

            # --- 1. Get the leaderboard (top users with points > 0) ---
            leaderboard_query = query_cte + """
                SELECT *,
                       (SELECT COUNT(*) + 1 FROM UserStats s2 WHERE s2.points > s1.points) as rank
                FROM UserStats s1
                WHERE s1.points > 0
                ORDER BY s1.points DESC, s1.total_complaints DESC, s1.total_votes DESC;
            """
            top_users = cursor.execute(leaderboard_query).fetchall()

            # --- 2. Get the current user's data and rank ---
            current_user_query = query_cte + """
                SELECT *,
                       (SELECT COUNT(*) + 1 FROM UserStats s2 WHERE s2.points > s1.points) as rank
                FROM UserStats s1
                WHERE s1.id = ?;
            """
            current_user_data = cursor.execute(current_user_query, (user_id,)).fetchone()
            
            if not current_user_data:
                app.logger.error(f"Failed to retrieve current user data for user {user_id}")
                return jsonify({'error': 'Failed to retrieve user data'}), 500

            return jsonify({
                "leaderboard": top_users,
                "currentUser": current_user_data
            })

    except sqlite3.Error as e:
        app.logger.error(f"Database error in leaderboard API: {e}")
        return jsonify({'error': 'A database error occurred'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in leaderboard API: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
