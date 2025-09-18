# Smart Pothole Detection & Civic Management

eNivaran is an ML-powered road safety and civic issue reporting system designed to help create safer, smarter cities through community participation and cutting-edge technology. Users can report civic issues like potholes, view existing complaints, and track their status, while administrators can manage and update these reports. The system utilizes advanced AI models for pothole detection and duplicate prevention, along with real-time communication features.

## Core Functionalities

*   **AI Pothole Detection:** Upload an image to the 'Tools' section to automatically detect potholes using a pre-trained Pytorch model (`pothole_detector_v1.onnx`). The system analyzes depth, estimates severity, and provides road condition assessment with annotated visualizations.

*   **AI Video Analysis:** Upload a video of a road to the 'Tools' section to get a frame-by-frame analysis. The system uses the `pothole_detector_v1.onnx` model to:
    *   Identify potholes in each frame.
    *   Produce an annotated output video showing the detections in real-time.

*   **Civic Issue Reporting:** Users can submit detailed complaints including:
    * Text description and issue categorization
    * Location (automatically converted to coordinates using Geopy)
    * Image evidence
    * Real-time status tracking
    * Two-way communication with administrators

*   **Advanced Duplicate Prevention:** Multi-modal duplicate detection system that:
    * Analyzes text similarity using Sentence-BERT embeddings
    * Compares image features using ResNet50
    * Checks location proximity with geospatial clustering
    * Uses XGBoost to classify potential duplicates with confidence scores

*   **User Authentication & Security:** 
    * Secure signup/login system with password hashing
    * Session-based authentication
    * Role-based access control (user/admin)
    * Secure file upload handling
    * Request validation and sanitization

*   **Complaint Management:**
    *   **Public View (`/complaints`):** View all non-duplicate complaints, sortable by time or upvotes. Includes basic search by ID. Users can upvote complaints.
    *   **User View (`/my_complaints`):** Logged-in users can view the status and details of complaints they have submitted.
    *   **Admin Dashboard (`/admin`):** Admins can view all non-duplicate complaints, update their status (e.g., Approved, Rejected, On Hold), and add remarks.

*   **Real-time Communication:**
    * Firebase-powered chat system between users and administrators
    * File sharing capabilities in chat
    * Unread message notifications
    * Chat history persistence
    * New chat notifications

*   **AI Assistance:**
    * Google Gemini-powered chatbot (J.A.R.V.I.S)
    * Context-aware responses
    * File upload support
    * Markdown-formatted responses
    * Chat history management

*   **Analytics & Statistics:** 
    * Real-time pothole statistics
    * Severity categorization (high, medium, low)
    * Road condition assessment
    * Complaint resolution tracking
    * User engagement metrics

## Technology Stack

### Frontend Architecture
* **React (Client-side Rendering)**
  - In-browser Babel for JSX compilation
  - Single Page Application (SPA) design
  - Component-based structure
  - State management with React hooks

* **UI Framework & Styling**
  - Bootstrap 5
  - Custom CSS
  - AOS animations
  - Responsive design
  - Bootstrap Icons

### Backend Systems
* **Core Backend:**  
  - Python 3.11
  - Flask framework
  - RESTful API design
  - Session management
  - File handling

* **Database Systems:**  
  - SQLite (Users, Complaints, Pothole Detection)
  - Firebase Realtime Database (Chat System)
  - Efficient blob storage for images
  - Transaction management

### Machine Learning & AI
* **Pothole Detection:**  
  - ONNX Runtime
  - Custom YOLO model
  - OpenCV integration
  - Depth estimation
  - Priority assessment

* **Duplicate Detection:**
  - ResNet50 for image features
  - Sentence-BERT for text analysis
  - XGBoost classifier
  - KMeans clustering
  - Cosine similarity

* **AI Chatbot:**
  - Google Generative AI (Gemini 2.5)
  - Context management
  - File processing
  - Response formatting

### Utilities & Support
* **Image Processing:**
  - OpenCV (cv2)
  - Pillow (PIL)
  - Matplotlib
  - Base64 encoding/decoding

* **Data Processing:**
  - NumPy
  - Scikit-learn
  - TF-IDF vectorization
  - StandardScaler

* **Location Services:**
  - Geopy
  - Coordinate conversion
  - Distance calculations
  - Location clustering

* **System Utilities:**
  - Werkzeug
  - Python Logging
  - Performance monitoring
  - Error handling

## Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Souravshukla007/eNivaran_Codex2.0_Dynamite_PS07.git
    cd eNivaran_Codex2.0_Dynamite_PS07
    ```

2.  **Environment Configuration:**
    Create a `.env` file in the root directory with:
    ```plaintext
    GEMINI_API_KEY=your_gemini_api_key
    FLASK_SECRET_KEY=your_secret_key
    FIREBASE_DATABASE_URL=your_firebase_url
    ```

3.  **Firebase Setup:**
    * Place your `firebase-service-account.json` in the root directory
    * Enable Realtime Database in your Firebase console
    * Configure database rules for security

4.  **Virtual Environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Initialize the Application:**
    ```bash
    flask run
    # or
    python flask_app.py
    ```
    Access at `http://127.0.0.1:5000/`

7.  **Database Setup:**
    * Automatic initialization of SQLite databases:
      - `users.db`
      - `complaints.db`
      - `pothole_data.db`
    * Tables created on first run

## System Access

1. **Admin Access:**
   * Username: `admin001`
   * Password: `admin$001`
   * Full complaint management
   * Status updates
   * User communication

2. **User Registration:**
   * Sign up at `/signup`
   * Required: username, full name, password
   * Automatic redirect to homepage
   * Access to all user features

## How It Works

1. **User Authentication Flow:**
   * Login/Signup validation
   * Session creation
   * Role-based routing
   * Secure password handling

2. **Pothole Detection Process:**
   * Image/video upload and validation
   * ONNX/YOLO model inference
   * Depth and severity analysis
   * Results storage and visualization
   * Statistics update

3. **Complaint Management:**
   * Form submission with validation
   * Geocoding of address
   * Duplicate detection check:
     - Text similarity analysis
     - Image feature comparison
     - Location proximity check
   * Storage with appropriate flags
   * Real-time notifications

4. **Communication System:**
   * Firebase-powered chat
   * File sharing
   * Message history
   * Read status tracking
   * Real-time updates

5. **Admin Operations:**
   * Complaint review
   * Status updates
   * User communication
   * Statistics monitoring

6. **AI Assistance:**
   * Context-aware responses
   * File processing
   * History management
   * Formatted output

---
