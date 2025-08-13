# eNivaran Project Features: A Detailed Analysis

## 1. Project Overview

eNivaran is a comprehensive web application designed to bridge the gap between citizens and municipal authorities in India. The platform's primary objective is to provide a streamlined, user-friendly interface for reporting civic issues, with a special focus on road-related problems like potholes. By leveraging modern technologies such as machine learning, real-time communication, and AI, eNivaran aims to make the process of civic complaint resolution more efficient, transparent, and accountable.

## 2. User Management

The platform supports two distinct user roles with different levels of access and functionality:

-   **Citizen Users:**
    -   **Signup:** Citizens can register for a new account by providing a username, full name, and password. The password is securely stored using a password hash.
    -   **Login:** Registered users can log in to the platform to access its features.
-   **Administrators:**
    -   **Admin Login:** A dedicated login for administrators with predefined credentials (`admin001`/`admin$001`). Admins have full access to the admin dashboard and all platform management features.

## 3. Complaint Management

The complaint management system is the core of the eNivaran platform, designed to handle the entire lifecycle of a civic issue report.

-   **Raise a Complaint:** Users can submit a new complaint through a detailed form that captures the following information:
    -   **Description:** A textual description of the issue.
    -   **Location:** The location of the issue, captured via street, city, state, and zipcode, which is then converted to latitude and longitude coordinates.
    -   **Evidence:** Users can upload an image or a video as evidence of the issue. The system can process both formats.
-   **View All Complaints:** A public-facing page that displays all non-duplicate complaints. Users can sort the complaints by submission time or the number of upvotes.
-   **"My Complaints" Page:** A personalized page where users can view and track the status of all the complaints they have submitted.
-   **Upvote System:** To help prioritize issues, users can upvote existing complaints. This feature helps authorities identify and address the most critical problems first.
-   **Duplicate Complaint Detection:** To prevent redundant reports, the system employs a sophisticated duplicate detection mechanism. It analyzes the location, issue type, and image similarity of new complaints to identify and flag potential duplicates.

## 4. Admin Dashboard

The admin dashboard is a powerful tool that provides administrators with complete control over the platform's content and operations.

-   **Comprehensive Complaint View:** Admins can view a detailed list of all complaints, including user information, submission time, and current status.
-   **Status and Remarks Updates:** Admins can update the status of a complaint (e.g., "Submitted," "In Progress," "Resolved") and add remarks to provide feedback to the user.
-   **Complaint Deletion:** Admins have the authority to delete complaints that are irrelevant, resolved, or spam.
-   **Advanced Filtering:** To facilitate efficient management, admins can filter complaints by their unique ID or by city.

## 5. Pothole Detection

eNivaran uses a machine learning model to automatically detect potholes in the evidence provided by users.

-   **Image-based Detection:** When a user uploads an image, it is processed by a pre-trained machine learning model (`pothole_detector_v1.onnx`). The model identifies and highlights potholes in the image, providing an annotated version as output.
-   **Video-based Road Assessment:** For video uploads, the system analyzes the video frame by frame to detect potholes and assess the overall condition of the road.

## 6. Video Analysis

The platform's video analysis capabilities provide a more in-depth understanding of road conditions.

-   **Pothole Detection in Videos:** The system processes video uploads to identify and locate potholes throughout the video's duration.
-   **Road Damage Assessment:** In addition to detecting potholes, the video analysis feature calculates an average damage score for the road, providing a quantitative measure of its condition.
-   **Processed Video Output:** After the analysis is complete, the system generates a new video with the detected potholes highlighted. This processed video is then made available to both the user and the administrators.

## 7. AI Chatbot

The platform includes an AI-powered chatbot to assist users with their queries and guide them through the complaint submission process.

-   **Integration with Google Gemini:** The chatbot is powered by Google's advanced Gemini AI model, which enables it to understand and respond to a wide range of user queries.
-   **Multi-lingual Support:** The chatbot is designed to detect the user's language and respond in the same language, making the platform more accessible to a diverse user base.
-   **Guided Complaint Submission:** The chatbot can guide users on how to report a complaint, providing them with the correct format and ensuring that all the necessary information is included.

## 8. Real-time Chat

To facilitate communication between users and administrators, eNivaran integrates a real-time chat feature.

-   **Firebase Integration:** The chat functionality is built on Google's Firebase Realtime Database, which allows for instant messaging between users and admins.
-   **Complaint-specific Chat:** Each complaint has its own dedicated chat thread, allowing for focused discussions about the issue.

## 9. Database

The application's data is stored in an SQLite database, which is a lightweight, serverless, and self-contained database engine.

-   **Database Schema:** The database consists of several tables, including:
    -   `users`: Stores user information, such as username, full name, and hashed password.
    -   `complaints`: Contains all the details of the submitted complaints, including the description, location, evidence, and status.
    -   `pothole_detections`: Stores the results of the pothole detection analysis.
    -   `pothole_stats`: Keeps track of the overall pothole statistics.
    -   `upvotes`: Manages the upvote data for each complaint.
