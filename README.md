# FetoVision: Fetal Abdominal Circumference Estimation System

An automated deep learning–based web system for estimating fetal abdominal circumference (AC) from ultrasound images.

FetoVision integrates deep learning image translation and medical image segmentation within a full-stack web application to support automated fetal biometry analysis for academic purposes.

To address differences in image quality, the system applies a BMI-aware inference approach that selects different deep learning pipelines based on maternal body composition.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deep Learning Model Weights](#deep-learning-model-weights)
- [Running Application](#running-application)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Built With](#built-with)
- [Contributing](#contributing)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## Features

* **BMI-Based Dual Deep Learning Pipeline**

  * BMI < 30: Direct fetal abdomen segmentation using U-Net
  * BMI ≥ 30: CycleGAN-based image translation followed by VNet2D segmentation
* **Deep Learning Domain Adaptation**

  * Image-to-image translation improves segmentation robustness in obese maternal cases
* **Deep Learning Medical Image Segmentation**

  * U-Net and VNet2D architectures implemented using PyTorch
* **Automated Abdominal Circumference Estimation**

  * Ellipse fitting and contour-based analysis from predicted segmentation masks
* **Near Real-Time Inference**

  * End-to-end prediction completed within seconds on CPU or GPU hardware
* **Visual Segmentation Feedback**

  * Segmentation masks overlaid on original ultrasound images
* **Prediction Record Management**

  * Storage and retrieval of historical prediction results
* **Cloud-Based Image Storage**

  * Ultrasound inputs and outputs managed via Cloudinary
* **Secure Web Interface**

  * Authenticated React frontend designed for research workflows

---

## System Architecture

```
          ┌─────────────────────────┐
          │        Frontend         │
          │        (React)          │
          └───────────┬─────────────┘
                      │ REST API
                      ▼
          ┌─────────────────────────┐
          │        Backend          │
          │       (FastAPI)         │
          └───────────┬─────────────┘
                      │
     ┌────────────────┼───────────────────┐
     ▼                ▼                   ▼
┌─────────┐   ┌────────────────┐   ┌──────────────┐
│ Database│   │ Deep Learning  │   │ Cloud Storage│
│ SQLite  │   │ Inference      │   │ Cloudinary   │
└─────────┘   │ • CycleGAN     │   └──────────────┘
              │ • U-Net        │
              │ • VNet2D       │
              └────────────────┘
```
The system uses **SQLite** as the database for storing user and prediction records, and **Cloudinary** for managing uploaded ultrasound images and generated segmentation outputs.  
For consistency with the provided implementation and configuration, it is recommended to use the same database and storage services.

---

## Project Structure

```
Fetal-Abdominal-Circumference-Estimation-System/
│
├── backend/
│   ├── app/              # FastAPI application (authentication, routes, database)
│   ├── model/            # Deep learning inference pipelines & architectures
│   ├── requirements.txt
│   └── .env.example      # Example backend environment variables
│
├── frontend/
│   ├── src/              # React source code
│   ├── public/
│   ├── .env.example      # Example frontend environment variables
│   └── package.json
│
├── .gitignore
├── .gitattributes        
└── README.md
```

---

## Technology Stack

### Backend

* Python 3.9+
* FastAPI
* SQLAlchemy
* Pydantic
* PyTorch
* OpenCV
* Pillow
* Cloudinary SDK

### Frontend

* React 18
* Vite
* React Router
* Fetch API
* CSS Modules

### Deep Learning

* PyTorch
* CycleGAN (image-to-image translation)
* U-Net (2D medical image segmentation)
* VNet2D (2D medical image segmentation)
* NumPy
* OpenCV (post-processing and contour analysis)

---

## Getting Started

Follow the instructions below to run the system locally for development or evaluation.

---

## Prerequisites

Ensure the following software is installed:

* Python 3.9 or later
* Node.js 16+
* npm
* Git
* Git LFS (required if using LFS-tracked model weights)

---

## Installation

### Clone the Repository

```
git clone https://github.com/joyeeteoh/Fetal-Abdominal-Circumference-Estimation-System.git
cd Fetal-Abdominal-Circumference-Estimation-System
```

### Backend Setup

```
cd backend
python -m venv venv
```

Activate the virtual environment:

**Windows**

```
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```
source venv/bin/activate
```

Install backend dependencies:

```
pip install -r requirements.txt
```

### Frontend Setup

```
cd frontend
npm install
```

---

## Configuration

### Backend Environment Variables

Create a `.env` file in the `backend` directory:

```
DATABASE_URL=sqlite:///./app.db

JWT_SECRET=change_me
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

CORS_ORIGINS=http://localhost:5173

CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
CLOUDINARY_UNSIGNED_PRESET=
CLOUDINARY_INPUT_FOLDER=
CLOUDINARY_OUTPUT_FOLDER=

AUTO_CREATE_TABLES=true
```

### Frontend Environment Variables

Create a `.env` file in the `frontend` directory:

```
VITE_API_BASE_URL=http://localhost:8000
```

---

## Deep Learning Model Weights

Place your trained deep learning model files at the following locations before running the application:


- **U-Net model**  
  `backend/model/weights/unet/unet_segmentation.pt`

- **VNet2D model**  
  `backend/model/weights/vnet/vnet_segmentation.pt`

- **CycleGAN generator (obese → non-obese)**  
  `backend/model/weights/cyclegan/obese2nonobese/latest_net_G_A.pth`


The system assumes **fixed filenames and directory structures**.  
Model files may be replaced or updated as long as the original paths and names are preserved.

---

## Running Application

### Start Backend Server

```
cd backend
uvicorn app.main:app --reload
```

API documentation will be available at:

```
http://localhost:8000/docs
```

### Start Frontend Server

```
cd frontend
npm run dev
```

Access the application at:

```
http://localhost:5173
```

---

## Usage

1. Register or log in to the system
2. Upload a fetal ultrasound image (PNG or JPG)
3. Provide:

   * Maternal BMI
   * Pixel-to-centimeter scale
4. Run abdominal circumference estimation
5. View:

   * Estimated AC value
   * Segmentation mask visualization
6. Save prediction results if required

### Deep Learning Pipeline Logic

* **BMI < 30**
  → U-Net segmentation applied directly

* **BMI ≥ 30**
  → CycleGAN image translation
  → VNet2D segmentation

---

## API Documentation

### Authentication Endpoints

* `POST /api/auth/register`
* `POST /api/auth/login`
* `GET /api/auth/me`
* `PUT /api/auth/me`

### Prediction Endpoints

* `POST /api/prediction/run`
* `POST /api/prediction/records`
* `GET /api/prediction/records`

Interactive API documentation is available via Swagger UI when the backend server is running.

---

## Troubleshooting

**Backend fails to start**

* Verify Python version
* Ensure `.env` file exists
* Confirm port 8000 is available

**Frontend cannot connect to backend**

* Confirm backend server is running
* Check `VITE_API_BASE_URL`
* Review CORS configuration

**Deep learning inference fails**

* Ensure model weight files exist
* Verify directory structure and filenames
* Confirm valid image input format

---

## Built With

* FastAPI
* React
* PyTorch
* Cloudinary
* SQLite

---

## Contributing

Contributions are welcome and appreciated. If you have suggestions for improvements, bug fixes, or feature enhancements, feel free to open an issue or submit a pull request.

---

## Authors

**Teoh Zhi Yee**  
Email: joyee0727@gmail.com  
Project Repository: https://github.com/joyeeteoh/Fetal-Abdominal-Circumference-Estimation-System

---

## Acknowledgments

This project would not have been possible without the support and contributions of the following individuals and institutions:

* **Dr. Saw Shier Nee**: Project supervisor, for her guidance, feedback, and academic support throughout the development of this work.
* **Mrs. Nurul Syazwani Jalil**: Clinical collaborator, for providing clinical insights and domain expertise related to fetal ultrasound imaging.
* **University of Malaya Medical Centre (UMMC)**: For providing the ultrasound data used in this project.
* **Faculty of Computer Science and Information Technology**: For access to academic resources and a supportive research environment.
