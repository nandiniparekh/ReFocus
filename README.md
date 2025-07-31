# focuSync

**focuSync** is an AI-powered attention detection system designed to help individuals reclaim their focus in a world full of distractions. Created during the AI4Good Lab 2025 by a diverse and passionate team, focuSync empowers users to recognize when they are distracted and take actionable steps toward improving their attention and productivity.  
🔗 [Visit the Project Website](https://almasoriaw.github.io/focuSync-website/index.html)

## 🌍 Project Vision

> *"How can we change the world if we’re constantly distracted?"*

In a world where:
- Over 60% of people feel their attention span has decreased,
- 70% report being distracted often or very often at work,

focuSync aims to build an intelligent, behavior-based system that understands when you're focused or distracted — and helps you take back control.


## 🧠 What It Does

focuSync uses **computer vision** and a **recurrent neural network (LSTM)** to classify whether a person is focused or distracted based on their facial behavior. The model is built using:

- **OpenFace 3.0** for facial landmark detection and preprocessing
- **Custom LSTM model** trained on processed facial data
- A pipeline that outputs **FOCUSED** or **DISTRACTED** classifications

This enables real-time or near real-time feedback to help users understand and manage their attention levels.


## 🔬 Methodology

### 📐 System Architecture

The focuSync system follows a modular pipeline architecture:

```
Video Input → OpenFace 3.0 Feature Extraction → Sequential Data Processing → LSTM Model → Focus State Prediction → Annotated Output
```


### 🧊 OpenFace 3.0 Integration

We leverage OpenFace 3.0 for sophisticated facial analysis:

- Facial landmark detection (68 points)
- Action Unit (AU) intensity estimation
- Head pose estimation
- Gaze direction tracking


### 🧪 Feature Extraction Implementation

Example: Head pose estimation using 3D orientation features:

```python
def estimate_head_pose(landmarks, image_shape):
    import numpy as np
    import cv2

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    image_points = np.array(landmarks, dtype="double")
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rot_vec)
    proj_matrix = np.hstack((rotation_matrix, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = [angle[0] for angle in euler_angles]
    return pitch, yaw, roll
```

### 🔄 LSTM Model Architecture

Implemented in PyTorch:

```python
class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, layer_dim=2, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
```


### ⚙️ Processing Pipeline

1. **Video Processing** – Frame extraction and preparation  
2. **Feature Extraction** – Using OpenFace 3.0 to extract facial features  
3. **Sequence Creation** – Building temporal sequences for LSTM input  
4. **Focus Classification** – Applying the trained model to detect focus states  
5. **Output Annotation** – Visualizing results on the original video  


### 🛠️ Implementation Environment

- **Python** 3.8+
- **PyTorch** for deep learning components
- **OpenCV** for image processing
- **OpenFace 3.0** for facial analysis
- **Google Colab** for accessible deployment


### ⚡ Efficient Frame-by-Frame Processing

- Batched frame processing for video analysis
- Optimized feature extraction pipeline
- Efficient LSTM implementation for temporal analysis
- Frame-by-frame annotation with minimal latency

> **Note:** While the current setup processes pre-recorded videos, future phases aim to support real-time streaming applications.

## 💡 Design Philosophy

Unlike assumption-based or one-size-fits-all focus tools, focuSync is:
- ✅ **Behavior-Based**
- ✅ **Automatic (No manual input required)**
- ✅ **Personalized and Transparent**


## 🔐 Ethical AI Principles

We take AI responsibility seriously. Here’s how:

- **Data Privacy & Security**
  - Every data point is collected with **informed consent**
  - Clear governance over how data is handled and protected
  - Anonymized datasets to safeguard identities

- **Transparency & Bias Mitigation**
  - Transparent model use and algorithmic logic
  - Control for users over what data is collected or deleted
  - Inclusive and diverse training data


## Contributors

- **Nandini Parekh** – Software Engineer  
- **Alma Soria** – ML Student  
- **Juliane Phan** – Computer Science Student  
- **Heba Iftikhar** – Project Manager
