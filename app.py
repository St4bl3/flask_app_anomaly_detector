import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import time

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = os.path.join('static', 'output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- PyTorch Model Definition (Copied from your notebook) ---
# This must match the architecture of the saved model exactly.

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size, dropout_rate):
        super(ConvLSTMCell, self).__init__()
        # In PyTorch, 'same' padding is not a string argument.
        # We calculate it manually if needed, but for a 3x3 kernel, a padding of 1 is standard 'same'.
        if padding == 'same':
            padding = kernel_size[0] // 2
        self.conv = nn.Conv2d(in_channels + out_channels, 4 * out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        i, f, o, g = torch.chunk(conv_output, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = self.activation(g)

        c_next = f * c + i * g
        h_next = o * self.activation(c_next)
        h_next = self.dropout(h_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size, return_sequences=True, dropout_rate=0.0):
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.return_sequences = return_sequences
        self.cell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size, dropout_rate)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        h = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
        c = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :, :, :], h, c)
            if self.return_sequences:
                outputs.append(h.unsqueeze(1))
        
        if self.return_sequences:
            return torch.cat(outputs, dim=1)
        return h

class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.spatial_enc1 = nn.Conv3d(1, 128, kernel_size=(1, 11, 11), stride=(1, 4, 4), padding=(0, 0, 0))
        self.spatial_enc2 = nn.Conv3d(128, 64, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 0, 0))
        
        self.temporal_enc1 = ConvLSTM(64, 64, (3, 3), 'same', torch.tanh, (26, 26), dropout_rate=0.4)
        self.temporal_enc2 = ConvLSTM(64, 32, (3, 3), 'same', torch.tanh, (26, 26), dropout_rate=0.3)
        
        self.temporal_dec = ConvLSTM(32, 64, (3, 3), 'same', torch.tanh, (26, 26), dropout_rate=0.5)
        
        self.spatial_dec1 = nn.ConvTranspose3d(64, 128, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 0, 0))
        self.spatial_dec2 = nn.ConvTranspose3d(128, 1, kernel_size=(1, 11, 11), stride=(1, 4, 4), padding=(0, 0, 0))
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.spatial_enc1(x))
        x = self.tanh(self.spatial_enc2(x))
        x = x.permute(0, 2, 1, 3, 4)
        x = self.temporal_enc1(x)
        x = self.temporal_enc2(x)
        x = self.temporal_dec(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.tanh(self.spatial_dec1(x))
        x = self.tanh(self.spatial_dec2(x))
        return x

# --- Model and Data Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyDetector()
model_path = 'checkpoints/Avenue_Model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load normalization stats and threshold
norm_stats = np.load('checkpoints/norm_stats.npy', allow_pickle=True).item()
train_mean = norm_stats['mean']
train_std = norm_stats['std']
threshold_data = np.load('checkpoints/anomaly_threshold.npy', allow_pickle=True).item()
threshold = threshold_data['threshold']

print("--- Model and data loaded successfully ---")
print(f"Using device: {device}")
print(f"Anomaly Threshold: {threshold}")

# --- Helper Functions for Inference ---
def ImgProcessInfer(frame, mean, std, shape=(227, 227)):
    frame = cv2.resize(frame, shape)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray = np.dot(frame, rgb_weights)
    gray = (gray - mean) / std
    gray = np.clip(gray, 0, 1)
    return gray

def overlay_text(frame, text):
    """Adds text with a background to a frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Position at top-left corner
    x, y = 10, 10
    
    # Create a black rectangle background
    cv2.rectangle(frame, (x, y), (x + text_w + 10, y + text_h + 10), (0,0,0), -1)
    
    # Add text
    cv2.putText(frame, text, (x + 5, y + text_h + 5), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return frame

# --- Main Anomaly Detection Function ---
def detect_anomalies(video_path):
    """Processes a video to detect anomalies and saves the output."""
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    
    # Generate a unique output filename
    timestamp = int(time.time())
    output_filename = f'output_{timestamp}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # UPDATED: Use a more browser-compatible codec (H.264/AVC)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Check if the VideoWriter was successfully created
    if not out.isOpened():
        print("Error: Could not open video writer with 'avc1' codec. Trying fallback 'mp4v'.")
        # Fallback to the original codec if the preferred one is not available
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
             raise IOError("Error: Could not open video writer with any of the available codecs.")

    frame_queue = []
    frame_count = 0
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_queue.append(frame)
            frame_count += 1
            
            if len(frame_queue) == 10:
                img_lst = [ImgProcessInfer(f, train_mean, train_std) for f in frame_queue]
                img_arr = np.array(img_lst).reshape(1, 1, 10, 227, 227).astype(np.float32)
                img_tensor = torch.from_numpy(img_arr).to(device)
                
                recon_tensor = model(img_tensor)
                loss_val = torch.mean((img_tensor - recon_tensor)**2).item()
                
                text = 'Normal'
                if loss_val > float(threshold):
                    text = 'Anomaly Detected!'
                
                display_text = f'Loss: {loss_val:.5f} | {text}'
                print(f"Frame {frame_count}: {display_text}")

                for f in frame_queue:
                    processed_frame = overlay_text(f.copy(), display_text)
                    out.write(processed_frame)
                
                frame_queue = [] # Reset queue
    
    # Process any remaining frames in the queue
    if frame_queue:
        for f in frame_queue:
            # For the last few frames, we just write them without text or use the last known status
            out.write(f)

    cap.release()
    out.release()
    
    # ADDED: Check if the file was created successfully
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise IOError("Failed to create a valid output video file.")

    print(f"Output video saved to: {output_path}")
    return output_filename

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    # List default videos from the static directory
    default_video_dir = os.path.join('static', 'videos')
    default_videos = [f for f in os.listdir(default_video_dir) if f.endswith('.avi')]
    return render_template('index.html', default_videos=default_videos)

@app.route('/process', methods=['POST'])
def process_video():
    """Handles video processing request."""
    video_path = None
    
    # Check if a default video was selected
    selected_video = request.form.get('default_video')
    if selected_video:
        video_path = os.path.join('static', 'videos', selected_video)
    
    # Check if a file was uploaded
    elif 'video_upload' in request.files:
        file = request.files['video_upload']
        if file.filename != '':
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            video_path = upload_path

    if not video_path:
        return jsonify({'error': 'No video selected or uploaded.'}), 400

    try:
        output_filename = detect_anomalies(video_path)
        video_url = url_for('static', filename=f'output/{output_filename}')
        return jsonify({'video_url': video_url})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during video processing.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
