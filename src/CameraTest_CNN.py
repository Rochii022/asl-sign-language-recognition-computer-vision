import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os


# 1. Network Architecture

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(SignLanguageCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

classes = [ 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

model = SignLanguageCNN(num_classes=len(classes)).to(device)

try:
    model.load_state_dict(torch.load("SignLanguageCNN.pth", map_location=device, weights_only=True))
    print("Weights of the model loaded correctly.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit() 

model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 2. Loop for the camera feed and prediction
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

print("Camera initialized.")
print("---> PRESS 'q' TO EXIT.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    # Coordinates for the box where the hand should be placed
    x_initial, y_initial = 100, 100
    x_end, y_end = 350, 350
    
    cv2.rectangle(frame, (x_initial, y_initial), (x_end, y_end), (0, 255, 0), 2)
    roi = frame[y_initial:y_end, x_initial:x_end]
    
    roi_black_background = None

    try:
        #HSV Segmentation to isolate the hand
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 30, 60], dtype=np.uint8) 
        upper_skin = np.array([30, 255, 255], dtype=np.uint8)
        
        mask_raw = cv2.inRange(roi_hsv, lower_skin, upper_skin)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask_clean)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, cv2.FILLED)

        # IMAGEN FINAL CON FONDO NEGRO
        roi_black_background = cv2.bitwise_and(roi, roi, mask=final_mask)
        cv2.imshow('Image with Black Background', roi_black_background)

        # Prepare the image for the model
        roi_rgb = cv2.cvtColor(roi_black_background, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(roi_rgb)
        
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilidades = F.softmax(outputs, dim=1)
            max_prob, predicted_idx = torch.max(probabilidades, 1)
            
            percentage = max_prob.item() * 100
            possible_letter = classes[predicted_idx.item()]

        umbral_seguridad = 80.0 
        
        if percentage > umbral_seguridad:
            texto = f"{possible_letter} ({percentage:.1f}%)"
            color_texto = (0, 255, 0)
        else:
            texto = "Waiting for hand..."
            color_texto = (0, 165, 255)
            
        cv2.putText(frame, texto, (x_initial, y_initial - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2, cv2.LINE_AA)
        
    except Exception as e:
        pass 

    cv2.imshow('Real Time ASL Translator', frame)

    # Key to exit the program
    tecla = cv2.waitKey(1) & 0xFF
    
    if tecla == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program closed correctly.")
