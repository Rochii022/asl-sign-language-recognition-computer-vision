import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Definir la Arquitectura de la CNN
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=24): # 24 letras (sin J ni Z por movimiento)
        super(SignLanguageCNN, self).__init__()
        # Entrada: Imagen RGB de 64x64 (3 canales)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Salida: 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Salida: 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Salida: 8x8
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Evita el overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 2. Configurar el entorno
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

# 3. Data Augmentation y Carga de Datos
# ¡CAMBIA ESTA RUTA por la carpeta de tu dataset!
DATASET_PATH = "./asl_dataset" 

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# Cargar todo el dataset
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
clases = dataset.classes # Guardamos los nombres de las letras (A, B, C...)
print(f"Clases detectadas ({len(clases)}): {clases}")

# --- NUEVO: Dividir en Entrenamiento (80%) y Testeo (20%) ---
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Crear los Loaders separados
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Shuffle=False para el test
print(f"Imágenes para entrenar: {train_size} | Imágenes para testear: {test_size}")

# 4. Inicializar modelo, función de pérdida y optimizador
model = SignLanguageCNN(num_classes=len(clases)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Bucle de Entrenamiento
epochs = 20
print("\n--- Iniciando entrenamiento ---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Época {epoch+1}/{epochs} - Pérdida: {running_loss/len(train_loader):.4f}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "modelo_signos.pth")
print("\nModelo guardado correctamente.")

# 6. --- NUEVO: Bucle de Evaluación (Test) ---
print("\n--- Iniciando Evaluación con Test Data ---")
model.eval() # Ponemos el modelo en modo evaluación (apaga el Dropout)
correct = 0
total = 0
muestra_actual = 1

# torch.no_grad() apaga el cálculo de gradientes para ahorrar memoria y hacer el test rápido
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # Cogemos la neurona con la puntuación más alta
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Imprimir Real vs Predicho para cada imagen del batch
        for i in range(len(labels)):
            letra_real = clases[labels[i].item()]
            letra_predicha = clases[predicted[i].item()]
            
            # Formatear el color del texto (verde si acierta, rojo si falla en consola)
            if letra_real == letra_predicha:
                resultado = f"✅ Real: {letra_real} | Predicción: {letra_predicha}"
            else:
                resultado = f"❌ Real: {letra_real} | Predicción: {letra_predicha}"
                
            print(f"Muestra {muestra_actual:04d} -> {resultado}")
            muestra_actual += 1

# Calcular e imprimir el Accuracy final
accuracy = 100 * correct / total
print("\n" + "="*40)
print(f"🏆 ACCURACY FINAL EN TEST DATA: {accuracy:.2f}%")
print("="*40)