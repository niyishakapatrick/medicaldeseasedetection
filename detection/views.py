import csv
import io  # Import the io module
import base64
from datetime import datetime
import os
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from django.shortcuts import render
from .forms import ImageUploadForm
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from django.conf import settings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict_single_image(image_path, model_path, class_names):
    # Load the model
    model = models.efficientnet_b2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Prepare the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

    # Get the predicted class
    predicted_class = class_names[predicted.item()]

    return predicted_class, probabilities

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_name = image.name
            
            # Make prediction
            model_path = os.path.join(settings.BASE_DIR, 't_v_c_n_best_model.pth')
            class_names = ['covid19', 'normal', 'tuberculosis', 'viral_pneumonia']
            predicted_class, probabilities = predict_single_image(image.file, model_path, class_names)
            
            # Plotting the probabilities (HTML version)
            plt.figure(figsize=(8, 6))
            colors = ['blue', 'green', 'orange', 'red']  # Define colors for each class
            plt.barh(class_names, probabilities, color=colors)
            plt.xlabel('Probabilities',fontsize=14, fontweight='bold')
            plt.title('Desease Probabilities',fontsize=14, fontweight='bold')
            plt.xticks(fontweight='bold',fontsize=10)
            plt.yticks(fontweight='bold',fontsize=14)
            plt.xlim(0, 1.0)  # Setting the limit for x-axis from 0 to 1 for probabilities
            plt.grid()
            plt.tight_layout()

            # Save the plot to a temporary buffer
            plot_buffer = io.BytesIO()
            plt.savefig(plot_buffer, format='png')
            plot_buffer.seek(0)

            # Convert plot buffer to base64 to embed in HTML
            plot_data = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
            plot_img = 'data:image/png;base64,' + plot_data

            # Append to CSV
            csv_path = os.path.join(settings.BASE_DIR, 'predictions.csv')
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_name] + list(probabilities))

            # Render result.html with prediction details
            return render(request, 'result.html', {
                'image_name': image_name,
                'predicted_class': predicted_class,
                'plot_img': plot_img
            })
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
