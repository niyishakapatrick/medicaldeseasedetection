import csv
from datetime import datetime
import os
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
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Get the predicted class and its probability
    predicted_class = class_names[predicted.item()]
    predicted_prob = probability[predicted.item()].item()

    return predicted_class, predicted_prob

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_name = image.name
            
            # Make prediction
            model_path = os.path.join(settings.BASE_DIR, 't_v_c_n_best_model.pth')
            class_names = ['covid19', 'normal', 'tuberculosis', 'viral_pneumonia']
            predicted_class, confidence = predict_single_image(image.file, model_path, class_names)
            
            # Append to CSV
            csv_path = os.path.join(settings.BASE_DIR, 'predictions.csv')
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_name, predicted_class, confidence, datetime.now()])
            
            return render(request, 'result.html', {
                'image_name': image_name,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
