import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Create two columns
col1, col2 = st.columns([1, 3])

# Function to load and predict image
def predict_single_image(image_path, model_path, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Get predicted class and all probabilities
    predicted_class = class_names[predicted.item()]
    probabilities = probabilities.cpu().numpy()

    return predicted_class, probabilities

# Define the Streamlit app
def main():
    # Column 1: Logo
    with col1:
        logo = 'AQS.png'  # Replace with the path to your logo file
        st.image(logo, width=200)  # Adjust width as needed
    # Column 2: Title and Description
    with col2:
        st.markdown("<h4 style="color: black;">COVID-19,Tuberculosis  Viral Pneumonia diagnosis from Chest X-ray </h4>")
    st.warning("Detects COVID-19, Tuberculosis, and Viral Pneumonia")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...",  type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Make prediction on the uploaded image
        model_path = 't_v_c_n_best_model.pth'  # Path to your model
        class_names = ['covid19', 'normal', 'tuberculosis', 'viral_pneumonia']

        predicted_class, probabilities = predict_single_image(uploaded_file, model_path, class_names)
        st.markdown(f'Detected desease: {predicted_class}')

        # Plot probabilities as horizontal bar chart with percentage labels
        fig, ax = plt.subplots()
        colors = ['blue', 'green', 'orange', 'red']
        bars = ax.barh(class_names, probabilities, color=colors)

        for bar, prob in zip(bars, probabilities):
            ax.text(bar.get_width() - 0.1, bar.get_y() + bar.get_height()/2, f'{prob*100:.1f}%', 
                    va='center', ha='right', color='white', fontsize=12, fontweight='bold')  # Make text bold

        ax.set_xlabel('Probability')
        ax.set_title('Probabilities')
        ax.set_xlim(0, 1.0)
        ax.grid(True)  # Add grid
        st.pyplot(fig)

    st.markdown("<small style='color: lightblue;'><hr></small>", unsafe_allow_html=True)
    st.markdown("<small style='color: darkblue;'>Keeza ~ Tech | +250788384528 | keey08@gmail.com |Noorsken - Kigali - Rwanda</small>", unsafe_allow_html=True)

    st.markdown(
        """
        <script>
        setInterval(function() {
            fetch('/stream');
        }, 60000);  // Ping the server every minute
        </script>
        """,
        unsafe_allow_html=True
    )


# Run the app
if __name__ == '__main__':
    main()
