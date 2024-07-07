from django.db import models

# Create your models here.

class Prediction(models.Model):
    image_name = models.CharField(max_length=255)
    predicted_class = models.CharField(max_length=100)
    confidence = models.FloatField()
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.image_name} - {self.predicted_class}"
