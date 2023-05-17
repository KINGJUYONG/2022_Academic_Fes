from django.db import models

# Create your models here.

class Timer(models.Model):
    BERT = models.IntegerField(default=0)
    LSTM = models.IntegerField(default=0)

class Profile(models.Model):
    name = models.CharField(max_length=30)
    audio = models.FileField(upload_to="audios/")
    def __str__(self):
        return self.title
