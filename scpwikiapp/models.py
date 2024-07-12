from django.db import models
from django.contrib.auth.models import User 
# Create your models here.

class ChatSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession,related_name='messages',on_delete=models.CASCADE)
    sender = models.CharField(max_length=255)
    message = models.TextField()