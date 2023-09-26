from django.db import models
from django.contrib.auth.models import User
# Create your models here.


# Create your models here.


class Book(models.Model):
    reader=  models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)
    name =models.CharField(max_length=100)
    pdf=models.FileField()
    def __str__(self):
        return str(self.name)
    

class Message(models.Model):
    user= models.ForeignKey(User, on_delete=models.CASCADE)
    human= models.TextField()
    ai= models.TextField()
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
         ordering = ['-created']
         
    def __str__(self):
        return self.human


    
    