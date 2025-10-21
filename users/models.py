from django.db import models

class RegisteredUser(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    mobile = models.CharField(max_length=15)
    password = models.CharField(max_length=100)  # store plain for demo; use hashing in prod!
    image = models.ImageField(upload_to='user_images/')
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return self.name
