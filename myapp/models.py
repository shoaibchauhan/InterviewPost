from django.db import models

class Filee(models.Model):
    id = models.AutoField(primary_key=True)  # Auto-incrementing integer ID
    filename = models.CharField(max_length=250, unique=True)
    filesummary = models.TextField()


    def __str__(self):
        return self.filename  # Return the filename as the string representation