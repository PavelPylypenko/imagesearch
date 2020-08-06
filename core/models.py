from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField
from django.db import models


class Image(models.Model):
    title = models.CharField(max_length=255)
    uploaded_by = models.ForeignKey(User, blank=True, null=True,
                                    on_delete=models.SET_NULL)
    keypoints = JSONField()
    descriptors = JSONField()

    def __str__(self):
        return self.title
