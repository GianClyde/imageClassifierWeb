from django import forms
from .models import Images

class UploadForm(forms.ModelForm):
    class Meta:
        model = Images
        fields = ['image']
        