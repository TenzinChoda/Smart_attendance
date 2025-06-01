from django import forms

class StudentRegistrationForm(forms.Form):
    name = forms.CharField(max_length=100)
    roll_number = forms.CharField(max_length=20) 