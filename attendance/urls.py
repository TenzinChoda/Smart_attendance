from django.urls import path
from . import views

urlpatterns = [
    path('', views.register_student, name='register'),
    path('register/', views.register_student, name='register'),
    path('capture_frame/', views.capture_frame, name='capture_frame'),
    path('train_model/', views.train_model, name='train_model'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('recognize/', views.video_feed, name='recognize'),
    path('attendance_logs/', views.view_attendance, name='attendance_logs'),
    path('delete_attendance/', views.delete_attendance, name='delete_attendance'),
    path('registrations/', views.view_registrations, name='registrations'),
    path('delete_registration/', views.delete_registration, name='delete_registration'),
] 