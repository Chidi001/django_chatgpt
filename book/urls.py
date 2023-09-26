from django.urls import  path
from book import views



urlpatterns = [
    path('',views.home,name='home'),
    path('upload',views.upload,name='upload'),
    path('review/<str:pk>/',views.conversation,name='review'),
    
]

