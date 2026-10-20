from django.urls import path
from . import views

app_name = 'customer_segmentation'

urlpatterns = [
    path('', views.home, name='home'),
    path('sample-data/', views.sample_data, name='sample_data'),
    path('history/', views.segmentation_history, name='segmentation_history'),
    path('export-csv/', views.export_history_csv, name='export_history_csv'),
    path('detail/<int:pk>/', views.segmentation_detail, name='segmentation_detail'),
    path('analytics/', views.analytics_dashboard, name='analytics'),
    path('model-performance/', views.model_performance, name='model_performance'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('delete/<int:pk>/', views.SegmentationDeleteView.as_view(), name='delete_segmentation'),
]
