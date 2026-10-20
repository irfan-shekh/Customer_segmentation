from django.contrib import admin
from .models import CustomerSegmentation, ModelPerformance

@admin.register(CustomerSegmentation)
class CustomerSegmentationAdmin(admin.ModelAdmin):
    """Admin interface for Customer Segmentation model"""
    
    list_display = [
        'id', 'gender', 'age', 'profession', 'predicted_segmentation', 
        'confidence_score', 'prediction_method', 'created_at'
    ]
    list_filter = [
        'predicted_segmentation', 'gender', 'profession', 'spending_score', 
        'ever_married', 'graduated', 'prediction_method'
    ]
    search_fields = ['profession', 'predicted_segmentation']
    readonly_fields = [
        'predicted_segmentation', 'confidence_score', 'prediction_method', 
        'cluster_id', 'processing_time', 'created_at'
    ]
    
    fieldsets = (
        ('Customer Information', {
            'fields': ('gender', 'ever_married', 'age', 'graduated', 'profession')
        }),
        ('Financial Information', {
            'fields': ('work_experience', 'spending_score', 'family_size', 'var_1')
        }),
        ('Prediction Results', {
            'fields': ('predicted_segmentation', 'confidence_score', 'prediction_method', 'cluster_id')
        }),
        ('Metadata', {
            'fields': ('processing_time', 'created_at'),
            'classes': ('collapse',)
        }),
    )
    
    def has_add_permission(self, request):
        """Disable manual addition through admin"""
        return False
    
    def get_readonly_fields(self, request, obj=None):
        """Make all fields readonly for existing objects"""
        if obj:
            return [field.name for field in self.model._meta.fields]
        return self.readonly_fields

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    """Admin interface for Model Performance model"""
    
    list_display = ['model_name', 'accuracy_score', 'training_date']
    list_filter = ['model_name', 'training_date']
    search_fields = ['model_name']
    readonly_fields = ['training_date']
    
    fieldsets = (
        ('Model Information', {
            'fields': ('model_name', 'accuracy_score')
        }),
        ('Parameters', {
            'fields': ('parameters',)
        }),
        ('Metadata', {
            'fields': ('training_date',)
        }),
    )
