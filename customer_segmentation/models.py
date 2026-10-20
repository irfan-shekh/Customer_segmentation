from django.db import models
from django.utils import timezone

class CustomerSegmentation(models.Model):
    """Model to store customer segmentation results"""
    
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]
    
    MARRIED_CHOICES = [
        ('Yes', 'Yes'),
        ('No', 'No'),
    ]
    
    GRADUATED_CHOICES = [
        ('Yes', 'Yes'),
        ('No', 'No'),
    ]
    
    PROFESSION_CHOICES = [
        ('Healthcare', 'Healthcare'),
        ('Engineer', 'Engineer'),
        ('Lawyer', 'Lawyer'),
        ('Entertainment', 'Entertainment'),
        ('Artist', 'Artist'),
        ('Executive', 'Executive'),
        ('Doctor', 'Doctor'),
        ('Homemaker', 'Homemaker'),
        ('Marketing', 'Marketing'),
    ]
    
    SPENDING_CHOICES = [
        ('Low', 'Low'),
        ('Average', 'Average'),
        ('High', 'High'),
    ]
    
    SEGMENTATION_CHOICES = [
        ('A', 'Segment A'),
        ('B', 'Segment B'),
        ('C', 'Segment C'),
        ('D', 'Segment D'),
    ]
    
    # Input fields
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    ever_married = models.CharField(max_length=3, choices=MARRIED_CHOICES)
    age = models.IntegerField()
    graduated = models.CharField(max_length=3, choices=GRADUATED_CHOICES)
    profession = models.CharField(max_length=20, choices=PROFESSION_CHOICES)
    work_experience = models.FloatField(null=True, blank=True)
    spending_score = models.CharField(max_length=10, choices=SPENDING_CHOICES)
    family_size = models.FloatField(null=True, blank=True)
    var_1 = models.CharField(max_length=10, null=True, blank=True)
    
    # Prediction results
    predicted_segmentation = models.CharField(max_length=1, choices=SEGMENTATION_CHOICES)
    confidence_score = models.FloatField()
    prediction_method = models.CharField(max_length=20)  # Random Forest, Decision Tree, KMeans
    
    # Clustering results (for KMeans)
    cluster_id = models.IntegerField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    processing_time = models.FloatField()  # in seconds
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Customer Segmentation"
        verbose_name_plural = "Customer Segmentations"
    
    def __str__(self):
        return f"Customer {self.id} - Segment {self.predicted_segmentation}"

class ModelPerformance(models.Model):
    """Model to store ML model performance metrics"""
    
    model_name = models.CharField(max_length=50)
    accuracy_score = models.FloatField()
    training_date = models.DateTimeField(default=timezone.now)
    parameters = models.JSONField(default=dict)  # Store model hyperparameters
    
    class Meta:
        ordering = ['-training_date']
        verbose_name = "Model Performance"
        verbose_name_plural = "Model Performances"
    
    def __str__(self):
        return f"{self.model_name} - {self.accuracy_score:.2%}"
