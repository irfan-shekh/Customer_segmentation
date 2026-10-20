import csv
import json
import random
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Avg
from django.urls import reverse_lazy
from django.views.generic.edit import DeleteView

from .models import CustomerSegmentation, ModelPerformance
from .forms import CustomerSegmentationForm
from .ml_service import CustomerSegmentationService

def home(request):
    """Home page with interactive customer segmentation form"""
    if request.method == 'POST':
        form = CustomerSegmentationForm(request.POST)
        if form.is_valid():
            customer_data = {
                'Gender': form.cleaned_data['gender'],
                'Ever_Married': form.cleaned_data['ever_married'],
                'Age': form.cleaned_data['age'],
                'Graduated': form.cleaned_data['graduated'],
                'Profession': form.cleaned_data['profession'],
                'Work_Experience': form.cleaned_data['work_experience'],
                'Spending_Score': form.cleaned_data['spending_score'],
                'Family_Size': form.cleaned_data['family_size'],
                'Var_1': form.cleaned_data.get('var_1') or 'Cat_4'
            }
            
            prediction_method = request.POST.get('prediction_method', 'random_forest')
            ml_service = CustomerSegmentationService()
            result = ml_service.predict_segmentation(customer_data, prediction_method)
            insights = ml_service.get_segment_insights(result['predicted_segmentation'])
            
            # Save to SQLite database
            segmentation = form.save(commit=False)
            segmentation.predicted_segmentation = result['predicted_segmentation']
            segmentation.confidence_score = result['confidence_score']
            segmentation.prediction_method = result['prediction_method']
            segmentation.cluster_id = result.get('cluster_id')
            segmentation.processing_time = result['processing_time']
            segmentation.save()
            
            # Support AJAX modal/result responses if requested
            if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.POST.get('ajax') == '1':
                return JsonResponse({
                    'success': True,
                    'redirect_url': f'/detail/{segmentation.id}/',
                    'result': result,
                    'insights': insights,
                    'id': segmentation.id
                })
            
            context = {
                'segmentation': segmentation,
                'insights': insights,
                'form': CustomerSegmentationForm(),
            }
            messages.success(request, f'Customer segmented successfully as Segment {result["predicted_segmentation"]}!')
            return render(request, 'customer_segmentation/result.html', context)
        else:
            messages.error(request, 'Please correct the highlighted form errors.')
    else:
        form = CustomerSegmentationForm()
    
    total_count = CustomerSegmentation.objects.count()
    recent_segmentations = CustomerSegmentation.objects.all()[:6]
    
    context = {
        'form': form,
        'recent_segmentations': recent_segmentations,
        'total_count': total_count,
    }
    return render(request, 'customer_segmentation/home.html', context)

def sample_data(request):
    """Generate realistic persona sample data or randomized test data"""
    preset = request.GET.get('preset', 'random')
    
    presets = {
        'executive': {
            'gender': 'Male',
            'ever_married': 'Yes',
            'age': 52,
            'graduated': 'Yes',
            'profession': 'Executive',
            'work_experience': 14.0,
            'spending_score': 'High',
            'family_size': 2.0,
            'var_1': 'Cat_6'
        },
        'young_pro': {
            'gender': 'Female',
            'ever_married': 'No',
            'age': 29,
            'graduated': 'Yes',
            'profession': 'Engineer',
            'work_experience': 4.5,
            'spending_score': 'Average',
            'family_size': 1.0,
            'var_1': 'Cat_4'
        },
        'family': {
            'gender': 'Male',
            'ever_married': 'Yes',
            'age': 38,
            'graduated': 'Yes',
            'profession': 'Artist',
            'work_experience': 8.0,
            'spending_score': 'Average',
            'family_size': 5.0,
            'var_1': 'Cat_6'
        },
        'conservative': {
            'gender': 'Female',
            'ever_married': 'No',
            'age': 22,
            'graduated': 'No',
            'profession': 'Healthcare',
            'work_experience': 1.0,
            'spending_score': 'Low',
            'family_size': 3.0,
            'var_1': 'Cat_3'
        }
    }
    
    if preset in presets:
        selected_data = presets[preset]
    else:
        selected_data = {
            'gender': random.choice(['Male', 'Female']),
            'ever_married': random.choice(['Yes', 'No']),
            'age': random.randint(22, 65),
            'graduated': random.choice(['Yes', 'No']),
            'profession': random.choice([
                'Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 
                'Artist', 'Executive', 'Doctor', 'Homemaker', 'Marketing'
            ]),
            'work_experience': round(random.uniform(0.5, 18.0), 1),
            'spending_score': random.choice(['Low', 'Average', 'High']),
            'family_size': random.randint(1, 6),
            'var_1': f'Cat_{random.randint(1, 7)}'
        }
    
    # If AJAX request from home page presets
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
        return JsonResponse(selected_data)
        
    form = CustomerSegmentationForm(initial=selected_data)
    context = {
        'form': form,
        'sample_data': selected_data,
        'active_preset': preset,
        'recent_segmentations': CustomerSegmentation.objects.all()[:5],
        'total_count': CustomerSegmentation.objects.count()
    }
    messages.info(request, f'Loaded sample customer profile ({preset.replace("_", " ").title()}). Click "Segment Customer" to test.')
    return render(request, 'customer_segmentation/home.html', context)

def segmentation_history(request):
    """View all segmentation records with search, filtering, and pagination"""
    queryset = CustomerSegmentation.objects.all()
    
    # Filtering by segment
    segment_filter = request.GET.get('segment')
    if segment_filter in ['A', 'B', 'C', 'D']:
        queryset = queryset.filter(predicted_segmentation=segment_filter)
        
    # Filtering by method
    method_filter = request.GET.get('method')
    if method_filter:
        queryset = queryset.filter(prediction_method=method_filter)
        
    # Search query
    search_q = request.GET.get('q', '').strip()
    if search_q:
        queryset = queryset.filter(profession__icontains=search_q)
    
    paginator = Paginator(queryset, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_count': CustomerSegmentation.objects.count(),
        'filtered_count': queryset.count(),
        'active_segment': segment_filter,
        'active_method': method_filter,
        'search_q': search_q,
    }
    return render(request, 'customer_segmentation/history.html', context)

def segmentation_detail(request, pk):
    """Detailed view for a specific segmentation prediction"""
    segmentation = get_object_or_404(CustomerSegmentation, pk=pk)
    ml_service = CustomerSegmentationService()
    insights = ml_service.get_segment_insights(segmentation.predicted_segmentation)
    
    context = {
        'segmentation': segmentation,
        'insights': insights,
    }
    return render(request, 'customer_segmentation/detail.html', context)

def analytics_dashboard(request):
    """Analytics dashboard with rich data for visual charts"""
    total_segmentations = CustomerSegmentation.objects.count()
    segment_distribution = list(
        CustomerSegmentation.objects.values('predicted_segmentation')
        .annotate(count=Count('predicted_segmentation'))
        .order_by('predicted_segmentation')
    )
    method_distribution = list(
        CustomerSegmentation.objects.values('prediction_method')
        .annotate(count=Count('prediction_method'))
        .order_by('prediction_method')
    )
    spending_distribution = list(
        CustomerSegmentation.objects.values('spending_score')
        .annotate(count=Count('spending_score'))
        .order_by('-count')
    )
    profession_distribution = list(
        CustomerSegmentation.objects.values('profession')
        .annotate(count=Count('profession'))
        .order_by('-count')[:6]
    )
    
    # Calculate averages
    metrics = CustomerSegmentation.objects.aggregate(
        avg_score=Avg('confidence_score'),
        avg_age=Avg('age'),
        avg_experience=Avg('work_experience')
    )
    
    recent_segmentations = CustomerSegmentation.objects.all()[:8]
    
    context = {
        'total_segmentations': total_segmentations,
        'segment_distribution': segment_distribution,
        'method_distribution': method_distribution,
        'spending_distribution': spending_distribution,
        'profession_distribution': profession_distribution,
        'avg_confidence': round(metrics['avg_score'] or 0.0, 3),
        'avg_age': round(metrics['avg_age'] or 0.0, 1),
        'avg_experience': round(metrics['avg_experience'] or 0.0, 1),
        'recent_segmentations': recent_segmentations,
    }
    return render(request, 'customer_segmentation/analytics.html', context)

def model_performance(request):
    """View ML model evaluation metrics and feature importances"""
    ml_service = CustomerSegmentationService()
    performance_data = ml_service.get_model_performance()
    feature_importances = ml_service.get_feature_importances()
    performance_records = ModelPerformance.objects.all()
    
    rf_data = performance_data.get('Random Forest', {})
    dt_data = performance_data.get('Decision Tree', {})
    km_data = performance_data.get('KMeans Clustering', {})
    
    performance_cards = [
        {
            'name': 'Random Forest Classifier',
            'tag': 'Ensemble Model',
            'badge_class': 'bg-primary-subtle text-primary border',
            'border_class': 'border-primary',
            'text_class': 'text-primary',
            'progress_class': 'bg-primary',
            'description': 'Trained on 100 decision trees with balanced subsampling, reducing variance and delivering robust multi-class probability scores.',
            'accuracy': round(float(rf_data.get('accuracy', 0.535)), 3),
            'f1_score': round(float(rf_data.get('f1_score', 0.527)), 3),
            'precision': round(float(rf_data.get('precision', 0.524)), 3),
            'recall': round(float(rf_data.get('recall', 0.535)), 3),
            'points': ['Multi-class classification (A, B, C, D)', 'Max tree depth: 12 estimators', 'Confidence probability calibration']
        },
        {
            'name': 'Decision Tree Classifier',
            'tag': 'Rule-Based',
            'badge_class': 'bg-info-subtle text-info border',
            'border_class': 'border-info',
            'text_class': 'text-info',
            'progress_class': 'bg-info',
            'description': 'Single hierarchical decision tree with pruned depth. High interpretability and instant inference latency.',
            'accuracy': round(float(dt_data.get('accuracy', 0.525)), 3),
            'f1_score': round(float(dt_data.get('f1_score', 0.515)), 3),
            'precision': round(float(dt_data.get('precision', 0.517)), 3),
            'recall': round(float(dt_data.get('recall', 0.525)), 3),
            'points': ['Max depth: 8 levels', 'Low inference latency (< 2ms)', 'Explicit decision rules']
        },
        {
            'name': 'KMeans Clustering',
            'tag': 'Unsupervised',
            'badge_class': 'bg-warning-subtle text-dark border',
            'border_class': 'border-warning',
            'text_class': 'text-warning',
            'progress_class': 'bg-warning',
            'description': 'Unsupervised geometric clusterer operating on customer age and family lifecycle profiles to discover organic cohorts.',
            'silhouette_score': round(float(km_data.get('silhouette_score', 0.531)), 3),
            'inertia': round(float(km_data.get('inertia', 203189.8)), 1),
            'n_clusters': 4,
            'points': ['Geometric distance metric (Euclidean)', 'Natural customer separation', 'Combined with supervised prediction']
        }
    ]
    
    context = {
        'performance_cards': performance_cards,
        'feature_importances': feature_importances,
        'performance_records': performance_records,
    }
    return render(request, 'customer_segmentation/model_performance.html', context)

def export_history_csv(request):
    """Export all customer segmentation records as a CSV download"""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="customer_segmentations.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'ID', 'Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',
        'Work_Experience', 'Spending_Score', 'Family_Size', 'Category_Var1',
        'Predicted_Segment', 'Confidence_Score', 'Model_Used', 'Created_At'
    ])
    
    for seg in CustomerSegmentation.objects.all():
        writer.writerow([
            seg.id, seg.gender, seg.ever_married, seg.age, seg.graduated,
            seg.profession, seg.work_experience, seg.spending_score,
            seg.family_size, seg.var_1, seg.predicted_segmentation,
            seg.confidence_score, seg.prediction_method, seg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ])
        
    return response

def api_predict(request):
    """REST API endpoint for real-time customer segmentation predictions"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            required_fields = ['gender', 'ever_married', 'age', 'graduated', 'profession', 'spending_score']
            for field in required_fields:
                if field not in data:
                    return JsonResponse({'error': f'Missing required field: {field}'}, status=400)
            
            # Map input to expected capitalized format
            customer_data = {
                'Gender': data.get('gender'),
                'Ever_Married': data.get('ever_married'),
                'Age': float(data.get('age', 30)),
                'Graduated': data.get('graduated'),
                'Profession': data.get('profession'),
                'Work_Experience': float(data.get('work_experience') or 0.0),
                'Spending_Score': data.get('spending_score'),
                'Family_Size': float(data.get('family_size') or 1.0),
                'Var_1': data.get('var_1') or 'Cat_4'
            }
            
            ml_service = CustomerSegmentationService()
            prediction_method = data.get('prediction_method', 'random_forest')
            result = ml_service.predict_segmentation(customer_data, prediction_method)
            insights = ml_service.get_segment_insights(result['predicted_segmentation'])
            
            return JsonResponse({
                'success': True,
                'prediction': result,
                'insights': insights
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

class SegmentationDeleteView(DeleteView):
    """Delete a segmentation record"""
    model = CustomerSegmentation
    template_name = 'customer_segmentation/confirm_delete.html'
    success_url = reverse_lazy('customer_segmentation:segmentation_history')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Customer segmentation record deleted successfully.')
        return super().delete(request, *args, **kwargs)
