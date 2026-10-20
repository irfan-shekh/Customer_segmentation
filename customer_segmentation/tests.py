import json
from django.test import TestCase, Client
from django.urls import reverse
from .models import CustomerSegmentation
from .ml_service import CustomerSegmentationService

class CustomerSegmentationTests(TestCase):
    """Automated test suite for customer segmentation ML web app"""
    
    def setUp(self):
        self.client = Client()
        self.sample_customer_data = {
            'gender': 'Male',
            'ever_married': 'Yes',
            'age': 45,
            'graduated': 'Yes',
            'profession': 'Executive',
            'work_experience': 10.0,
            'spending_score': 'High',
            'family_size': 2.0,
            'var_1': 'Cat_6',
            'prediction_method': 'random_forest'
        }
    
    def test_home_page_status(self):
        """Test home view loads successfully"""
        response = self.client.get(reverse('customer_segmentation:home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Customer Segmentation')
        self.assertContains(response, 'Launch Profiler')
    
    def test_form_submission_creates_record(self):
        """Test customer form submission runs prediction and saves record"""
        response = self.client.post(reverse('customer_segmentation:home'), data=self.sample_customer_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Customer Classification Outcome')
        self.assertEqual(CustomerSegmentation.objects.count(), 1)
        
        record = CustomerSegmentation.objects.first()
        self.assertIn(record.predicted_segmentation, ['A', 'B', 'C', 'D'])
        self.assertGreater(record.confidence_score, 0.0)
    
    def test_sample_data_endpoint(self):
        """Test sample data generator loads preset and returns 200"""
        response = self.client.get(reverse('customer_segmentation:sample_data') + '?preset=executive')
        self.assertEqual(response.status_code, 200)
        
        # JSON preset test
        json_resp = self.client.get(reverse('customer_segmentation:sample_data') + '?preset=executive&format=json')
        self.assertEqual(json_resp.status_code, 200)
        data = json_resp.json()
        self.assertEqual(data.get('profession'), 'Executive')
    
    def test_history_page(self):
        """Test history archive page renders and filters work"""
        # Create a test record
        CustomerSegmentation.objects.create(
            gender='Female',
            ever_married='No',
            age=30,
            graduated='Yes',
            profession='Engineer',
            work_experience=5.0,
            spending_score='Average',
            family_size=1.0,
            var_1='Cat_4',
            predicted_segmentation='B',
            confidence_score=0.85,
            prediction_method='random_forest',
            processing_time=0.02
        )
        response = self.client.get(reverse('customer_segmentation:segmentation_history'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Engineer')
        self.assertContains(response, 'Segment B')
    
    def test_analytics_page(self):
        """Test analytics dashboard loads with chart data"""
        response = self.client.get(reverse('customer_segmentation:analytics'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Customer Cohort Analytics')
    
    def test_model_performance_page(self):
        """Test model diagnostics page loads successfully"""
        response = self.client.get(reverse('customer_segmentation:model_performance'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Machine Learning Engines')
        self.assertContains(response, 'Random Forest Classifier')
    
    def test_export_csv_endpoint(self):
        """Test CSV export endpoint generates downloadable CSV file"""
        CustomerSegmentation.objects.create(
            gender='Male',
            ever_married='Yes',
            age=40,
            graduated='Yes',
            profession='Artist',
            work_experience=6.0,
            spending_score='High',
            family_size=3.0,
            var_1='Cat_6',
            predicted_segmentation='A',
            confidence_score=0.88,
            prediction_method='random_forest',
            processing_time=0.01
        )
        response = self.client.get(reverse('customer_segmentation:export_history_csv'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        self.assertIn('Predicted_Segment', response.content.decode('utf-8'))
        self.assertIn('Artist', response.content.decode('utf-8'))
    
    def test_api_predict_endpoint(self):
        """Test REST API predict endpoint returns JSON response"""
        payload = {
            'gender': 'Male',
            'ever_married': 'Yes',
            'age': 35,
            'graduated': 'Yes',
            'profession': 'Doctor',
            'work_experience': 7.0,
            'spending_score': 'High',
            'family_size': 3.0,
            'var_1': 'Cat_6',
            'prediction_method': 'random_forest'
        }
        response = self.client.post(
            reverse('customer_segmentation:api_predict'),
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get('success'))
        self.assertIn(data['prediction']['predicted_segmentation'], ['A', 'B', 'C', 'D'])
        self.assertIn('insights', data)
    
    def test_ml_service_standalone(self):
        """Test CustomerSegmentationService directly"""
        service = CustomerSegmentationService()
        result = service.predict_segmentation({
            'Gender': 'Female',
            'Ever_Married': 'No',
            'Age': 24,
            'Graduated': 'No',
            'Profession': 'Healthcare',
            'Work_Experience': 1.0,
            'Spending_Score': 'Low',
            'Family_Size': 4.0,
            'Var_1': 'Cat_3'
        }, method='random_forest')
        
        self.assertIn(result['predicted_segmentation'], ['A', 'B', 'C', 'D'])
        self.assertIsInstance(result['confidence_score'], float)
        self.assertIsInstance(result['processing_time'], float)
