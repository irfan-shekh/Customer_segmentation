from django import forms
from .models import CustomerSegmentation

VAR1_CHOICES = [
    ('', '-- Select Category (Optional) --'),
    ('Cat_1', 'Category 1 (Cat_1)'),
    ('Cat_2', 'Category 2 (Cat_2)'),
    ('Cat_3', 'Category 3 (Cat_3)'),
    ('Cat_4', 'Category 4 (Cat_4 - Common)'),
    ('Cat_5', 'Category 5 (Cat_5)'),
    ('Cat_6', 'Category 6 (Cat_6)'),
    ('Cat_7', 'Category 7 (Cat_7)'),
]

class CustomerSegmentationForm(forms.ModelForm):
    """Form for customer segmentation input with enhanced UX widgets"""
    
    var_1 = forms.ChoiceField(
        choices=VAR1_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_var_1'}),
        help_text="Anonymized customer classification category"
    )
    
    class Meta:
        model = CustomerSegmentation
        fields = [
            'gender', 'ever_married', 'age', 'graduated', 'profession',
            'work_experience', 'spending_score', 'family_size', 'var_1'
        ]
        widgets = {
            'gender': forms.Select(attrs={'class': 'form-select', 'id': 'id_gender'}),
            'ever_married': forms.Select(attrs={'class': 'form-select', 'id': 'id_ever_married'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': 18, 'max': 100, 'id': 'id_age'}),
            'graduated': forms.Select(attrs={'class': 'form-select', 'id': 'id_graduated'}),
            'profession': forms.Select(attrs={'class': 'form-select', 'id': 'id_profession'}),
            'work_experience': forms.NumberInput(attrs={
                'class': 'form-control', 'min': 0, 'max': 40, 'step': 0.5, 'id': 'id_work_experience'
            }),
            'spending_score': forms.Select(attrs={'class': 'form-select', 'id': 'id_spending_score'}),
            'family_size': forms.NumberInput(attrs={
                'class': 'form-control', 'min': 1, 'max': 15, 'step': 1, 'id': 'id_family_size'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['work_experience'].required = False
        self.fields['family_size'].required = False
        self.fields['var_1'].required = False
        
        self.fields['gender'].empty_label = "-- Select Gender --"
        self.fields['ever_married'].empty_label = "-- Marital Status --"
        self.fields['graduated'].empty_label = "-- Education Level --"
        self.fields['profession'].empty_label = "-- Select Profession --"
        self.fields['spending_score'].empty_label = "-- Spending Score --"
        
        # Help text
        self.fields['work_experience'].help_text = "Years of professional work experience (0 - 40)"
        self.fields['family_size'].help_text = "Total members in customer household"
        
        # Placeholders
        self.fields['age'].widget.attrs['placeholder'] = "e.g. 32"
        self.fields['work_experience'].widget.attrs['placeholder'] = "e.g. 5.0"
        self.fields['family_size'].widget.attrs['placeholder'] = "e.g. 3"
    
    def clean_age(self):
        age = self.cleaned_data.get('age')
        if age is not None and (age < 18 or age > 100):
            raise forms.ValidationError("Customer age must be between 18 and 100.")
        return age
    
    def clean_work_experience(self):
        work_experience = self.cleaned_data.get('work_experience')
        if work_experience is not None and (work_experience < 0 or work_experience > 40):
            raise forms.ValidationError("Work experience must be between 0 and 40 years.")
        return work_experience
    
    def clean_family_size(self):
        family_size = self.cleaned_data.get('family_size')
        if family_size is not None and (family_size < 1 or family_size > 15):
            raise forms.ValidationError("Family size must be between 1 and 15.")
        return family_size
