/**
 * Customer Segmentation ML - Interactive Frontend Controller
 */

// Preset Persona Definitions for 1-Click Form Pre-fill
const CUSTOMER_PRESETS = {
  executive: {
    gender: 'Male',
    ever_married: 'Yes',
    age: 52,
    graduated: 'Yes',
    profession: 'Executive',
    work_experience: 14.0,
    spending_score: 'High',
    family_size: 2.0,
    var_1: 'Cat_6',
    name: 'Segment A: Corporate Executive'
  },
  young_pro: {
    gender: 'Female',
    ever_married: 'No',
    age: 29,
    graduated: 'Yes',
    profession: 'Engineer',
    work_experience: 4.5,
    spending_score: 'Average',
    family_size: 1.0,
    var_1: 'Cat_4',
    name: 'Segment B: Rising Tech Specialist'
  },
  family: {
    gender: 'Male',
    ever_married: 'Yes',
    age: 38,
    graduated: 'Yes',
    profession: 'Artist',
    work_experience: 8.0,
    spending_score: 'Average',
    family_size: 5.0,
    var_1: 'Cat_6',
    name: 'Segment C: Household Builder'
  },
  conservative: {
    gender: 'Female',
    ever_married: 'No',
    age: 22,
    graduated: 'No',
    profession: 'Healthcare',
    work_experience: 1.0,
    spending_score: 'Low',
    family_size: 3.0,
    var_1: 'Cat_3',
    name: 'Segment D: Prudent Saver'
  }
};

/**
 * Apply persona preset directly to the form
 */
function applyPreset(presetKey) {
  const data = CUSTOMER_PRESETS[presetKey];
  if (!data) return;

  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) {
      el.value = val;
      el.dispatchEvent(new Event('change'));
    }
  };

  setVal('id_gender', data.gender);
  setVal('id_ever_married', data.ever_married);
  setVal('id_age', data.age);
  setVal('id_graduated', data.graduated);
  setVal('id_profession', data.profession);
  setVal('id_work_experience', data.work_experience);
  setVal('id_spending_score', data.spending_score);
  setVal('id_family_size', data.family_size);
  setVal('id_var_1', data.var_1);

  // Update visual state of preset pills
  document.querySelectorAll('.preset-badge').forEach(pill => {
    pill.classList.remove('active');
  });
  const activePill = document.querySelector(`[data-preset="${presetKey}"]`);
  if (activePill) {
    activePill.classList.add('active');
  }

  // Smooth pulse highlight on the form
  const formCard = document.getElementById('segmentationFormCard');
  if (formCard) {
    formCard.style.transition = 'box-shadow 0.3s ease';
    formCard.style.boxShadow = '0 0 0 4px rgba(79, 70, 229, 0.35)';
    setTimeout(() => {
      formCard.style.boxShadow = '';
    }, 800);
  }
}

/**
 * Initialize on page load
 */
document.addEventListener('DOMContentLoaded', () => {
  // Preset buttons event listener
  document.querySelectorAll('[data-preset]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const presetKey = btn.getAttribute('data-preset');
      applyPreset(presetKey);
    });
  });

  // Handle form submission animation
  const form = document.getElementById('customerForm');
  if (form) {
    form.addEventListener('submit', () => {
      const submitBtn = document.getElementById('submitBtn');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
          <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
          Analyzing Customer Features...
        `;
      }
    });
  }

  // Auto-dismiss alert banners after 5 seconds
  const alerts = document.querySelectorAll('.alert-dismissible');
  alerts.forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) bsAlert.close();
    }, 6000);
  });

  // Auto-sync dynamic progress-bar widths from data-width or aria-valuenow
  // This avoids invalid CSS syntax errors in editor linters caused by Django tags in style=""
  document.querySelectorAll('.progress-bar[data-width], .progress-bar[aria-valuenow], .mini-progress-bar[data-width]').forEach(bar => {
    const width = bar.getAttribute('data-width') || bar.getAttribute('aria-valuenow');
    if (width) {
      bar.style.width = `${width}%`;
    }
  });
});
