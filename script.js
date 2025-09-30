const imageInput = document.getElementById('imageInput');
const dropzone = document.getElementById('dropzone');
const browseBtn = document.getElementById('browseBtn');
const preview = document.getElementById('preview');
const form = document.getElementById('uploadForm');
const results = document.getElementById('results');
const predictionText = document.getElementById('predictionText');
const probabilitiesEl = document.getElementById('probabilities');
const predictBtn = document.getElementById('predictBtn');

function resetResults() {
  results.classList.add('hidden');
  predictionText.textContent = '—';
  probabilitiesEl.innerHTML = '';
}

function setPreview(file) {
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.classList.remove('hidden');
}

browseBtn.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    alert('File too large. Max 10 MB.');
    imageInput.value = '';
    return;
  }
  setPreview(file);
  resetResults();
});

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  imageInput.files = e.dataTransfer.files;
  setPreview(file);
  resetResults();
});

form.addEventListener('submit', async e => {
  e.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    alert('Please select an image first.');
    return;
  }
  predictBtn.disabled = true;
  predictBtn.textContent = 'Analyzing…';

  try {
    const formData = new FormData();
    formData.append('image', file);
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');

    results.classList.remove('hidden');
    predictionText.textContent = data.prediction;
    probabilitiesEl.innerHTML = '';
    (data.probabilities || []).forEach(item => {
      const wrap = document.createElement('div');
      wrap.className = 'prob';
      const pct = Math.round(item.probability * 100);
      wrap.innerHTML = `<div>${item.class} — <strong>${pct}%</strong></div>
        <div class="bar"><span style="width:${pct}%;"></span></div>`;
      probabilitiesEl.appendChild(wrap);
    });
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Analyze Image';
  }
});

