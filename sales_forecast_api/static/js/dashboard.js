/* dashboard.js — Sales Forecast API interactive dashboard */
'use strict';

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  const badge = document.getElementById('healthBadge');
  const dot   = badge.querySelector('.health-dot');
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    if (data.status === 'ok') {
      badge.innerHTML = '<span class="health-dot online"></span> API Online';
    } else {
      throw new Error('not ok');
    }
  } catch {
    badge.innerHTML = '<span class="health-dot offline"></span> API Offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);


// ── Single prediction form ────────────────────────────────────────────────────
const form        = document.getElementById('predictForm');
const predictBtn  = document.getElementById('predictBtn');
const resultCard  = document.getElementById('resultCard');
const placeholder = document.getElementById('resultPlaceholder');
const resultContent = document.getElementById('resultContent');
const resultValue   = document.getElementById('resultValue');
const resultMeta    = document.getElementById('resultMeta');
const resultTime    = document.getElementById('resultTime');
const resultError   = document.getElementById('resultError');
const requestJson   = document.getElementById('requestJson');
const responseJson  = document.getElementById('responseJson');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  predictBtn.disabled = true;
  predictBtn.textContent = 'Predicting...';
  resultError.classList.add('hidden');
  resultContent.classList.add('hidden');
  placeholder.classList.add('hidden');

  const payload = {
    Store_id:      parseInt(document.getElementById('store_id').value),
    Store_Type:    document.getElementById('store_type').value,
    Location_Type: document.getElementById('location_type').value,
    Region_Code:   document.getElementById('region_code').value,
    Date:          document.getElementById('date').value,
    Holiday:       parseInt(document.getElementById('holiday').value),
    Discount:      document.getElementById('discount').value,
  };

  requestJson.textContent = JSON.stringify(payload, null, 2);

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    responseJson.textContent = JSON.stringify(data, null, 2);

    if (!res.ok) {
      resultError.textContent = `Error ${res.status}: ${data.error || 'Unknown error'}`;
      resultError.classList.remove('hidden');
      placeholder.classList.add('hidden');
    } else {
      const sales = data.predicted_sales.toLocaleString('en-IN', {
        style: 'currency', currency: 'INR', minimumFractionDigits: 2,
      });
      resultValue.textContent = sales;
      resultMeta.innerHTML = [
        `<strong>Store ${data.store_id}</strong> · ${data.store_type}`,
        `${data.location_type} · ${data.region_code}`,
        `Date: ${data.date}`,
        `Discount: ${data.discount} · Holiday: ${data.holiday === 1 ? 'Yes' : 'No'}`,
      ].join('<br>');
      resultTime.textContent = `Inference: ${data.inference_time_ms} ms`;
      resultContent.classList.remove('hidden');
    }
  } catch (err) {
    responseJson.textContent = `{ "error": "${err.message}" }`;
    resultError.textContent = `Network error: ${err.message}`;
    resultError.classList.remove('hidden');
    placeholder.classList.add('hidden');
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict Sales';
  }
});


// ── Batch prediction ──────────────────────────────────────────────────────────
const csvFileInput  = document.getElementById('csvFile');
const fileLabelEl   = document.getElementById('fileLabel');
const batchBtn      = document.getElementById('batchBtn');
const batchResult   = document.getElementById('batchResult');
const batchSummary  = document.getElementById('batchSummary');
const batchTableBody = document.getElementById('batchTableBody');
const downloadBtn   = document.getElementById('downloadBtn');

let batchPredictions = [];

csvFileInput.addEventListener('change', () => {
  const file = csvFileInput.files[0];
  if (file) {
    fileLabelEl.textContent = `📄 ${file.name}`;
    batchBtn.disabled = false;
  } else {
    fileLabelEl.textContent = '📂 Choose CSV file';
    batchBtn.disabled = true;
  }
});

batchBtn.addEventListener('click', async () => {
  const file = csvFileInput.files[0];
  if (!file) return;

  batchBtn.disabled = true;
  batchBtn.textContent = 'Running...';
  batchResult.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res  = await fetch('/predict/batch', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      alert(`Error ${res.status}: ${data.error || 'Unknown error'}`);
      return;
    }

    batchPredictions = data.predictions;
    const total = batchPredictions.reduce((s, r) => s + r.predicted_sales, 0);
    const avg   = total / batchPredictions.length;

    batchSummary.innerHTML =
      `<strong>${data.count.toLocaleString()} records</strong> predicted in ` +
      `<strong>${data.inference_time_ms} ms</strong> · ` +
      `Total: <strong>₹${total.toLocaleString('en-IN', {maximumFractionDigits: 0})}</strong> · ` +
      `Avg/record: <strong>₹${avg.toLocaleString('en-IN', {maximumFractionDigits: 0})}</strong>`;

    batchTableBody.innerHTML = '';
    const preview = batchPredictions.slice(0, 50);
    for (const row of preview) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${row.store_id}</td>
        <td>${row.date}</td>
        <td>${row.store_type}</td>
        <td>${row.discount}</td>
        <td>${row.holiday === 1 ? 'Yes' : 'No'}</td>
        <td><strong>₹${row.predicted_sales.toLocaleString('en-IN', {minimumFractionDigits: 2})}</strong></td>
      `;
      batchTableBody.appendChild(tr);
    }

    if (batchPredictions.length > 50) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td colspan="6" style="text-align:center;color:#888;">
        Showing 50 of ${batchPredictions.length} rows. Download CSV for full results.
      </td>`;
      batchTableBody.appendChild(tr);
    }

    batchResult.classList.remove('hidden');

  } catch (err) {
    alert(`Network error: ${err.message}`);
  } finally {
    batchBtn.disabled = false;
    batchBtn.textContent = 'Run Batch Prediction';
  }
});

downloadBtn.addEventListener('click', () => {
  if (!batchPredictions.length) return;
  const headers = ['store_id','date','store_type','discount','holiday','predicted_sales'];
  const rows = batchPredictions.map(r =>
    headers.map(h => r[h]).join(',')
  );
  const csv  = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'sales_predictions.csv';
  a.click();
  URL.revokeObjectURL(url);
});
