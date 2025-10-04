// --- Canvas ---
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function resetCanvas() {
  ctx.fillStyle = 'black';      // fondo negro (como MNIST invertido)
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
resetCanvas();

let drawing = false;

canvas.addEventListener('mousedown', () => { drawing = true; });
canvas.addEventListener('mouseup',   () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mouseleave',() => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);

function draw(e){
  if(!drawing) return;
  ctx.lineWidth = 16;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'white';
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

document.getElementById('clearBtn').addEventListener('click', () => {
  resetCanvas();
  document.getElementById('prediction').innerText = 'Predicción: -';
  if (window.probChart) {
    window.probChart.destroy();
    window.probChart = null;
  }
});

document.getElementById('predictBtn').addEventListener('click', async () => {
  try {
    const dataURL = canvas.toDataURL('image/png'); // data:image/png;base64,...
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL })
    });

    const out = await res.json();
    if (out.error) {
      alert('Error: ' + out.error);
      return;
    }

    document.getElementById('prediction').innerText = 'Predicción: ' + out.prediction;
    renderProbabilities(out.probs);
  } catch (err) {
    console.error(err);
    alert('No se pudo predecir. Abre la consola (F12) para ver detalles.');
  }
});

