// --- Canvas setup ---
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Fondo negro para simular MNIST (trazo blanco)
function resetCanvas() {
  ctx.fillStyle = 'black';
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
  // coordenadas relativas al canvas
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

// Botón limpiar
document.getElementById('clearBtn').addEventListener('click', () => {
  resetCanvas();
  document.getElementById('prediction').innerText = 'Predicción: -';
  if (window.probChart) {
    window.probChart.destroy();
    window.probChart = null;
  }
});

// Botón predecir
document.getElementById('predictBtn').addEventListener('click', async () => {
  try {
    const dataURL = canvas.toDataURL('image/png');
    const fd = new FormData();
    fd.append('image_base64', dataURL);

    const res = await fetch('/predict', {
      method: 'POST',
      body: fd
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
    alert('No se pudo predecir. Revisa la consola (F12).');
  }
});

