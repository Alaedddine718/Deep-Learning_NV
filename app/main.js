const canvas = document.getElementById('draw');
const ctx = canvas.getContext('2d');
let drawing = false;
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.strokeStyle = '#000';

canvas.addEventListener('mousedown', e => { drawing = true; draw(e); });
canvas.addEventListener('mouseup',   () => drawing = false);
canvas.addEventListener('mouseout',  () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e){
  if(!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 0.1, y + 0.1);
  ctx.stroke();
  if(window.socket){
    window.socket.emit('stroke', {points:[[x,y]]});
  }
}

document.getElementById('clear').onclick = () => {
  ctx.clearRect(0,0,canvas.width, canvas.height);
};

function canvasToDataURL(){
  return canvas.toDataURL('image/png');
}

async function predictBase64(dataURL){
  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({image_base64: dataURL})
  });
  return res.json();
}

document.getElementById('predict').onclick = async () => {
  const dataURL = canvasToDataURL();
  const out = await predictBase64(dataURL);
  renderPrediction(out);
};

document.getElementById('uploadBtn').onclick = async () => {
  const f = document.getElementById('fileInput').files[0];
  if(!f){ alert('Selecciona un archivo'); return; }
  const fd = new FormData();
  fd.append('image', f);
  const res = await fetch('/predict', {method:'POST', body:fd});
  const out = await res.json();
  renderPrediction(out);
};

function renderPrediction(pred){
  const div = document.getElementById('result');
  if(pred.error){ div.textContent = pred.error; return; }
  div.innerHTML = `<h3>Predicción: ${pred.prediction}</h3>`;
  if(pred.probs){ renderProbsChart(pred.probs); }
}
