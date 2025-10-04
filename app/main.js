// --- Configuración del canvas ---
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener("mousedown", () => { drawing = true; });
canvas.addEventListener("mouseup", () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener("mousemove", draw);

function draw(event) {
    if (!drawing) return;
    ctx.lineWidth = 15;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.offsetX, event.offsetY);
}

// --- Botón limpiar ---
document.getElementById("clearBtn").addEventListener("click", () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

// --- Botón predecir ---
document.getElementById("predictBtn").addEventListener("click", async () => {
    const imageData = canvas.toDataURL("image/png");

    const formData = new FormData();
    formData.append("image_base64", imageData);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
        } else {
            // Mostrar predicción final
            document.getElementById("prediction").innerText =
                "Predicción: " + data.prediction;

            // Dibujar gráfico de probabilidades
            renderProbabilities(data.probs);
        }
    } catch (err) {
        console.error("Error en la predicción:", err);
        alert("Error al comunicarse con el servidor.");
    }
});
