let probChart = null;

function renderProbabilities(probs) {
    const ctx = document.getElementById('probChart').getContext('2d');

    // Si ya existe el gráfico, destruir antes de redibujar
    if (probChart) {
        probChart.destroy();
    }

    probChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [...Array(10).keys()], // 0 al 9
            datasets: [{
                label: 'Probabilidad',
                data: probs,
                backgroundColor: 'rgba(54, 162, 235, 0.7)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0
                }
            }
        }
    });
}
