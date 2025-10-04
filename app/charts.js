window.probChart = null;

function renderProbabilities(probs) {
  const ctx = document.getElementById('probChart').getContext('2d');
  if (window.probChart) {
    window.probChart.destroy();
  }
  window.probChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [...Array(10).keys()].map(String), // "0"..."9"
      datasets: [{
        label: 'Probabilidad',
        data: probs,
        backgroundColor: 'rgba(54, 162, 235, 0.7)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 1.0 }
      }
    }
  });
}

