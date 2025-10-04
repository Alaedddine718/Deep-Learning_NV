let probsChart = null;
function renderProbsChart(probs){
  const ctx = document.getElementById('probsChart').getContext('2d');
  if(probsChart){ probsChart.destroy(); }
  probsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [...Array(10).keys()].map(String),
      datasets: [{ label:'Prob', data: probs }]
    },
    options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
  });
}
