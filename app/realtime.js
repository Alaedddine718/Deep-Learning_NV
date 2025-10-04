// Conexión Socket.IO
window.socket = io();

window.socket.on('connect', () => {
  console.log('✅ Conectado al servidor Socket.IO');
});

window.socket.on('stroke_ack', (data) => {
  console.log('Trazo recibido por el servidor:', data);
});
