function actualizarPlacaDetectada() {
    fetch('/placa_info')
        .then(response => response.json())
        .then(data => {
            const placaDetectada = document.getElementById('placa-detectada');
            const mensajeEstado = document.getElementById('mensaje-estado');
            const placaImg = document.getElementById('placa-img');
            const ocrTexto = document.getElementById('ocr-texto');
            const verificacionTexto = document.getElementById('verificacion-texto');

            ocrTexto.textContent = `OCR: ${data.placa_texto}`;
            verificacionTexto.textContent = `Verificación: ${data.verificacion}`;

            placaDetectada.style.display = 'block';
            mensajeEstado.style.display = 'none';
        });
}

setInterval(actualizarPlacaDetectada, 1000);