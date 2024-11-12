import cv2

# Lista todas las cámaras disponibles
def list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
        else:
            is_reading, img = camera.read()
            if is_reading:
                working_ports.append(dev_port)
            else:
                available_ports.append(dev_port)
            camera.release()
        dev_port += 1
    return working_ports, available_ports

working_ports, available_ports = list_ports()
print(f"Puertos de cámara funcionando: {working_ports}")
print(f"Puertos de cámara disponibles pero no funcionando: {available_ports}")