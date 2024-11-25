import re
from datetime import datetime

def validar_formato_placa(placa_texto):
    """
    Valida si el texto corresponde a un formato válido de placa colombiana.
    Retorna un diccionario con el resultado de la validación y el tipo de vehículo.
    """
    # Limpiamos la placa de espacios al inicio y al final, y la convertimos a mayúsculas
    placa_texto = placa_texto.strip().upper()
    
    # Patrones de formato para cada tipo de vehículo con espacio incluido
    patrones = {
        'particular': r'^[A-Z]{3} \d{3}$',  # AAA 000
        'publico': r'^[A-Z]{2} \d{4}$',     # AA 0000
        'moto': r'^[A-Z]{3} \d{2}[A-Z]?$'   # AAA 00 o AAA 00A
    }
    
    for tipo, patron in patrones.items():
        if re.match(patron, placa_texto):
            return {
                'es_valida': True,
                'tipo_vehiculo': tipo,
                'placa': placa_texto,
                'mensaje': f"Placa válida de {tipo}"
            }
    
    return {
        'es_valida': False,
        'tipo_vehiculo': None,
        'placa': placa_texto,
        'mensaje': f"Formato de placa '{placa_texto}' no válido"
    }

def verificar_pico_y_placa(placa_texto, confidence):
    """
    Verifica si un vehículo tiene pico y placa basado en su número de placa.
    Incluye el nivel de confianza en el resultado.
    """
    # Primero validamos el formato de la placa
    validacion = validar_formato_placa(placa_texto)
    
    if not validacion['es_valida']:
        return {
            'message': validacion['mensaje'],
            'status': 'error',
            'confidence': confidence
        }
    
    # Obtenemos el último dígito de la placa
    digitos = ''.join(filter(str.isdigit, placa_texto))
    if not digitos:
        return {
            'message': f"No se encontraron dígitos en la placa '{placa_texto}'.",
            'status': 'error',
            'confidence': confidence
        }

    ultimo_digito = int(digitos[-1])
    dia_semana = datetime.today().weekday()
    
    # Lógica de pico y placa según el día de la semana y el último dígito
    restriccion = {
        0: [7, 8],  # Lunes
        1: [9, 0],  # Martes
        2: [1, 2],  # Miércoles
        3: [3, 4],  # Jueves
        4: [5, 6]   # Viernes
    }
    
    if dia_semana in restriccion and ultimo_digito in restriccion[dia_semana]:
        return {
            'message': f"La placa '{placa_texto}' ({validacion['tipo_vehiculo']}) tiene pico y placa hoy.",
            'status': 'warning',
            'confidence': confidence
        }
    else:
        return {
            'message': f"La placa '{placa_texto}' ({validacion['tipo_vehiculo']}) no tiene pico y placa hoy.",
            'status': 'success',
            'confidence': confidence
        }