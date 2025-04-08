# Ejercicios:

## Clase 3: 
- Crear un endpoint que devuelva la hora al momento de hacer un request 
```python
@app.get('/time')
async def time():
    return {"Hora": f'{datetime.now()}'}
```

## Clase 4:
- Crear un endpoint que reciba una variable en formato get y que automaticamente pueda habilitar el formato de hora. Por ejemplo devolver la hora en un formato de 24 horas si el usuario lo requiere.

```python
country_timezone = {
    'CO': 'America/Bogota',
    'MX': 'America/Mexico_City',
    'AR': 'America/Argentina/Buenos_Aires',
    'BR': 'America/Sao_Paulo',
    'PE': 'America/Lima'}

format_hr = {
    '12': '%I:%M %p',
    '24': '%H:%M %p'
}
@app.get('/time/{iso_code}/{formato}')
async def time_format(iso_code:str, formato:str):

    time_zone = country_timezone.get(iso_code.upper()) 
    tz = zoneinfo.ZoneInfo(time_zone)
    hora = datetime.now(tz)

    formato_hora = hora.strftime(format_hr.get(formato))

    return {"Hora": formato_hora}
```

## Clase 6
Modificar el modelo de customer el agregar la validacion correcta para el correo, probar con varios tipos de correso para ver como funciona la validacion.

