from fastapi import FastAPI
from datetime import datetime
import zoneinfo
from models import Customer, Transaction, Invoice, CustomerCreate
from typing import List
from db import SessionDep, create_all_tables

app = FastAPI(lifespan=create_all_tables)
# fastapi dev "para correr la api en modo dev"

@app.get("/")
async def root():
    return {"message": "Configuramos!"}

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

@app.get('/time/{iso_code}')
async def time(iso_code:str):
    iso = iso_code.upper() # convierte todo a mayuscula
    timezone_str = country_timezone.get(iso) # Obtenemnos el codigo del dict
    tz = zoneinfo.ZoneInfo(timezone_str)
    return {"Hora": f'{datetime.now(tz)}'}

@app.get('/time/{iso_code}/{formato}') # Dos parametros de rutas
async def time_format(iso_code:str, formato:str):

    time_zone = country_timezone.get(iso_code.upper()) # Codigo de zona horaria 
    tz = zoneinfo.ZoneInfo(time_zone)   # ZOna horaria
    hora = datetime.now(tz) # Hora en la zona horaria

    formato_hora = hora.strftime(format_hr.get(formato)) # Formato

    return {"Hora": formato_hora}

current_id: int = 0
db_customers: List[Customer] = []

# CREACION DE UN CUSTOMER
@app.post('/customers', response_model = Customer) # Usamos el metodo post para la creacion
async def create_customer(customer_data: CustomerCreate, session: SessionDep):    # Recibe los datos basicos para crear un Customer
    customer = Customer.model_validate(customer_data.model_dump()) # Convierte los datos a un diccionario y valida la informacion ingresada
                                                                   # teniendo como base el modelo de Customer
                                                                   # Se crea un nuevo customer todos estos nuevos datos
    session.add(customer)
    session.commit()
    session.refresh(customer)

    # Ahora que la base de datos "sqlite3" genera el id, no necesitamos estas lineas
    # customer.id = len(db_customers)     # Ya que customer contiene el campo id, aqui actualizamos el id teniendo en cuenta los ingresos anteriores de customers
    # db_customers.append(customer)       # Agregamos el nuevo customer a nuestra  "base de datos" provicional
    return customer

# OBTENER UN CUSTOMER (GET)
@app.get('/customers', response_model = List[Customer])
async def list_customers():
    return db_customers

@app.get('/customers/{id_customer}', response_model = Customer)
async def get_customer(id_customer: int):
    customer = db_customers[id_customer]
    return customer



@app.post('/transactions') # Usamos el metodo post para la creacion
async def create_transaction(transaction_data: Transaction):
    return transaction_data

@app.post('/invoices') # Usamos el metodo post para la creacion
async def create_invoice(inovice_data: Invoice):
    return inovice_data

