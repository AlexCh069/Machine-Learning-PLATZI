from pydantic import BaseModel, EmailStr

# Modelo para validar datos dentro de un endpoint 
class CustomerBase(BaseModel):
    name: str 
    description: str | None # Puede tener o no tener descripcion
    email: EmailStr
    age: int

class Customer(CustomerBase):
    id: int | None = None       # No olvidar asignar el valor (asi sea nulo)

class CustomerCreate(CustomerBase):
    pass

class Transaction(BaseModel):
    id: int         
    ammount: int    # Es mejor tener los datos financieros en formato int
    description: str # obligatorio

class Invoice(BaseModel):
    # Conectamos los dos modelos dentro de este. Ya que un invoice (factura)
    # va a tener refiera a una Transaction realizada por un Customer

    id:int 
    customer: Customer 
    transaction : list[Transaction] # Un invoice cuenta la lista de transactions
    total: int

    @property # Disponibiliza esta variable como una variable de clase 
    def ammount_total(self):
        return sum(transactions.ammount for transactions in self.transaction)
