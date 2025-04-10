from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field

# Modelo para validar datos dentro de un endpoint 
class CustomerBase(SQLModel):
    name: str = Field(default=None) 
    description: str | None  = Field(default=None)# Puede tener o no tener descripcion
    email: EmailStr = Field(default=None)
    age: int = Field(default=None)

class Customer(CustomerBase, table = True):
    id: int | None = Field(default= None, primary_key=True)       # No olvidar asignar el valor (asi sea nulo)

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
