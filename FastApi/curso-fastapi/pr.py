from typing import List 

def add_vectors(vectors:List[List[float]]) -> List[float]:
    # Comprobar que los vectores no estan vacios 
    assert vectors, "no vectors provided" 

    # Comprobar que los vectores tienen el mismo tamaño
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "Tamaño diferente entre vectores"

    return [sum(vector[i] for vector in vectors)
                for i in range(num_elements)]

def multiply_scalar(scalar:float, vector: List[float]) -> List[float]:
    assert scalar, "No hay escalar"
    assert vector, "No hay vector"

    return [scalar*i for i in vector]

print(add_vectors([[1, 2], [12, 4]]))

print(multiply_scalar(2,[1,3,4]))