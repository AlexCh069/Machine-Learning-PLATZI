from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
import sklearn.linear_model as LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def filtrar_correlaciones(df, umbral=0.7, objetivo=None):
    """
    Filtra las correlaciones mayores al umbral dado y excluye las correlaciones con la variable objetivo,
    evitando correlaciones duplicadas.
    
    Parámetros:
    - df_corr: DataFrame de correlaciones (variables en filas y columnas).
    - umbral: Valor mínimo de correlación a filtrar (default = 0.7).
    - objetivo: Nombre de la variable objetivo que se quiere excluir (default = None).
    
    Retorna:
    - DataFrame con las combinaciones de variables que tienen correlación mayor al umbral, sin duplicados.
    """
    df_corr = df.corr()

    # Si se da una variable objetivo, excluimos sus filas y columnas
    if objetivo:
        df_corr = df_corr.drop(index=objetivo, columns=objetivo)


    # Eliminar duplicados manteniendo solo las correlaciones por encima de la diagonal
    df_corr_upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    
    # Filtrar valores mayores al umbral
    df_corr_upper_filtrado = df_corr_upper[df_corr_upper > umbral]

    # Convertir en formato largo
    df_corr_stack_sin_duplicados = df_corr_upper_filtrado.stack()
    pd.set_option('display.max_rows', None) # Para mostrar todo el contenido
    
    return df_corr_stack_sin_duplicados


def calcular_vif(df: pd.DataFrame, umbral_vif = None):

    """
    Calcula el VIF (Variance Inflation Factor). El VIF mide cuánto aumenta la varianza de los 
    coeficientes debido a la colinealidad.
    Parámetros:
    - df: DataFrame de DATOS 
    - umbral_vif: Opcional en caso se desee filtral los datos de VIF

    Retorna:
    - DataFrame con las variables y su valor VIF
    """

    vif = pd.DataFrame()
    vif["Variable"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    if umbral_vif:
        return vif[vif.VIF > umbral_vif]

    return vif

# Modelo de clasificacion Binaria
def graf(model_red: LogisticRegression, x_test, y_test,X):

    y_probs = model_red.predict_proba(x_test)[:, 1]  # Probabilidades de la clase 1 (positiva)

    # 1. Matriz de correlacion
    cm=confusion_matrix(model_red.predict(x_test),y_test)

    # 2. Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)  # Calcular el valor de AUC (opcional)

    # 3. Curva Presicion-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    average_precision = average_precision_score(y_test, y_probs)  # Calcular el valor de AP (opcional)

    # 4. Grafica de coeficientes
    weights = pd.Series(model_red.coef_[0], index=X.columns.values).sort_values(ascending=False)

    # Crear la figura con gridspec
    fig = plt.figure(figsize=(18,14))

    # Definir el layout usando gridspec
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])  # 2 filas, 3 columnas

    # Primera fila - tres gráficas
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Segunda fila - una gráfica que abarque todas las columnas
    ax4 = fig.add_subplot(gs[1, :])  # gs[1, :] indica que abarca todas las columnas de la segunda fila

    #----------------   

    # Paso 2: Obtener las probabilidades de la clase positiva
    y_probs = model_red.predict_proba(x_test)[:, 1]  # Probabilidades de la clase 1 (positiva)

    # Paso 3: Calcular los valores para la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    average_precision = average_precision_score(y_test, y_probs)  # Calcular el valor de AP (opcional)


    # Gráfica 1 - Matriz de confusión
    sns.heatmap(
        cm,
        annot=True,
        cmap='gray',
        cbar=False,
        square=True,
        fmt="d",
        annot_kws={"size": 20},  # Cambia el tamaño de los números en las celdas
        ax=ax1  # Agregar el heatmap en el ax1 (primera columna)
    )
    ax1.set_ylabel('Real Label', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=16)
    ax1.set_title('Matriz de Confusión', fontsize=20)

    # Grafica 2: Graficar la curva ROC
    ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random guessing')  # Línea diagonal de azar
    ax2.set_title('Curva ROC', fontsize=20)
    ax2.legend(loc='lower right')
    ax2.legend(fontsize=15)  # Ajusta el tamaño de la fuente de la leyenda
    ax2.tick_params(axis='both', which='major', labelsize=17)  # Ticks más grandes

    # Grafica 3: Graficar la curva Precision-Recall
    ax3.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})', color='blue')
    ax3.set_title('Curva Precision-Recall', fontsize=20)
    ax3.legend(loc='lower left')
    ax3.legend(fontsize=15)  # Ajusta el tamaño de la fuente de la leyenda
    ax3.grid(True)
    ax3.tick_params(axis='both', which='major', labelsize=17)  # Ticks más grandes



    # Grafica de Coeficientes
    weights.plot(kind='bar', ax = ax4)
    ax4.set_title('Coeficientes del Modelo', fontsize=20)
    ax4.set_ylabel('Peso', fontsize=17)  # Etiqueta del eje Y con fuente más grande
    ax4.set_xlabel('Variables', fontsize=17)  # Etiqueta del eje X con fuente más grande
    ax4.tick_params(axis='both', which='major', labelsize=17)  # Ticks más grandes

    # Ajustar el espacio entre las filas
    fig.subplots_adjust(hspace=7)  # Controla el espacio vertical entre las filas


    # Ajustar el layout para que no haya superposición
    plt.tight_layout()


# Modelo Tree decision

def tree_division(tree_model: DecisionTreeClassifier, x_cols:list):
    """Funcion que nos muestra el arbol generado de nuestro modelo
    tree_model: Modelo Tree Decision
    x_cols: Lista de nombres de las columnas de acuerdo a como se usaron en el entrenamiento del modelo"""


    # Asegúrate de que los nombres de las características sean cadenas
    feature_names = x_cols

    # Convierte las clases en cadenas de texto
    class_names = list(map(str, tree_model.classes_))

    # Dibuja el árbol de decisión
    plt.figure(figsize=(20,10))  # Ajusta el tamaño si es necesario
    tree.plot_tree(tree_model, 
                feature_names=feature_names,  # Nombres de las columnas (características)
                class_names=class_names,      # Nombres de las clases
                filled=True,                  # Colorea las hojas del árbol
                rounded=True,                 # Usa bordes redondeados
                fontsize=10)                  # Tamaño de la fuente
    plt.show()

def tree_important_features(tree_model:DecisionTreeClassifier, x_cols):
    # Obtenemos las importancias de las características
    importances = pd.Series(tree_model.feature_importances_, index=x_cols).sort_values(ascending=False)

    # Configuramos la paleta de colores y la saturación
    sns.set(style="whitegrid")  # Configuración del estilo
    palette = sns.color_palette("viridis", len(importances))  # Paleta de colores personalizada

    # Creación del gráfico de barras
    sns.barplot(x=importances, y=importances.index, palette=palette, hue=importances.index, saturation=0.9, legend=False)

    # Añadimos título y etiquetas
    plt.title('Importancia de cada Feature')
    plt.xlabel('Importancia')
    plt.ylabel('Características')

    # Mostramos el gráfico
    plt.show()