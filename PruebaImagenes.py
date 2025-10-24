import pandas as pd

df_train = pd.read_csv('landmarks_entrenamiento.csv')
df_val = pd.read_csv('landmarks_validacion.csv')

# Columnas de datos (excluyendo la columna 'clase')
columnas_datos = df_train.columns.drop('clase')

# Crear una columna temporal que indique si la fila tiene todos los valores 0
df_train['todos_0'] = (df_train[columnas_datos] == 0).all(axis=1)
df_val['todos_0'] = (df_val[columnas_datos] == 0).all(axis=1)

# Contar por clase cuántas filas tienen todos los valores 0
conteo_train_0 = df_train[df_train['todos_0']].groupby('clase').size()
conteo_val_0 = df_val[df_val['todos_0']].groupby('clase').size()

print("Conteo de filas con todos los valores 0 por clase en train:\n", conteo_train_0)
print("\nConteo de filas con todos los valores 0 por clase en val:\n", conteo_val_0)