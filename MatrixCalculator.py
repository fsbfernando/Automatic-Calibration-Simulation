import os
import numpy as np
import open3d as o3d

# ==============================
# CONFIGURAÇÕES
# ==============================
BASE_DIR = os.path.join(os.path.dirname(__file__), 'PointClouds/Run 1')  # Diretório com as capturas
output_path = os.path.join(os.path.dirname(__file__), 'T_cf_final.npy')
output_txt_path = os.path.join(os.path.dirname(__file__), 'T_cf_final.txt')

# ==============================
# LISTAR OS ARQUIVOS DE NUVEM DE PONTOS
# ==============================
files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith('.ply')])
if len(files) == 0:
    raise FileNotFoundError(f"Nenhuma nuvem de pontos encontrada em: {BASE_DIR}")

print(f"🔎 {len(files)} arquivos de nuvem de pontos encontrados para calibração.")

# ==============================
# FUNÇÃO PARA GERAR MATRIZ T_CF POR ICP
# ==============================
T_cf_list = []
first_pcd = None

for i, file in enumerate(files):
    file_path = os.path.join(BASE_DIR, file)
    
    print(f"\n📥 Processando arquivo {i + 1}/{len(files)}: {file_path}")
    
    # Carregar nuvem de pontos
    pcd = o3d.io.read_point_cloud(file_path)
    
    if first_pcd is None:
        first_pcd = pcd
        continue
    
    # ICP para alinhar em relação à primeira nuvem de pontos
    threshold = 0.02  # Limite para o ajuste (em metros)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, first_pcd, threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    # Extrai a matriz T_cf obtida pelo ICP
    T_cf = reg_p2p.transformation
    
    # Adiciona à lista de matrizes
    T_cf_list.append(T_cf)
    
    print(f"📐 Matriz T_CF obtida para o arquivo {i + 1}:\n{T_cf}")

# ==============================
# COMBINAR AS MATRIZES COM MEDIANA
# ==============================
if len(T_cf_list) == 0:
    raise ValueError("Nenhuma matriz T_CF foi gerada.")

# Converte para numpy array para poder calcular a mediana
T_cf_list = np.array(T_cf_list)

# Usa mediana para maior robustez a ruídos
T_cf_final = np.median(T_cf_list, axis=0)

# ==============================
# SALVAR A MATRIZ FINAL
# ==============================
np.save(output_path, T_cf_final)
np.savetxt(output_txt_path, T_cf_final, fmt="%.6f")

print("\n✅ Matriz final T_CF gerada com sucesso:")
print(T_cf_final)
print(f"\n✅ Matriz final T_CF salva em: {output_path}")
print(f"✅ Matriz final T_CF salva como texto em: {output_txt_path}")

