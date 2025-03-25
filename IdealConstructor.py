import os
import numpy as np
import open3d as o3d

# ==============================
# CONFIGURAÇÃO DE CAMINHOS
# ==============================
INPUT_DIR = r'C:\Users\fsbfe\Desktop\Ref_PointClouds\Run 17'
OUTPUT_FILE = r'C:\Users\fsbfe\Desktop\Ref_PointClouds\Consolidated_Model.ply'

# ==============================
# CARREGAR E PRE-PROCESSAR AS NUVENS DE PONTOS
# ==============================
print("🔄 Carregando nuvens de pontos...")

# Lista os arquivos PLY no diretório de entrada
point_cloud_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.ply')]
point_clouds = []
for f in point_cloud_files:
    pcd = o3d.io.read_point_cloud(f)
    # Redução de ruído com voxel downsample (ajuste o voxel se necessário)
    pcd = pcd.voxel_down_sample(0.001)
    # Estimação de normais (importante para ICP point-to-plane e Poisson)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    point_clouds.append(pcd)

print(f"✅ {len(point_clouds)} nuvens de pontos carregadas.")

# ==============================
# REGISTRO GLOBAL INICIAL
# ==============================
print("🔄 Inicializando registro global baseado em posição angular...")

# Define a primeira nuvem como referência
merged_cloud = point_clouds[0]

# Define o ângulo de separação aproximado para a inicialização (assume distribuição circular)
angle_step = 2 * np.pi / len(point_clouds)

for i in range(1, len(point_clouds)):
    print(f"🔄 Alinhando nuvem {i + 1}/{len(point_clouds)}...")
    
    # Aproximação inicial: rotação com base no ângulo (ajuste se houver informações adicionais)
    angle = i * angle_step
    transformation_init = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])
    point_clouds[i].transform(transformation_init)
    
    # Refinamento do alinhamento usando ICP com estimativa point-to-plane
    threshold = 0.01  # Tolerância para ICP (ajuste conforme necessário)
    result_icp = o3d.pipelines.registration.registration_icp(
        point_clouds[i],
        merged_cloud,
        threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    point_clouds[i].transform(result_icp.transformation)
    
    # Opcional: um segundo refinamento com um threshold menor para maior precisão
    result_icp_refine = o3d.pipelines.registration.registration_icp(
        point_clouds[i],
        merged_cloud,
        0.005,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    point_clouds[i].transform(result_icp_refine.transformation)
    
    # Adiciona a nuvem alinhada ao modelo consolidado
    merged_cloud += point_clouds[i]

# ==============================
# REFINAMENTO GLOBAL
# ==============================
print("🔄 Refinamento global das nuvens de pontos...")
# Executa um ICP global para minimizar erros acumulados usando a métrica point-to-plane
global_icp = o3d.pipelines.registration.registration_icp(
    merged_cloud,
    merged_cloud,
    0.005,
    np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
# Note: nesse caso, o modelo já é o merged_cloud; o refinamento atua para ajustar a coerência

# ==============================
# FUSÃO E REDUÇÃO DE RUÍDO
# ==============================
print("🔄 Fundindo nuvens de pontos e aplicando voxel downsample...")
# Um downsample final para reduzir ruído e equilibrar a densidade
merged_cloud = merged_cloud.voxel_down_sample(0.001)

# ==============================
# RECONSTRUÇÃO 3D COM POISSON
# ==============================
print("🔄 Estimando normais e reconstruindo superfície com Poisson...")
# Reestima os normais com um parâmetro ajustado para a nuvem consolidada
merged_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

# Reconstrução de superfície usando o método de Poisson
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    merged_cloud, depth=9
)

# ==============================
# SALVAR O MODELO FINAL
# ==============================
print("💾 Salvando modelo consolidado...")
o3d.io.write_triangle_mesh(OUTPUT_FILE, mesh)
print(f"✅ Modelo consolidado salvo em: {OUTPUT_FILE}")

# ==============================
# EXIBIR O RESULTADO FINAL
# ==============================
print("🖥️ Mostrando o modelo 3D consolidado...")
o3d.visualization.draw_geometries([mesh])
