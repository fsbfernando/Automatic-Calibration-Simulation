import sim
import time
import numpy as np

# ============================
# CONECTAR AO COPPELIASIM
# ============================
print('🔌 Tentando conectar ao CoppeliaSim...')
sim.simxFinish(-1)  # Fecha conexões antigas
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID != -1:
    print('✅ Conexão bem-sucedida com a Remote API!')
else:
    print('❌ Falha na conexão. Verifique se a simulação está rodando.')
    exit()

# ============================
# OBTENDO O HANDLE DAS JUNTAS (NOMES CORRETOS)
# ============================
joint_names = [
    'LBR_iiwa_14_R820_joint1',
    'LBR_iiwa_14_R820_joint2',
    'LBR_iiwa_14_R820_joint3',
    'LBR_iiwa_14_R820_joint4',
    'LBR_iiwa_14_R820_joint5',
    'LBR_iiwa_14_R820_joint6',
    'LBR_iiwa_14_R820_joint7'
]

joint_handles = []
for joint in joint_names:
    result, handle = sim.simxGetObjectHandle(clientID, joint, sim.simx_opmode_blocking)
    if result == sim.simx_return_ok:
        joint_handles.append(handle)
        print(f'✅ Handle de {joint} obtido com sucesso.')
    else:
        print(f'❌ Falha ao obter handle de {joint}')
        exit()

# ============================
# PERGUNTAR AO USUÁRIO OS VALORES
# ============================
num_poses = int(input("Quantas poses você deseja passar ao robô? "))
time_per_pose = float(input("Quanto tempo o robô deve permanecer em cada pose (em segundos)? "))

# Coletar ângulos para cada pose
poses = []
for i in range(num_poses):
    print(f"\n🔢 Pose {i + 1}:")
    angles = []
    for j in range(len(joint_names)):
        angle = float(input(f" - Ângulo para {joint_names[j]} (em graus): "))
        angles.append(np.radians(angle))  # Converte para radianos
    poses.append(angles)

# ============================
# MOVIMENTAÇÃO DO ROBÔ
# ============================
for pose_idx, pose in enumerate(poses):
    print(f"\n➡️ Movendo para a pose {pose_idx + 1}...")
    for i, joint_handle in enumerate(joint_handles):
        sim.simxSetJointTargetPosition(clientID, joint_handle, pose[i], sim.simx_opmode_oneshot)
    
    # Esperar o tempo especificado para a pose
    time.sleep(time_per_pose)
    print(f"✅ Pose {pose_idx + 1} concluída.")

# ============================
# FINALIZAR A CONEXÃO
# ============================
sim.simxFinish(clientID)
print('🔌 Conexão com CoppeliaSim encerrada.')
