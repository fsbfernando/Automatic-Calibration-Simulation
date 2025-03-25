CoppeliaSim:

CenaBaseIdeal.ttt - Robô avulso com a câmera orbitando ao redor da base (Coppelia)

CenaCalibracao.ttt - Cena do robô sobre a base fixa, com a câmera fixa (Coppelia)

Scripts Desenvolvidos em Python:

CloudPointExtractor.py - Responsável por mover o robô para 4 posições pré definidas (que posicionam o flange em direção a base) e registrar a "visão de Realsense" ao estar apontada para base.

TakingIdeal.py - Responsável por capturar as diversas nuvens de pontos que servem de matéria prima para a reconstrução da base a partir de diversas nuvens de pontos da base.

IdealConstructor.py - Responsável por reconstruir o modelo 3D a partir das diversas nuvens de pontos obtidas pelo TakingIdeal.py

ScriptComando.py - Responsável por movimentar o robô da simulação para quantas posições forem desejadas. Utilizado para testar quais posições de juntas apontam o flange para a base.

MatrixCalculator.py - Primeira tentativa de extrair a matriz de transformação desejada a partir do alinhamentos das nuvens de pontos

ExtrinsicMatrix.py - Segunda tentativa de extrair a matriz de transformação desejada a partir do alinhamento das nuvens de pontos

Scripts Python para Integração Python + Coppelia:

sim.py

simConst.py

RemoteAPI.dll também é responsável pela integração do Python com o Coppelia
