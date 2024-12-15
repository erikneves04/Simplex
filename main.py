import argparse
from enum import Enum
import numpy as np

# Tipo das variáveis
class TypesOfVariables(Enum):
    Free = "0"
    LessThanOrEqualToZero = "-1"
    GreaterThanOrEqualToZero = "1"

    def __str__(self):
        return self.value

# Tipo das restrições
class TypesOfRestrictions(Enum):
    Equal = "=="
    GreaterThanOrEqual = "<="
    LessThanOrEqual = ">="

    def __str__(self):
        return self.value

# Variáveis de controle
VAR_COUNT = 0
VAR_TYPES = []
OBJ_COEFFICIENTS = []
RESTRICTIONS_COUNT = 0
RESTRICTIONS_TYPE = []
MATRIX = None # inclui o vetor b a direita

def parseArgs():
    parser = argparse.ArgumentParser(description="Aplicação solução de problemas de programação linear usando o simplex")
    parser.add_argument("--input", type=str, required=True, help="Arquivo de entrada com os dados do problema.")
    return parser.parse_args()

def LoadPL(filename):
    global VAR_COUNT, RESTRICTIONS_COUNT, VAR_TYPES, OBJ_COEFFICIENTS, RESTRICTIONS_TYPE, MATRIX

    with open(filename, "r") as file:
        lines = file.readlines()

    VAR_COUNT = int(lines[0].strip())
    RESTRICTIONS_COUNT = int(lines[1].strip())

    varTypes = list(map(int, lines[2].strip().split()))
    for type in varTypes:
        if type == -1:
            VAR_TYPES.append(TypesOfVariables.LessThanOrEqualToZero)
        elif type == 1:
            VAR_TYPES.append(TypesOfVariables.GreaterThanOrEqualToZero)
        else:
            VAR_TYPES.append(TypesOfVariables.Free)

    OBJ_COEFFICIENTS = np.array(list(map(int, lines[3].strip().split())))

    coef = []
    b = []
    for line in lines[4:]:
        line = line.strip()

        if "<=" in line:
            left, right = line.split("<=")
            RESTRICTIONS_TYPE.append(TypesOfRestrictions.LessThanOrEqual)
        elif ">=" in line:
            left, right = line.split(">=")
            RESTRICTIONS_TYPE.append(TypesOfRestrictions.GreaterThanOrEqual)
        elif "==" in line:
            left, right = line.split("==")
            RESTRICTIONS_TYPE.append(TypesOfRestrictions.Equal)
        
        coef.append(list(map(int, left.strip().split())))
        b.append(int(right.strip()))

    MATRIX = np.hstack((np.array(coef, dtype=int), np.array(b, dtype=int).reshape(-1, 1)))