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
ORIGINAL_VAR_COUNT = 0
VAR_COUNT = 0
VAR_TYPES = []
OBJ_COEFFICIENTS = []
RESTRICTIONS_COUNT = 0
RESTRICTIONS_TYPE = []
MATRIX = None # inclui o vetor b a direita
IS_MAXIMIZATION = False
SOLUTION_GETTERS = None

def parseArgs():
    parser = argparse.ArgumentParser(description="Aplicação solução de problemas de programação linear usando o simplex")
    parser.add_argument("--input", type=str, required=True, help="Arquivo de entrada com os dados do problema.")
    parser.add_argument("--decimals", type=int, required=False, default=3, help="Número de casas decimais para imprimir valores numéricos.")
    parser.add_argument("--digits", type=int, required=False, default=7, help="Número total de dígitos para imprimir valores numéricos.")

    return parser.parse_args()

def LoadPL(filename):
    global VAR_COUNT, ORIGINAL_VAR_COUNT, RESTRICTIONS_COUNT, VAR_TYPES, OBJ_COEFFICIENTS, RESTRICTIONS_TYPE, MATRIX, IS_MAXIMIZATION

    with open(filename, "r") as file:
        lines = file.readlines()

    VAR_COUNT = int(lines[0].strip())
    ORIGINAL_VAR_COUNT = VAR_COUNT
    RESTRICTIONS_COUNT = int(lines[1].strip())

    varTypes = list(map(int, lines[2].strip().split()))
    for type in varTypes:
        if type == -1:
            VAR_TYPES.append(TypesOfVariables.LessThanOrEqualToZero)
        elif type == 1:
            VAR_TYPES.append(TypesOfVariables.GreaterThanOrEqualToZero)
        else:
            VAR_TYPES.append(TypesOfVariables.Free)

    objective = lines[3].strip().split()
    OBJ_COEFFICIENTS = np.array(list(map(int, objective[1:])))
    IS_MAXIMIZATION = objective[0] == "max"

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
        elif "=" in line:
            left, right = line.split("=")
            RESTRICTIONS_TYPE.append(TypesOfRestrictions.Equal)
        
        coef.append(list(map(int, left.strip().split())))
        b.append(int(right.strip()))

    MATRIX = np.hstack((np.array(coef, dtype=int), np.array(b, dtype=int).reshape(-1, 1)))

def ExtendTableau():
    objective_row = np.concatenate((
            np.zeros(RESTRICTIONS_COUNT),
            OBJ_COEFFICIENTS * -1,
            [0]
        ))

    identity_matrix = np.eye(RESTRICTIONS_COUNT, dtype=int)
    extended_matrix = np.hstack((identity_matrix, MATRIX))
    extended_matrix = np.vstack((objective_row, extended_matrix))

    return extended_matrix

def PrintTableau(tableau, decimals, digits):
    rows, cols = tableau.shape
    left_cols = rows - 1
    main_cols = cols - left_cols - 1

    format_string = f"{{:>{digits}.{decimals}f}}"
    def format_row(row):
        left_part = " | ".join(format_string.format(row[j]) for j in range(left_cols))
        main_part = " | ".join(format_string.format(row[j + left_cols]) for j in range(main_cols))
        rhs_part = format_string.format(row[-1]) 
        return f"{left_part} || {main_part} || {rhs_part}"

    formatted_rows = [format_row(tableau[i, :]) for i in range(rows)]
    line_separator = "=" * len(formatted_rows[0])

    table = [line_separator] + [formatted_rows[0]] + [line_separator] + formatted_rows[1:] + [line_separator]
    
    print("\n".join(table))

def ConvertToFPI():
    global MATRIX, OBJ_COEFFICIENTS, IS_MAXIMIZATION, VAR_COUNT, VAR_TYPES, RESTRICTIONS_TYPE, SOLUTION_GETTERS

    # Convertendo o objetivo para maximização
    if not IS_MAXIMIZATION:
        IS_MAXIMIZATION = True
        OBJ_COEFFICIENTS = OBJ_COEFFICIENTS * -1
    
    # Adicionando variáveis de folga
    added_count = 0
    for i in range(RESTRICTIONS_COUNT):
        if RESTRICTIONS_TYPE[i] == TypesOfRestrictions.LessThanOrEqual:
            new_column = np.zeros(RESTRICTIONS_COUNT)
            new_column[i] = 1

            MATRIX = np.insert(MATRIX, -1, new_column, axis=1)
            VAR_TYPES.append(TypesOfVariables.GreaterThanOrEqualToZero)
            added_count += 1
        elif RESTRICTIONS_TYPE[i] == TypesOfRestrictions.GreaterThanOrEqual:
            new_column = np.zeros(RESTRICTIONS_COUNT)
            new_column[i] = -1

            MATRIX = np.insert(MATRIX, -1, new_column, axis=1)
            VAR_TYPES.append(TypesOfVariables.GreaterThanOrEqualToZero)
            added_count += 1

    VAR_COUNT += added_count
    OBJ_COEFFICIENTS = np.append(OBJ_COEFFICIENTS, [0] * added_count)

    # Garantindo que todas as variáveis sejam não-negativas
    SOLUTION_GETTERS = []
    added_count = 0
    lastAddedIndex = VAR_COUNT - 1
    for i in range(VAR_COUNT):
        var_type = VAR_TYPES[i]
        if var_type == TypesOfVariables.Free:
            # Variáveis livres: x = x' - x''
            
            reference_column = MATRIX[:, i]
            new_column = reference_column * -1

            MATRIX = np.insert(MATRIX, -1, new_column, axis=1)
            VAR_TYPES.append(TypesOfVariables.GreaterThanOrEqualToZero)

            lastAddedIndex += 1
            added_count += 1
            getter = lambda solutions_list: (solutions_list[i]) - (solutions_list[lastAddedIndex])
            SOLUTION_GETTERS.append(getter)
            VAR_TYPES[i] = TypesOfVariables.GreaterThanOrEqualToZero
        elif var_type == TypesOfVariables.LessThanOrEqualToZero:
            # Variáveis <= 0: x = -x'

            reference_column = MATRIX[:, i]
            new_column = reference_column * -1
            MATRIX[:, i] = new_column

            getter = lambda solutions_list: (solutions_list[i]) * -1
            SOLUTION_GETTERS.append(getter)
            VAR_TYPES[i] = TypesOfVariables.GreaterThanOrEqualToZero
        else:
            getter = lambda solutions_list: solutions_list[i]
            SOLUTION_GETTERS.append(getter)

    OBJ_COEFFICIENTS = np.append(OBJ_COEFFICIENTS, [0] * added_count)
            
def main():
    # Leitura dos parâmetros
    args = parseArgs()

    # Carregandos os dados da PL e colocando em FPI
    LoadPL(args.input)
    ConvertToFPI()
    
    # Removendo restrições com dependências linear
    #MakeMatrixFullRank()

    # Criando o tableau estendido
    tableau = ExtendTableau()

    PrintTableau(tableau, args.decimals, args.digits)

if __name__ == "__main__":
    main() 