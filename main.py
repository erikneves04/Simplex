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

class InviableLPException(Exception):
    """
    Exceção personalizada para indicar que o problema de Programação Linear (LP)
    não possui solução viável.
    """
    def __init__(self, message="O problema de Programação Linear não possui solução viável."):
        self.message = message
        super().__init__(self.message)

class UnboundedLPException(Exception):
    """
    Exceção personalizada para indicar que o problema de Programação Linear (LP)
    é ilimitado (unbounded).
    """
    def __init__(self, message="O problema de Programação Linear é ilimitado."):
        self.message = message
        super().__init__(self.message)

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

    # Garantindo que o B seja positivo
    for i in range(0, RESTRICTIONS_COUNT):
        row = MATRIX[i, :]
        b = row[-1]

        if b < 0:
            MATRIX[i, :] = row * -1

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

def GetViableBasis(tableau):
    base = []
    for i in range(RESTRICTIONS_COUNT, tableau.shape[1] -1):
        for j in range (1, RESTRICTIONS_COUNT + 1):
            expected_colum = np.zeros(RESTRICTIONS_COUNT + 1)
            expected_colum[j] = 1
            column = tableau[:,i]

            if np.array_equal(expected_colum, column):
                base.append(i)

    return base[0:RESTRICTIONS_COUNT]

def Simplex(tableau, base):
    while np.any(tableau[0, RESTRICTIONS_COUNT:-1] < 0):
        selected_column = np.argmin(tableau[0, RESTRICTIONS_COUNT:-1]) + RESTRICTIONS_COUNT
        if selected_column in base:
            break 

        selected_row_index = None
        min_ratio = np.inf
        removed_from_base = None

        # Encontrar a linha pivô
        for i in range(1, RESTRICTIONS_COUNT + 1):
            coef = tableau[i, selected_column]
            b = tableau[i, -1]
            if coef <= 0:
                continue
            ratio = b / coef
            if ratio < min_ratio:
                min_ratio = ratio
                selected_row_index = i

        if selected_row_index is None:
            raise UnboundedLPException()

        # Atualiza a base
        for j, base_col in enumerate(base):
            if tableau[selected_row_index, base_col] == 1:
                removed_from_base = base_col
                break

        base.remove(removed_from_base)
        base.append(selected_column)

        # Normaliza a linha pivô
        pivot_row = tableau[selected_row_index, :]
        pivot_element = pivot_row[selected_column]
        tableau[selected_row_index, :] = pivot_row / pivot_element

        # Elimina os valores na coluna pivô das outras linhas
        for i in range(RESTRICTIONS_COUNT + 1):
            if i == selected_row_index:
                continue
            coef = tableau[i, selected_column]
            tableau[i, :] -= coef * tableau[selected_row_index, :]

        # Checa se existe colunas compostas só por valores <= 0
        for i in range(RESTRICTIONS_COUNT, RESTRICTIONS_COUNT*2):
            if np.all(tableau[1:, i] < 0):
                raise UnboundedLPException()

    return tableau, base

def ExtractPrimalSolution(tableau, base):
    primal = []
    b = tableau[1:, -1]

    for i in base:
        column = tableau[1:, i]
        for j in range(0, RESTRICTIONS_COUNT):
            if column[j] == 1:
                primal.append(b[j])

    return primal

def HasMultipleSolutions(tableau, base):
    objective = tableau[0, :]
    for i in range(RESTRICTIONS_COUNT, RESTRICTIONS_COUNT + ORIGINAL_VAR_COUNT):
        if i in base or objective[i] != 0: 
            continue

        return True, i
    return False, None

def ExtractSolutions(tableau, base):
    value = tableau[0, -1]
    dual = tableau[0, 0:RESTRICTIONS_COUNT]
    primal = ExtractPrimalSolution(tableau, base)
    solutions = [primal]

    multipleSolutions, columnToJoin = HasMultipleSolutions(tableau, base)
    if multipleSolutions:
        print ("Tem multiplas soluções!")
        #solutions.append()

    return value, solutions, dual

def FormatNumber(number, decimals, digits):
    format_string = f"{{:>{digits}.{decimals}f}}"
    return format_string.format(number)

def PrintTableau(tableau, decimals, digits):
    rows, cols = tableau.shape
    left_cols = rows - 1
    main_cols = cols - left_cols - 1

    def format_row(row):
        left_part = " | ".join(FormatNumber(row[j], decimals, digits) for j in range(left_cols))
        main_part = " | ".join(FormatNumber(row[j + left_cols], decimals, digits) for j in range(main_cols))
        rhs_part = FormatNumber(row[-1], decimals, digits)
        return f"{left_part} || {main_part} || {rhs_part}"

    formatted_rows = [format_row(tableau[i, :]) for i in range(rows)]
    line_separator = "=" * len(formatted_rows[0])

    table = [line_separator] + [formatted_rows[0]] + [line_separator] + formatted_rows[1:] + [line_separator]
    
    print("\n".join(table))

def PrintSolutions(primal_solutions, dual_solution, value, decimals, digits):
    print("otima")
    print(FormatNumber(value, decimals, digits))

    for solution in primal_solutions:
        print(" ".join(FormatNumber(j, decimals, digits) for j in solution))

    print(" ".join(FormatNumber(j, decimals, digits) for j in dual_solution))

def main():
    # Leitura dos parâmetros
    args = parseArgs()

    try:
        # Carregandos os dados da PL e colocando em FPI
        LoadPL(args.input)
        ConvertToFPI()
        
        # Removendo restrições com dependências linear
        #MakeMatrixFullRank()

        # Criando o tableau estendido
        tableau = ExtendTableau()

        # Busca uma base viável para o problema
        base = GetViableBasis(tableau)        
        tableau, base = Simplex(tableau, base)
        value, primal_solutions, dual_solution = ExtractSolutions(tableau, base)

        PrintSolutions(primal_solutions, dual_solution, value, args.decimals, args.digits)
        #PrintTableau(tableau, args.decimals, args.digits)
    except InviableLPException:
        print ("inviavel")
    except UnboundedLPException:
        print ("ilimitada")

if __name__ == "__main__":
    main() 