#!/bin/bash

# Cores ANSI
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Diretórios
input_dir="inputs"
output_dir="outputs"
expected_output_dir="expected-outputs"

# Verifica se o diretório de saída existe, senão cria
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

# Inicializa contador de testes
total_tests=0
passed_tests=0

# Função para imprimir linha da tabela com cor
print_table_line() {
    local test_number=$1
    local status=$2

    if [[ $status == "PASSED" ]]; then
        printf "| %-2s | ${GREEN}%-6s${NC} |\n" "$test_number" "$status"
    else
        printf "| %-2s | ${RED}%-6s${NC} |\n" "$test_number" "$status"
    fi
}

echo "Iniciando os testes da aplicação:"
echo

# Tabela de resultados
printf "+%s+%s+\n" "----" "--------"
printf "| %-2s | %-6s |\n" "ID" "STATUS"
printf "+%s+%s+\n" "----" "--------"

# Para cada arquivo de entrada no diretório de entradas
for input_file in "$input_dir"/*; do
    # Nome do arquivo de saída correspondente
    output_file="$output_dir/$(basename "$input_file" .txt)"
    expected_output_file="$expected_output_dir/$(basename "$input_file")"
    
    # Remove caracteres especiais do Windows (supressão de erros com redirecionamento)
    sed -i 's/\r$//' "$expected_output_file" 2>/dev/null

    # Executa o programa com o arquivo de entrada e redireciona a saída para o arquivo de saída (supressão de erros)
    python3 main.py --input "$input_file" --decimals 7 > "$output_file" 2>/dev/null

    # Compara a saída gerada com a saída esperada (supressão de erros)
    if diff -q "$output_file" "$expected_output_file" > /dev/null 2>/dev/null; then
        status="PASSED"
        passed_tests=$((passed_tests + 1))
        rm -f "$output_file"
    else
        status="FAILED"
    fi

    # Imprime o resultado do teste na tabela com cor
    print_table_line "$(basename "$input_file" .txt)" "$status"

    total_tests=$((total_tests + 1))
done
printf "+%s+%s+\n" "----" "--------"

# Resultados finais
echo
if [[ $total_tests -eq $passed_tests ]]; then
    rm -rf "$output_dir"
    echo -e "Todos os testes ${GREEN}passaram${NC}."
else
    echo "Consulte as saídas com problemas em '$output_dir'."
fi

echo "Testes concluídos."
