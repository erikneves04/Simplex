# Resolvedor de Programação Linear - Método Simplex

Este projeto implementa o método Simplex para resolver problemas de Programação Linear (PL).  
O código lê um problema de PL, converte-o para a Forma Padrão de Igualdades (FPI), constrói o tableau e resolve utilizando o método Simplex.

## Autor

Desenvolvido por **Erik Neves**.

## Funcionalidades

- Leitura de problemas de Programação Linear.
- Conversão para a Forma Padrão de Igualdades (FPI).
- Construção do tableau do Simplex.
- Execução do Simplex para encontrar a solução ótima.
- Suporte a múltiplas soluções ótimas.
- Exibição de soluções primais e dual.

## Dependências

O código foi escrito em Python e pode exigir as seguintes bibliotecas:

```bash
pip install numpy
```

## Como Usar

Execute o script principal passando o arquivo de entrada com os dados da PL:

```bash
python main.py --input problema.lp
```

### Opções

- `--detail`: Exibe detalhes sobre a execução do algoritmo.
- `--decimals`: Define a precisão decimal dos cálculos.
- `--digits`: Controla a exibição de números no tableau.
- `--policy`: Define a política de escolha de pivô no Simplex.
- `--show_tableau`: Exibe os tableaus intermediários.

Exemplo de uso com detalhes e precisão decimal de 4 casas:

```bash
python main.py --input problema.lp --detail True --decimals 4
```

## Tratamento de Erros

O código trata dois tipos principais de erro:

- `InviableLPException`: Caso o problema não tenha solução viável.
- `UnboundedLPException`: Caso o problema seja ilimitado.

Se ocorrer um desses casos, o script imprimirá:

```plaintext
Status: inviavel
```

ou

```plaintext
Status: ilimitada
```

## Testes  

O script possui um conjunto de testes automatizados que verificam a corretude da implementação. Os testes são definidos na pasta `/inputs`, e as saídas esperadas estão em `/expected-outputs`. Caso uma saída gerada não corresponda à esperada, ela será armazenada na pasta `/outputs`.  

Para executar os testes, utilize o comando:  

```bash
bash test.sh
```

Caso haja falhas, os resultados divergentes serão salvos em `/outputs` para análise.  
