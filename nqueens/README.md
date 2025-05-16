# N-Queens Solver: Hill Climbing vs. Simulated Annealing

Este projeto compara duas abordagens clássicas de otimização — **Hill Climbing** e **Simulated Annealing** — aplicadas ao problema das 8 rainhas.

---

## Sobre o Problema

O **Problema das N Rainhas** consiste em posicionar N rainhas em um tabuleiro NxN de forma que nenhuma se ataque.  
Isso significa evitar que duas rainhas compartilhem a mesma **linha**, **coluna** ou **diagonal**.

---

## Algoritmos Implementados

### Hill Climbing
Um algoritmo de busca local que sempre tenta se mover para o vizinho com menor custo.  
Rápido  
Pode ficar preso em **mínimos locais**, falhando em encontrar a melhor solução.

### Simulated Annealing
Algoritmo inspirado no processo de resfriamento de metais.  
Aceita movimentos piores com uma certa probabilidade, permitindo explorar melhor o espaço de busca.  
Mais robusto e flexível  
Levemente mais lento, mas com maior chance de encontrar a solução ótima.

---

## Metodologia

Foram realizadas **500 execuções** para cada algoritmo, utilizando estados iniciais aleatórios.  
Os critérios avaliados foram:

- Tempo de execução médio  
- Pontuação média (número de conflitos restantes)  
- Melhor e ❌ pior pontuação  
- Porcentagem de soluções ótimas encontradas  

---

## Resultados

### Hill Climbing

- **Tempo médio:** 0.0006 segundos  
- **Pontuação média:** 2.46  
- **Melhor pontuação:** 0  
- **Pior pontuação:** 6  
- **Porcentagem de soluções ótimas:** 17%  

### Simulated Annealing

- **Tempo médio:** 0.0411 segundos  
- **Pontuação média:** 1.67  
- **Melhor pontuação:** 0  
- **Pior pontuação:** 6  
- **Porcentagem de soluções ótimas:** 32.6%  

---

## Conclusão

A partir dos testes realizados, podemos observar o seguinte:

#### Hill Climbing
- Extremamente rápido.
- Eficiente para encontrar soluções rápidas, mas muitas vezes parciais.
- Limitações evidentes na hora de escapar de mínimos locais.

#### Simulated Annealing
- Mais lento, porém muito mais consistente.
- Alta capacidade de escapar de armadilhas locais.
- Quase **o dobro de acertos** na solução ótima em relação ao Hill Climbing.

**Resumo prático:**  
Se você precisa de **velocidade**, vá de *Hill Climbing*.  
Se busca **qualidade e robustez**, *Simulated Annealing* é a melhor escolha.

---

## Como Executar

Para rodar os experimentos:

```bash
python main.py
```
