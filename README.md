# 🧠 N-Queens Solver: Hill Climbing vs. Simulated Annealing

Este projeto compara dois algoritmos clássicos de otimização — **Hill Climbing** e **Simulated Annealing** — para resolver o famoso problema das 8 rainhas.

## ♟️ Sobre o Problema

O **Problema das N Rainhas** consiste em posicionar N rainhas em um tabuleiro NxN de forma que nenhuma delas se ataque. Ou seja, não pode haver duas rainhas na mesma linha, coluna ou diagonal.

---

## ⚙️ Algoritmos Implementados

### 🔼 Simple Hill Climbing
É uma abordagem gulosa que procura mover-se sempre para o vizinho com menor custo imediato. Porém, pode ficar preso em ótimos locais (minímos locais) e não explorar tanto o espaço de soluções.

### 🔥 Simulated Annealing
É uma meta-heurística inspirada no processo de resfriamento de metais. Aceita soluções piores com uma certa probabilidade (que diminui com o tempo), permitindo escapar de mínimos locais.

---

## 🧪 Experimentos

Foram realizadas **500 execuções** para cada algoritmo, usando estados iniciais aleatórios, e medindo:

- Tempo de execução médio
- Pontuação média (custo final da solução)
- Melhor/Pior estado
- Melhor/Pior pontuação
- Melhor/Pior tempo

---

## 📊 Resultados Obtidos

### 🧗 Hill Climbing:
- ⏱️ **Tempo médio:** 0.0006 segundos
- 🎯 **Pontuação média:** 2.4000
- 🥇 **Melhor estado:** `[0, 2, 4, 1, 7, 0, 3, 6]`
- ✅ **Melhor pontuação:** 2
- 🐇 **Melhor tempo:** 0.00025 segundos
- ❌ **Pior pontuação:** 2
- 🐢 **Pior tempo:** 0.00108 segundos

### 🌡️ Simulated Annealing:
- ⏱️ **Tempo médio:** 0.0412 segundos
- 🎯 **Pontuação média:** 1.6280
- 🥇 **Melhor estado:** `[0, 2, 6, 1, 7, 7, 3, 3]`
- ✅ **Melhor pontuação:** 4
- 🐇 **Melhor tempo:** 0.03155 segundos
- ❌ **Pior pontuação:** 4
- 🐢 **Pior tempo:** 0.04796 segundos

---

### 📌 Conclusão

Os testes mostram uma diferença clara entre os dois métodos:

- **Hill Climbing** é extremamente rápido, mas limitado. Ele tende a encontrar soluções razoáveis com agilidade, mas por seguir sempre o caminho "mais promissor", pode acabar preso em mínimos locais — ou seja, soluções que parecem boas, mas não são as melhores.

- **Simulated Annealing**, por outro lado, se destaca pela sua flexibilidade. Apesar de ser mais lento, ele é capaz de explorar melhor o espaço de soluções, aceitando temporariamente estados piores para, potencialmente, encontrar soluções mais próximas do ótimo global.

💡 **Resumo prático:**  
Se você quer **velocidade**, vá de *Hill Climbing*.  
Se busca **qualidade de solução**, *Simulated Annealing* leva vantagem.

---

## 🏁 Como Executar

```bash
python main.py
```

Isso irá rodar os dois algoritmos 500 vezes e exibir os resultados consolidados no terminal.

