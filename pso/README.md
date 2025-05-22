# **Comparação entre algoritmos de PSO:**

#### **Wierstrass**:

* **Configuração 1 (30 partículas, 500 iterações)**:

  * **Best Fitness**: -21.9883
  * O PSO com uma população pequena (30 partículas) e poucas iterações conseguiu encontrar um bom valor de fitness, mas o desvio padrão e a média indicam que há uma certa dispersão nas soluções encontradas, o que sugere que o espaço de busca não foi suficientemente explorado.

* **Configuração 2 (50 partículas, 1000 iterações)**:

  * **Best Fitness**: -21.9979
  * Aumento da população e iterações ajudou a encontrar uma solução ligeiramente melhor e mais consistente, com um desvio padrão menor (0.002) e uma mediana muito próxima da melhor solução encontrada.

* **Configuração 3 (100 partículas, 2000 iterações)**:

  * **Best Fitness**: -21.9981
  * A configuração com maior população e iterações convergiu para uma solução ainda mais precisa e consistente, com um desvio padrão muito baixo (0.00098), indicando uma boa estabilidade nas soluções encontradas.

#### **Rotated High Conditional Elliptic**:

* **Configuração 1 (30 partículas, 500 iterações)**:

  * **Best Fitness**: 9549.68
  * O PSO com uma população pequena (30 partículas) e poucas iterações encontrou uma solução razoável, mas com alta variabilidade nas soluções, evidenciada pelo desvio padrão de 2603.33.

* **Configuração 2 (50 partículas, 1000 iterações)**:

  * **Best Fitness**: 6682.35
  * O PSO com uma população maior e mais iterações reduziu a variabilidade (desvio padrão de 2389.20) e encontrou uma solução melhor.

* **Configuração 3 (100 partículas, 2000 iterações)**:

  * **Best Fitness**: 7254.74
  * A configuração com 100 partículas e 2000 iterações forneceu a melhor solução entre as três, com uma mediana mais baixa e uma variabilidade (desvio padrão de 1612.12) mais controlada.

### **Conclusão Crítica (PSO)**:

* **Wierstrass**: O PSO foi muito eficiente em encontrar soluções ótimas com pequenas populações e poucas iterações, especialmente na **Configuração 3**, que teve uma convergência muito precisa.
* **Rotated High Conditional Elliptic**: O PSO teve mais dificuldades em encontrar soluções de alta qualidade com as populações e iterações menores. No entanto, com o aumento da população e das iterações, o algoritmo foi capaz de encontrar soluções mais estáveis e de melhor qualidade, especialmente na **Configuração 3**.
