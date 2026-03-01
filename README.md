# Análise Estatística da Resistência do Concreto: Validando a Lei de Abrams 🏗️📊

Este projeto é uma aplicação de **Ciência de Dados e Estatística Inferencial** voltada para a Engenharia Civil. O objetivo principal é analisar um *dataset* de amostras de concreto e provar, de forma puramente estatística e baseada em dados, a **Lei de Abrams**: a regra fundamental que dita que a resistência do concreto é inversamente proporcional à relação Água/Cimento (A/C).

## 📌 O Estudo

Na dosagem de concreto, a água é necessária para a hidratação do cimento e trabalhabilidade da massa. Contudo, o excesso de água cria porosidade, reduzindo drasticamente a capacidade estrutural da peça. 

Neste projeto, em vez de assumir essa lei como uma verdade absoluta, utilizamos **Testes de Hipóteses** e **Modelagem Estatística** para verificar se os dados reais comprovam este fenômeno.

## 🧮 Metodologia e Modelagem Estatística

O projeto foi construído num pipeline completo de Análise Exploratória de Dados (EDA) e Inferência:

1. **Limpeza e Tratamento:** Verificação de dados nulos (NaN) e remoção de duplicatas para evitar enviesamento estatístico.
2. **Feature Engineering:** Criação da variável independente central do estudo:
   $$Relação_{A/C} = \frac{Agua}{Cimento}$$
3. **Análise de Distribuição:** * Utilização do Teste de **Shapiro-Wilk** e do método **QQ-Plot** para verificar a normalidade da resistência das amostras com relação à compressão.
4.  ** Aplicação da biblioteca `Fitter` para encontrar a distribuição matemática que melhor modela os dados reais.
5. **Teste de Correlação (Pearson):** Avaliação da correlação entre a resistência à compressão e a relação A/C
6. **Regressão Linear Simples:** Cálculo da equação da reta e do Coeficiente de Determinação ($R^2$).

## 📈 Principais Insights e Resultados

* **A Hipótese foi Confirmada:** O P-valor tendendo a zero no teste de Pearson nos permitiu rejeitar a Hipótese Nula. Existe uma relação **negativa** entre a relação A/C e a resistência, validando a Lei de Abrams.
* **O Limite da Relação A/C:** Através do R-Quadrado ($R^2$), descobrimos que a relação Água/Cimento explica **apenas cerca de 24%** da variação total da resistência do concreto neste dataset. 
* **Conclusão Prática:** Isso prova estatisticamente que controlar apenas a água e o cimento não garante a resistência estrutural. Os outros 75% dependem obrigatoriamente de outras *features* (como tempo de cura, uso de superplastificantes e tipos de agregados), abrindo espaço para futuras modelagens de **Machine Learning**.


## 🛠️ Bibliotecas Utilizadas

* **Python 3.12.6**
* **Pandas & NumPy:** Manipulação e limpeza de dados.
* **SciPy & Fitter:** Testes estatísticos de Shapiro, Pearson, QQ-Plot e cálculo de regressão.
* **Seaborn & Matplotlib:** Visualização de dados e plotagem de modelos estatísticos.
