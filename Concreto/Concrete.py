import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter
from scipy.stats import shapiro, probplot, pearsonr, linregress

# Fazendo a Leitura do arquivo
file_path = 'Concreto/concrete.csv'
concrete = pd.read_csv(file_path, sep=',')

print('=' * 70)
print(' VERFICANDO OS DADOS DO ARQUIVO ')
print('=' * 70)

# Verificando informações importantes
print(concrete.describe())
print(concrete.info()) # Todos os dados estão em float/int
print(concrete.head())
print(concrete.columns) # Aqui mostra as colunas em formato de lista

# Início do tratamento dos dados
print('=' * 70)
print(' INICIANDO EDA ')
print('=' * 70)

valores_nan = concrete.isnull().sum().sum() # Aqui retorna a quantidade de valores NaN
if valores_nan == 0:
    print('Ótimo! Não temos valores NaN')
else:
    print(f'Temos valores {valores_nan} NaN')

concrete_clean = concrete.drop_duplicates()
if concrete.shape[0] != concrete_clean.shape[0]:
    print('Linhas duplicadas foram encontradas e removidas!')
print(f'Temos na planilha {concrete_clean.shape[0]} concretos e {concrete_clean.shape[1]} features')

print('=' * 70)
print(' ANÁLISE ESTATÍSTICA ')
print('=' * 70)

print(f'A média dos valores de resistência à compressão é de: {concrete_clean.strength.mean():.3f} MPa')
print(f"O desvio padrão das amostras é de: {concrete_clean['strength'].std():.3f} MPa")

print('=' * 70)

print(' MÉTODO DE SHAPIRO-WILK ')
stats, p_value = shapiro(concrete_clean['strength'])
print(f'Estatística de Teste: {stats:.3f}')
print(f'Cálculo do p-valor: {p_value:.10f}')
if p_value < 0.05:
    print('Rejeitamos a hipótese de que a distribuição é uma distribuição normal')
else:
    print('Falha ao rejeitar a hipótese nula. O gráfico parece seguir uma distribuição normal')  
print('=' * 70)

print('=' * 70)
print(' QQ-PLOT ')
fig, ax = plt.subplots(figsize=(5,5))
sample = concrete_clean['strength']
slope, r_squared = probplot(sample, plot=ax, rvalue=True)
ax.set_title('Distribuição da QQ')
plt.show()
print('=' * 70)
print('Podemos perceber que a distribuição pelo método QQ-Plot não satisfaz para uma distribuição normal')
print(f'Apesar de R² estar bem ajustado {r_squared[2]:.3f} podemos perceber pelo gráfico que a curva não segue uma normal')

print('=' * 70)
print(' UTILIZANDO O FITTER PARA VERIFICAÇÃO ')
func_distr = Fitter(concrete_clean.strength)
func_distr.fit()
print(func_distr.summary())
plt.show()
print("A melhor distribuição matemática para estes dados é:")
print(func_distr.get_best(method='sumsquare_error'))
print('=' * 70)

print('=' * 70)
print(' PLOTANDO GRÁFICOS PARA A RELAÇÃO ÁGUA/CIMENTO ')
print('=' * 70)

# Aqui criei uma nova series baseada num conceito muito importante para a resistência
concrete_clean['Relação água/cimento'] = concrete_clean['water'].values / concrete_clean['cement'].values
print(concrete_clean.head())

# Criando a quantidade de gráficos que serão representados
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,15))

# Fazendo as primeiras relações
# Relação entre a resistência do concreto e a relação água/cimento
corr_matrix = concrete_clean.corr()
print(f'A matriz de correlação é dada por: \n{corr_matrix}')
print(f'As maiores correlações da Resistência à Compressão são entre: {corr_matrix['strength'].drop(index='strength').idxmax()}, {corr_matrix.strength.idxmin()}')

sns.histplot(data=concrete_clean, x='strength', kde=True, ax=axes[0], color='green')
axes[0].set_title('Distribuição da resistência (MPa)')
axes[0].set_ylabel('Distribuição')
sns.scatterplot(data=concrete_clean, x='Relação água/cimento', y='strength', ax=axes[1], color='purple')
axes[1].set_title('Resistência (MPa) x Relação a/c')
axes[1].set_ylabel('Resistência à compressão (MPa)')
axes[1].set_xlabel('Relação a/c')
# hspace = espaço horizontal (altura) entre as linhas
# wspace = espaço vertical (largura) entre as colunas
plt.subplots_adjust(hspace=0.4, wspace=0.3)

""" Criamos aqui a seguinte hipótese: Não existe relação nenhuma entre a quantidade de água/cimento e a força do concreto
    Ou seja, queremos verificar se a correlação é igual ou muito proxima de zero 
    Assim temos a hipótese alternativa: O coeficiente de correlação é diferente de 0 """

print('=' * 70)
print(' CALCULANDO PELO MÉTODO DE PEARSON ')
print('=' * 70)
print('Iniciando o teste estatístico (Pearson)...')
r, p_valor = pearsonr(concrete_clean['Relação água/cimento'], concrete_clean.strength)
print(f'Valor calculado do P-Valor: {p_valor:.10f}')
print(f'Coeficiente de correlação de Pearson: {r:.3f}')
if p_valor < 0.05:
    print('Rejeitamos a Hipótese Nula (H0)')
    if r < 0:
        print('Aceitamos a Hipótese Alternativa (H1)')
        print('Logo temos uma relação entre o coeficiente a/c e a resistência do concreto')
else:
    print('Falha em rejeitar a hipótese nula!')
print('=' * 70)

print('=' * 70)
print(' CALCULANDO PELO MÉTODO R-QUADRADO ')
print('=' * 70)
print('Iniciando o teste estatístico (R-QUADRADO)...')
slope, intercept, r, p_value, std_slope = linregress(concrete_clean['Relação água/cimento'], concrete_clean.strength)
print(f'Valor calculado do P-Valor: {p_value:.10f}')
print(f'Coeficiente de R-QUADRADO: {r**2:.3f}')
print(f'Valor da taxa de crescimento: {slope:.3f}')
print('Temos uma relação sim entre os valores de a/c e da resistência à compressão, entretando com base\n' \
'no índice de R-QUADRADO temos que essa relação representa apenas cerca de 24% na influência da resistência do concreto' \
'por isso devemos buscar novas correlações!')

print('Gerando gráfico da relação...')
plt.figure(figsize=(8, 5))
sns.regplot(data = concrete_clean, 
            x='Relação água/cimento', 
            y='strength', 
            line_kws={'color':'red', 'linewidth': 3},
            scatter_kws={'alpha': 0.4, 'color': 'green'})
plt.title(f'Lei de Abrams: Relação A/C vs Resistência')
plt.xlabel('Relação Água / Cimento (A/C)')
plt.ylabel('Resistência à Compressão (MPa)')
print('=' * 70)

plt.tight_layout()
plt.show()

