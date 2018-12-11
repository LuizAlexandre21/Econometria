##Pacotes
from linearmodels import*
import pandas as pd
import numpy as np
from statistics import variance
scipy.stats.ttest_ind
##Importando os Dados
local="/home/luiz-alexandre/Documentos/PIBIC/Pobreza no Brasil(2004-2014)/Painel.xlsx"
dados=pd.read_excel(local,skiprows=1)
dados2=pd.read_excel(local,skiprows=1)
##Ajustando a base
dados=dados.set_index(["Year","ID"])
dadoestil=dados.pivot_table(index=["Estado","Year"])


##Estimando resultados
#Extrema Pobreza
#GINI

#Pooled
modelo1=PooledOLS.from_formula('np.log(ExtrPobres)~1 + np.log(Gini) + np.log(Rendapc)',dados)
print(modelo1)
rm1=modelo1.fit()
print(rm1)

#Pooled com ajuste clusted
modelo2=PooledOLS.from_formula('np.log(ExtrPobres)~1+ np.log(Gini) + np.log(Rendapc)',dados)
rm2=modelo2.fit(cov_type='clustered',cluster_entity=True)
print(rm2)

#Efeito Fixo tempo

modelo3=PanelOLS.from_formula('np.log(ExtrPobres)~1+np.log(Gini)+np.log(Rendapc)+EntityEffects',dados)
rm3=modelo3.fit()
print(rm3)

#Efeito Fixo Cross section

modelo4=PanelOLS.from_formula('np.log(ExtrPobres)~1+np.log(Gini)+np.log(Rendapc)+TimeEffects',dados)
rm4=modelo4.fit()
print(rm4)

#Efeito Aleatório tempo

modelo5=RandomEffects.from_formula('np.log(ExtrPobres)~1+np.log(Gini)+np.log(Rendapc)+EntityEffects',dados)
rm5=modelo5.fit()
print(rm5)

#Efeito Aleatorio com ajuste
modelo6=RandomEffects.from_formula('np.log(ExtrPobres)~1+np.log(Gini)+np.log(Rendapc)+EntityEffects',dados)
rm6=modelo6.fit(cov_type='clustered',cluster_entity=True)
print(rm6)

#Escolha do Modelo Para estimação
#Teste Pooled
