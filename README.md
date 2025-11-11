## 🛍️ Sistema de Recomendação de E-commerce — Modelo SVD

Um *pipeline* de Machine Learning compacto e reproduzível que implementa um sistema de **Filtragem Colaborativa** (SVD) para prever o valor de compra de um cliente em produtos que ele ainda não conhece.

Este projeto simula um desafio real do varejo, focando em **personalização** e demonstrando a capacidade de **diagnosticar e corrigir problemas de modelagem**.

-----

## ✨ Recursos Principais

  * **Modelo de Decomposição Matricial (SVD):** Utiliza a Fatoração de Matrizes para aprender os "fatores latentes" (preferências não óbvias) de usuários e itens.
  * **Normalização Logarítmica:** Utiliza a função $\log(1+x)$ para normalizar o valor monetário da transação, garantindo que o modelo personalize as recomendações e não caia na média global de gastos.
  * **Avaliação Realista:** Utiliza o **RMSE (Root Mean Square Error)** para medir o erro de previsão do valor de compra do cliente na escala logarítmica.
  * **Pipeline Completo de Varejo:** Simula todo o processo, desde a limpeza de dados de transação (remover cancelamentos e nulos) até a geração da lista de itens recomendados para um cliente específico.

-----

## 📂 Conteúdo do Repositório

  * `systemrecomend.py` — O *script* Python completo contendo o pré-processamento, a limpeza dos dados, o treinamento do modelo SVD e a geração das recomendações finais.
  * `online_retail.csv` — O *dataset* de transações de varejo utilizado (necessário para rodar o script).
  * `README.md` — Este arquivo.

## ⚙️ Requisitos

Este projeto requer bibliotecas que permitem manipulação de dados e modelagem de sistemas de recomendação.

Instale as dependências no seu ambiente virtual:

```bash
pip install pandas numpy scikit-surprise
```

⚠️ **Nota de Compatibilidade:** Se houver erro de instalação, pode ser necessário fazer o *downgrade* do **NumPy** para uma versão compatível com `scikit-surprise` (ex: `numpy<2`):

```bash
pip uninstall numpy
pip install numpy==1.26.4
```

-----

## 🏃 Como Rodar

1.  **Coloque o Dataset:** Certifique-se de que o arquivo `online_retail.csv` esteja na mesma pasta que o script `systemrecomend.py`.
2.  **Ative o Ambiente Virtual:**
    ```bash
    .\rec_sys_env\Scripts\activate
    ```
3.  **Execute o Script:**
    ```bash
    python systemrecomend.py
    ```

O script imprimirá no console todas as etapas: limpeza de dados, dimensões da matriz, RMSE do modelo e as 10 principais recomendações de `StockCode` para o primeiro cliente do *dataset*.

-----

## 🧠 Funcionamento do Modelo: Pré-processamento e Correção

O sucesso deste projeto se deve à etapa de pré-processamento, que transformou dados brutos de transação em uma métrica de preferência funcional para o SVD.

### 1\. Limpeza de Dados de Transação

  * **Tratamento de Cancelamentos:** Transações com `InvoiceNo` iniciando com 'C' são removidas, garantindo que apenas compras efetivas sejam consideradas.
  * **Tratamento de Nulos:** Linhas sem `CustomerID` são descartadas, pois a Filtragem Colaborativa depende da identificação única do usuário.

### 2\. Ação e Correção Crítica: O Protagonismo na Modelagem

O `rating` (a métrica de preferência) é baseado no **Valor Total da Compra** (`Quantity * UnitPrice`).

  * **Diagnóstico:** Ao usar o valor absoluto, o modelo SVD inicial falhou (RMSE altíssimo), pois estava sendo "puxado" por transações de valores extremos, caindo na **média global**.
  * **Ação Corretiva (Normalização):** O *rating* foi redefinido como **logaritmo do Valor Total** ($\log(1 + \text{Valor Total})$). Esta transformação reduz a dispersão e permite que o SVD **personalize** as previsões, resultando em um **RMSE baixo e relevante** (em torno de $0.53$), provando que o modelo está de fato aprendendo as preferências do cliente.

### 3\. Geração de Recomendações

O SVD treinado prevê o *rating logarítmico* que o cliente daria a itens que ele ainda não comprou. Os itens com o maior *rating* previsto são apresentados como as melhores recomendações, maximizando a chance de *cross-selling* no e-commerce.

-----

## ✍️ Motivação

Este projeto foi desenvolvido para portfolio e estudo. Ele demonstra proficiência em **AI-driven**, **Aprendizagem Contínua** ao aplicar métodos de *Machine Learning* para resolver um desafio de negócio central no varejo: a personalização da experiência do cliente.
