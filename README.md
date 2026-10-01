# Quantitative B2B Energy Churn Audit & Financial Risk Optimization

### Boston Consulting Group (BCG X) Job Simulation — PowerCo Case Study

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble-orange.svg)](https://scikit-learn.org/)
[![Model Risk Management](https://img.shields.io/badge/Governance-SR%2011--7%20Aligned-darkgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### Language Navigation / Selección de Idioma

- [English Version — Statistical Audit, MRM & Asymmetric Risk Evaluation](#english-version)
- [Versión en Español — Auditoría Estadística, Gestión de Riesgo de Modelo (MRM) y Optimización Asimétrica](#versión-en-español)

---

<a name="english-version"></a>

# [EN] Quantitative B2B Energy Churn Audit & Financial Risk Optimization

## 1. Executive Summary & Audit Mandate

This project conducts an empirical audit and econometric risk evaluation of customer churn for **PowerCo**, a major B2B utility provider. Based on a commercial hypothesis posed by BCG leadership—_that churn is primarily driven by price sensitivity and tariff inflation_—this investigation subjects the hypothesis to rigorous statistical testing, establishes a deterministic data governance framework, and develops an operational model risk management (MRM) pipeline aligned with **Federal Reserve SR 11-7 / OCC 2011-12** standards.

### Key Audit Findings:

1. **Refutation of the Price Sensitivity Hypothesis:** Unweighted price variance ($\Delta P = P_{\text{Dec}} - P_{\text{Jan}}$) exhibits near-zero predictive correlation with churn ($r \approx 0.01$). Churn is primarily driven by **volumetric consumption exposure** ($\text{cons\_12m}$) and **margin sensitivity** ($\text{margin\_net\_pow\_ele}$, $\text{margin\_gross\_pow\_ele}$).
2. **The 90.3% Accuracy Fallacy:** The portfolio displays severe class imbalance ($90.3\%$ retained, $9.7\%$ churned). A standard non-intervention baseline achieves $90.3\%$ accuracy while failing to detect $100\%$ of departing clients, resulting in catastrophic loss of net margin ($-\$60,315.68$ in the test set).
3. **Asymmetric Loss Threshold Optimization:** Operating the `RandomForestClassifier` at the default threshold $p = 0.50$ produces operational blindness ($\text{Recall} = 13\%-18\%$). By calibrating the decision threshold to **$p^* = 0.25$** under an asymmetric corporate payoff matrix, PowerCo mitigates False Negative leakage by $>50\%$, preserving **$+\$27,420.58$ in net margin** compared to the zero-intervention baseline.

---

## 2. Quantitative Architecture & Pipeline Integrity

```
[price_data.csv] (193k monthly records)  ──┐
                                           ├──> [Feature Engineering] ──> [Stratified Split (80/20)] ──> [RandomForest] ──> [Bayesian Loss Sweep]
[client_data.csv] (14.6k corporate SME)  ──┘     - Price Deltas (Dec-Jan)     - Preserve 9.7% Base Rate     - Balanced Weights    - Threshold p* = 0.25
                                                 - Financial Shock (ΔS)       - Zero Data Leakage           - 100 Estimators      - Net Margin Maximization
                                                 - Continuous Tenures
```

### 2.1 Component Repository Links

- **Data Quality & Exploratory Analysis:** [`PowerCO_EDA.ipynb`](./PowerCO_EDA.ipynb)
- **Econometric Feature Engineering:** [`Features_engineering.ipynb`](./Features_engineering.ipynb)
- **Risk Modeling & Threshold Sweep:** [`Random_Forest.ipynb`](./Random_Forest.ipynb)
- **Transformed Dataset:** [`clean_data_after_eda.csv`](./clean_data_after_eda.csv)
- **Scored Production Matrix:** [`data_for_predictions.csv`](./data_for_predictions.csv)

---

## 3. Data Governance & Statistical Audit (EDA)

An audit of 14,606 corporate accounts and 192,975 price observations revealed structural distribution anomalies requiring strict pre-modeling controls:

| Audit Domain                 | Issue Identified                                                                                                                                                                            | Quantitative Remediation                                                                                                                                               |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Heavy-Tailed Volumetrics** | Electricity consumption ($\text{cons\_12m}$) exhibited extreme right-skew ($\text{Skewness} > 15$, Kurtosis $> 300$). Mean was non-representative ($\mu = 159,222$, but Median $= 14,183$). | Preserved heavy tails without arbitrary outlier trimming; models rely on rank-invariant decision trees robust to monotonic scaling.                                    |
| **Missingness Mechanism**    | `channel_sales` missingness ($28.8\%$) violated MCAR (Missing Completely at Random).                                                                                                        | Handled via categorical imputation token (`MISSING`), preventing sample size degradation and preserving structural non-reporting signal.                               |
| **Temporal Data Leakage**    | Historical dates (`date_activ`, `date_end`, `date_modif_prod`, `date_renewal`) risked look-ahead bias.                                                                                      | Engineered relative tenure vectors referenced strictly to retrospective anchor date `2016-01-01` ($\text{act\_months}$, $\text{ren\_months}$, $\text{modif\_months}$). |
| **Class Imbalance Risk**     | Churn baseline rate is $9.71\%$ ($N=1,419$) vs $90.29\%$ non-churn ($N=13,187$).                                                                                                            | Stratified random partitioning ($80/20$, seed $42$) to guarantee constant probability priors across training and test partitions.                                      |

---

## 4. Econometric Feature Engineering

Raw tariff data were converted into client-specific cost-shock metrics:

### 4.1 Vectorized Tariff Price Dynamics ($\Delta P$)

Unit prices for energy ($\text{var}$) and power capacity ($\text{fix}$) across peak, off-peak, and mid-peak periods were differenced between December ($t_{12}$) and January ($t_1$) of the baseline year:
$$\Delta P_{i, \tau}^{\text{type}} = P_{i, 12, \tau}^{\text{type}} - P_{i, 1, \tau}^{\text{type}}, \quad \forall \tau \in \{\text{off\_peak}, \text{peak}, \text{mid\_peak}\}, \; \text{type} \in \{\text{var}, \text{fix}\}$$

### 4.2 Total Financial Shock ($\Delta \mathcal{S}$)

The microeconomic impact of price changes is conditioned by corporate consumption volume. We formulated the **Total Financial Shock** ($\Delta \mathcal{S}_i$) as the inner product of volumetric vectors and tariff deltas:
$$\Delta \mathcal{S}_i = \sum_{\tau} \left( \Delta P_{i, \tau}^{\text{var}} \times \text{cons\_12m}_i \right) + \sum_{\tau} \left( \Delta P_{i, \tau}^{\text{fix}} \times \text{pow\_max}_i \right)$$

This metric proved to be the **4th most important predictive feature** overall ($5.51\%$ feature importance), far surpassing isolated price differentials ($< 1.0\%$).

---

## 5. Model Risk Management: The Asymmetric Loss Function

### 5.1 Rejection of Symmetrical Loss (Accuracy & $p = 0.50$)

Under symmetrical cost assumptions ($C(\text{FP}) = C(\text{FN})$), standard algorithms apply a default threshold $p = 0.50$. In utility risk management, costs are deeply asymmetric:

- **False Positive (Type I Error, $\alpha$):** Offering a $20\%$ commercial discount to a customer who had no intention of leaving.
  $$\text{Loss}_{\text{FP}} = 0.20 \times \text{net\_margin}_i$$
- **False Negative (Type II Error, $\beta$):** Failing to identify an exiting customer, leading to complete and permanent loss of account value (absorbing barrier).
  $$\text{Loss}_{\text{FN}} = 1.00 \times \text{net\_margin}_i$$

The commercial penalty ratio is asymmetric:
$$\frac{C(\text{FN})}{C(\text{FP})} = \frac{1.00}{0.20} = 5.0$$
Failing to intervene is **5 times more costly** than an accidental discount giveaway.

### 5.2 Global Economic Objective Function

For any probability cut-off $p$, the net portfolio impact $\mathcal{U}(p)$ relative to account margin $M_i = \text{net\_margin}_i$ is defined as:
$$\mathcal{U}(p) = \sum_{i \in \text{TP}(p)} 0.80 M_i - \sum_{j \in \text{FP}(p)} 0.20 M_j - \sum_{k \in \text{FN}(p)} 1.00 M_k$$

### 5.3 Empirical Threshold Calibration Sweep

| Decision Threshold ($p$) | Saved Net Margin ($\text{TP}$) | Accidental Giveaway ($\text{FP}$) | Unseen Churn Losses ($\text{FN}$) | Net Portfolio Margin ($\mathcal{U}$) | Operational Verdict                                             |
| :----------------------: | :----------------------------: | :-------------------------------: | :-------------------------------: | :----------------------------------: | :-------------------------------------------------------------- |
|         **0.10**         |         $\$44,123.67$          |           $\$69,896.13$           |           $\$6,213.84$            |            $-\$31,986.30$            | Excessive budget dilution                                       |
|         **0.20**         |         $\$32,263.86$          |           $\$34,284.01$           |           $\$21,038.61$           |            $-\$23,058.77$            | High sensitivity regime                                         |
|    **0.25 (OPTIMAL)**    |       **$\$28,813.78$**        |         **$\$21,345.54$**         |         **$\$25,351.21$**         |          **$-\$17,882.97$**          | **Optimal Corporate Threshold ($+\$32.5\text{k}$ vs baseline)** |
|         **0.30**         |         $\$23,652.99$          |           $\$14,008.52$           |           $\$31,802.19$           |            $-\$22,157.72$            | Conservative intervention                                       |
|    **0.50 (DEFAULT)**    |          $\$5,543.34$          |           $\$1,482.73$            |           $\$54,439.25$           |            $-\$50,378.64$            | **Defective: Catastrophic churn leakage**                       |
|  **Zero Intervention**   |            $\$0.00$            |             $\$0.00$              |           $\$60,315.68$           |            $-\$60,315.68$            | Absorbing boundary breakdown                                    |

**Audit Conclusion:** Recalibrating from $p=0.50$ to $p^*=0.25$ achieves optimal balance between discount preservation and capital protection, generating a **$+\$32,495.67$ direct net profit improvement** across the holdout cohort alone.

---

## 6. Econometric Audit of Feature Importances

Gini-impurity decomposition ($I(f)$) across 100 balanced decision trees confirms that structural account size and margin drive churn, not tariff inflation:

```
margin_gross_pow_ele   [████████████████████] 5.97%
margin_net_pow_ele     [███████████████████ ] 5.85%
cons_12m               [██████████████████  ] 5.69%
total_financial_shock  [█████████████████   ] 5.51%
forecast_meter_rent    [█████████████████   ] 5.49%
net_margin             [███████████████     ] 4.87%
forecast_cons_12m      [███████████████     ] 4.81%
cons_last_month        [████████████        ] 3.92%
var_year_price_off     [███████████         ] 3.74%
act_months             [███████████         ] 3.71%
```

_Tariff differentials alone ($\Delta P$) accounted for $<1.2\%$ individual importance._ Price increases only matter when scaled against corporate consumption intensity ($\Delta \mathcal{S}$).

---

<a name="versión-en-español"></a>

# [ES] Auditoría Cuantitativa de Churn B2B y Optimización Asimétrica del Riesgo Financiero

## 1. Resumen Ejecutivo y Mandato de Auditoría

Este proyecto ejecuta una auditoría empírica y evaluación econométrica del riesgo de abandono (_churn_) en clientes B2B para **PowerCo**, una empresa proveedora de energía. Frente a la hipótesis estratégica inicial de la consultora (BCG)—_que el churn es impulsado primordialmente por la sensibilidad al precio y los incrementos tarifarios_—este trabajo somete la premisa a prueba estadística formal, diseña un marco determinista de gobernanza de datos y construye un protocolo de Gestión de Riesgo de Modelos (MRM) bajo los estándares internacionales **Federal Reserve SR 11-7 / OCC 2011-12**.

### Hallazgos Principales de Auditoría:

1. **Refutación de la Hipótesis de Sensibilidad al Precio:** La variación marginal de precios no ponderada ($\Delta P = P_{\text{Dic}} - P_{\text{Ene}}$) tiene correlación casi nula con el churn ($r \approx 0.01$). El abandono está dominado por la **exposición volumétrica de consumo** ($\text{cons\_12m}$) y la **sensibilidad de los márgenes comerciales** ($\text{margin\_net\_pow\_ele}$, $\text{margin\_gross\_pow\_ele}$).
2. **La Falacia de la Exactitud (Accuracy del 90.3%):** La cartera presenta un desbalance severo ($90.3\%$ retenidos, $9.7\%$ churners). Un clasificador ingenuo que prediga retención total obtiene un $90.3\%$ de precisión formal mientras oculta la fuga del $100\%$ de clientes en riesgo, provocando la pérdida íntegra del margen neto ($-\$60,315.68$ en el conjunto de prueba).
3. **Optimización del Umbral de Pérdida Asimétrica:** Usar `RandomForestClassifier` bajo el umbral estándar $p = 0.50$ induce ceguera operativa ($\text{Sensibilidad/Recall} = 13\%-18\%$). Al calibrar el punto de corte probabilístico en **$p^* = 0.25$** bajo una matriz de pagos financieros asimétrica, PowerCo reduce las pérdidas por Falsos Negativos en más del $50\%$, preservando **$+\$27,420.58$ en margen neto** frente al escenario base sin intervención.

---

## 2. Enlaces a los Componentes del Sistema

- **Auditoría de Integridad y EDA:** [`PowerCO_EDA.ipynb`](./PowerCO_EDA.ipynb)
- **Ingeniería de Características y Choque Financiero:** [`Features_engineering.ipynb`](./Features_engineering.ipynb)
- **Modelado y Barrido de Umbrales Financieros:** [`Random_Forest.ipynb`](./Random_Forest.ipynb)
- **Datos Auditados Post-EDA:** [`clean_data_after_eda.csv`](./clean_data_after_eda.csv)
- **Matriz de Predicción:** [`data_for_predictions.csv`](./data_for_predictions.csv)

---

## 3. Gobernanza de Datos y Auditoría Estadística

La auditoría sobre 14,606 cuentas B2B y 192,975 registros tarifarios mensuales reveló anomalías distributivas estructurales mitigadas mediante controles técnicos:

| Dominio Auditado                          | Condición Identificada                                                                                                                                                                                          | Mitigación Cuantitativa                                                                                                                                    |
| :---------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Distribuciones con Colas Pesadas**      | El consumo de energía ($\text{cons\_12m}$) exhibe asimetría positiva extrema ($\text{Asimetría} > 15$, Curtosis $> 300$). La media está completamente distorsionada ($\mu = 159,222$, pero Mediana $= 14,183$). | No se truncaron valores extremos arbitrariamente; se emplearon ensambles de árboles de decisión invariantes a transformaciones monótonas.                  |
| **Mecanismo de Datos Faltantes**          | Vacíos en `channel_sales` ($28.8\%$) no eran aleatorios (no-MCAR).                                                                                                                                              | Se preservó la señal estructural mediante codificación explícita de imputación (`MISSING`), evitando sesgo de selección por eliminación de filas.          |
| **Fuga Temporal de Datos (Data Leakage)** | Fechas de contrato (`date_activ`, `date_end`, etc.) introducían sesgo de anticipación.                                                                                                                          | Se construyeron vectores continuos de antigüedad con fecha de corte retrospectiva fija al `2016-01-01` ($\text{act\_months}$, $\text{ren\_months}$, etc.). |
| **Riesgo por Desbalance de Clases**       | Tasa base de churn del $9.71\%$ frente al $90.29\%$ de retención.                                                                                                                                               | Particionamiento estratificado determinista ($80/20$, semilla 42) asegurando priors idénticos en entrenamiento y testeo.                                   |

---

## 4. Ingeniería Econométrica de Características

### 4.1 Dinámica Temporal de Tarifas ($\Delta P$)

Se vectorizó la resta matricial entre diciembre ($t_{12}$) y enero ($t_1$) para tarifas de energía ($\text{var}$) y potencia fija ($\text{fix}$) en períodos valle, pico y medio:
$$\Delta P_{i, \tau}^{\text{tipo}} = P_{i, 12, \tau}^{\text{tipo}} - P_{i, 1, \tau}^{\text{tipo}}$$

### 4.2 Choque Financiero Total ($\Delta \mathcal{S}$)

El impacto microeconómico de un cambio tarifario depende de la exposición volumétrica del cliente. Se calculó el **Choque Financiero Total** ($\Delta \mathcal{S}_i$) integrando el volumen anual contratado:
$$\Delta \mathcal{S}_i = \sum_{\tau} \left( \Delta P_{i, \tau}^{\text{var}} \times \text{cons\_12m}_i \right) + \sum_{\tau} \left( \Delta P_{i, \tau}^{\text{fix}} \times \text{pow\_max}_i \right)$$

Este indicador resultó ser el **4to predictor más relevante de todo el modelo** ($5.51\%$ de importancia), superando holgadamente a los diferenciales de precios aislados ($< 1.0\%$).

---

## 5. Gestión del Riesgo de Modelo (MRM): Función de Pérdida Asimétrica

### 5.1 Desestimación de la Función de Costo Simétrica

En auditoría y análisis cuantitativo, asumir que el Error Tipo I ($\alpha$, Falso Positivo) equivale al Error Tipo II ($\beta$, Falso Negativo) es una falla metodológica grave:

- **Falso Positivo (FP):** Descuento comercial del $20\%$ otorgado a un cliente que no planeaba retirarse.
  $$\text{Costo}_{\text{FP}} = 0.20 \times \text{net\_margin}_i$$
- **Falso Negativo (FN):** No detectar un cliente en riesgo, provocando la pérdida permanente y total del valor del contrato (barrera absorbente).
  $$\text{Costo}_{\text{FN}} = 1.00 \times \text{net\_margin}_i$$

Relación asimétrica de costos:
$$\frac{C(\text{FN})}{C(\text{FP})} = \frac{1.00}{0.20} = 5.0$$
Cometer un Falso Negativo es **5 veces más destructivo financieramente** que cometer un Falso Positivo.

### 5.2 Función de Recompensa Económica Neta

Para cualquier umbral de probabilidad $p$, el impacto financiero neto $\mathcal{U}(p)$ sobre el margen $M_i = \text{net\_margin}_i$ es:
$$\mathcal{U}(p) = \sum_{i \in \text{TP}(p)} 0.80 M_i - \sum_{j \in \text{FP}(j)} 0.20 M_j - \sum_{k \in \text{FN}(k)} 1.00 M_k$$

### 5.3 Tabla de Calibración de Umbrales de Decisión

|     Umbral ($p$)     | Margen Salvado ($\text{TP}$) | Desperdicio en Descuentos ($\text{FP}$) | Pérdida por Churn No Visto ($\text{FN}$) | Impacto Neto en Cartera ($\mathcal{U}$) | Diagnóstico de Auditoría                                 |
| :------------------: | :--------------------------: | :-------------------------------------: | :--------------------------------------: | :-------------------------------------: | :------------------------------------------------------- |
|       **0.10**       |        $\$44,123.67$         |              $\$69,896.13$              |               $\$6,213.84$               |             $-\$31,986.30$              | Sobre-intervención costosa                               |
|       **0.20**       |        $\$32,263.86$         |              $\$34,284.01$              |              $\$21,038.61$               |             $-\$23,058.77$              | Régimen conservador                                      |
|  **0.25 (ÓPTIMO)**   |      **$\$28,813.78$**       |            **$\$21,345.54$**            |            **$\$25,351.21$**             |           **$-\$17,882.97$**            | **Punto Óptimo Empresarial ($+\$32.5\text{k}$ vs base)** |
|       **0.30**       |        $\$23,652.99$         |              $\$14,008.52$              |              $\$31,802.19$               |             $-\$22,157.72$              | Pérdida de cobertura                                     |
|  **0.50 (DEFECTO)**  |         $\$5,543.34$         |              $\$1,482.73$               |              $\$54,439.25$               |             $-\$50,378.64$              | **Defectuoso: Fuga masiva de clientes**                  |
| **Sin Intervención** |           $\$0.00$           |                $\$0.00$                 |              $\$60,315.68$               |             $-\$60,315.68$              | Colapso por barrera absorbente                           |

**Dictamen de Auditoría:** Mover el umbral a **$p^* = 0.25$** minimiza la pérdida total esperada y salvaguarda **$+\$32,495.67$** en comparación con el umbral por defecto $0.50$.

---

## 6. Auditoría Econométrica de Importancia de Variables

La descomposición de impureza de Gini ($I(f)$) demuestra que la estructura de márgenes y volumen gobierna el comportamiento de los clientes corporativos:

```
margin_gross_pow_ele   [████████████████████] 5.97%
margin_net_pow_ele     [███████████████████ ] 5.85%
cons_12m               [██████████████████  ] 5.69%
total_financial_shock  [█████████████████   ] 5.51%
forecast_meter_rent    [█████████████████   ] 5.49%
net_margin             [███████████████     ] 4.87%
forecast_cons_12m      [███████████████     ] 4.81%
cons_last_month        [████████████        ] 3.92%
var_year_price_off     [███████████         ] 3.74%
act_months             [███████████         ] 3.71%
```

Las variaciones puras de precios no tienen fuerza explicativa per se; solo adquieren tracción al ponderarse frente a la masa de consumo del cliente corporativo.

---

## 7. Instrucciones de Reproducibilidad y Entorno Local

Optimizada para arquitecturas UNIX / macOS (Apple Silicon) gestionadas con `uv`:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Giorferp/PowerCo_Simulation.git
cd PowerCo_Simulation

# 2. Inicializar entorno virtual e instalar dependencias con uv
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# 3. Ejecución de pipelines de análisis
jupyter notebook PowerCO_EDA.ipynb
jupyter notebook Features_engineering.ipynb
jupyter notebook Random_Forest.ipynb
```

---

**Consultor / Auditor Cuantitativo:** Giovanni Fermin
**Especialidad:** Ciencias Estadísticas, Universidad Central de Venezuela (UCV)  
**Rol:** Analista Cuantitativo & Consultor de Riesgo de Modelos (BCG X Simulation)
