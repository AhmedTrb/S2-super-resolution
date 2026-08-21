# Analyse des résultats d'entraînement et de prédiction des modèles DL

## 1. Objectif des expériences

Quatre expériences ont été réalisées afin d’évaluer l’influence de la résolution des images et de la représentation spectrale sur la prédiction de l’indice de jaunissement (*yellowness*).

| Expérience | Résolution | Variables d’entrée | Taille des patchs | Canaux | Paramètres entraînables |
|---|---:|---|---:|---:|---:|
| E1 | LR | Bandes brutes (`raw10`) | 128 × 128 | 10 | 168 921 |
| E2 | LR | Indices proposés dans l’article (`paper_indices`) | 128 × 128 | 8 | 168 489 |
| E3 | SR | Bandes brutes (`raw10`) | 256 × 256 | 10 | 168 921 |
| E4 | SR | Indices proposés dans l’article (`paper_indices`) | 256 × 256 | 8 | 168 489 |

Les modèles possèdent un nombre de paramètres très proche (environ 169 000). Les différences de performance observées peuvent donc être principalement attribuées à la résolution spatiale et au type de variables spectrales utilisé, plutôt qu’à une variation importante de capacité du modèle.

Les métriques utilisées sont :

- **MAE** (*Mean Absolute Error*) : erreur absolue moyenne ; une valeur faible est souhaitable.
- **RMSE** (*Root Mean Squared Error*) : erreur quadratique moyenne ; elle pénalise davantage les erreurs importantes.
- **R²** : proportion de la variabilité de la cible expliquée par le modèle ; une valeur proche de 1 est souhaitable.

---

## 2. Comparaison globale des performances

### 2.1 Résultats quantitatifs

| Expérience | MAE entraînement | MAE validation | MAE test | RMSE test | R² test |
|---|---:|---:|---:|---:|---:|
| E1 — LR + raw10 | 12,67 | 16,55 | 17,99 | 24,40 | 0,628 |
| E2 — LR + paper_indices | 13,33 | 16,11 | 19,76 | 28,14 | 0,506 |
| E3 — SR + raw10 | 13,82 | **13,68** | 18,29 | 24,22 | 0,634 |
| E4 — SR + paper_indices | **9,94** | 15,36 | **17,50** | **22,84** | **0,674** |

L’expérience **E4** fournit les meilleures performances sur le jeu de test. Elle atteint une MAE de **17,50**, un RMSE de **22,84** et un coefficient de détermination de **R² = 0,674**. Le modèle explique donc approximativement **67,4 % de la variabilité** observée dans les valeurs de jaunissement sur les données de test.

Par rapport à E1, qui constitue une référence avec les bandes brutes à basse résolution, E4 réduit :

- la MAE de **17,99 à 17,50**, soit une amélioration d’environ **2,7 %** ;
- le RMSE de **24,40 à 22,84**, soit une amélioration d’environ **6,4 %** ;
- et augmente le R² de **0,628 à 0,674**.

L’amélioration de la MAE entre E1 et E4 reste modérée, mais la baisse plus importante du RMSE indique que E4 réduit surtout certaines erreurs de forte amplitude. Ce point est important car les erreurs importantes concernent les observations présentant des niveaux de jaunissement élevés ou des comportements spectraux atypiques.

### 2.2 Classement des modèles sur le jeu de test

Selon les métriques de test, le classement global est le suivant :

1. **E4 — SR + paper_indices** : meilleur modèle global, avec le RMSE le plus faible et le R² le plus élevé.
2. **E3 — SR + raw10** : deuxième meilleur modèle selon le RMSE et le R².
3. **E1 — LR + raw10** : légèrement meilleur que E3 selon la MAE, mais moins performant selon le RMSE et le R².
4. **E2 — LR + paper_indices** : modèle le moins performant sur le jeu de test.

Le choix final du modèle peut donc être justifié par l’expérience **E4**, car elle présente le meilleur compromis entre précision moyenne, robustesse face aux grandes erreurs et capacité explicative.

---

## 3. Effet de la résolution spatiale

### 3.1 Comparaison E1 et E3 : bandes brutes

Les expériences E1 et E3 utilisent les mêmes 10 bandes brutes, mais E1 est basée sur une résolution basse (LR) alors que E3 utilise les images super-résolues (SR).

| Métrique test | E1 — LR + raw10 | E3 — SR + raw10 |
|---|---:|---:|
| MAE | **17,99** | 18,29 |
| RMSE | 24,40 | **24,22** |
| R² | 0,628 | **0,634** |

Le passage à la super-résolution avec les bandes brutes n’apporte pas une amélioration nette de la MAE. Toutefois, E3 améliore légèrement le RMSE et le R². Cela suggère que la résolution supérieure permet de mieux représenter certains cas complexes, mais que cette information spatiale supplémentaire ne se traduit pas systématiquement par une réduction de l’erreur absolue moyenne.

En revanche, E3 obtient la meilleure performance sur l’ensemble de validation, avec une MAE de **13,68** et un R² de **0,654**. Cette observation indique une bonne capacité de généralisation sur les données de validation, même si le gain est moins marqué sur le jeu de test.

### 3.2 Comparaison E2 et E4 : indices spectraux

Les expériences E2 et E4 utilisent les mêmes indices spectraux, avec une différence de résolution.

| Métrique test | E2 — LR + indices | E4 — SR + indices |
|---|---:|---:|
| MAE | 19,76 | **17,50** |
| RMSE | 28,14 | **22,84** |
| R² | 0,506 | **0,674** |

L’effet de la super-résolution est beaucoup plus marqué lorsque les variables d’entrée sont les indices spectraux. Le passage de E2 à E4 entraîne :

- une réduction de la MAE d’environ **11,4 %** ;
- une réduction du RMSE d’environ **18,8 %** ;
- une hausse du R² de **0,506 à 0,674**.

Ces résultats montrent que les indices spectraux semblent davantage bénéficier de l’augmentation de résolution spatiale. La super-résolution peut améliorer la représentation des motifs spatiaux et des hétérogénéités présentes dans les parcelles, ce qui permet au modèle de mieux relier les signatures spectrales au niveau de jaunissement.

---

## 4. Effet de la représentation spectrale

### 4.1 À basse résolution : E1 contre E2

À résolution basse, l’utilisation des bandes brutes (E1) est plus performante que l’utilisation des indices spectraux (E2).

| Métrique test | E1 — raw10 | E2 — paper_indices |
|---|---:|---:|
| MAE | **17,99** | 19,76 |
| RMSE | **24,40** | 28,14 |
| R² | **0,628** | 0,506 |

Les bandes brutes conservent une information spectrale plus complète que les indices calculés. À basse résolution, cette information supplémentaire semble utile pour compenser la perte de détails spatiaux. Les indices spectraux seuls ne permettent donc pas, dans E2, de représenter suffisamment la variabilité du jaunissement.

### 4.2 À super-résolution : E3 contre E4

À super-résolution, l’utilisation des indices spectraux devient plus favorable.

| Métrique test | E3 — raw10 | E4 — paper_indices |
|---|---:|---:|
| MAE | 18,29 | **17,50** |
| RMSE | 24,22 | **22,84** |
| R² | 0,634 | **0,674** |

Le modèle E4 dépasse E3 sur les trois métriques de test. Cette observation suggère que la combinaison entre les indices spectraux et les images super-résolues constitue la configuration la plus adaptée parmi les expériences évaluées.

Les indices peuvent fournir une représentation plus directement liée à l’état physiologique des plantes, tandis que la résolution spatiale plus élevée permet de mieux exploiter les structures locales visibles dans les patchs. La combinaison de ces deux sources d’information améliore donc la prédiction finale.

---

## 5. Analyse des courbes d’apprentissage

### 5.1 Convergence générale

Les quatre expériences présentent une dynamique d’apprentissage similaire au début de l’entraînement :

- les pertes d’entraînement et de validation diminuent fortement durant les premières époques ;
- le R², initialement négatif, devient progressivement positif ;
- les modèles apprennent donc une relation utile entre les données d’entrée et la variable cible.

Un R² négatif aux premières époques indique que le modèle est initialement moins performant qu’une prédiction constante égale à la moyenne de la cible. La progression vers des valeurs positives puis supérieures à 0,5 confirme l’apprentissage progressif de structures prédictives pertinentes.

### 5.2 E1 : LR avec bandes brutes

Pour E1, les pertes diminuent jusqu’à environ l’époque 20–30, puis les métriques de validation deviennent plus variables. La MAE d’entraînement continue à diminuer jusqu’à environ 13,3 en fin d’entraînement, alors que la MAE de validation reste proche de 16,5.

L’écart entre entraînement et validation indique un **surapprentissage modéré** en fin d’entraînement. Le modèle continue à s’adapter aux exemples d’entraînement, mais les gains ne sont pas entièrement transférés aux données non vues.

### 5.3 E2 : LR avec indices spectraux

E2 montre une convergence correcte, mais ses performances de validation et de test restent inférieures à celles de E1. Après les premières améliorations, les courbes de validation oscillent autour d’un plateau.

L’utilisation des indices spectraux à basse résolution semble limiter la capacité du modèle à capturer toute la variabilité de la cible. Le modèle apprend néanmoins une relation utile, comme l’indique le R² de test positif de 0,506, mais sa généralisation demeure limitée par rapport aux autres configurations.

### 5.4 E3 : SR avec bandes brutes

E3 présente la meilleure généralisation sur l’ensemble de validation. La diminution de la perte de validation est relativement rapide, et le R² de validation atteint des valeurs élevées au cours de l’entraînement.

L’écart entre les métriques d’entraînement et de validation est plus faible que pour E1 et E4, ce qui indique une convergence relativement équilibrée. Cette stabilité peut expliquer les bonnes performances de E3 sur les données de validation.

### 5.5 E4 : SR avec indices spectraux

E4 atteint les meilleures performances sur le jeu de test, mais présente également l’écart entraînement-validation le plus marqué :

- MAE entraînement : **9,94** ;
- MAE validation : **15,36** ;
- MAE test : **17,50**.

Le modèle apprend donc très bien les données d’entraînement, mais une partie de ce gain ne se généralise pas totalement aux ensembles validation et test. Cela indique un **surapprentissage modéré**, particulièrement après les époques où la performance de validation atteint son optimum.

Malgré cet écart, E4 reste le meilleur modèle sur le jeu de test. Les résultats suggèrent que sa meilleure capacité d’apprentissage compense partiellement le risque de surapprentissage.

---

## 6. Remarque importante sur l’époque optimale et les métriques exportées

Les fichiers `metrics.json` indiquent une `best_epoch` correspondant à l’époque minimisant la perte de validation Huber :

| Expérience | Époque indiquée comme optimale |
|---|---:|
| E1 | 21 |
| E2 | 31 |
| E3 | 22 |
| E4 | 37 |

Cependant, les valeurs contenues dans les fichiers `metrics.json` correspondent aux métriques de la dernière époque enregistrée dans les fichiers `history.csv`, et non aux métriques affichées à l’époque `best_epoch`.

Par exemple :

- pour E3, l’époque 22 présente une perte de validation minimale et un R² de validation d’environ 0,748 ;
- les métriques exportées dans `metrics.json` correspondent à la dernière époque, avec un R² de validation de 0,654 ;
- pour E4, l’époque 37 possède la perte de validation minimale, tandis que les métriques exportées correspondent à la dernière époque d’entraînement.

Cette différence doit être clarifiée avant la version finale des résultats. Deux situations sont possibles :

1. les prédictions de test ont été produites avec le modèle de la dernière époque ;
2. ou les prédictions ont été produites avec le meilleur checkpoint, mais les fichiers d’historique et de métriques n’ont pas été exportés de manière cohérente.

Pour une évaluation finale rigoureuse, les prédictions train, validation et test doivent être générées à partir du même checkpoint, idéalement `best_model.pt`, sélectionné selon la perte de validation minimale. Les valeurs présentées dans ce chapitre correspondent aux fichiers CSV et JSON actuellement exportés.

---

## 7. Analyse des prédictions par rapport aux observations

### 7.1 Comportement général

Les graphiques et fichiers `predictions_train.csv`, `predictions_val.csv` et `predictions_test.csv` montrent une relation globale positive entre les observations et les prédictions : les valeurs élevées de jaunissement tendent à recevoir des prédictions plus élevées que les valeurs faibles.

Cependant, la dispersion autour de la droite idéale `y_pred = y_true` reste importante. Cela est cohérent avec les R² de test compris entre 0,506 et 0,674 : les modèles expliquent une part substantielle de la variabilité, mais certaines observations demeurent difficiles à prédire.

### 7.2 Prédiction des faibles niveaux de jaunissement

Les modèles prédisent généralement correctement les très faibles valeurs, proches de 0 à 5. Les prédictions restent souvent comprises entre 1 et 5 pour ces observations.

Toutefois, certaines prédictions sont positives lorsque la valeur observée est nulle ou très faible. Cette tendance peut être expliquée par :

- l’absence de contrainte stricte imposant une sortie minimale égale à zéro ;
- la difficulté à distinguer les niveaux très faibles de jaunissement ;
- la présence potentielle de bruit dans les observations ou les images.

Les modèles peuvent également produire occasionnellement des valeurs négatives ou des valeurs légèrement supérieures à la plage théorique de la cible, comme certaines valeurs supérieures à 100. Cela indique que la sortie de régression n’est pas bornée. Une contrainte de sortie ou un post-traitement dans l’intervalle `[0, 100]` pourrait être envisagé si cela est cohérent avec la définition physique de l’indice de jaunissement.

### 7.3 Prédiction des niveaux intermédiaires

Les niveaux intermédiaires, notamment entre 10 et 60, constituent une zone où les erreurs restent fréquentes. Les modèles peuvent sous-estimer ou surestimer certaines observations de manière importante.

Par exemple, dans le jeu de test :

- certaines observations proches de 20 ou 25 sont prédites à des niveaux très faibles ;
- certaines observations faibles peuvent être prédites à des niveaux nettement plus élevés ;
- des observations proches de 50 à 70 peuvent être sous-estimées selon la parcelle considérée.

Cette dispersion indique que les signatures spectrales ou spatiales associées aux niveaux intermédiaires sont moins facilement séparables que celles des niveaux très faibles ou très élevés.

### 7.4 Prédiction des niveaux élevés

Les observations élevées, particulièrement celles proches de 80 à 100, sont souvent sous-estimées. Plusieurs valeurs observées à 100 reçoivent des prédictions comprises approximativement entre 50 et 85 selon l’expérience et la parcelle.

Cette tendance à la sous-estimation des valeurs extrêmes peut être liée à plusieurs facteurs :

- un déséquilibre dans la distribution des niveaux de jaunissement ;
- une saturation des signatures spectrales pour les niveaux de jaunissement élevés ;
- une variabilité importante entre les parcelles ;
- une tendance des modèles de régression à produire des valeurs proches de la moyenne lorsque les exemples sont ambigus.

Malgré cette tendance, E4 reproduit globalement mieux les niveaux élevés que les autres expériences pour plusieurs parcelles du jeu de test, ce qui contribue à son RMSE plus faible.

### 7.5 Exemples d’erreurs importantes

Les erreurs les plus importantes sont observées sur certaines séquences de mesures appartenant à une même parcelle. Par exemple, pour certaines observations de test, le modèle prédit correctement la progression globale vers des niveaux élevés, mais ne reproduit pas précisément les valeurs individuelles.

Des erreurs fortes restent présentes lorsque :

- une valeur observée intermédiaire ou élevée est associée à une prédiction très faible ;
- une faible valeur observée reçoit une prédiction élevée ;
- le comportement d’une parcelle diffère de celui appris pendant l’entraînement.

Ces observations confirment que le modèle capte la tendance générale du jaunissement, mais reste sensible aux particularités locales et aux cas atypiques.

---

## 8. Conclusion

Les résultats montrent que les quatre modèles apprennent une relation significative entre les images Sentinel-2 et l’indice de jaunissement. Les R² de test, compris entre 0,506 et 0,674, démontrent que les informations spectrales et spatiales disponibles permettent d’expliquer une part importante de la variabilité de la cible.

L’expérience **E4**, combinant les images super-résolues et les indices spectraux, est retenue comme la meilleure configuration. Elle obtient :

- la meilleure MAE de test : **17,50** ;
- le meilleur RMSE de test : **22,84** ;
- le meilleur R² de test : **0,674**.

La super-résolution améliore particulièrement les résultats lorsque les indices spectraux sont utilisés. Les bandes brutes restent plus efficaces que les indices à basse résolution, tandis qu’à haute résolution, les indices spectraux permettent d’obtenir les meilleures performances globales.

Les principales limites observées concernent la dispersion des prédictions autour de la valeur réelle, la sous-estimation fréquente de certaines valeurs élevées et un surapprentissage modéré pour les modèles les plus performants. Une sélection stricte du meilleur checkpoint de validation, ainsi qu’une analyse complémentaire des résidus par niveau de jaunissement et par parcelle, permettraient de renforcer la robustesse de l’évaluation finale.