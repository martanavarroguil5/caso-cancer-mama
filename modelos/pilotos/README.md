# Paquetes de pilotos

Estos paquetes conservan evidencia experimental con un número de folds inferior
al de las versiones finales. No forman parte de `modelos/catalogo.json`, no
reutilizan números `vNNN` y no sustituyen el modelo activo de la aplicación.

| Piloto | Pesos | Resultado |
|---|---:|---|
| [p001_antioverfit_20261021](p001_antioverfit_20261021/) | 16: control/candidato × dos BCE × dos semillas × dos folds | AUC media ponderada 0,5338 → 0,5571; filtro exploratorio cumplido, mejora no confirmada y mínimo 0,60 no alcanzado |

[Informe completo](../../resultados/16_informe_piloto_antioverfit_20261021/INFORME_RESULTADOS.md).
Los pesos contienen solo configuración y estado de inferencia, con código
congelado, hashes y comprobación aislada. Se mantienen separados de los
checkpoints de optimización y del dataset. V002 continúa activo.
