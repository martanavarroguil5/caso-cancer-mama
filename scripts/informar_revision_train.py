"""Cierra auditoría y protocolo sin entrenar ni modificar datos."""
from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess

import pandas as pd

from cancer_mama import revision_train as r
from cancer_mama import entrenamiento as base


def main():
    out=r.OUTPUT;a=json.loads((out/'auditoria.json').read_text())
    protocol_dir=out/'protocolo_cinco_folds';p=json.loads((protocol_dir/'protocolo.json').read_text())
    metrics=pd.read_csv(out/'avisos_para_revision.csv');all_metrics=pd.read_csv(out/'cortes_auditados.csv')
    files=pd.read_csv(out/'inventario_png.csv')
    selected=pd.read_csv(out/'muestras_revision_visual.csv')
    repeated=[]
    for sid,g in files.groupby('sample_id'):
        for digest,same in g.groupby('pixel_sha256'):
            if len(same)>1:repeated.append({'patient_id':g.patient_id.iloc[0],'sample_id':sid,'identical_phases':same.phase.tolist(),'pixel_sha256':digest})
    r.write(out/'fases_repetidas_confirmadas.json',repeated)
    visual=[]
    notes={
        'ISPY1_1139':('alta','EARLY/LATE idénticas; confirmado por hash e índices, no solo apariencia'),
        'ISPY1_1193':('alta','LATE tiene una escala extrema; no se identifica la causa con los PNG'),
        'ISPY1_1069':('alta','Diferencias marcadas de contorno PRE/post; revisar correspondencia/registro'),
        'ISPY1_1077':('media','La media global disminuye; esto no demuestra por sí solo inversión de fases'),
        'ISPY2-446263':('media','Contornos PRE/post diferentes; revisar registro, sin atribuir causa'),
        'ISPY1_1217':('media','Diferencias aparentes de contorno/posición; revisar, sin corregir')}
    for row in selected.itertuples():
        priority,note=notes.get(row.patient_id,('media','Cambios de realce y/o contorno; la traslación automática no queda confirmada visualmente'))
        visual.append({'patient_id':row.patient_id,'sample_id_reviewed':row.sample_id,'priority':priority,'technical_visual_note':note,
            'scope':'Revisión técnica exploratoria, no dictamen radiológico ni autorización de exclusión'})
    pd.DataFrame(visual).to_csv(out/'revision_visual.csv',index=False)
    counts=metrics.groupby('patient_id').size();motion_patients=metrics[metrics['flags'].str.contains('posible_movimiento')].patient_id.nunique()
    cohort_rows='\n'.join(f'| {c["cohort"]} | {c["patients"]} | {c["pcr"]} | {c["prevalence"]:.2%} | {c["median_energy_center_128"]:.2%} |' for c in a['cohorts'])
    role_rows='\n'.join(f'| {f["fold"]} | {f["counts"]["fit"]} | {f["counts"]["selection"]} | {f["counts"]["outer"]} | {f["pos_weight"]:.4f} |' for f in p['folds'])
    report=f'''# Auditoría completa de train y protocolo con mayor aprovechamiento de pacientes

{r.now()}. Alcance autorizado: auditar y preparar; **no entrenar, corregir datos ni activar modelos**.

## Resultado de la auditoría

Se abrieron y validaron los **{a['opened_png']:,} PNG** de **{a['cuts']:,} cortes y {a['patients']:,} pacientes de train**. Se verificaron formato PNG, escala de grises L, uint8, tamaño256×256, rango, nombres de fases, correspondencia paciente/corte/z, integridad de lectura y coherencia de los CSV. Los promedios de intensidad se acumulan en float64.

- Archivos ausentes/corruptos o de formato incorrecto: **0**.
- PNG no indexados en train: **0**.
- Discrepancias de etiqueta/cohorte entre índices: **0**.
- Pacientes cruzados entre los splits públicos disponibles en metadata: **0**. No se han abierto imágenes ni usado etiquetas de test/validation para el análisis.
- Imágenes exactamente iguales entre pacientes de train: **0**; tripletas repetidas entre pacientes o cortes de una misma paciente: **0**.
- Avisos heurísticos: **101 cortes de18 pacientes**, {18/1097:.2%} de las pacientes. Se inspeccionó una muestra representativa de cada una de esas18, además de ejemplos por cohorte/clase. Un aviso no equivale a un dato erróneo.

El inventario por PNG y todas las métricas por corte permanecen locales; los resúmenes, incidencias, hashes y figuras son revisables. No se ha comparado el contenido de train contra píxeles de test/validation: por tanto no se afirma ausencia absoluta de duplicados entre esos conjuntos.

## Incidencia confirmada: EARLY y LATE repetidas

**ISPY1_1139**, paciente de train: EARLY y LATE son **idénticas píxel a píxel en sus diez cortes**. El CSV indica `n_times=3`, `pre=0`, `post_early=2`, `post_late=2`. Los nombres cumplen el esquema, pero no representan tres adquisiciones diferentes.

La evidencia está en fases_repetidas_confirmadas.json, con los diez hashes y muestras. No se ha reasignado otra adquisición, duplicado una fase distinta ni excluido la paciente. Con estos PNG no puede recuperarse la adquisición intermedia correcta. La política sobre este caso debe aclararse con el responsable del dataset; se entrega un borrador de consulta, sin enviarlo. Una paciente no explica por sí sola el bajo AUC global.

## Avisos que requieren interpretación

- **ISPY1_1193**: LATE extremadamente brillante en los diez cortes; el origen de la diferencia de escala no se determina con los PNG. No se normaliza cada fase por separado.
- **ISPY1_1069 / ISPY1_1077**: realce medio de foreground negativo al promediar sus cortes. Se marcaron16 cortes con media EARLY−PRE<−0,01. Esto no prueba inversión de fases: la media de foreground no mide exclusivamente tumor y es sensible a desplazamientos y adquisición.
- **{motion_patients} pacientes** presentan alguna sugerencia de traslación. PRE/EARLY:16 cortes; EARLY/LATE:23; PRE/LATE:51. Son conteos por comparación, con solapamientos. No se suman como pacientes diferentes.

Se usó correlación normalizada de magnitudes de gradiente, suavizado y resolución4px, con búsqueda±24px. Los avisos exigen desplazamiento≥8px, ganancia de correlación≥0,10 y correlación≥0,50, sin máximo en el borde. Los tests recuperan traslaciones sintéticas conocidas y no detectan movimiento al aplicar solo un cambio afín de intensidad.

Un máximo de esa métrica puede deberse al realce; no valida una corrección anatómica. Los máximos en el límite se conservan como ambiguos, no como alineados. Tampoco se detectan necesariamente rotaciones o deformaciones. En algunos ejemplos se aprecian diferencias de contorno PRE/post, pero la revisión es técnica y no sustituye evaluación radiológica.

![Ejemplos por cohorte y clase](01_ejemplos_cohortes.png)
![Casos para revisión](02_avisos_revision.png)

## Distribución y posición del realce

| Cohorte | Pacientes | pCR | Prevalencia | Mediana de energía de realce positivo en centro128 |
|---|---:|---:|---:|---:|
{cohort_rows}

La media EARLY supera PRE en **{a['patient_mean_early_gt_pre_fraction']:.2%}** de las pacientes. Esto describe el conjunto; no certifica el instante fisiológico real de cada PNG.

La fracción de energía en centro128 **no es la fracción de tumor incluida**: vasos, corazón y parénquima también realzan. Los límites tumorales del CSV están en coordenadas del volumen fuente y no hay una transformación documentada al PNG que permita usarlos como máscaras locales. Por ello **no se concluye que el tumor quede fuera del recorte ni se cambia la arquitectura por esa cifra**. V007 ya conserva una vista global256 además de sus centros128/64.

![Distribuciones exploratorias](03_distribuciones.png)

## Protocolo preparado: cinco folds sin un bloque adicional de calibración

Se conservan los folds originales y la selección interna estratificada por cohorte/clase. Los roles concretos y sus hashes ya están guardados, sin entrenar.

| Fold exterior | Fit | Selección de época/pérdida/política | Comprobación exterior | N0/N1 en fit |
|---|---:|---:|---:|---:|
{role_rows}

Cada paciente recibe exactamente una predicción exterior por receta, producida por la pareja que no la utilizó para fit ni selección. **Nunca se aplica el ensemble de los cinco folds a sus propias pacientes de train para llamarlo OOF**.

Respecto al último piloto, fit pasa de614 a746 en fold0 y de613 a745 en fold3: **132 pacientes adicionales por red**, aproximadamente21,5 %. Ya no existe un bloque de132 pacientes dedicado solo a calibración. Las1.097 participan en el desarrollo por validación cruzada; eso no significa que todas entrenen cada red.

El compromiso es explícito: **selección comparte época, pérdida, calibración y umbral**. Sus resultados son optimistas y se identifican como selección, no como una comprobación independiente. El fold exterior queda fuera de esas decisiones y evalúa el procedimiento completo. Se conserva un exterior precisamente para estimar ese efecto. Esto aprovecha más datos, pero no elimina todo el sesgo ni garantiza mejor AUC. Véase [Collins et al., BMJ2024](https://www.bmj.com/content/384/bmj-2023-074819) sobre la pérdida de información por fragmentar muestras pequeñas; no implica que ese artículo avale una receta concreta de CNN.

### Decisiones ya explicitadas

- Referencia: CNNMultiescalaCinetica, **593.521 parámetros**, desde cero, mismos recortes, filtros y dropout de V007. Sin variables clínicas como entrada.
- Ambas BCE, semillas42/2026, cinco folds: **20 ejecuciones de referencia si se autoriza después**. No hay otra matriz en ejecución.
- LR0,0003, WD0,01, lote32, máximo40, mínimo8, paciencia8 y min_delta0,001; receta modal de decisiones internas históricas, no elegida mirando test.
- Contribución igual por paciente con BCE por corte y peso1/n_cortes; pos_weight por pacientes de fit.
- Época elegida solo en selección. La pérdida se elige con AUC del ensemble de semillas de selección; empate a favor de normal.
- Agregación fija: media de probabilidades de todos los cortes y de ambas semillas. AUC principal: media±DE de los cinco folds; agrupadas cruda/calibrada como secundarias.
- PlattC1 con pendiente no negativa, ajustado en selección sobre la misma pareja. Umbral primario: máxima sensibilidad con especificidad≥95 % en selección; secundario macro-F1, sin escoger después entre políticas mirando exterior.
- Las cinco elecciones y políticas se congelarán antes de evaluar exterior. Se informarán curvas, brechas en la época elegida, AP, calibración, sensibilidad/especificidad, F1 por clase y comparación con siempre no pCR.

## Futuro modelo final: diseño, no entrenamiento

La partición final ya prefijada contiene **932 pacientes para fit y165 para selección/política**, estratificadas por cohorte/clase. Si se llega a autorizar una receta cerrada, serían cuatro entrenamientos: normal/ponderada por dos semillas, con elección y política solo en esas165.

El paquete resultante sería **una pareja fija de dos semillas**. Su calibración y umbral se ajustarían con las predicciones de esa misma pareja y media de cortes. Se evita trasladar una calibración aprendida con puntuaciones de dos modelos a una media de diez. No se sustituyen ahora V007/V008 ni la app.

La CV evalúa el procedimiento con fit745–746, no garantiza el rendimiento de los futuros pesos con932. La selección final no es una evaluación independiente. El test histórico permanece sin volver a consultar; una confirmación nueva corresponde al profesor con su validación privada y el procedimiento congelado. No se promete alcanzar0,60.

## Condiciones antes de cualquier entrenamiento

1. Aclarar la fase repetida de ISPY1_1139 y la política autorizada sobre la incidencia; revisar la escala extrema de ISPY1_1193.
2. Mantener trazabilidad de cualquier corrección que llegase a autorizarse; sin cambios individuales según etiqueta, error del modelo o test.
3. Registrar una hipótesis de un solo cambio, comparador y criterio de ampliación antes de entrenar. Este protocolo prepara la evaluación, no elige una arquitectura nueva.
4. Nueva autorización de entrenamiento: **en este trabajo no se ha iniciado ninguno**.

## Evidencia e integridad

- Código de auditoría/protocolo: commit local96a98cf3056062c07572060c96728340548c1320.
-99 pruebas pasaron: caminos cerrados a train, formatos, etiquetas coherentes, recuperación de traslación, invariancia a intensidad afín, cinco folds disjuntos y ausencia de entrenamiento.
- Árbol de datos train, SHA-256 agregado: `{a['data_tree_sha256']}`.
- Metadatos, PNG y versiones históricas se conservaron; modelos y aplicación no se modificaron. No se hicieron push ni ramas nuevas.
- El inventario de32.835 archivos y las métricas exhaustivas por corte permanecen locales; los hashes, resúmenes e incidencias permiten localizar la evidencia.

## Revisión visual de las18 pacientes señaladas

![Revisión1](04_revision_avisos_1.png)
![Revisión2](04_revision_avisos_2.png)
![Revisión3](04_revision_avisos_3.png)
'''
    (out/'INFORME_AUDITORIA.md').write_text(report)
    draft='''# Incidencias de train para consultar al responsable del dataset

Borrador preparado; no se ha enviado ningún mensaje ni modificado ningún dato.

Al revisar únicamente el train de BreastDCEDL MinCrop encontramos:

1. **ISPY1_1139:** los diez EARLY y LATE respectivos son idénticos píxel a píxel. En patients.csv constan n_times=3, pre=0, post_early=2 y post_late=2. ¿La repetición es intencionada? ¿Puede facilitarse la exportación correcta o indicarse una política común para este caso? No podemos recuperar otra adquisición a partir de los PNG suministrados.
2. **ISPY1_1193:** LATE tiene intensidad media de foreground próxima a0,955 en los diez cortes, frente a PRE≈0,046 y EARLY≈0,059. Solicitamos confirmar si corresponde a la adquisición/escala esperada; no hemos aplicado normalización independiente ni excluido el caso.
3. Existen casos con cambios de contorno entre fases (por ejemplo ISPY1_1069). La correlación de gradientes es solo una alarma, no confirma error de correspondencia ni justifica registrar imágenes automáticamente. ¿Hay información sobre el alineamiento aplicado al generar MinCrop?

No hemos utilizado test ni validación privada para detectar o decidir estas incidencias. La mayoría de train supera los controles de formato y correspondencia. Son incidencias concretas, no una explicación demostrada del bajo rendimiento global.

Evidencia: fases_repetidas_confirmadas.json y las figuras del informe de auditoría.
'''
    (out/'INCIDENCIAS_PARA_PROFESOR.md').write_text(draft)
    doc=r.PROJECT_ROOT/'docs/AUDITORIA_TRAIN_Y_PROTOCOLO.md'
    doc.write_text('''# Auditoría de train y preparación de cinco folds

25/10/2026. Se han ejecutado únicamente los pasos de auditoría y preparación, sin entrenamiento ni corrección de datos.

-32.835 PNG,10.945 cortes,1.097 pacientes de train; sin archivos ausentes/corruptos ni imágenes idénticas entre pacientes de train.
-18 pacientes con avisos; confirmado EARLY=LATE en los diez cortes de ISPY1_1139, con índices de origen2/2. No se excluye ni se reconstruye una fase sin autorización del responsable.
-Datos y versiones históricas conservados. Test/validation no se abren para el análisis; no se afirma ausencia de duplicados por píxeles con esos conjuntos.
-Cinco folds originales con745–746 fit,132 selección y219–220 exterior. Calibración/umbral comparten selección; solo el exterior evalúa el procedimiento completo.
-Futuro paquete de una pareja fija,932 fit y165 selección/política; no se entrena ni activa ahora. La CV no valida por sí sola esos futuros pesos.
-99 pruebas; código de auditoría96a98cf. Todo permanece en main local, sin push.

[Informe completo, figuras y límites](../resultados/26_auditoria_train_protocolo_20261025/INFORME_AUDITORIA.md).
[Incidencias para consultar al profesor](../resultados/26_auditoria_train_protocolo_20261025/INCIDENCIAS_PARA_PROFESOR.md).
[Protocolo ejecutable preparado](../resultados/26_auditoria_train_protocolo_20261025/protocolo_cinco_folds/protocolo.json).
''')
    for path in [out/'INFORME_AUDITORIA.md',out/'INCIDENCIAS_PARA_PROFESOR.md',doc]:
        text=path.read_text()
        text=re.sub(r'(?m)^-(?=\S)','- ',text)
        text=re.sub(r'\b(de|con|por|en|las|los|tamaño|resolución|búsqueda|lote|máximo|mínimo|semillas)(?=\d)',r'\1 ',text)
        text=text.replace('PlattC1','Platt C=1').replace('Protocolo ejecutable preparado','Protocolo y roles preparados')
        path.write_text(text)
    r.write(out/'condiciones_para_entrenar.json',{'prepared_at':r.now(),'training_authorized_this_step':False,'training_started':False,
        'source_case_requiring_policy':'ISPY1_1139: EARLY y LATE repetidas; sin corrección/exclusión automática',
        'required':['Aclarar política de datos','Elegir hipótesis única y protocolo de comparación','Autorización posterior para entrenar'],
        'audit_sha256':base.sha256(out/'auditoria.json'),'protocol_sha256':base.sha256(protocol_dir/'protocolo.json')})
    # Comprobaciones independientes sobre roles y metadatos, sin cargar imágenes.
    outer=[]
    for item in p['folds']:
        roles=pd.read_csv(protocol_dir/f'roles_fold_{item["fold"]}.csv')
        assert len(roles)==1097 and not roles.patient_id.duplicated().any()
        assert roles.loc[roles.role.eq('outer'),'fold'].eq(item['fold']).all()
        assert base.sha256(protocol_dir/f'roles_fold_{item["fold"]}.csv')==item['sha256']
        outer.extend(roles.loc[roles.role.eq('outer'),'patient_id'])
    assert len(outer)==len(set(outer))==1097
    assert base.sha256(protocol_dir/'protocolo.json')==(protocol_dir/'protocolo.json.sha256').read_text().strip()
    assert base.sha256(protocol_dir/'roles_modelo_final.csv')==p['final_model']['split_sha256']
    for name,digest in json.loads((out/'versiones_antes.json').read_text()).items():assert base.sha256(r.PROJECT_ROOT/name)==digest,name
    for name,digest in a['metadata_sha256'].items():assert base.sha256(r.DATA_DIR/'metadata'/name)==digest,name
    for name,digest in a['source_sha256'].items():assert base.sha256(r.PROJECT_ROOT/name)==digest,name
    r.write(out/'verificacion_final.json',{'at':r.now(),'png_checked':32835,'patient_oof_roles_verified':1097,'training_started':False,
        'data_modified':False,'versions_unchanged':True,'test_images_opened':0,'private_images_opened':0,'tests':99,
        'confirmed_repeated_phases':len(repeated),'audit_code_commit':'96a98cf3056062c07572060c96728340548c1320',
        'report_generator_sha256':base.sha256(Path(__file__))})
    inventory={str(f.relative_to(out)):base.sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='indice_entrega.json'}
    r.write(out/'indice_entrega.json',{'at':r.now(),'files':inventory,'local_only':['inventario_png.csv','cortes_auditados.csv']})
    print(json.dumps({'report':str(out/'INFORME_AUDITORIA.md'),'training_started':False,'verified':True},ensure_ascii=False))


if __name__=='__main__':main()
