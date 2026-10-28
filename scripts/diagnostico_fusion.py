"""Exporta diagnósticos de la fusión usando solo resultados existentes de train."""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from pcr_party import fusion, metricas
from pcr_party.recursos_app import modelo_fusion, RECURSOS
import pandas as pd


def main():
    interna = fusion.RUNS / "cv_interna_fusion"
    externa = fusion.RUNS / fusion.CV_EXTERIOR
    necesarias = [interna / "plan.json", externa / f"oof_{fusion.BRAZO_IMAGEN}.csv",
                  interna / "evaluacion" / "resumen.json", interna / "evaluacion" / "oof_fusion.csv"]
    faltan = [str(p) for p in necesarias if not p.exists()]
    if faltan:
        raise SystemExit("Faltan resultados de train; ejecutar en el equipo de entrenamiento:\n" + "\n".join(faltan))
    plan = json.loads((interna / "plan.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(necesarias[1]).set_index("patient_id")
    clinica = fusion.tabla_clinica("train").join(oof[["fold"]], how="inner")
    if len(clinica) != len(oof):
        raise RuntimeError("Las pacientes clínicas y las OOF no coinciden")
    diag = []
    for k in sorted(clinica["fold"].unique()):
        ajuste = clinica[clinica["fold"] != k].copy()
        validacion = clinica[clinica["fold"] == k].copy()
        ajuste["imagen"] = metricas.logit(fusion.imagen_interna(interna, k, len(plan["semillas"])).reindex(ajuste.index))
        validacion["imagen"] = metricas.logit(oof["p"].reindex(validacion.index))
        if ajuste["imagen"].isna().any() or validacion["imagen"].isna().any():
            raise RuntimeError(f"Faltan puntuaciones de imagen en fold {k}")
        submodelos = fusion.ajustar_submodelos(ajuste, False)
        diag.append(fusion.diagnostico_ajuste(submodelos, ajuste, validacion, k))
    resumen_path = interna / "evaluacion" / "resumen.json"
    resumen = json.loads(resumen_path.read_text(encoding="utf-8"))
    resumen["diagnostico_generalizacion"] = diag
    resumen_path.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    recurso = modelo_fusion()
    destino = RECURSOS / "modelo_fusion.json"
    destino.write_text(json.dumps(recurso, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Exportados {len(diag)} folds, PR, calibración y parámetros a {destino}")


if __name__ == "__main__":
    main()
