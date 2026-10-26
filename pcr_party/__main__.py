"""CLI: python -m pcr_party {cache,humo,cv,resumen,final,predecir,evaluar-test}."""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np


def _humo(args) -> None:
    """Prueba de extremo a extremo en GPU con pocas pacientes de train y dos épocas."""
    import torch

    from . import datos
    from .cv import pixeles_en_dispositivo
    from .entrenar import RECETA, cargar_modelo, entrenar, predecir
    from .modelo import PartyNet, numero_parametros

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indice = datos.cargar_indice("train")
    pacientes = datos.tabla_pacientes(indice)
    roles = datos.roles_fold(pacientes, 0, RECETA["fraccion_seleccion"])
    elegidas = set(roles.groupby("rol", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), args.pacientes), random_state=0))["patient_id"])
    indice = indice[indice["patient_id"].isin(elegidas)].reset_index(drop=True)
    rol = indice["patient_id"].map(roles.set_index("patient_id")["rol"]).to_numpy()
    inicio = time.time()
    pixeles = pixeles_en_dispositivo(indice, dispositivo)
    print(f"{len(indice)} cortes cargados en {time.time() - inicio:.1f} s; "
          f"PartyNet {numero_parametros(PartyNet(**RECETA['modelo'])):,} parámetros")
    receta = deepcopy(RECETA)
    receta["epocas_min"] = 1
    receta["pacientes_diagnostico_fit"] = 30
    for lambda_aux in (0.0, 1.0):
        resultado = entrenar(receta, pixeles, indice, np.flatnonzero(rol == "fit"),
                             np.flatnonzero(rol == "seleccion"), 42, "ponderada",
                             lambda_aux, epocas=args.epocas)
        modelo = cargar_modelo(receta, resultado.estado, dispositivo)
        p, aux = predecir(modelo, pixeles, np.flatnonzero(rol == "exterior"))
        assert np.isfinite(p).all() and np.isfinite(aux).all()
        print(f"λ={lambda_aux}: {resultado.segundos:.1f} s, memoria {resultado.memoria_max_mb:.0f} MB, "
              f"época elegida {resultado.epoca}, p exterior ∈ [{p.min():.3f}, {p.max():.3f}]")


def main(argv=None) -> None:
    analizador = argparse.ArgumentParser(prog="python -m pcr_party")
    sub = analizador.add_subparsers(dest="orden", required=True)

    sub.add_parser("cache", help="Lee y valida todos los PNG de train y crea la caché uint8")

    humo = sub.add_parser("humo", help="Prueba corta en GPU")
    humo.add_argument("--pacientes", type=int, default=40, help="Pacientes por rol")
    humo.add_argument("--epocas", type=int, default=2)

    cv = sub.add_parser("cv", help="Validación cruzada por paciente (5 folds)")
    cv.add_argument("--nombre", default="cv_party")
    cv.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    cv.add_argument("--semillas", type=int, nargs="+", default=[42, 2026, 7])
    cv.add_argument("--perdidas", nargs="+", default=["normal", "ponderada"],
                    choices=["normal", "ponderada"])
    cv.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0],
                    help="Peso de las tareas auxiliares; 0 = control sin multitarea")

    resumen = sub.add_parser("resumen", help="Recalcula el resumen de una CV")
    resumen.add_argument("--nombre", default="cv_party")

    final = sub.add_parser("final", help="Entrena el paquete final con todo train")
    final.add_argument("--cv", default="cv_party")
    final.add_argument("--perdida", required=True, choices=["normal", "ponderada"])
    final.add_argument("--lambda-aux", type=float, required=True)
    final.add_argument("--semillas", type=int, nargs="+", default=[42, 2026, 7, 1234, 31415])
    final.add_argument("--nombre", default="party_final")
    final.add_argument("--epocas", type=int, default=None,
                       help="Por defecto, mediana de épocas elegidas en la CV del brazo")

    pred = sub.add_parser("predecir", help="Predice una tripleta PRE/EARLY/LATE")
    pred.add_argument("--modelo", default="modelos/party_final")
    pred.add_argument("--pre", required=True)
    pred.add_argument("--early", required=True)
    pred.add_argument("--late", required=True)

    test = sub.add_parser("evaluar-test", help="Evaluación final ÚNICA de un paquete congelado")
    test.add_argument("--modelo", default="modelos/party_final")
    test.add_argument("--confirmo-modelo-congelado", action="store_true", required=True)
    test.add_argument("--forzar", action="store_true")

    args = analizador.parse_args(argv)
    if args.orden == "cache":
        from . import datos
        indice = datos.cargar_indice("train")
        inicio = time.time()
        pixeles = datos.cargar_imagenes(indice)
        print(f"{pixeles.shape} uint8 en {time.time() - inicio:.1f} s")
    elif args.orden == "humo":
        _humo(args)
    elif args.orden == "cv":
        from .cv import validacion_cruzada
        validacion_cruzada(args.nombre, args.folds, args.semillas, args.perdidas, args.lambdas)
    elif args.orden == "resumen":
        from .cv import RUNS, resumir
        resumir(RUNS / args.nombre)
    elif args.orden == "final":
        from .cv import RUNS
        from .final import entrenar_final
        print(entrenar_final(RUNS / args.cv, args.perdida, args.lambda_aux, args.semillas,
                             args.nombre, args.epocas))
    elif args.orden == "predecir":
        from .inferencia import Paquete, leer_tripleta
        paquete = Paquete(Path(args.modelo))
        p = paquete.probabilidad_cortes(leer_tripleta(args.pre, args.early, args.late))[0]
        print(json.dumps(paquete.decidir(p), indent=2, ensure_ascii=False))
    elif args.orden == "evaluar-test":
        from .inferencia import evaluar_test
        print(json.dumps(evaluar_test(Path(args.modelo), args.forzar), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
