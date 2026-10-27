"""Recursos cacheados que comparten las páginas (paquetes, ejemplos, predicciones e imágenes)."""
import numpy as np
import streamlit as st

import motor


@st.cache_resource(show_spinner="Verificando SHA-256 y cargando las 5 redes…")
def paquete_imagen():
    return motor.cargar_imagen()


@st.cache_resource(show_spinner="Cargando el modelo combinado…")
def paquete_fusion():
    return motor.cargar_fusion()


@st.cache_data(show_spinner=False)
def ejemplos():
    return motor.ejemplos()


@st.cache_data(show_spinner=False)
def cortes_ejemplo(paciente: str):
    return motor.cortes_de_ejemplo(paciente)


@st.cache_data(show_spinner="Prediciendo: 5 redes × 4 reflexiones por corte…")
def prediccion(pixeles: np.ndarray, nombres: tuple[str, ...]) -> motor.Prediccion:
    return motor.predecir_cortes(paquete_imagen(), pixeles, list(nombres))


@st.cache_data(show_spinner="Propagando el corte por la red…")
def recorrido(corte: np.ndarray) -> dict:
    return motor.recorrido(paquete_imagen().modelos[0], corte)


@st.cache_data(show_spinner=False)
def visor(pixeles: np.ndarray, nombres: tuple[str, ...], p_cortes: tuple[float, ...]) -> list[dict]:
    return motor.visor(pixeles, list(nombres), list(p_cortes))


@st.cache_data(show_spinner=False)
def portada() -> dict:
    return motor.portada()


@st.cache_data(show_spinner=False)
def capas_red() -> list[dict]:
    return motor.capas_red()
