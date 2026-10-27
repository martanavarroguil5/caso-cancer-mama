"""PartyNet: CNN 2D propia, desde pesos aleatorios, con cabezas auxiliares.

El tronco reproduce el diseño de V007 (la mejor media de CV solo imagen):
fases PRE/EARLY/LATE más tres diferencias firmadas, tres vistas centrales
(256, 128 y 64 px reescaladas a 128) y un backbone compartido. La novedad es
el aprendizaje multitarea: durante el entrenamiento la misma representación
predice también HR, HER2 y log-volumen tumoral. En inferencia solo entran
píxeles y sale un único logit de pCR.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

AUXILIARES = ("HR", "HER2", "log_volumen")
CONFIG_DEFECTO = {
    "anchos": [24, 48, 96, 160],
    "oculto": 64,
    "dropout": 0.35,
    "dropout_espacial": 0.10,
    "escalas": [256, 128, 64],
    "lado": 128,
}


class Bloque(nn.Module):
    """Dos convoluciones 3×3 con BatchNorm y ReLU, seguidas de MaxPool 2×2."""

    def __init__(self, entrada: int, salida: int):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Conv2d(entrada, salida, 3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            nn.Conv2d(salida, salida, 3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.capas(x)


class PartyNet(nn.Module):
    def __init__(self, anchos=(24, 48, 96, 160), oculto=64, dropout=0.35,
                 dropout_espacial=0.10, escalas=(256, 128, 64), lado=128, cabezas=1):
        super().__init__()
        self.config = {"anchos": list(anchos), "oculto": oculto, "dropout": dropout,
                       "dropout_espacial": dropout_espacial, "escalas": list(escalas),
                       "lado": lado}
        if cabezas != 1:
            self.config["cabezas"] = cabezas
        if any(t > 256 or t < 8 for t in escalas):
            raise ValueError("Cada vista central debe medir entre 8 y 256 píxeles")
        self.escalas = tuple(escalas)
        self.lado = lado
        self.backbone = nn.Sequential(*[
            Bloque(entrada, salida)
            for entrada, salida in zip((6, *anchos[:-1]), anchos)
        ])
        self.dropout_espacial = nn.Dropout2d(dropout_espacial)
        self.tronco = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(anchos[-1] * 2 * len(escalas), oculto),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Con varias cabezas (una por subtipo) el sesgo es común: la cabeza
        # elegida solo cambia los pesos, no el efecto principal del grupo.
        self.cabeza_pcr = nn.Linear(oculto, cabezas, bias=cabezas == 1)
        if cabezas != 1:
            self.sesgo_pcr = nn.Parameter(torch.zeros(1))
        self.cabeza_aux = nn.Linear(oculto, len(AUXILIARES))
        self.apply(self._inicializar)

    @property
    def sesgo(self) -> torch.Tensor:
        return self.cabeza_pcr.bias if self.cabeza_pcr.bias is not None else self.sesgo_pcr

    def _pcr(self, z: torch.Tensor) -> torch.Tensor:
        salida = self.cabeza_pcr(z)
        return salida if self.cabeza_pcr.bias is not None else salida + self.sesgo_pcr

    @staticmethod
    def _inicializar(capa: nn.Module) -> None:
        if isinstance(capa, nn.Conv2d):
            nn.init.kaiming_normal_(capa.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(capa, nn.Linear):
            nn.init.kaiming_uniform_(capa.weight, nonlinearity="relu")
            if capa.bias is not None:
                nn.init.zeros_(capa.bias)
        elif isinstance(capa, nn.BatchNorm2d):
            nn.init.ones_(capa.weight)
            nn.init.zeros_(capa.bias)

    @staticmethod
    def canales(x: torch.Tensor) -> torch.Tensor:
        """[N,3,H,W] → [N,6,H,W]: fases y realce EARLY−PRE, LATE−PRE, EARLY−LATE."""
        pre, early, late = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return torch.cat((x, early - pre, late - pre, early - late), dim=1)

    def vistas(self, x: torch.Tensor) -> torch.Tensor:
        """[N,3,256,256] → [N,S,3,lado,lado] con recortes centrados."""
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError(f"Se espera [N,3,256,256], recibido {tuple(x.shape)}")
        vistas = []
        for tamano in self.escalas:
            inicio = (256 - tamano) // 2
            vista = x[:, :, inicio:inicio + tamano, inicio:inicio + tamano]
            if tamano != self.lado:
                vista = F.interpolate(vista, size=(self.lado, self.lado), mode="bilinear",
                                      align_corners=False, antialias=tamano > self.lado)
            vistas.append(vista)
        return torch.stack(vistas, dim=1)

    def representacion(self, x: torch.Tensor) -> torch.Tensor:
        vistas = self.vistas(x)
        n, s = vistas.shape[:2]
        z = self.dropout_espacial(self.backbone(self.canales(vistas.flatten(0, 1))))
        z = torch.cat((z.mean(dim=(2, 3)), z.amax(dim=(2, 3))), dim=1)
        return self.tronco(z.reshape(n, s * z.shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contrato de inferencia: un único logit de pCR por corte, [N,1] ([N,K] con K cabezas)."""
        return self._pcr(self.representacion(x))

    def forward_todo(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logit de pCR [N,K] y logits/regresión auxiliares [N,3] (solo entrenamiento)."""
        z = self.representacion(x)
        return self._pcr(z), self.cabeza_aux(z)


def numero_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
